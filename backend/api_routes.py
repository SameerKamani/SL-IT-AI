"""
SL-IT-AI FastAPI Routes
- FastAPI endpoints and route handlers
"""
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from config import session_storage, logger
from models import ChatRequest, ChatResponse, TicketRequest, TicketResponse
from agents import (
    extract_user_info_from_history, generate_ticket_artifact,
    load_template_fields, build_ordered_ticket,
    get_template_path_for_issue_type
)
from agents import classify_issue_type_llm, fill_ticket_with_llm_and_fuzzy
from langgraph_workflow import compiled_graph
import os
import shutil

router = APIRouter()

# --- API ROUTES ---
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint that processes user messages"""
    try:
        session_id = request.session_id
        user_message = request.message
        conversation_history = session_storage.get(session_id, [])
        
        print(f"[DEBUG] chat_endpoint: Processing message for session {session_id}")
        print(f"[DEBUG] chat_endpoint: User message: {user_message}")
        print(f"[DEBUG] chat_endpoint: Conversation history length: {len(conversation_history)}")
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})
        
        # Extract user context from conversation
        user_context = extract_user_info_from_history(user_message, conversation_history)
        
        # If user_info is provided in the request, merge it with the extracted context
        if request.user_info:
            user_context.update(request.user_info)
            print(f"[DEBUG] chat_endpoint: Using provided user_info: {request.user_info}")
        else:
            # Use default employee data for testing if no user_info provided
            user_context.update({
                "employee_name": "Employee_4",
                "SL_competency": "VSI H - AI", 
                "floor_information": "2",
                "employee_id": "E004"
            })
            print(f"[DEBUG] chat_endpoint: Using default employee data for testing")
        
        # Prepare state for LangGraph workflow
        state = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "session_id": session_id,
            "context": user_context
        }
        
        # Run the workflow
        result = await compiled_graph.ainvoke(state)
        
        # Extract response from workflow result
        response_text = result.get("response", "I'm sorry, I couldn't process your request.")
        ticket = result.get("ticket", {})
        ticket_artifact = result.get("ticket_artifact", {})
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        # Update session storage
        session_storage[session_id] = conversation_history
        
        print(f"[DEBUG] chat_endpoint: Generated response: {response_text[:100]}...")
        print(f"[DEBUG] chat_endpoint: Ticket created: {bool(ticket)}")
        
        return ChatResponse(
            response=response_text,
            ticket=ticket,
            ticket_artifact=ticket_artifact
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def ticket_endpoint(request: TicketRequest) -> TicketResponse:
    """Dedicated ticket creation endpoint"""
    try:
        session_id = request.session_id
        user_message = request.message
        conversation_history = session_storage.get(session_id, [])
        
        print(f"[DEBUG] ticket_endpoint: Creating ticket for session {session_id}")
        
        # Extract context from conversation
        context = extract_user_info_from_history(user_message, conversation_history)
        
        # Classify issue type
        issue_type = await classify_issue_type_llm(user_message, conversation_history)
        template_path = get_template_path_for_issue_type(issue_type)
        
        # Load template and fill ticket
        template_fields = load_template_fields(template_path)
        ticket = await fill_ticket_with_llm_and_fuzzy(
            template_fields, user_message, conversation_history, context
        )
        
        # Build ordered ticket
        ordered_ticket = build_ordered_ticket(ticket, template_fields)
        
        # Create ticket artifact
        ticket_artifact = generate_ticket_artifact(
            context.get("employee_name", ""),
            context.get("problem_description", "")
        )
        
        print(f"[DEBUG] ticket_endpoint: Created ticket: {ordered_ticket}")
        
        return TicketResponse(
            ticket=ordered_ticket,
            ticket_artifact=ticket_artifact,
            issue_type=issue_type
        )
        
    except Exception as e:
        logger.error(f"Error in ticket endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def session_endpoint(session_id: str) -> Dict[str, Any]:
    """Get session information"""
    try:
        conversation_history = session_storage.get(session_id, [])
        return {
            "session_id": session_id,
            "conversation_history": conversation_history,
            "message_count": len(conversation_history)
        }
    except Exception as e:
        logger.error(f"Error in session endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def clear_session_endpoint(session_id: str) -> Dict[str, str]:
    """Clear session data"""
    try:
        if session_id in session_storage:
            del session_storage[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/ticket_with_attachments")
async def ticket_with_attachments(ticket: str = Form(...), attachments: list[UploadFile] = File(None)):
    print("[DEBUG] ticket_with_attachments endpoint called")
    # Parse ticket JSON
    try:
        ticket_data = json.loads(ticket)
    except Exception as e:
        return {"success": False, "error": f"Invalid ticket JSON: {e}"}
    # Save files (example: save to ./uploads)
    upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    saved_files = []
    if attachments:
        for file in attachments:
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_files.append(file.filename)
    # You can now process ticket_data and saved_files as needed
    return {"success": True, "files": saved_files}

# --- CORS MIDDLEWARE ---
def add_cors_middleware(app: FastAPI):
    """Add CORS middleware to the FastAPI app"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ) 