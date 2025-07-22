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
from rapidfuzz import fuzz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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

# Load MiniLM model (reuse your RAG code if possible)
minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Paths for FAISS index and embeddings
ticket_embeddings_dir = os.path.join(os.path.dirname(__file__), '..', 'Tickets_generated', 'ticket_embeddings')
os.makedirs(ticket_embeddings_dir, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(ticket_embeddings_dir, 'faiss.index')
EMBEDDINGS_PATH = os.path.join(ticket_embeddings_dir, 'ticket_embeddings.npy')
EMBEDDINGS_META_PATH = os.path.join(ticket_embeddings_dir, 'ticket_embeddings_meta.json')

# Load or initialize FAISS index and embeddings
embedding_dim = 384  # MiniLM-L6-v2 output dim
try:
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(EMBEDDINGS_META_PATH, 'r', encoding='utf-8') as f:
        embeddings_meta = json.load(f)
except Exception:
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    embeddings = np.zeros((0, embedding_dim), dtype='float32')
    embeddings_meta = []

def save_faiss_and_meta():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(EMBEDDINGS_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(embeddings_meta, f, ensure_ascii=False, indent=2)

@router.post("/api/ticket_with_attachments")
async def ticket_with_attachments(ticket: str = Form(...), attachments: list[UploadFile] = File(None)):
    print("[DEBUG] ticket_with_attachments endpoint called")
    import uuid
    # Parse ticket JSON
    try:
        ticket_data = json.loads(ticket)
    except Exception as e:
        return {"success": False, "error": f"Invalid ticket JSON: {e}"}
    # Extract requester name (try multiple keys for robustness)
    requester = ticket_data.get("Requester(Required)") or ticket_data.get("employee_name") or ticket_data.get("Requester") or "Unknown"
    requester = str(requester).replace("/", "_").replace("\\", "_")  # sanitize for folder name
    # Generate ticket ID
    ticket_id = ticket_data.get("ticket_id") or str(uuid.uuid4())
    # Prepare save paths
    root_dir = os.path.join(os.path.dirname(__file__), "..", "Tickets_generated")
    requester_dir = os.path.join(root_dir, requester)
    os.makedirs(requester_dir, exist_ok=True)
    ticket_filename = f"ticket_{ticket_id}.json"
    ticket_path = os.path.join(requester_dir, ticket_filename)
    # MiniLM+FAISS duplicate detection
    new_text = (ticket_data.get("Subject(Required)", "") + " " + ticket_data.get("Description", "")).strip()
    new_emb = minilm_model.encode(new_text, normalize_embeddings=True).astype('float32')
    duplicate_found = False
    duplicate_file = None
    threshold = 0.85
    if faiss_index.ntotal > 0:
        D, I = faiss_index.search(new_emb.reshape(1, -1), k=3)
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            if score > threshold:
                duplicate_found = True
                meta = embeddings_meta[idx]
                duplicate_file = meta.get('ticket_file')
                break
    if duplicate_found:
        print("DUPLICATE DETECTED")
        return {"success": False, "duplicate": True, "duplicate_file": duplicate_file, "message": f"A similar ticket already exists: {duplicate_file}"}
    # Save ticket JSON
    with open(ticket_path, "w", encoding="utf-8") as f:
        json.dump(ticket_data, f, ensure_ascii=False, indent=2)
    saved_files = []
    # Save attachments if present
    if attachments:
        attach_dir = os.path.join(requester_dir, f"ticket_{ticket_id}_attachments")
        os.makedirs(attach_dir, exist_ok=True)
        for file in attachments:
            file_path = os.path.join(attach_dir, file.filename)
            with open(file_path, "wb") as f_out:
                shutil.copyfileobj(file.file, f_out)
            saved_files.append(file.filename)
    # Add new embedding to FAISS and persist
    global embeddings
    embeddings = np.vstack([embeddings, new_emb]) if embeddings.shape[0] > 0 else new_emb.reshape(1, -1)
    faiss_index.add(new_emb.reshape(1, -1))
    embeddings_meta.append({"ticket_file": ticket_filename, "requester": requester, "ticket_id": ticket_id})
    save_faiss_and_meta()
    return {"success": True, "ticket_file": ticket_filename, "attachments": saved_files, "requester": requester, "ticket_id": ticket_id}

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