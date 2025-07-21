"""
SL-IT-AI Main Application
- FastAPI app with MCP integration
- Main chat endpoint with LangGraph workflow
"""
import os
import json
import logging
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import aiohttp
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# Import modules
from config import (
    ROOT_DIR, TEMPLATES_DIR, EMPLOYEE_DATA_DIR,
    TEMPLATE_PATH, EMPLOYEE_PATH, ELECTRIC_TEMPLATE_PATH
)
from models import RAGRequest, RAGResponse
from langgraph_workflow import compiled_graph
from agents import extract_user_info_from_history
from policy_rag import PolicyRAGAgent
from api_routes import router as api_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace Redis with in-memory storage (exact from old code)
session_storage = {}  # type: Dict[str, dict]

# RAG agent - use lazy initialization
from config import get_rag_agent

# Initialize FastAPI app
app = FastAPI()

# Add api_routes router
app.include_router(api_router)
print("[DEBUG] api_routes router included in FastAPI app")

# Add timeout configuration
import asyncio

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Add timeout middleware to handle long-running requests"""
    try:
        # Set a reasonable timeout for all requests
        response = await asyncio.wait_for(call_next(request), timeout=60.0)
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"detail": "Request timeout - the operation took too long to complete"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
if os.path.exists(TEMPLATES_DIR):
    app.mount("/static/templates", StaticFiles(directory=TEMPLATES_DIR), name="static-templates")
else:
    logger.warning(f"Templates directory not found at {TEMPLATES_DIR}")

# --- API ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "ok", "message": "IT Helpdesk Chatbot API is running"}

@app.post("/main_rag_chat", operation_id="main_rag_chat", summary="Main RAG agent chat tool")
async def main_rag_chat(request: RAGRequest):
    print("[DEBUG] main_rag_chat: Received conversation_history:")
    print(request.conversation_history)
    print("[DEBUG] main_rag_chat: Received user_message:")
    print(request.message)
    try:
        session_id = request.session_id or str(uuid.uuid4())
        user_message = request.message
        conversation_history = request.conversation_history or []
        context = session_storage.get(session_id, {})
        
        # Use user_info from frontend if provided
        if hasattr(request, "user_info") and request.user_info:
            context.update(request.user_info)
        
        # Extract user info from conversation (exact logic from old code)
        newly_extracted_context = extract_user_info_from_history(user_message, conversation_history)
        if newly_extracted_context.get("employee_name"):
            context["employee_name"] = newly_extracted_context["employee_name"]
        if newly_extracted_context.get("problem_description"):
            context["problem_description"] = newly_extracted_context["problem_description"]
        
        print(f"[DEBUG] Starting LangGraph workflow for message: {user_message}")
        
        # Use LangGraph workflow for proper agent flow (exact from old code)
        from langgraph_workflow import AgentState
        
        state = AgentState(
            messages=[],
            user_message=user_message,
            conversation_history=conversation_history,
            context=context,
            session_id=session_id
        )
        
        # Run the MCP-aware LangGraph workflow using async invoke with timeout
        try:
            print(f"[DEBUG] Starting LangGraph workflow with state: {state}")
            
            # Add timeout protection for the entire workflow
            workflow_task = asyncio.create_task(compiled_graph.ainvoke(state))
            
            try:
                result = await asyncio.wait_for(workflow_task, timeout=45.0)  # 45 second timeout
                print(f"[DEBUG] MCP-aware LangGraph result: {result}")
            except asyncio.TimeoutError:
                print("[ERROR] LangGraph workflow timed out after 45 seconds")
                # Fallback to direct processing
                rag_agent = get_rag_agent()
                dynamic_policy_response = rag_agent.generate_dynamic_policy_response(user_message)
                return RAGResponse(
                    response=dynamic_policy_response,
                    context=context if 'context' in locals() else {}
                )
                
        except Exception as e:
            print(f"[ERROR] MCP-aware LangGraph workflow failed: {str(e)}")
            print(f"[ERROR] Exception type: {type(e)}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            # Fallback to direct processing
            rag_agent = get_rag_agent()
            dynamic_policy_response = rag_agent.generate_dynamic_policy_response(user_message)
            return RAGResponse(
                response=dynamic_policy_response,
                context=context if 'context' in locals() else {}
            )
        
        if not result:
            print("[DEBUG] LangGraph returned empty result, using fallback")
            rag_agent = get_rag_agent()
            dynamic_policy_response = rag_agent.generate_dynamic_policy_response(user_message)
            return RAGResponse(
                response=dynamic_policy_response,
                context=context if 'context' in locals() else {}
            )
        
        # Store updated context (exact from old code)
        session_storage[session_id] = result.get("context", context)
        
        # Return response based on workflow result
        return RAGResponse(
            response=result.get("response", "No response generated"),
            citations=result.get("citations", []),
            context=result.get("context", context),
            ticket=result.get("ticket"),
        )
        
    except Exception as e:
        print(f"[ERROR] Unexpected error in main_rag_chat: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        # Fallback to dynamic policy RAG response (exact from old code)
        rag_agent = get_rag_agent()
        dynamic_policy_response = rag_agent.generate_dynamic_policy_response(request.message)
        return RAGResponse(
            response=dynamic_policy_response,
            context=context if 'context' in locals() else {}
        )

@app.post("/create_ticket", operation_id="create_ticket", summary="Create IT support ticket")
async def create_ticket(user_message: str, session_id: str = "default_session", conversation_history: list = []):
    try:
        # Use MCP tool-based extraction and ticket filling
        from agents import load_template_fields, build_ordered_ticket
        from agents import fill_ticket_with_llm_and_fuzzy
        template_fields = load_template_fields(TEMPLATE_PATH)
        ticket = await fill_ticket_with_llm_and_fuzzy(template_fields, user_message, conversation_history)
        response_ticket = build_ordered_ticket(ticket or {}, template_fields)
        return {
            "response": "Your ticket has been created and will be processed by IT support.",
            "ticket": response_ticket,
            "missing_fields": [],
            "template_fields": template_fields
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/complete_ticket", operation_id="complete_ticket", summary="Complete a ticket with missing field values")
async def complete_ticket(ticket: dict, field_values: dict):
    try:
        ticket.update(field_values)
        return {
            "success": True,
            "ticket": ticket
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- A2A ENDPOINTS (exact from old code) ---

@app.post("/a2a/task", tags=["A2A"], operation_id="a2a_task", summary="A2A JSON-RPC endpoint for agent tasks")
async def a2a_task(request: Request):
    try:
        data = await request.json()
        print(f"[DEBUG] /a2a/task called. Raw data: {data}")
        params = data.get("params", {})
        target_agent = params.get("target_agent", "it_helpdesk_agent")
        source_agent = params.get("source_agent", "main_agent")
        message = params.get("message", {})
        
        print(f"[A2A] {source_agent} communicating with {target_agent}")
        print(f"[A2A] Message: {message}")
        
        # Route to appropriate specialized agent
        if target_agent == "it_helpdesk_agent":
            result = await handle_it_helpdesk_agent(message)
        elif target_agent == "electric_agent":
            result = await handle_electric_agent(message)
        else:
            return {"jsonrpc": "2.0", "id": data.get("id"), "error": {"code": -32000, "message": f"Unknown agent: {target_agent}"}}
        
        return {"jsonrpc": "2.0", "id": data.get("id"), "result": result}
        
    except Exception as e:
        import traceback
        print(f"[DEBUG] Exception in /a2a/task: {e}\n{traceback.format_exc()}")
        logger.error(f"Agent communication error in /a2a/task: {e}")
        return {"jsonrpc": "2.0", "id": data.get("id", None), "error": {"code": -32000, "message": str(e)}}

async def handle_it_helpdesk_agent(message: dict):
    """Handle IT Helpdesk Agent requests"""
    try:
        user_message = message.get("user_message", "")
        conversation_history = message.get("conversation_history", [])
        context = message.get("context", {})
        
        # Use MCP tools for enhanced processing
        from agents import get_template_path_for_issue_type, load_template_fields, build_ordered_ticket
        template_path = get_template_path_for_issue_type("it")
        template_fields = load_template_fields(template_path)
        
        # Fill ticket using MCP-enhanced function
        from agents import fill_ticket_with_llm_and_fuzzy
        ticket = await fill_ticket_with_llm_and_fuzzy(template_fields, user_message, conversation_history, context)
        response_ticket = build_ordered_ticket(ticket or {}, template_fields)
        
        # Add agent identifier
        response_ticket["Agent_Type"] = "IT Helpdesk Agent"
        response_ticket["Processing_Notes"] = "Ticket created by IT Helpdesk Agent via A2A"
        
        return {"jsonrpc": "2.0", "id": 1, "result": {
            "response": f"Thank you, {context.get('employee_name', 'Unknown')}. I will create an IT support ticket for your issue.",
            "ticket": response_ticket,
            "context": context
        }}
    except Exception as e:
        print(f"[A2A] IT Helpdesk Agent error: {e}")
        return {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": str(e)}}

async def handle_electric_agent(message: dict):
    """Handle Electric Agent requests"""
    try:
        user_message = message.get("user_message", "")
        conversation_history = message.get("conversation_history", [])
        context = message.get("context", {})
        
        # Use electric template
        template_path = os.path.join(TEMPLATES_DIR, "SL - Electric Issues.jsonl")
        with open(template_path, "r", encoding="utf-8") as f:
            template = json.loads(f.readline())
        template_fields = template["fields"]
        
        # Fill ticket using MCP-enhanced function
        from agents import fill_ticket_with_llm_and_fuzzy
        ticket = await fill_ticket_with_llm_and_fuzzy(template_fields, user_message, conversation_history, context)
        response_ticket = build_ordered_ticket(ticket or {}, template_fields)
        
        # Add agent identifier
        response_ticket["Agent_Type"] = "Electric Issues Agent"
        response_ticket["Processing_Notes"] = "Ticket created by Electric Issues Agent via A2A"
        
        return {"jsonrpc": "2.0", "id": 1, "result": {
            "response": f"Thank you, {context.get('employee_name', 'Unknown')}. I will create an electric issue ticket for your problem.",
            "ticket": response_ticket,
            "context": context
        }}
    except Exception as e:
        print(f"[A2A] Electric Agent error: {e}")
        return {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": str(e)}}
    except Exception as e:
        import traceback
        print(f"[DEBUG] Exception in /a2a/task: {e}\n{traceback.format_exc()}")
        logger.error(f"Ticket generation error in /a2a/task: {e}")
        return {"jsonrpc": "2.0", "id": data.get("id", None), "error": {"code": -32000, "message": str(e)}}

# --- Electric Agent Router (exact from old code) ---
from fastapi import APIRouter

electric_router = APIRouter(prefix="/electric")

@electric_router.post("/a2a/task")
async def electric_a2a_task(request: Request):
    try:
        data = await request.json()
        logging.info(f"Electric A2A /a2a/task called")

        template_path = os.path.join(TEMPLATES_DIR, "SL - Electric Issues.jsonl")
        if not os.path.exists(template_path):
            logging.error(f"Electric template not found at {template_path}")
            return {"response": "Ticket template not found. Please contact IT support.", "ticket": {}}

        with open(template_path, "r", encoding="utf-8") as f:
            template = json.loads(f.readline())
        template_fields = template["fields"]
        
        # Extract context from the general agent
        user_message = data.get("message", "")
        conversation_history = data.get("conversation_history", [])
        context = data.get("context", {})
        citations = data.get("citations", [])
        
        # Enhance context with citations from general agent
        if citations:
            # Add relevant policy information to the context for better ticket filling
            citation_summary = "\n".join([f"• {citation}" for citation in citations[:3]])
            context["policy_context"] = citation_summary
            context["problem_description"] = context.get("problem_description", user_message)
        
        # Create ticket using enhanced context
        from agents import build_ordered_ticket
        from agents import fill_ticket_with_llm_and_fuzzy
        ticket = await fill_ticket_with_llm_and_fuzzy(template_fields, user_message, conversation_history, context)
        response_ticket = build_ordered_ticket(ticket or {}, template_fields)
        
        # Add electric agent identifier
        response_ticket["Agent_Type"] = "Electric Issues Agent"
        response_ticket["Processing_Notes"] = "Ticket created by Electric Issues Agent with policy context from General Agent"
        
        return {
            "response": f"Thank you, {context.get('employee_name', 'Unknown')}. I will create an electric issue ticket for your problem. Here are the details:\n\nName: {context.get('employee_name', 'Unknown')}\nIssue: {context.get('problem_description', user_message)}\n\nI will escalate this to the Electric Issues team for further investigation. You will receive updates on your ticket shortly.",
            "ticket": response_ticket,
            "context": context
        }

    except Exception as e:
        logging.error(f"Ticket generation error in /electric/a2a/task: {e}")
        return {
            "response": "An error occurred while generating your electric issue ticket. Please contact support.",
            "ticket": {}
        }

# Include the electric router
app.include_router(electric_router)

# --- MCP SERVER MOUNTING with specialized tools ---
# Do NOT mount mcp.app; FastMCP is not a FastAPI app. Run MCP server as a separate process using mcp_tools.py
# mcp = create_mcp_tools(app)  # <-- REMOVE or COMMENT OUT this line if present

@app.get("/mcp/tools", tags=["MCP"], operation_id="list_mcp_tools", summary="List available MCP tools")
async def list_mcp_tools():
    """List all available MCP tools"""
    try:
        tools = [
            {
                "name": "search_policies",
                "description": "Search company policies and solutions for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "k": {"type": "integer", "description": "Number of results", "default": 3}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "extract_user_info",
                "description": "Extract employee information from conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_message": {"type": "string"},
                        "conversation_history": {
                            "type": "array",
                            "items": {"type": "object"}
                        }
                    },
                    "required": ["conversation_history"]
                }
            },
            {
                "name": "classify_issue_type",
                "description": "Classify if an issue is IT or Electric-related",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_message": {"type": "string"},
                        "conversation_history": {
                            "type": "array",
                            "items": {"type": "object"}
                        }
                    },
                    "required": ["user_message"]
                }
            },
            {
                "name": "create_ticket",
                "description": "Create a support ticket with the given information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_message": {"type": "string"},
                        "conversation_history": {
                            "type": "array",
                            "items": {"type": "object"}
                        },
                        "context": {"type": "object"}
                    },
                    "required": ["user_message"]
                }
            },
            {
                "name": "a2a_communicate",
                "description": "Communicate with another agent via A2A",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent": {"type": "string"},
                        "source_agent": {"type": "string"},
                        "message": {"type": "object"}
                    },
                    "required": ["target_agent", "source_agent", "message"]
                }
            }
        ]
        return {"tools": tools}
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}")
        return {"error": str(e)}

# Add MCP proxy endpoints
@app.post("/api/mcp/proxy")
async def mcp_proxy(request: dict):
    """Proxy MCP requests from frontend to MCP server"""
    try:
        # Convert frontend request to proper MCP format
        mcp_request = {
            "jsonrpc": "2.0",
            "id": request.get("id", 1),
            "method": request.get("method", "tools/call"),
            "params": request.get("params", {})
        }
        
        print(f"[DEBUG] MCP Proxy: Sending request to MCP server: {mcp_request}")
        
        # For SSE transport, we need to use a different approach
        # FastMCP v2.10.5 SSE transport doesn't support direct HTTP POST
        # Instead, we'll use the MCP tools directly
        
        # Extract tool name and arguments
        tool_name = request.get("params", {}).get("name")
        arguments = request.get("params", {}).get("arguments", {})
        
        if tool_name:
            # Call the MCP tool directly using the mcp_tools module
            from mcp_tools import (
                main_rag_chat_tool, create_ticket_tool, complete_ticket_tool,
                a2a_task_tool, search_policies_tool, extract_user_info_tool,
                classify_issue_type_tool
            )
            
            # Map tool names to functions
            tool_map = {
                'main_rag_chat': main_rag_chat_tool,
                'create_ticket': create_ticket_tool,
                'complete_ticket': complete_ticket_tool,
                'a2a_task': a2a_task_tool,
                'search_policies': search_policies_tool,
                'extract_user_info': extract_user_info_tool,
                'classify_issue_type': classify_issue_type_tool
            }
            
            tool_func = tool_map.get(tool_name)
            if tool_func:
                try:
                    # Call the tool asynchronously - FastMCP tools have a 'fn' attribute
                    if hasattr(tool_func, 'fn'):
                        actual_func = tool_func.fn
                    else:
                        actual_func = tool_func
                    
                    if asyncio.iscoroutinefunction(actual_func):
                        result = await actual_func(**arguments)
                    else:
                        result = actual_func(**arguments)
                    
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id", 1),
                        "result": result
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id", 1),
                        "error": {
                            "code": -32603,
                            "message": f"Tool execution failed: {str(e)}"
                        }
                    }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id", 1),
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found"
                    }
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id", 1),
                "error": {
                    "code": -32602,
                    "message": "No tool name provided"
                }
            }
            
    except Exception as e:
        print(f"[ERROR] MCP Proxy: Unexpected error - {e}")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id", 1),
            "error": {
                "code": -32603,
                "message": f"MCP proxy error: {str(e)}"
            }
        }

@app.get("/api/mcp/status")
async def mcp_status():
    """Check if MCP server is running"""
    try:
        # Check if MCP tools are available by importing the module
        from mcp_tools import (
            main_rag_chat_tool, create_ticket_tool, complete_ticket_tool,
            a2a_task_tool, search_policies_tool, extract_user_info_tool,
            classify_issue_type_tool
        )
        
        # Check if tools are available (FastMCP tools are objects, not callable functions)
        tools = [
            main_rag_chat_tool, create_ticket_tool, complete_ticket_tool,
            a2a_task_tool, search_policies_tool, extract_user_info_tool,
            classify_issue_type_tool
        ]
        
        # FastMCP tools have a 'fn' attribute that contains the actual function
        available_tools = [tool for tool in tools if hasattr(tool, 'fn') and callable(tool.fn)]
        
        if available_tools:
            return {"status": "connected", "tools_count": len(available_tools)}
        else:
            return {"status": "error", "message": "No MCP tools found"}
            
    except Exception as e:
        print(f"[ERROR] MCP status check failed: {e}")
        return {"status": "disconnected", "error": str(e)}

@app.get("/api/mcp/tools")
async def mcp_tools():
    """Get available MCP tools"""
    try:
        from mcp_tools import (
            main_rag_chat_tool, create_ticket_tool, complete_ticket_tool,
            a2a_task_tool, search_policies_tool, extract_user_info_tool,
            classify_issue_type_tool
        )
        
        # Define tool information
        tools = [
            {
                "name": "main_rag_chat",
                "description": "Main RAG chat tool that handles conversation and ticket creation"
            },
            {
                "name": "create_ticket",
                "description": "Create a support ticket with the given information"
            },
            {
                "name": "complete_ticket",
                "description": "Complete a ticket with missing field values"
            },
            {
                "name": "a2a_task",
                "description": "Agent-to-agent communication via A2A"
            },
            {
                "name": "search_policies",
                "description": "Search company policies and solutions for relevant information"
            },
            {
                "name": "extract_user_info",
                "description": "Extract employee information from conversation"
            },
            {
                "name": "classify_issue_type",
                "description": "Classify if an issue is IT or Electric-related"
            }
        ]
        
        return {"tools": tools}
    except Exception as e:
        print(f"[ERROR] Failed to get MCP tools: {e}")
        return {"tools": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 