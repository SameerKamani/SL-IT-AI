"""
MCP Tools for SL-IT-AI
- Convert existing functions into MCP-compatible tools
- Only includes functions that are actually used in the codebase
"""
import os
import json
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from agents import (
    classify_issue_type_llm, extract_user_info_from_history,
    fill_ticket_with_llm_and_fuzzy, generate_subject_from_description,
    get_employee_info, load_template_fields, build_ordered_ticket,
    get_template_path_for_issue_type, generate_ticket_artifact,
    analyze_conversation_and_extract_problem
)
from policy_rag import PolicyRAGAgent

# Initialize RAG agent - use lazy initialization
from config import get_rag_agent

# Create MCP server with proper FastMCP v2.x syntax
mcp = FastMCP('sl-it-mcp-tools')

@mcp.tool()
async def main_rag_chat(message: str, conversation_history: List[Dict[str, str]] = None, session_id: str = None, user_info: Dict[str, Any] = None, force_create_ticket: bool = False) -> Dict[str, Any]:
    """Main RAG chat tool that handles conversation and ticket creation"""
    try:
        from langgraph_workflow import compiled_graph
        from models import AgentState
        
        # Prepare context with user information
        context = {}
        print(f"[MCP] main_rag_chat: Received user_info parameter: {user_info}")
        if user_info:
            context.update(user_info)
            print(f"[MCP] main_rag_chat: Using provided user_info: {user_info}")
        else:
            print("[MCP] main_rag_chat: No user_info provided")
        
        # Prepare state for LangGraph workflow
        state = AgentState(
            messages=[],
            user_message=message,
            conversation_history=conversation_history or [],
            context=context,
            session_id=session_id or "default"
        )
        
        # Run the workflow
        result = await compiled_graph.ainvoke(state)

        print("[DEBUG] main_rag_chat: Workflow result:", result)

        # If result is a dict with 'response', use it
        response_message = None
        ticket = None
        ticket_artifact = None
        if isinstance(result, dict):
            response_message = result.get("response")
            ticket = result.get("ticket")
            ticket_artifact = result.get("ticket_artifact")
        else:
            response_message = getattr(result, "response", None)
            ticket = getattr(result, "ticket", None)
            ticket_artifact = getattr(result, "ticket_artifact", None)

        response = {
            "response": response_message or "No response generated",
            "citations": result.get("citations", []) if isinstance(result, dict) else [],
            "context": result.get("context", {}) if isinstance(result, dict) else {}
        }
        if ticket and bool(ticket):
            response["ticket"] = ticket
        if ticket_artifact and bool(ticket_artifact):
            response["ticket_artifact"] = ticket_artifact
        return response
    except Exception as e:
        return {"error": str(e), "response": "I encountered an error processing your request."}

@mcp.tool()
async def create_ticket(user_message: str, conversation_history: List[Dict[str, str]] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a support ticket with the given information"""
    try:
        if context is None:
            context = {}
        
        # Use LLM to analyze the entire conversation and extract the problem
        conversation_analysis = None
        if conversation_history:
            try:
                conversation_analysis = await analyze_conversation_and_extract_problem(conversation_history, user_message)
                print(f"[MCP] Conversation analysis result: {conversation_analysis}")
            except Exception as e:
                print(f"[ERROR] Conversation analysis failed: {e}")
        
        # Extract the original problem description
        original_problem = user_message
        if conversation_analysis and conversation_analysis.get("problem_description"):
            original_problem = conversation_analysis.get("problem_description")
            print(f"[MCP] Using LLM-extracted problem: {original_problem}")
        elif conversation_history:
            for msg in conversation_history:
                if msg.get("role") == "user":
                    content = msg.get("content", "").lower()
                    # Look for messages that describe actual problems (not ticket requests)
                    if any(word in content for word in ["wifi", "network", "internet", "connect", "printer", "computer", "electric", "power", "light", "outlet", "not working", "broken", "issue", "problem"]) and not any(word in content for word in ["create ticket", "generate ticket", "open ticket", "make ticket"]):
                        original_problem = msg.get("content", "")
                        print(f"[MCP] Found original problem in conversation: {original_problem}")
                        break
        
        # Update context with the original problem and conversation analysis
        context["problem_description"] = original_problem
        context["original_problem"] = original_problem
        if conversation_analysis:
            context["conversation_analysis"] = conversation_analysis
        
        # Classify issue type using the original problem
        issue_type = await classify_issue_type_llm(original_problem, conversation_history or [])
        template_path = get_template_path_for_issue_type(issue_type)
        template_fields = load_template_fields(template_path)
        
        # Fill ticket using the original problem description
        ticket = await fill_ticket_with_llm_and_fuzzy(
            template_fields, original_problem, conversation_history or [], context
        )
        ordered_ticket = build_ordered_ticket(ticket, template_fields)
        return {
            "ticket": ordered_ticket,
            "issue_type": issue_type,
            "template_path": template_path,
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@mcp.tool()
async def complete_ticket(ticket: Dict[str, Any], field_values: Dict[str, Any]) -> Dict[str, Any]:
    """Complete a ticket with missing field values"""
    try:
        ticket.update(field_values)
        return {
            "success": True,
            "ticket": ticket
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@mcp.tool()
async def a2a_task(target_agent: str, source_agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """Agent-to-agent communication via A2A"""
    try:
        from agent_communication import route_to_agent
        
        user_message = message.get("user_message", "")
        conversation_history = message.get("conversation_history", [])
        context = message.get("context", {})
        
        print(f"[MCP_A2A] Routing to {target_agent} with message: {user_message}")
        
        # Route to the appropriate agent
        result = await route_to_agent(target_agent, user_message, conversation_history, context)
        
        print(f"[MCP_A2A] Agent {target_agent} response: {result}")
        return result
    except Exception as e:
        print(f"[MCP_A2A] Error: {e}")
        return {"error": str(e)}

# Removed old handler functions - now using agent_communication.py

@mcp.tool()
async def search_policies(query: str, k: int = 3) -> Dict[str, Any]:
    """Search policies and solutions using RAG"""
    try:
        rag_agent = get_rag_agent()
        results = rag_agent.search(query)
        return {"results": results, "query": query}
    except Exception as e:
        return {"error": str(e), "results": []}

@mcp.tool()
async def extract_user_info(current_message: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract user information from conversation"""
    try:
        return extract_user_info_from_history(current_message, conversation_history)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def classify_issue_type(user_message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Classify the type of issue (IT or Electric)"""
    try:
        issue_type = await classify_issue_type_llm(user_message, conversation_history or [])
        return {"issue_type": issue_type, "message": user_message}
    except Exception as e:
        return {"error": str(e)}

# Add tool discovery endpoint
@mcp.tool()
async def list_available_tools() -> Dict[str, Any]:
    """List all available MCP tools"""
    tools = [
        "main_rag_chat",
        "create_ticket", 
        "complete_ticket",
        "a2a_task",
        "search_policies",
        "extract_user_info",
        "classify_issue_type",
        "list_available_tools"
    ]
    return {"tools": tools, "count": len(tools)}

# Add health check endpoint
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check MCP server health"""
    return {"status": "healthy", "server": "sl-it-mcp-tools"}

# Only run the server if executed directly
if __name__ == "__main__":
    print("Starting SL-IT-AI MCP Server...")
    print("Server will be available via STDIO transport")
    print("Use the FastAPI proxy at: http://localhost:8000/api/mcp/proxy")
    # FastMCP v2.10.5 STDIO transport - simplest and most reliable for local tools
    mcp.run(transport="stdio")  # STDIO is the best choice for local tools

# Export tools for direct access
main_rag_chat_tool = main_rag_chat
create_ticket_tool = create_ticket
complete_ticket_tool = complete_ticket
a2a_task_tool = a2a_task
search_policies_tool = search_policies
extract_user_info_tool = extract_user_info
classify_issue_type_tool = classify_issue_type 