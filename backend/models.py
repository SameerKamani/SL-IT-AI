"""
SL-IT-AI Data Models and Pydantic Schemas
- All Pydantic models, TypedDict definitions, and data structures
"""
from typing import Optional, List, Dict, Any, Annotated
from pydantic import BaseModel
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import add_messages

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    session_id: Optional[str] = None
    user_info: Optional[dict] = None

class RAGRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []
    session_id: Optional[str] = None
    user_info: Optional[dict] = None

class RAGResponse(BaseModel):
    response: str
    citations: Optional[List[str]] = []
    context: Optional[dict] = {}
    ticket: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    ticket: Optional[dict] = None
    ticket_artifact: Optional[dict] = None

class TicketRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class TicketResponse(BaseModel):
    ticket: dict
    ticket_artifact: dict
    issue_type: str

# LangGraph State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_message: str
    conversation_history: NotRequired[list]
    context: NotRequired[dict]
    ticket: NotRequired[dict]
    intent: NotRequired[str]
    response: NotRequired[str]
    citations: NotRequired[list]
    agent_type: NotRequired[str]
    session_id: NotRequired[str]

# Simple EventQueue class for compatibility
class EventQueue:
    def __init__(self):
        self.status_message = None
        self.artifact = None
    
    def submit(self):
        pass
    
    def update_status(self, state, message=None, artifact=None):
        self.status_message = message
        if artifact:
            self.artifact = artifact 