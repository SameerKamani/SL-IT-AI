"""
Agent Communication System for SL-IT-AI
- A2A routing and specialized agent handling
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from agents import (
    classify_issue_type_llm, fill_ticket_with_llm_and_fuzzy,
    get_template_path_for_issue_type, load_template_fields, build_ordered_ticket
)

class AgentCommunication:
    def __init__(self):
        self.specialized_agents = {
            "it_helpdesk_agent": self._handle_it_agent,
            "electric_agent": self._handle_electric_agent
        }
    
    async def route_to_specialized_agent(self, user_message: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate specialized agent based on issue type"""
        try:
            # Classify issue type
            issue_type = await classify_issue_type_llm(user_message, conversation_history)

            if issue_type == "other":
                return {
                    "response": "Sorry, this issue is not related to IT or Electric support and cannot be ticketed by this helpdesk.",
                    "ticket": {},
                    "agent_type": "none",
                    "context": context
                }
            if issue_type == "electric":
                return await self._handle_electric_agent(user_message, conversation_history, context)
            else:
                return await self._handle_it_agent(user_message, conversation_history, context)
                
        except Exception as e:
            print(f"[A2A] Agent routing failed: {e}")
            return {"error": str(e)}
    
    async def _handle_it_agent(self, user_message: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle IT Helpdesk Agent requests"""
        try:
            issue_type = await classify_issue_type_llm(user_message, conversation_history)
            if issue_type == "other":
                return {
                    "response": "Sorry, this issue is not related to IT or Electric support and cannot be ticketed by this helpdesk.",
                    "ticket": {},
                    "agent_type": "none",
                    "context": context
                }
            template_path = get_template_path_for_issue_type("it")
            template_fields = load_template_fields(template_path)
            
            ticket = await fill_ticket_with_llm_and_fuzzy(template_fields, user_message, conversation_history, context)
            response_ticket = build_ordered_ticket(ticket or {}, template_fields)
            
            response_ticket["Agent_Type"] = "IT Helpdesk Agent"
            response_ticket["Processing_Notes"] = "Ticket created by IT Helpdesk Agent via A2A"
            
            return {
                "response": f"Thank you, {context.get('employee_name', 'Unknown')}. I will create an IT support ticket for your issue.",
                "ticket": response_ticket,
                "agent_type": "it_helpdesk_agent",
                "context": context
            }
        except Exception as e:
            print(f"[A2A] IT Agent error: {e}")
            return {"error": str(e)}
    
    async def _handle_electric_agent(self, user_message: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Electric Agent requests"""
        try:
            issue_type = await classify_issue_type_llm(user_message, conversation_history)
            if issue_type == "other":
                return {
                    "response": "Sorry, this issue is not related to IT or Electric support and cannot be ticketed by this helpdesk.",
                    "ticket": {},
                    "agent_type": "none",
                    "context": context
                }
            template_path = get_template_path_for_issue_type("electric")
            template_fields = load_template_fields(template_path)
            
            ticket = await fill_ticket_with_llm_and_fuzzy(template_fields, user_message, conversation_history, context)
            response_ticket = build_ordered_ticket(ticket or {}, template_fields)
            
            response_ticket["Agent_Type"] = "Electric Issues Agent"
            response_ticket["Processing_Notes"] = "Ticket created by Electric Issues Agent via A2A"
            
            return {
                "response": f"Thank you, {context.get('employee_name', 'Unknown')}. I will create an electric issue ticket for your problem.",
                "ticket": response_ticket,
                "agent_type": "electric_agent",
                "context": context
            }
        except Exception as e:
            print(f"[A2A] Electric Agent error: {e}")
            return {"error": str(e)}

# Global instance
agent_comm = AgentCommunication()

async def route_to_agent(target_agent: str, user_message: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Dict[str, Any]:
    """Route to specific agent"""
    return await agent_comm.route_to_specialized_agent(user_message, conversation_history, context) 