"""
LangGraph workflow for SL-IT-AI chatbot
Matches the exact behavior of the old monolithic code
"""
import os
import json
import logging
import re
import time
from typing import TypedDict, NotRequired, List, Dict, Any
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from rapidfuzz import process, fuzz
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_VERSION,
    ROOT_DIR, TEMPLATES_DIR, EMPLOYEE_DATA_DIR,
    TEMPLATE_PATH, EMPLOYEE_PATH, ELECTRIC_TEMPLATE_PATH
)
from agents import (
    fill_ticket_with_llm_and_fuzzy, classify_issue_type_llm,
    extract_user_info_from_history, generate_subject_from_description,
    load_template_fields, build_ordered_ticket,
    get_template_path_for_issue_type, generate_ticket_artifact,
    is_ticket_creation_intent_dynamic, is_confirmation_intent_dynamic,
    generate_dynamic_response, generate_dynamic_error_message,
    validate_field_dynamically, determine_routing_dynamically,
    generate_dynamic_tool_description, generate_dynamic_system_prompt,
    MCPAgent, analyze_conversation_and_extract_problem
)
from policy_rag import PolicyRAGAgent

# Add LangChain MCP adapters imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_API_BASE,
    api_version=AZURE_OPENAI_API_VERSION
)

# RAG agent - use lazy initialization
from config import get_rag_agent

# Load employee data for autofill
try:
    with open(EMPLOYEE_PATH, 'r', encoding='utf-8') as f:
        EMPLOYEE_DATA = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    logging.warning(f"Employee data file not found at {EMPLOYEE_PATH}, using empty list")
    EMPLOYEE_DATA = []

# --- Missing function definitions ---

async def search_policies_llm(query: str) -> str:
    """Search policies using RAG agent"""
    try:
        rag_agent = get_rag_agent()
        results = rag_agent.search(query)
        return results
    except Exception as e:
        print(f"[ERROR] Policy search failed: {e}")
        return ""

async def extract_user_info_llm(current_message: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract user info using the existing function"""
    try:
        return extract_user_info_from_history(current_message, conversation_history)
    except Exception as e:
        print(f"[ERROR] User info extraction failed: {e}")
        return {}

async def get_mcp_tools():
    mcp_client = MultiServerMCPClient({
        "sl-it-ai": {
            "command": "python",
            "args": ["mcp_tools.py"],
            "transport": "stdio",  # STDIO transport - simplest and most reliable
        }
    })
    async with mcp_client.session("sl-it-ai") as session:
        tools = await load_mcp_tools(session)
        return tools

# --- LangGraph State ---
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

# --- MCP Client Initialization ---
async def get_mcp_client():
    """Initialize MCP client with SL-IT-AI tools"""
    try:
        mcp_client = MultiServerMCPClient({
            "sl-it-ai": {
                "command": "python",
                "args": ["mcp_tools.py"],
                "transport": "stdio",  # STDIO transport - simplest and most reliable
            }
        })
        return mcp_client
    except Exception as e:
        print(f"[MCP] Failed to initialize MCP client: {e}")
        return None

# --- LangGraph Node Functions ---
async def rag_node(state: AgentState) -> AgentState:
    """RAG node that provides solutions first, then offers ticket creation (like old code)"""
    print("[DEBUG] rag_node called with user_message:", state["user_message"])
    user_message = state["user_message"]
    conversation_history = state.get("conversation_history", [])
    context = state.get("context", {})
    
    # Check if this is a ticket creation request (explicit)
    def is_explicit_ticket_request(msg):
        if not msg: return False
        lower = msg.lower()
        explicit_phrases = [
            'create ticket', 'generate ticket', 'open ticket', 'file ticket', 'submit ticket',
            'raise ticket', 'make ticket', 'support ticket', 'can you generate a ticket', 
            'can you create a ticket', 'can you open a ticket', 'can you file a ticket', 
            'can you submit a ticket', 'can you raise a ticket', 'can you make a ticket',
            'i want to create a ticket', 'i want to open a ticket', 'i want to file a ticket',
            'i want to submit a ticket', 'i want to raise a ticket', 'i want to make a ticket'
        ]
        return any(phrase in lower for phrase in explicit_phrases)
    
    # If user explicitly requests ticket creation, skip solution phase
    if is_explicit_ticket_request(user_message):
        print("[DEBUG] rag_node: Explicit ticket request detected, skipping solution phase")
        state["intent"] = "create_ticket"
        return state
    
    # Otherwise, provide solutions first (like old code)
    print("[DEBUG] rag_node: Providing solutions first, then offering ticket creation")
    
    # Use RAG to search for relevant policies and solutions
    try:
        print("[DEBUG] rag_node: Starting RAG search")
        rag_agent = get_rag_agent()
        policy_results = rag_agent.search(user_message)
        print(f"[DEBUG] rag_node: RAG search completed, got {len(policy_results) if policy_results else 0} results")
        
        citations = []
        rag_context = []
        if policy_results and isinstance(policy_results, list):
            for r in policy_results:
                if isinstance(r, dict) and 'text' in r:
                    text = str(r['text'])
                    source = str(r.get('source', 'unknown'))
                    score = float(r.get('score', 0.0))
                    
                    # Add to citations for display
                    if len(text) > 150:
                        text = text[:150] + "..."
                    citations.append(text)
                    
                    # Add to context for agent communication
                    rag_context.append({
                        "text": text,
                        "source": source,
                        "score": score,
                        "relevance": "high" if score < 0.5 else "medium"
                    })
        elif isinstance(policy_results, str):
            citations = [policy_results[:150] + "..." if len(policy_results) > 150 else policy_results]
            rag_context = [{"text": policy_results, "source": "solutions.txt", "score": 0.0, "relevance": "high"}]
        
        print(f"[DEBUG] rag_node: Processed {len(citations)} citations")
        
        # Store RAG context for agent communication
        context['rag_results'] = rag_context
        context['policy_citations'] = citations
        context['rag_query'] = user_message
        
        # Generate helpful response with solutions (like old code)
        system_prompt = """You are an IT support assistant for Systems Limited. Use the provided context to answer questions. Be helpful, professional, and concise.

        IMPORTANT FORMATTING RULES:
        1. Always format troubleshooting steps in a clear, readable way
        2. Use numbered steps (1., 2., 3., etc.) for troubleshooting procedures
        3. Use bullet points (•) for lists of options or items
        4. Separate different sections with blank lines
        5. Keep responses concise and focused on the user's specific issue
        6. Don't overwhelm users with too much information at once
        7. If providing multiple solutions, present them in order of likelihood to solve the issue
        8. When answering, use Markdown formatting. Use bullet points, numbered lists, and clear section headers. Separate each policy point with a newline. Each troubleshooting step should be on its own line.
        9. Unless the query is about any of the policies, make sure to ask the user if they would like to create a ticket for this issue; Always ask this question at the end of the response in a separate paragraph. e.g. "Would you like to create a support ticket for this issue?"
        

        When giving troubleshooting steps:
        - Start with the most common and simple solutions first
        - Use clear, numbered steps
        - Explain what each step does briefly
        - If a step doesn't work, suggest the next step
        - Keep the total response under 200 words unless the issue is complex

        When answering policy questions:
        - Only answer if the information is available in the following context, otherwise answer "I don't know this"
        - If the user asks for an item, look for a numbered or bullet point in the context.
        - Use the following context to answer questions about Systems Limited policy
        - If there is a match, please state the information word-for-word, don't introduce your own formatting

        """
        
        # Prepare the context string from policy_results
        if policy_results:
            if isinstance(policy_results, list):
                context_text = "\n\n".join([
                    f"• {item.get('content', str(item))}"
                    for item in policy_results[:3]
                ])
            else:
                context_text = f"• {str(policy_results)}"
        else:
            context_text = "No relevant policy information found."

        # Combine system prompt and context
        combined_system_prompt = (
            system_prompt
            + "\n\nRelevant policy information to help with this issue:\n"
            + context_text
        )

        messages = [
            {"role": "system", "content": combined_system_prompt},
            *[{"role": msg["role"], "content": msg["content"]} for msg in conversation_history],
            # {"role": "user", "content": user_message}
        ]
        
        # Print the full messages being sent to the LLM
        # for debugging purposes
        print("\n\n--- MESSAGES SENT TO LLM COMPLETION ---\n")
        import pprint
        pprint.pprint(messages, width=120)
        print("\n--- END OF MESSAGES ---\n\n")
        
        print("[DEBUG] rag_node: Calling LLM for response generation")
        llm_start = time.time()
        completion = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.6,
            max_tokens=800
        )
        llm_end = time.time()
        print(f"[RAG_NODE] LLM completion took {llm_end - llm_start:.2f}s")
        response_text = completion.choices[0].message.content or "No response generated."
        
        # IMPORTANT: Provide solutions first, then ask for ticket creation (like old code)
        state["response"] = str(response_text)
        state["citations"] = citations if citations else None
        
        # Set context to indicate we're awaiting ticket confirmation
        context['awaiting_ticket_confirmation'] = True
        context['has_rag_results'] = len(rag_context) > 0
        state["context"] = context
        
        print(f"[DEBUG] rag_node response: {state['response'][:100]}...")
        
    except Exception as e:
        print(f"[ERROR] RAG node failed: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        # Fallback response
        state["response"] = "I understand you're having an issue. Let me help you with that.\n\nWould you like to create a support ticket for this issue?"
        state["citations"] = None
        context['awaiting_ticket_confirmation'] = True
        context['has_rag_results'] = False
        state["context"] = context
    
    return state

async def intent_detection_node(state: AgentState) -> AgentState:
    """Modern LLM-based intent detection with rich examples and explicit output format"""
    print(f"[DEBUG] intent_detection_node called with user_message: {state['user_message']}")
    user_message = state["user_message"]
    context = state.get("context", {})
    conversation_history = state.get("conversation_history", [])

    # Modern LLM intent classification prompt
    system_prompt = """
Role: Intent Classifier for IT Helpdesk Chatbot.
Task: Identify and classify user intents from their inputs with a virtual assistant, aligning each with a specific intent from the list provided.
Possible Intents:
- create_ticket: The user wants to create or confirm creation of a support ticket.
- qa: The user is asking a question or describing a problem, but not explicitly asking for a ticket.
- confirmation: The user is confirming or agreeing to a previous ticket offer.
- other: The user's message does not fit any of the above.

Instructions:
- Use the conversation history for context.
- Output a JSON object with the intent, e.g. {"intent": "create_ticket"}.
- If the user is asking for help, troubleshooting, or describing an issue (see examples), use qa.
- If the user is confirming or agreeing to create a ticket, use confirmation.
- If the user is explicitly asking for a ticket, use create_ticket.
- If the message is unrelated, use other.

Examples:
User: "please create a ticket for this issue" -> create_ticket
User: "can you open a ticket?" -> create_ticket
User: "I want to log a support ticket" -> create_ticket
User: "could you please create a ticket for this issue" -> create_ticket
User: "I need a ticket for this problem" -> create_ticket
User: "open a support ticket for me" -> create_ticket
User: "log this as a ticket" -> create_ticket
User: "file a ticket" -> create_ticket
User: "submit a ticket for this" -> create_ticket
User: "raise a ticket" -> create_ticket
User: "can you please generate a ticket?" -> create_ticket
User: "I want you to create a ticket" -> create_ticket
User: "register this issue as a ticket" -> create_ticket
User: "ticket please" -> create_ticket
User: "support ticket needed" -> create_ticket
User: "I need to report this as a ticket" -> create_ticket
User: "create a new ticket" -> create_ticket
User: "generate a ticket for this" -> create_ticket
User: "I want to submit a ticket" -> create_ticket
User: "can you log a ticket for me?" -> create_ticket
User: "yes, create a ticket" -> confirmation
User: "yes" -> confirmation
User: "please do" -> confirmation
User: "go ahead" -> confirmation
User: "confirm" -> confirmation
User: "sure, create the ticket" -> confirmation
User: "okay, please create it" -> confirmation
User: "yes, please" -> confirmation
User: "alright, make the ticket" -> confirmation
User: "do it" -> confirmation
User: "yes, submit the ticket" -> confirmation
User: "yes, log it" -> confirmation
User: "yes, file the ticket" -> confirmation
User: "yes, open the ticket" -> confirmation
User: "yes, raise the ticket" -> confirmation
User: "yes, register the ticket" -> confirmation
User: "yes, support ticket" -> confirmation
User: "yes, go ahead" -> confirmation
User: "yes, proceed" -> confirmation
User: "yes, please do" -> confirmation
User: "yes, that's fine" -> confirmation
User: "yes, that's correct" -> confirmation
User: "thank you" -> other
User: "never mind" -> other
User: "no, that's all" -> other

# QA Examples (IT Helpdesk):
User: "my wifi isn’t working" -> qa
User: "I can't print" -> qa
User: "how do I reset my password?" -> qa
User: "printer not working" -> qa
User: "browser keeps crashing" -> qa
User: "docker service is down" -> qa
User: "I need help with Android Studio" -> qa
User: "VPN not connecting" -> qa
User: "website is not accessible" -> qa
User: "certificate error on web access" -> qa
User: "keyboard issue on my workstation" -> qa
User: "display issue on my computer" -> qa
User: "battery problem on my laptop" -> qa
User: "IP phone not working" -> qa
User: "Forticlient login issues" -> qa
User: "Pulse Secure not working" -> qa
User: "cartridge empty in printer" -> qa
User: "paper jam issue" -> qa
User: "host file needs update" -> qa
User: "need help with Dot Net Framework" -> qa
User: "headphones not working" -> qa
User: "LAN issue on my workstation" -> qa
User: "need to install new software" -> qa

# QA Examples (Electric Issues):
User: "power outlet is not working" -> qa
User: "cable damage in the office" -> qa
User: "voltage fluctuation in my room" -> qa
User: "UPS supply issue" -> qa
User: "breaker keeps tripping" -> qa
User: "network cable testing needed" -> qa
User: "face plate is damaged" -> qa
User: "projector VGA cable connection issue" -> qa
User: "RFID system networking problem" -> qa
User: "VOIP phone installation needed" -> qa
User: "generator service required" -> qa
User: "insect killer maintenance needed" -> qa
User: "data center power room maintenance" -> qa
User: "short circuit failure" -> qa
User: "power plug is burned" -> qa
User: "need new patch cord" -> qa
User: "switch side rack to user desk issue" -> qa
User: "access port enabling required" -> qa
User: "periodic generator service" -> qa
User: "multiconverter or power board required" -> qa
User: "overload and short circuit failure" -> qa
User: "safety breaker tripped" -> qa

"""
    # Compose LLM messages
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # Add last assistant message for context if available
    if conversation_history:
        for turn in conversation_history[-2:]:
            if turn["role"] == "assistant":
                messages.append({"role": "assistant", "content": turn["content"]})
    # Add user message
    messages.append({"role": "user", "content": user_message})

    # Call LLM for intent classification
    try:
        import time
        llm_start = time.time()
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0,
            max_tokens=30
        )
        llm_end = time.time()
        print(f"[INTENT] LLM intent classification took {llm_end - llm_start:.2f}s")
        content = response.choices[0].message.content
        print(f"[INTENT] LLM raw output: {content}")
        # Parse JSON output
        intent = "qa"
        try:
            data = json.loads(content)
            intent = data.get("intent", "qa").strip().lower()
        except Exception as e:
            print(f"[INTENT] Failed to parse LLM output as JSON: {e}")
            # Fallback: try to extract intent from text
            if "create_ticket" in content:
                intent = "create_ticket"
            elif "confirmation" in content:
                intent = "confirmation"
            elif "other" in content:
                intent = "other"
            else:
                intent = "qa"
        print(f"[INTENT] Final classified intent: {intent}")
    except Exception as e:
        print(f"[INTENT] LLM intent classification error: {e}")
        intent = "qa"

    # Set context and intent as before
    state["intent"] = intent
    state["context"] = context
    print(f"[DEBUG] intent_detection_node: final intent = {intent}")
    return state

async def ticket_creation_node(state: AgentState) -> AgentState:
    """Ticket creation node with direct processing and proper error handling"""
    print("[DEBUG] ticket_creation_node: Starting ticket creation")
    user_message = state["user_message"]
    conversation_history = state.get("conversation_history", [])
    context = state.get("context", {})
    print(f"[DEBUG] ticket_creation_node: context = {context}")
    print(f"[DEBUG] ticket_creation_node: conversation_history = {conversation_history}")
    
    try:
        # Add timeout for the entire ticket creation process
        import asyncio
        ticket_creation_task = asyncio.create_task(_create_ticket_direct(
            user_message, conversation_history, context
        ))
        # Wait for ticket creation with timeout
        try:
            result = await asyncio.wait_for(ticket_creation_task, timeout=30.0)
            response_ticket, response_text, issue_type = result
        except asyncio.TimeoutError:
            print("[ERROR] Ticket creation timed out after 30 seconds")
            raise Exception("Ticket creation timed out")
        print(f"[DEBUG] Ticket creation completed successfully")
        print(f"[DEBUG] Response ticket: {response_ticket}")
        # If issue_type is 'other', only return the refusal message and do not create ticket or artifact
        if issue_type == "other":
            # If response_text is None, get the message from response_ticket["response"]
            if not response_text and isinstance(response_ticket, dict) and "response" in response_ticket:
                response_text = response_ticket["response"]
            return {
                "response": response_text,
                "ticket": {},
                "ticket_artifact": {},
                "context": context
            }
        # If ticket is empty, only return the refusal message and do not include ticket or artifact
        if not response_ticket:
            state["response"] = response_text
            state["ticket"] = {}
            state["ticket_artifact"] = {}
            context["ticket_created"] = False
            state["context"] = context
            return state
        # Create ticket artifact
        employee_name = context.get('employee_name', 'Unknown')
        problem_description = context.get('problem_description', user_message)
        ticket_artifact = generate_ticket_artifact(employee_name, problem_description)
        print(f"[DEBUG] Created ticket artifact")
        # Set response and state
        state["response"] = response_text
        state["ticket"] = response_ticket
        state["ticket_artifact"] = ticket_artifact
        # Update context
        context["ticket_created"] = True
        context["ticket"] = response_ticket
        context["ticket_artifact"] = ticket_artifact
        state["context"] = context
        print(f"[DEBUG] Ticket creation completed successfully")
    except Exception as e:
        print(f"[ERROR] Ticket creation failed: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        state["response"] = "I apologize, but I encountered an error while creating the ticket. Please try again or contact IT support directly."
        state["ticket"] = {}
        state["ticket_artifact"] = {}
    return state

async def _create_ticket_direct(user_message, conversation_history, context):
    """Direct ticket creation with proper agent communication flow"""
    print("[DEBUG] _create_ticket_direct: Starting ticket creation with agent communication")
    
    try:
        # Use LLM to analyze the entire conversation and extract the problem
        conversation_analysis = None
        if conversation_history:
            try:
                conversation_analysis = await analyze_conversation_and_extract_problem(conversation_history, user_message)
                print(f"[DEBUG] Conversation analysis result: {conversation_analysis}")
            except Exception as e:
                print(f"[ERROR] Conversation analysis failed: {e}")
        
        # Extract the original problem description
        original_problem = user_message
        if conversation_analysis and conversation_analysis.get("problem_description"):
            original_problem = conversation_analysis.get("problem_description")
            print(f"[DEBUG] Using LLM-extracted problem: {original_problem}")
        elif conversation_history:
            for msg in conversation_history:
                if msg.get("role") == "user":
                    content = msg.get("content", "").lower()
                    # Look for messages that describe actual problems (not ticket requests)
                    if any(word in content for word in ["wifi", "network", "internet", "connect", "printer", "computer", "electric", "power", "light", "outlet", "not working", "broken", "issue", "problem"]) and not any(word in content for word in ["create ticket", "generate ticket", "open ticket", "make ticket"]):
                        original_problem = msg.get("content", "")
                        print(f"[DEBUG] Found original problem in conversation: {original_problem}")
                        break
        
        # Update context with the original problem and conversation analysis
        context["problem_description"] = original_problem
        context["original_problem"] = original_problem
        if conversation_analysis:
            context["conversation_analysis"] = conversation_analysis
        
        # Classify issue type using the original problem
        issue_type = await classify_issue_type_llm(original_problem, conversation_history)
        print(f"[DEBUG] Classified issue type: {issue_type}")

        if issue_type == "other":
            return {}, "Sorry, this issue is not related to IT or Electric support and cannot be ticketed by this helpdesk.", issue_type
        
        # Try MCP agent communication first
        try:
            print("[DEBUG] Attempting MCP agent communication...")
            
            # Use new agent communication system
            from agent_communication import agent_comm
            
            # Route to appropriate specialized agent
            print(f"[A2A] Routing to specialized agent for issue type: {issue_type}")
            agent_response = await agent_comm.route_to_specialized_agent(original_problem, conversation_history, context)
            
            if agent_response and not agent_response.get("error") and agent_response.get("ticket"):
                # Use response from specialized agent
                response_ticket = agent_response["ticket"]
                response_text = agent_response.get("response", "Ticket created successfully")
                agent_type = agent_response.get("agent_type", "unknown")
                print(f"[A2A] Successfully received ticket from {agent_type}")
                return response_ticket, response_text, issue_type
            else:
                print(f"[A2A] No response from specialized agent, falling back to direct processing")
                if agent_response.get("error"):
                    print(f"[A2A] Agent error: {agent_response['error']}")
                
        except Exception as e:
            print(f"[MCP] Agent communication failed: {e}, falling back to direct processing")
        
        # Fallback to direct processing
        print("[DEBUG] Using direct processing fallback...")
        
        # Get template path and load fields
        template_path = get_template_path_for_issue_type(issue_type)
        print(f"[DEBUG] Using template path: {template_path}")
        
        template_fields = load_template_fields(template_path)
        print(f"[DEBUG] Loaded {len(template_fields)} template fields")
        
        # Fill ticket using the original problem description
        ticket = await fill_ticket_with_llm_and_fuzzy(
            template_fields, original_problem, conversation_history, context
        )
        print(f"[DEBUG] Filled ticket with {len(ticket)} fields")
        
        # Build ordered ticket
        response_ticket = build_ordered_ticket(ticket, template_fields)
        print(f"[DEBUG] Built ordered ticket with {len(response_ticket)} fields")
        
        # Add agent identifier
        if issue_type == "electric":
            response_ticket["Agent_Type"] = "Electric Issues Agent"
            response_text = (
                f"Thank you, {context.get('employee_name', 'Unknown')}. I will create an electric issue ticket for your problem. "
                f"Here are the details:\n\nName: {context.get('employee_name', 'Unknown')}\n"
                f"Issue: {original_problem}\n\n"
                "I will escalate this to the Electric Issues team for further investigation. You will receive updates on your ticket shortly."
            )
        else:
            response_ticket["Agent_Type"] = "IT Helpdesk Agent"
            response_text = (
                f"Thank you, {context.get('employee_name', 'Unknown')}. I will create an IT support ticket for your issue. "
                f"Here are the details:\n\nName: {context.get('employee_name', 'Unknown')}\n"
                f"Issue: {original_problem}\n\n"
                "I will escalate this to the IT team for further investigation. You will receive updates on your ticket shortly."
            )
        
        return response_ticket, response_text, issue_type
        
    except Exception as e:
        print(f"[ERROR] Direct ticket creation failed: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise e

# --- LangGraph Workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("rag", rag_node)
workflow.add_node("intent_detection", intent_detection_node)
workflow.add_node("ticket_creation", ticket_creation_node)
workflow.set_entry_point("rag")
workflow.add_edge("rag", "intent_detection")

async def route_after_intent(state: AgentState):
    """Dynamic routing based on state and context"""
    intent = state.get("intent", "")
    context = state.get("context", {})
    
    print(f"[DEBUG] route_after_intent: intent = {intent}")
    print(f"[DEBUG] route_after_intent: context keys = {list(context.keys())}")
    
    # If the intent is 'other', return the refusal message immediately
    if intent == "other":
        return {"response": "Sorry, this issue is not related to IT or Electric support and cannot be ticketed by this helpdesk."}
    # Simple routing logic based on intent
    if intent == "create_ticket":
        print(f"[DEBUG] route_after_intent: routing to ticket_creation")
        return "ticket_creation"
    elif intent in ["awaiting_confirmation", "awaiting_details"]:
        print(f"[DEBUG] route_after_intent: routing to end (awaiting)")
        return END
    else:
        print(f"[DEBUG] route_after_intent: routing to end (qa)")
        return END

# Add conditional edges with proper mapping
workflow.add_conditional_edges(
    "intent_detection", 
    route_after_intent, 
    {
        "ticket_creation": "ticket_creation",
        END: END,
    }
)
workflow.add_edge("ticket_creation", END)

# Compile the workflow
compiled_graph = workflow.compile()

print("[DEBUG] LangGraph workflow compiled successfully")
print(f"[DEBUG] Workflow nodes: {list(workflow.nodes.keys())}")
print(f"[DEBUG] Workflow edges: {list(workflow.edges)}") 