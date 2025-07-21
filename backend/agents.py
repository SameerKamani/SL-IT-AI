"""
SL-IT-AI Agent Logic
- Agent functions for ticket creation, classification, and field extraction
"""
import os
import json
import logging
import re
import uuid
from typing import Optional, List, Dict, Any
from rapidfuzz import process, fuzz
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_VERSION,
    ROOT_DIR, TEMPLATES_DIR, EMPLOYEE_DATA_DIR,
    TEMPLATE_PATH, EMPLOYEE_PATH, ELECTRIC_TEMPLATE_PATH
)
import time
import traceback

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_API_BASE,
    api_version=AZURE_OPENAI_API_VERSION
)

# MCP Agent Base Class
class MCPAgent:
    def __init__(self, agent_type: str, mcp_command: str = "python", mcp_args: list = None):
        self.agent_type = agent_type
        self.mcp_command = mcp_command
        self.mcp_args = mcp_args or ["mcp_tools.py"]
        self.available_tools = []
        self.session_id = None
    
    async def discover_tools(self):
        """Discover available MCP tools using STDIO transport"""
        try:
            # For STDIO transport, tools are discovered through the MCP client
            # This will be handled by the LangGraph workflow
            print(f"[MCP] {self.agent_type} using STDIO transport - tools discovered via MCP client")
            self.available_tools = ["main_rag_chat", "create_ticket", "search_policies"]  # Default tools
        except Exception as e:
            print(f"[MCP] Tool discovery failed: {e}")
            self.available_tools = []
    
    async def invoke_tool(self, tool_name: str, params: dict):
        """Invoke MCP tool using STDIO transport"""
        try:
            # For STDIO transport, tool invocation is handled by the MCP client
            # This will be handled by the LangGraph workflow
            print(f"[MCP] {self.agent_type} invoking tool {tool_name} via STDIO transport")
            return {"status": "success", "tool": tool_name, "params": params}
        except Exception as e:
            print(f"[MCP] Tool invocation failed: {e}")
            return None
    
    async def communicate_with_agent(self, target_agent: str, message: dict):
        """A2A communication with another agent with timeout"""
        try:
            import requests
            import asyncio
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "a2a/task",
                "params": {
                    "target_agent": target_agent,
                    "message": message,
                    "source_agent": self.agent_type
                }
            }
            
            # Use asyncio to add timeout
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.mcp_endpoint}/a2a/task", json=payload, timeout=15.0)
            )
            
            if response.status_code == 200:
                return response.json().get("result", {})
            else:
                print(f"[A2A] Agent communication failed with status {response.status_code}")
                return None
        except Exception as e:
            print(f"[A2A] Agent communication failed: {e}")
            return None

# Load employee data for autofill
try:
    with open(EMPLOYEE_PATH, 'r', encoding='utf-8') as f:
        EMPLOYEE_DATA = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    logging.warning(f"Employee data file not found at {EMPLOYEE_PATH}, using empty list")
    EMPLOYEE_DATA = []

# --- AGENT FUNCTIONS ---

async def classify_issue_type_llm(message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Classify issue type using LLM with MCP tool integration and fallback"""
    print("[DEBUG] classify_issue_type_llm: Using MCP-aware classification with fallback")
    print("[DEBUG] classify_issue_type_llm: Raw conversation_history object:")
    print(conversation_history)
    if conversation_history and (not isinstance(conversation_history, list) or not all(isinstance(turn, dict) and 'role' in turn and 'content' in turn for turn in conversation_history)):
        print("[WARNING] classify_issue_type_llm: conversation_history is not a list of dicts with 'role' and 'content'.")
    
    # Try MCP tools first with comprehensive error handling
    try:
        # Create MCP agent for enhanced classification
        classifier_agent = MCPAgent("issue_classifier")
        await classifier_agent.discover_tools()
        
        # Use MCP tools for enhanced classification if available
        if "search_policies" in classifier_agent.available_tools:
            try:
                # Use policy search to enhance classification
                policy_result = await classifier_agent.invoke_tool("search_policies", {"query": message})
                if policy_result:
                    print(f"[MCP] Policy search result used for classification: {policy_result}")
            except Exception as e:
                print(f"[MCP] Policy search failed during classification: {e}")
                print(f"[MCP] Full traceback: {traceback.format_exc()}")
    except Exception as e:
        print(f"[MCP] MCP tool integration failed: {e}")
        import traceback
        print(f"[MCP] Full traceback: {traceback.format_exc()}")
        print("[MCP] Continuing with direct classification")
    
    prompt = (
        "You are an expert IT support agent. Classify the main issue described in the following conversation as one of: [IT Issue, Electric Issue, Other].\n"
        "You have access to MCP tools for enhanced classification. Consider all user and assistant messages for context.\n"
        "Focus on the main user problem, not just the last message.\n"
        "If the user confirms ticket creation, use the main issue discussed previously for classification.\n"
        "\n"
        "AVAILABLE MCP TOOLS FOR CLASSIFICATION:\n"
        "- search_policies: Search company policies for relevant information\n"
        "- extract_user_info: Extract employee information from conversation\n"
        "- lookup_employee: Look up employee details\n"
        "\n"
        "IMPORTANT CLASSIFICATION GUIDELINES:\n"
        "\n"
        "ELECTRIC ISSUES include:\n"
        "- Physical power problems: power outage, electric outlet not working, socket broken, lamp not working, circuit tripped, voltage fluctuation, fuse blown, breaker tripped\n"
        "- Physical electrical equipment: UPS, generator, extension cord, sparking, burned plug, cable damage, short circuit, power supply, power board, safety breaker, PDU\n"
        "- Physical network infrastructure: network cable issue, patch cord, face plate damaged, RJ45 issue, cable tracing, physical network not working, projector/LCD cable, RFID system networking, VOIP phone installation, servers/racks/switches mounting, PDU installation\n"
        "- Electrical services: generator service, UPS service, stabilizer service, insect killer maintenance, lift room/data center power room maintenance\n"
        "\n"
        "IT ISSUES include:\n"
        "- Software problems: Android Studio, browser, Docker service, Dot Net Framework, host file, software installation, application error, others (specify software)\n"
        "- Network connectivity (logical): WiFi not working, internet connection issues, VPN connection problems, network configuration issues\n"
        "- Hardware problems: printer issues (cartridge empty, paper jam, printer not working), workstation problems (battery, charger issue, display issue, headphones issue, keyboard issue, computer not starting, password reset)\n"
        "- Authentication issues: Forticlient login issues, Fortitoken reissuance, Pulse Secure not working, VPN login problems\n"
        "- Web access issues: certificate error, website not working, web access denied\n"
        "\n"
        "KEY DIFFERENTIATION:\n"
        "- WiFi/Network connectivity issues are IT issues (logical network problems)\n"
        "- Physical network cabling issues are Electric issues (physical infrastructure)\n"
        "- Software and configuration problems are IT issues\n"
        "- Physical electrical problems are Electric issues\n"
        "\n"
        "Example classifications:\n"
        "User: My WiFi is not working\n"
        "Category: IT Issue (network connectivity problem)\n"
        "\n"
        "User: The network cable is broken\n"
        "Category: Electric Issue (physical infrastructure)\n"
        "\n"
        "User: My computer won't turn on\n"
        "Category: IT Issue (workstation hardware problem)\n"
        "\n"
        "User: The power outlet is not working\n"
        "Category: Electric Issue (physical electrical problem)\n"
        "\n"
        "User: The lamp above my workstation isn't working\n"
        "Category: Electric Issue (physical electrical problem)\n"
        "\n"
        "Now classify this issue:\n"
        f"User message: {message}\n"
        "Category:"
    )
    
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert IT support agent. Classify issues accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        end_time = time.time()
        
        classification = completion.choices[0].message.content.strip().lower()
        print(f"[AGENT] classify_issue_type_llm LLM completion took {end_time - start_time:.2f}s")
        print(f"[AGENT] LLM raw classification output: {classification}")
        
        # Log routing decision
        print(f"[AGENT_ROUTING] Message: '{message[:50]}...', Classification: {classification}, Time: {end_time - start_time:.2f}s")
        
        # Map classification to template path
        if "electric" in classification:
            return "electric"
        elif "it" in classification:
            return "it"
        elif "other" in classification:
            return "other"
        else:
            return "it"  # Default fallback, but now only if not 'other'
            
    except Exception as e:
        print(f"[ERROR] Issue classification failed: {e}")
        print(f"[AGENT_ROUTING] Message: '{message[:50]}...', Classification: ERROR, Fallback: it")
        return "it"  # Default fallback

async def analyze_conversation_and_extract_problem(conversation_history: List[Dict[str, str]], current_message: str = None) -> Dict[str, Any]:
    """Analyze entire conversation history to extract user's problem, generate subject and description"""
    try:
        # Format conversation for LLM analysis
        conversation_text = ""
        for msg in conversation_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            conversation_text += f"{role}: {content}\n"
        
        if current_message:
            conversation_text += f"Current: {current_message}\n"
        
        # Use LLM to analyze the conversation and extract key information
        analysis_prompt = f"""Analyze this IT support conversation and extract the user's problem, generate a subject line, and create a detailed description.

Conversation:
{conversation_text}

Please provide:
1. The main problem the user is experiencing (be specific and technical)
2. A clear, specific subject line for the support ticket
3. A detailed description including context and any troubleshooting steps already attempted

IMPORTANT GUIDELINES:
- Focus on the actual technical problem, not ticket creation requests
- Be specific about the issue (e.g., "WiFi connectivity problems" not just "network issues")
- Extract the original problem from the conversation, not the current request
- For network issues, specify if it's WiFi, LAN, VPN, etc.
- For hardware issues, specify the component (battery, charger, display, etc.)

Respond in JSON format:
{{
    "problem_description": "The specific technical issue the user reported",
    "subject": "Clear, specific subject line",
    "description": "Detailed description with context and troubleshooting steps"
}}

Examples:
- WiFi issue: "WiFi connectivity problems on laptop"
- Printer issue: "Printer not responding to print jobs"
- VPN issue: "VPN connection problems"
"""
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert IT support analyst. Analyze conversations to extract specific technical problems and create professional ticket content."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"[DEBUG] Conversation analysis result: {result_text}")
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            return {
                "problem_description": result.get("problem_description", ""),
                "subject": result.get("subject", ""),
                "description": result.get("description", "")
            }
        else:
            print("[WARNING] LLM did not return valid JSON for conversation analysis")
            return {
                "problem_description": "",
                "subject": "",
                "description": ""
            }
            
    except Exception as e:
        print(f"[ERROR] Conversation analysis failed: {e}")
        return {
            "problem_description": "",
            "subject": "",
            "description": ""
        }

def get_employee_info(requester):
    """Get employee info by name or ID (from old code)"""
    # Try to match by employee_name or employee_id
    for emp in EMPLOYEE_DATA:
        if requester.lower() in (emp.get('employee_name', '').lower(), emp.get('employee_id', '').lower()):
            return emp
    return None

def extract_user_info_from_history(current_message, conversation_history):
    """Extract user info from conversation history (from old code)"""
    context = {}
    # Combine all previous user messages to get the full context
    full_text = " ".join([msg['content'] for msg in conversation_history if msg['role'] == 'user'] + [current_message])

    # Extract employee name/id (keep as before)
    name_match = re.search(r"(?:my name is|I am|I'm)\s+([a-zA-Z0-9_]+)", full_text, re.IGNORECASE)
    if name_match:
        emp_id = name_match.group(1)
        context['employee_id'] = emp_id  # Store as employee_id to match the data file
        # Look up in employee data
        try:
            with open(EMPLOYEE_PATH, 'r') as f:
                for line in f:
                    employee = json.loads(line)
                    if employee.get('employee_id', '').lower() == emp_id.lower():
                        context['employee_name'] = employee.get('employee_name', emp_id)
                        break
                if 'employee_name' not in context:
                    context['employee_name'] = emp_id  # Fallback to using emp_id as name
        except FileNotFoundError:
            print(f"[ERROR] Employee data file not found at {EMPLOYEE_PATH}")
            context['employee_name'] = emp_id  # Fallback to using emp_id as name

    # --- LLM-based problem description extraction ---
    try:
        llm_prompt = (
            "Given the following IT support chat transcript, extract a concise and clear problem description. "
            "Summarize what the user's issue is, and include any solutions or troubleshooting steps the user has already tried. "
            "Be specific and include all relevant details.\n\n"
            f"Chat transcript:\n{full_text}\n\nProblem Description:"
        )
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert IT helpdesk assistant. Extract a clear, concise problem description from the chat, including any solutions the user tried and what the issue is."},
                {"role": "user", "content": llm_prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        problem_description = response.choices[0].message.content
        if problem_description:
            context['problem_description'] = problem_description.strip()
    except Exception as e:
        print(f"[ERROR] LLM problem description extraction failed: {e}")
        # fallback: leave problem_description empty or use previous regex if desired
        context['problem_description'] = ''

    return context

def generate_subject_from_description(description: str, conversation_history: Optional[list] = None) -> str:
    """Generate subject from description (from old code)"""
    prompt = (
        "Given the following IT support issue description, generate a clear and precise subject line. "
        "The subject should be concise, specific, and similar in style to: Website is not accessible, Software access required, Issues with the air conditioning, etc.\n\n"
    )
    if conversation_history:
        for turn in conversation_history:
            role = turn.get('role', 'user').capitalize()
            content = turn.get('content', '')
            prompt += f"{role}: {content}\n"
    prompt += f"Description: {description}\nSubject:"

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert IT helpdesk assistant. Generate a clear, precise subject line for a support ticket."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20
        )
        subject = response.choices[0].message.content
        if subject is not None:
            subject = subject.strip()
            subject = subject.split('\n')[0].strip()
            return subject
        else:
            return "IT Support Ticket"
    except Exception as e:
        logging.error(f"LLM subject generation failed: {e}")
        # Fallback to old logic
        if not description:
            return "IT Support Ticket"
        subject = description.strip().split('.')[0]
        words = subject.split()
        if len(words) > 8:
            subject = ' '.join(words[:8]) + "..."
        return subject.capitalize()

def generate_ticket_artifact(employee_name, problem_description):
    """Generate ticket artifact (from old code)"""
    return {
        "ticket_id": str(uuid.uuid4()),
        "employee_name": employee_name or "Unknown",
        "problem_description": problem_description or "No description provided",
        "status": "Open"
    }

def load_template_fields(template_path):
    """Load template fields from file with proper error handling"""
    try:
        print(f"[DEBUG] load_template_fields: Loading from {template_path}")
        
        if not os.path.exists(template_path):
            print(f"[ERROR] Template file not found at {template_path}")
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"[DEBUG] load_template_fields: Read {len(content)} characters from file")
            template = json.loads(content)
        
        fields = template.get("fields", [])
        print(f"[DEBUG] load_template_fields: Loaded {len(fields)} fields")
        return fields
        
    except FileNotFoundError as e:
        print(f"[ERROR] Template file not found: {e}")
        raise e
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in template file: {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] Error loading template fields: {e}")
        raise e

def build_ordered_ticket(ticket, template_fields):
    """Build ordered ticket (from old code)"""
    ordered_ticket = {}
    for field in template_fields:
        name = field["name"]
        value = ticket.get(name)
        if value is not None and value != "":
            ordered_ticket[name] = value
        elif "default" in field:
            ordered_ticket[name] = field["default"]
        else:
            ordered_ticket[name] = "Not Specified"
    return ordered_ticket

def get_template_path_for_issue_type(issue_type):
    """Get template path for issue type with proper error handling"""
    print(f"[DEBUG] get_template_path_for_issue_type called with issue_type: {issue_type}")
    
    try:
        if issue_type == "electric":
            path = ELECTRIC_TEMPLATE_PATH
        else:
            path = TEMPLATE_PATH
        
        # Verify the file exists
        if not os.path.exists(path):
            print(f"[WARNING] Template file not found at {path}")
            # Fallback to default template
            path = TEMPLATE_PATH
            if not os.path.exists(path):
                print(f"[ERROR] Default template file not found at {path}")
                raise FileNotFoundError(f"Template file not found: {path}")
        
        print(f"[DEBUG] Template path selected: {path}")
        return path
    except Exception as e:
        print(f"[ERROR] Error getting template path: {e}")
        # Return default template path as fallback
        return TEMPLATE_PATH

async def fill_ticket_with_llm_and_fuzzy(template_fields, user_message, conversation_history, original_context=None, threshold=70):
    """Fill ticket with LLM and fuzzy matching using MCP tools with fallback"""
    print("[DEBUG] fill_ticket_with_llm_and_fuzzy: Called with:")
    print(f"  template_fields: {[f['name'] for f in template_fields]}")
    print(f"  user_message: {user_message}")
    print(f"  conversation_history: {conversation_history}")
    print(f"  original_context: {original_context}")
    print("[DEBUG] fill_ticket_with_llm_and_fuzzy: Using MCP-aware ticket filling with fallback")
    
    try:
        # Add timeout protection for the entire function
        import asyncio
        
        # Try MCP tools first
        try:
            print("[DEBUG] Attempting MCP tool usage...")
            ticket_agent = MCPAgent("ticket_filler")
            await ticket_agent.discover_tools()
            
            # Use MCP tools for enhanced field extraction
            if "extract_user_info" in ticket_agent.available_tools:
                try:
                    user_info_result = await ticket_agent.invoke_tool("extract_user_info", {
                        "current_message": user_message,
                        "conversation_history": conversation_history
                    })
                    if user_info_result:
                        print(f"[MCP] Enhanced user info extraction: {user_info_result}")
                        if original_context:
                            original_context.update(user_info_result)
                        else:
                            original_context = user_info_result
                except Exception as e:
                    print(f"[MCP] User info extraction failed: {e}")
                    import traceback
                    print(f"[MCP] Full traceback: {traceback.format_exc()}")
        except Exception as e:
            print(f"[MCP] MCP tool usage failed: {e}")
            import traceback
            print(f"[MCP] Full traceback: {traceback.format_exc()}")
            print("[MCP] Continuing with direct processing")
        
        # Only fill non-autofill fields with agent extraction
        fields_to_fill = [f for f in template_fields if f.get("type") != "autofill"]
        print(f"[DEBUG] fill_ticket_with_llm_and_fuzzy: Processing {len(fields_to_fill)} non-autofill fields")
        
        # Set timeout for LLM operations
        llm_start = time.time()
        
        # Use asyncio.wait_for to add timeout protection
        extracted_fields_task = asyncio.create_task(
            _extract_fields_with_timeout(user_message, conversation_history, fields_to_fill)
        )
        
        try:
            extracted_fields = await asyncio.wait_for(extracted_fields_task, timeout=15.0)
        except asyncio.TimeoutError:
            print("[ERROR] Field extraction timed out after 15 seconds")
            extracted_fields = {}
        
        llm_end = time.time()
        print(f"[AGENT] fill_ticket_with_llm_and_fuzzy field extraction LLM took {llm_end - llm_start:.2f}s")
        
        # Merge agent-extracted fields with original context and user_message for robust fallback
        merged_context = {}
        if original_context:
            merged_context.update(original_context)
        merged_context.update(extracted_fields)
        merged_context['user_message'] = user_message
        
        # Use direct ticket filling with MCP support
        result = fill_ticket_from_context_with_fuzzy_and_employee(template_fields, merged_context, threshold)
        print(f"[DEBUG] fill_ticket_with_llm_and_fuzzy: Completed successfully")
        return result
        
    except Exception as e:
        print(f"[ERROR] fill_ticket_with_llm_and_fuzzy failed: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        # Return empty ticket as fallback
        return {}

async def _extract_fields_with_timeout(user_message, conversation_history, fields_to_fill):
    """Extract fields with timeout protection"""
    try:
        return extract_fields_with_agent(user_message, conversation_history, fields_to_fill)
    except Exception as e:
        print(f"[ERROR] Field extraction failed: {e}")
        return {}

def extract_fields_with_agent(user_message, conversation_history, fields_to_fill):
    """Extract fields using MCP-aware agent with timeout protection"""
    try:
        # Filter out subcategory and item fields since they're handled by template-based classification
        field_names = [f["name"] for f in fields_to_fill if "subcategory" not in f["name"].lower() and "item" not in f["name"].lower()]
        
        # Only extract fields that actually exist in the template
        available_fields = [f["name"] for f in fields_to_fill]
        print(f"[DEBUG] Available fields in template: {available_fields}")
        
        # Build dynamic prompt based on available fields
        fields_to_extract = []
        for field_name in field_names:
            if field_name in available_fields:
                fields_to_extract.append(field_name)
        
        if not fields_to_extract:
            print("[DEBUG] No fields to extract - all handled by template classification")
            return {}
        
        prompt_template = f"""You are an IT helpdesk ticketing agent. Extract the following fields from the user's message and conversation. Only include these fields (as JSON, with empty string if not found):
{chr(10).join([f"- {field}" for field in fields_to_extract])}

Note: Subcategory and Item are handled separately by template-based classification.

Return the extracted fields as JSON.

Conversation:
{{chat_history}}
User: {{input}}
"""
        prompt = prompt_template

        # Compose conversation for context
        conversation = ""
        if conversation_history:
            for turn in conversation_history:
                role = turn.get('role', 'user').capitalize()
                content = turn.get('content', '')
                conversation += f"{role}: {content}\n"
        conversation += f"User: {user_message}\n"
        agent_input = f"{prompt}\nConversation:\n{conversation}"
        print(f"[DEBUG] Agent Extraction Prompt:\n{agent_input}")
        
        # Add timeout protection for LLM call
        import asyncio
        import concurrent.futures
        
        def call_llm_sync():
            try:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are a smart IT helpdesk ticketing agent. Extract fields from the user's message and return as JSON."},
                        {"role": "user", "content": agent_input}
                    ],
                    temperature=0,
                    max_tokens=200
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}")
                return ""
        
        # Use ThreadPoolExecutor to add timeout protection
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_llm_sync)
            try:
                result_text = future.result(timeout=10.0)  # 10 second timeout
            except concurrent.futures.TimeoutError:
                print("[ERROR] LLM call timed out after 10 seconds")
                return {}
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}")
                return {}
        
        print(f"[DEBUG] Agent final answer: {result_text}")

        # Try to parse JSON from the agent's final answer
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            extracted = json.loads(match.group(0))
        else:
            print(f"[WARNING] Agent did not return a parsable JSON object in its final answer.")
            extracted = {}
            
        print(f"[DEBUG] Agent extracted fields: {extracted}")
        return extracted
        
    except Exception as e:
        print(f"[ERROR] Agent extraction failed: {e}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        return {}

def fill_ticket_from_context_with_fuzzy_and_employee(template_fields, context, threshold=70):
    """Fill ticket from context with fuzzy matching (from old code)"""
    print("[DEBUG] fill_ticket_from_context_with_fuzzy_and_employee called!")
    print("[DEBUG] Context received:", context)
    print("[DEBUG] Context keys:", list(context.keys()) if context else "None")
    ticket = {}
    sanitized_context = {str(k): v for k, v in context.items() if isinstance(v, (str, int, float, bool))}
    print("[DEBUG] Sanitized context for fuzzy matching:", sanitized_context)
    print("[DEBUG] Context values for fuzzy matching:", list(sanitized_context.values()))
    emp_info = None
    requester_input = sanitized_context.get('employee_name') or sanitized_context.get('employee_id') or sanitized_context.get('Requester(Required)', '')
    requester_input = str(requester_input)
    all_emp_ids = [emp['employee_id'] for emp in EMPLOYEE_DATA]
    all_emp_names = [emp['employee_name'] for emp in EMPLOYEE_DATA]
    best_emp, score, _ = process.extractOne(requester_input, all_emp_ids + all_emp_names, scorer=fuzz.token_sort_ratio) if requester_input else (None, 0, None)
    for emp in EMPLOYEE_DATA:
        if emp['employee_id'] == best_emp or emp['employee_name'] == best_emp:
            emp_info = emp
            break
    if emp_info:
        print(f"[DEBUG] Fuzzy matched requester '{requester_input}' to '{best_emp}' (score={score}) -> {emp_info}")
    else:
        print(f"[WARNING] No employee match found for requester '{requester_input}'. Using default values where needed.")
    
    # Extract the original problem from conversation history if available
    problem_description = sanitized_context.get("problem_description", "") or sanitized_context.get("user_message", "")
    
    # If we have conversation history, try to find the original problem description
    if sanitized_context.get("conversation_history"):
        conversation_history = sanitized_context.get("conversation_history", [])
        # Look for the first user message that describes an actual problem
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                # Look for messages that describe actual problems (not ticket requests)
                if any(word in content for word in ["wifi", "network", "internet", "connect", "printer", "computer", "electric", "power", "light", "outlet", "not working", "broken", "issue", "problem"]) and not any(word in content for word in ["create ticket", "generate ticket", "open ticket", "make ticket"]):
                    problem_description = msg.get("content", "")
                    print(f"[DEBUG] Found original problem in conversation: {problem_description}")
                    break
    
    # If no problem found in conversation, try to extract from context
    if not problem_description or problem_description == sanitized_context.get("user_message", ""):
        # Look for problem_description in context
        if sanitized_context.get("problem_description"):
            problem_description = sanitized_context.get("problem_description")
            print(f"[DEBUG] Using problem_description from context: {problem_description}")
        else:
            # Try to extract from the original user message if it's not a ticket request
            original_message = sanitized_context.get("user_message", "")
            if original_message and not any(word in original_message.lower() for word in ["create ticket", "generate ticket", "open ticket", "make ticket"]):
                problem_description = original_message
                print(f"[DEBUG] Using original user message as problem: {problem_description}")
            else:
                problem_description = "Not Specified"
                print(f"[DEBUG] No problem description found, using default")
    
    # Use LLM to analyze conversation and extract problem, subject, and description
    conversation_analysis = None
    if sanitized_context.get("conversation_history"):
        try:
            # Import the async function and run it synchronously
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            conversation_analysis = loop.run_until_complete(
                analyze_conversation_and_extract_problem(
                    sanitized_context.get("conversation_history", []),
                    sanitized_context.get("user_message", "")
                )
            )
            loop.close()
            
            if conversation_analysis and conversation_analysis.get("problem_description"):
                problem_description = conversation_analysis.get("problem_description")
                print(f"[DEBUG] LLM extracted problem: {problem_description}")
        except Exception as e:
            print(f"[ERROR] Conversation analysis failed: {e}")
    
    def call_llm(prompt):
        result = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a smart helpdesk assistant. Select the most appropriate subcategory and item for this issue."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        return result.choices[0].message.content or ""
    
    subcat_val, item_val = llm_select_subcategory_item(problem_description, template_fields, call_llm)
    
    # DYNAMIC mapping from template autofill field names to employee data keys
    # This adapts based on the template structure
    AUTOFILL_FIELD_MAP = {}
    for field in template_fields:
        name = field["name"]
        if "requester" in name.lower():
            AUTOFILL_FIELD_MAP[name] = "employee_name"
        elif "id" in name.lower():
            AUTOFILL_FIELD_MAP[name] = "employee_id"
        elif "competency" in name.lower():
            AUTOFILL_FIELD_MAP[name] = "SL_competency"
        elif "floor" in name.lower():
            AUTOFILL_FIELD_MAP[name] = "floor_information"
        elif "machine" in name.lower():
            AUTOFILL_FIELD_MAP[name] = "machine_name"
    
    print(f"[DEBUG] Dynamic AUTOFILL_FIELD_MAP: {AUTOFILL_FIELD_MAP}")
    
    for field in template_fields:
        name = field["name"]
        ftype = field.get("type")
        print(f"[DEBUG] Filling field '{name}' of type '{ftype}'")
        
        if ftype == "autofill":
            # DYNAMIC autofill field handling based on field name and context
            mapped_key = AUTOFILL_FIELD_MAP.get(name)
            
            # Check if we have extracted fields from LLM first
            if "SL Competency(Required)" in name and sanitized_context.get("SL Competency(Required)"):
                ticket[name] = sanitized_context["SL Competency(Required)"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif "SL Competency" in name and sanitized_context.get("SL Competency"):
                ticket[name] = sanitized_context["SL Competency"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif "Floor Information(Required)" in name and sanitized_context.get("Floor Information(Required)"):
                ticket[name] = sanitized_context["Floor Information(Required)"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif "Floor Information" in name and sanitized_context.get("Floor Information"):
                ticket[name] = sanitized_context["Floor Information"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif "Machine Name(Required)" in name and sanitized_context.get("Machine Name(Required)"):
                ticket[name] = sanitized_context["Machine Name(Required)"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif "Machine Name" in name and sanitized_context.get("Machine Name"):
                ticket[name] = sanitized_context["Machine Name"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif mapped_key and sanitized_context.get(mapped_key):
                # Use context data if available (from frontend user info)
                ticket[name] = sanitized_context[mapped_key]
                print(f"[DEBUG] Field '{name}' autofilled from context: {ticket[name]}")
            elif mapped_key and emp_info and mapped_key in emp_info:
                # Use employee data if available (fallback)
                ticket[name] = emp_info[mapped_key]
                print(f"[DEBUG] Field '{name}' autofilled from employee data: {ticket[name]}")
            elif field.get("default"):
                # Use template default
                ticket[name] = field["default"]
                print(f"[DEBUG] Field '{name}' autofilled from template default: {ticket[name]}")
            else:
                # Dynamic default based on field name
                if "requester" in name.lower():
                    ticket[name] = sanitized_context.get("employee_name", "Not Specified")
                elif "type" in name.lower():
                    ticket[name] = "Incident"
                elif "status" in name.lower():
                    ticket[name] = "Open"
                elif "mode" in name.lower():
                    ticket[name] = "Web Form"
                elif "category" in name.lower():
                    # Use the category from the template
                    ticket[name] = field.get("default", "IT Helpdesk Issues")
                elif "priority" in name.lower():
                    ticket[name] = "Not Specified"
                elif "technician" in name.lower():
                    ticket[name] = "Not Specified"
                else:
                    ticket[name] = "Not Specified"
                print(f"[DEBUG] Field '{name}' autofilled with dynamic default: {ticket[name]}")
            
        elif ftype == "dropdown" and "subcategory" in name.lower():
            # Always use the template-based classification first
            ticket[name] = subcat_val
            print(f"[DEBUG] Field '{name}' set by template-based LLM: {subcat_val}")
        elif ftype == "dependent_dropdown" and "item" in name.lower():
            # Always use the template-based classification first
            ticket[name] = item_val
            print(f"[DEBUG] Field '{name}' set by template-based LLM: {item_val}")
        elif ftype == "dropdown":
            # DYNAMIC dropdown field handling with dynamic validation
            options = field.get("options", [])
            mapped_key = AUTOFILL_FIELD_MAP.get(name)
            
            # Try employee data first
            if mapped_key and emp_info and mapped_key in emp_info:
                emp_value = emp_info[mapped_key]
                if emp_value in options:
                    ticket[name] = emp_value
                    print(f"[DEBUG] Field '{name}' set from employee data: {emp_value}")
                else:
                    # Fuzzy match employee data to options
                    best_option, score, _ = process.extractOne(emp_value, options, scorer=fuzz.token_sort_ratio)
                    if score > 70:
                        ticket[name] = best_option
                        print(f"[DEBUG] Field '{name}' fuzzy matched from employee data: {best_option} (score={score})")
                    else:
                        # Use dynamic validation to determine if first option is appropriate
                        if validate_field_dynamically(name, options[0] if options else "Not Specified", "dropdown", options):
                            ticket[name] = options[0] if options else "Not Specified"
                            print(f"[DEBUG] Field '{name}' set to first option: {ticket[name]}")
                        else:
                            ticket[name] = "Not Specified"
                            print(f"[DEBUG] Field '{name}' set to Not Specified due to validation")
            
            # Try context data
            elif mapped_key and sanitized_context.get(mapped_key):
                context_value = sanitized_context[mapped_key]
                if context_value in options:
                    ticket[name] = context_value
                    print(f"[DEBUG] Field '{name}' set from context: {context_value}")
                else:
                    # Fuzzy match context data to options
                    best_option, score, _ = process.extractOne(context_value, options, scorer=fuzz.token_sort_ratio)
                    if score > 70:
                        ticket[name] = best_option
                        print(f"[DEBUG] Field '{name}' fuzzy matched from context: {best_option} (score={score})")
                    else:
                        # Use dynamic validation
                        if validate_field_dynamically(name, options[0] if options else "Not Specified", "dropdown", options):
                            ticket[name] = options[0] if options else "Not Specified"
                            print(f"[DEBUG] Field '{name}' set to first option: {ticket[name]}")
                        else:
                            ticket[name] = "Not Specified"
                            print(f"[DEBUG] Field '{name}' set to Not Specified due to validation")
            
            # Try problem description matching
            elif options and problem_description:
                best_option, score, _ = process.extractOne(problem_description, options, scorer=fuzz.token_sort_ratio)
                if score > 50:
                    ticket[name] = best_option
                    print(f"[DEBUG] Field '{name}' fuzzy matched from problem: {best_option} (score={score})")
                else:
                    # Use dynamic validation
                    if validate_field_dynamically(name, options[0] if options else "Not Specified", "dropdown", options):
                        ticket[name] = options[0] if options else "Not Specified"
                        print(f"[DEBUG] Field '{name}' set to first option: {ticket[name]}")
                    else:
                        ticket[name] = "Not Specified"
                        print(f"[DEBUG] Field '{name}' set to Not Specified due to validation")
            
            # Fallback to first option with dynamic validation
            else:
                if validate_field_dynamically(name, options[0] if options else "Not Specified", "dropdown", options):
                    ticket[name] = options[0] if options else "Not Specified"
                    print(f"[DEBUG] Field '{name}' set to first option: {ticket[name]}")
                else:
                    ticket[name] = "Not Specified"
                    print(f"[DEBUG] Field '{name}' set to Not Specified due to validation")
        elif ftype == "dependent_dropdown":
            dep = field.get("dependency")
            dep_val = ticket.get(dep, list(field.get("options", {}).keys())[0])
            options = field.get("options", {}).get(dep_val, ["Not Specified"])
            problem_text = problem_description
            if options:
                best_option, option_score, _ = process.extractOne(problem_text, options, scorer=fuzz.token_sort_ratio)
                # Use dynamic validation for dependent dropdown
                if validate_field_dynamically(name, best_option, "dependent_dropdown", options):
                    ticket[name] = best_option
                    print(f"[DEBUG] Field '{name}' dependent fuzzy matched: {ticket[name]} (score={option_score})")
                else:
                    ticket[name] = "Not Specified"
                    print(f"[DEBUG] Field '{name}' set to Not Specified due to validation")
            else:
                ticket[name] = "Not Specified"
                print(f"[WARNING] Field '{name}' dependent dropdown has no options, set to Not Specified")
        elif "subject" in name.lower():
            # Check if we have extracted fields from LLM
            if sanitized_context.get("Subject(Required)"):
                ticket[name] = sanitized_context["Subject(Required)"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif sanitized_context.get("Subject"):
                ticket[name] = sanitized_context["Subject"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif conversation_analysis and conversation_analysis.get("subject"):
                # Use LLM-generated subject from conversation analysis
                ticket[name] = conversation_analysis.get("subject")
                print(f"[DEBUG] Field '{name}' generated from conversation analysis: {ticket[name]}")
            elif problem_description and problem_description != "Not Specified" and problem_description != sanitized_context.get("user_message", ""):
                try:
                    # Use LLM to generate a specific subject based on the actual problem
                    subject_prompt = f"""Generate a clear, specific subject line for this IT support issue. The subject should be concise and descriptive.

Problem: {problem_description}

Examples of good subjects:
- "WiFi connectivity issues on laptop"
- "Printer not responding to print jobs"
- "Computer won't start up"
- "VPN connection problems"
- "Software installation error"
- "Network connectivity problems"
- "Internet connection issues"

Subject:"""
                    
                    response = client.chat.completions.create(
                        model=AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=[
                            {"role": "system", "content": "You are an IT helpdesk assistant. Generate a clear, specific subject line for support tickets."},
                            {"role": "user", "content": subject_prompt}
                        ],
                        temperature=0,
                        max_tokens=30
                    )
                    subject = response.choices[0].message.content.strip()
                    if subject:
                        ticket[name] = subject
                        print(f"[DEBUG] Field '{name}' generated from problem_description: {subject}")
                    else:
                        # Fallback to simple subject generation
                        subject = generate_subject_from_description(problem_description)
                        ticket[name] = subject
                        print(f"[DEBUG] Field '{name}' generated with fallback: {subject}")
                except Exception as e:
                    print(f"[ERROR] Subject generation failed: {e}")
                    # Fallback to simple subject generation
                    subject = generate_subject_from_description(problem_description)
                    ticket[name] = subject
                    print(f"[DEBUG] Field '{name}' generated with error fallback: {subject}")
            else:
                # Dynamic default based on issue type
                issue_type = sanitized_context.get("issue_type", "it")
                if issue_type == "electric":
                    ticket[name] = "Electric Issue Support Request"
                else:
                    ticket[name] = "IT Support Request"
                print(f"[DEBUG] Field '{name}' using dynamic default: {ticket[name]}")
        elif "description" in name.lower():
            # Check if we have extracted fields from LLM
            if sanitized_context.get("Description"):
                ticket[name] = sanitized_context["Description"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif sanitized_context.get("Description(Required)"):
                ticket[name] = sanitized_context["Description(Required)"]
                print(f"[DEBUG] Field '{name}' set from extracted fields: {ticket[name]}")
            elif conversation_analysis and conversation_analysis.get("description"):
                # Use LLM-generated description from conversation analysis
                ticket[name] = conversation_analysis.get("description")
                print(f"[DEBUG] Field '{name}' generated from conversation analysis: {ticket[name]}")
            elif problem_description and problem_description != "Not Specified" and problem_description != sanitized_context.get("user_message", ""):
                try:
                    # Use LLM to generate a detailed description based on the actual problem
                    desc_prompt = f"""Generate a detailed description for this IT support ticket. Include relevant context and troubleshooting steps already provided.

Problem: {problem_description}

Conversation Context: {conversation_history if sanitized_context.get("conversation_history") else "No additional context"}

Generate a comprehensive description that includes:
1. The specific issue reported
2. Any troubleshooting steps already attempted
3. Current status of the problem
4. Impact on user's work

Description:"""
                    
                    response = client.chat.completions.create(
                        model=AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=[
                            {"role": "system", "content": "You are an IT helpdesk assistant. Generate detailed, professional descriptions for support tickets."},
                            {"role": "user", "content": desc_prompt}
                        ],
                        temperature=0,
                        max_tokens=200
                    )
                    desc = response.choices[0].message.content.strip()
                    if desc:
                        ticket[name] = desc
                        print(f"[DEBUG] Field '{name}' generated from problem_description: {desc}")
                    else:
                        # Fallback to problem description
                        ticket[name] = problem_description
                        print(f"[DEBUG] Field '{name}' filled with problem_description: {problem_description}")
                except Exception as e:
                    print(f"[ERROR] Description generation failed: {e}")
                    # Fallback to problem description
                    ticket[name] = problem_description
                    print(f"[DEBUG] Field '{name}' filled with error fallback: {problem_description}")
            else:
                # Dynamic default based on context
                issue_type = sanitized_context.get("issue_type", "it")
                user_message = sanitized_context.get("user_message", "")
                if user_message and not any(word in user_message.lower() for word in ["create ticket", "generate ticket", "open ticket", "make ticket"]):
                    ticket[name] = user_message
                    print(f"[DEBUG] Field '{name}' filled from user_message: {user_message}")
                else:
                    ticket[name] = "Support ticket created via chatbot"
                    print(f"[DEBUG] Field '{name}' using default description")
        elif "machine" in name.lower():
            # For machine name, use a default or extract from context
            ticket[name] = sanitized_context.get("machine_name", "Not Specified")
            print(f"[DEBUG] Field '{name}' filled from context: {ticket[name]}")
        else:
            value = sanitized_context.get(name, field.get("default", "Not Specified"))
            ticket[name] = value
            if value == field.get("default", "Not Specified"):
                print(f"[WARNING] Field '{name}' filled with default: {value}")
            else:
                print(f"[DEBUG] Field '{name}' filled from context: {value}")
    print("[DEBUG] Ticket returned from fill_ticket_from_context_with_fuzzy_and_employee:", ticket)
    return ticket

def llm_select_subcategory_item(problem_description, template_fields, llm):
    """Use LLM to select subcategory and item with improved accuracy using exact template options"""
    # Find subcategory and item fields in the template
    subcat_field = None
    item_field = None
    for f in template_fields:
        if f["type"] == "dropdown" and "subcategory" in f["name"].lower():
            subcat_field = f
        if f["type"] == "dependent_dropdown" and "item" in f["name"].lower():
            item_field = f
    if not subcat_field or not item_field:
        print("[WARNING] No subcategory or item field found in template for LLM selection.")
        return ("Not Specified", "Not Specified")
    
    subcat_options = subcat_field.get("options", [])
    item_options_dict = item_field.get("options", {})
    
    # Determine template type for better examples
    template_type = "IT Helpdesk" if "Workstation" in subcat_options else "Electric Issues"
    
    # Create formatted strings for the prompt
    subcat_str = "\n".join([f"- {option}" for option in subcat_options])
    item_str = "\n".join([f"- {subcat}: {', '.join(items)}" for subcat, items in item_options_dict.items()])
    
    if template_type == "IT Helpdesk":
        prompt = f"""You are an expert IT helpdesk assistant. Given the following problem description, select the MOST APPROPRIATE subcategory and item from the EXACT options provided.

Problem Description: "{problem_description}"

Available subcategories:
{subcat_str}

Available items for each subcategory:
{item_str}

CRITICAL CLASSIFICATION RULES FOR IT HELPDESK:

WORKSTATION subcategory is for:
- Hardware issues: battery, charger, display, headphones, keyboard, computer not starting
- Network connectivity issues: WiFi not working, LAN issues, network connection problems, internet connectivity
- Device-specific problems: laptop issues, desktop issues, mobile device issues
- Physical hardware problems: broken hardware, device malfunctions

SOFTWARE TROUBLESHOOTING subcategory is for:
- Software installation problems
- Application errors and crashes
- Development tools: Android Studio, Docker, etc.
- System software issues
- Program-specific problems

PRINTER subcategory is for:
- Printer hardware issues: paper jams, cartridge problems, printer not working
- Printer connectivity issues
- Print quality problems

VPN subcategory is for:
- VPN connection issues
- Authentication problems: Forticlient, Fortitoken, Pulse Secure
- Remote access problems

VOIP subcategory is for:
- IP phone issues
- Voice communication problems
- Phone system issues

WEB ACCESS subcategory is for:
- Website access problems
- Certificate errors
- Web application issues
- Browser-specific problems

SPECIFIC EXAMPLES FOR NETWORK ISSUES:
- "WiFi not working" → Workstation subcategory, "LAN Issue" item
- "WiFi connectivity problems" → Workstation subcategory, "LAN Issue" item
- "Can't connect to WiFi" → Workstation subcategory, "LAN Issue" item
- "Internet not working" → Workstation subcategory, "LAN Issue" item
- "Network connection issues" → Workstation subcategory, "LAN Issue" item

EXAMPLES FOR OTHER ISSUES:
- "My computer won't start" → Workstation subcategory, "Display Issue" item (closest match)
- "Printer is not working" → Printer subcategory, "Printer Not Working" item
- "Can't access website" → Web Access subcategory, "Website Stop Working" item
- "VPN not connecting" → VPN subcategory, "Forticlient Login Issues" item
- "Android Studio not working" → Software Troubleshooting subcategory, "Android Studio" item

IMPORTANT: WiFi and network connectivity issues ALWAYS go to Workstation subcategory with "LAN Issue" item.

Select the most appropriate subcategory and item for this issue. Respond in JSON as:
{{"subcategory": "...", "item": "..."}}
If none match, use "Not Specified".
"""
    else:  # Electric Issues template
        prompt = f"""You are an expert electric issues assistant. Given the following problem description, select the most appropriate subcategory and item from the EXACT options provided.

Problem Description: "{problem_description}"

Available subcategories:
{subcat_str}

Available items for each subcategory:
{item_str}

IMPORTANT CLASSIFICATION GUIDELINES FOR ELECTRIC ISSUES:

PASSIVE NETWORKING ISSUES subcategory is for:
- Network cable issues: cable tracing, face plates, RJ45 problems
- Network infrastructure: switches, servers, racks, PDU installation
- Network connectivity: ports, VLAN, IP configuration
- Physical network problems: patch cords, network testing

POWER ISSUES subcategory is for:
- Electrical power problems: power supply, voltage fluctuations
- Electrical safety: circuit breakers, short circuits, overload
- Electrical equipment: UPS, power boards, converters
- Electrical infrastructure: plugs, sockets, cables

SERVICES subcategory is for:
- Maintenance services: generator, UPS, stabilizer maintenance
- Facility services: lift room, data center maintenance
- Specialized services: insect killer maintenance

EXAMPLES:
- "Network cable is broken" → Passive networking issues subcategory, "LAN/Network CAT-6/CAT-6A cable required" item
- "Power outlet not working" → Power Issues subcategory, "Power plug and sockets faulty, burned/damaged" item
- "UPS not working" → Power Issues subcategory, "Power issue (no power supply, damage, short circuit, UPS supply issue)" item
- "Generator needs service" → Services subcategory, "Periodic Generator service and maintenance" item

Select the most appropriate subcategory and item for this issue. Respond in JSON as:
{{"subcategory": "...", "item": "..."}}
If none match, use "Not Specified".
"""
    
    print("[DEBUG] LLM subcategory/item selection prompt:\n" + prompt)
    
    try:
        llm_response = llm(prompt)
        print(f"[DEBUG] LLM subcategory/item response: {llm_response}")
        
        # Extract JSON from the response
        match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            subcat = result.get("subcategory", "Not Specified")
            item = result.get("item", "Not Specified")
            
            # Validate against available options
            if subcat not in subcat_options:
                print(f"[WARNING] LLM subcategory '{subcat}' not in options, using 'Not Specified'")
                subcat = "Not Specified"
            valid_items = item_options_dict.get(subcat, ["Not Specified"])
            if item not in valid_items:
                print(f"[WARNING] LLM item '{item}' not in options for subcategory '{subcat}', using 'Not Specified'")
                item = "Not Specified"
            
            # CRITICAL FALLBACK: If this is a WiFi/network issue, force correct classification
            problem_lower = problem_description.lower()
            if any(word in problem_lower for word in ["wifi", "wi-fi", "wireless", "network", "internet", "connect"]) and template_type == "IT Helpdesk":
                if "workstation" in subcat_options and "lan issue" in [i.lower() for i in item_options_dict.get("Workstation", [])]:
                    print(f"[DEBUG] WiFi issue detected, forcing Workstation/LAN Issue classification")
                    subcat = "Workstation"
                    item = "LAN Issue"
            
            return (subcat, item)
        else:
            print("[WARNING] LLM did not return valid JSON for subcategory/item.")
            # CRITICAL FALLBACK: If this is a WiFi/network issue, force correct classification
            problem_lower = problem_description.lower()
            if any(word in problem_lower for word in ["wifi", "wi-fi", "wireless", "network", "internet", "connect"]) and template_type == "IT Helpdesk":
                if "workstation" in subcat_options and "lan issue" in [i.lower() for i in item_options_dict.get("Workstation", [])]:
                    print(f"[DEBUG] WiFi issue detected, forcing Workstation/LAN Issue classification")
                    return ("Workstation", "LAN Issue")
            return ("Not Specified", "Not Specified")
    except Exception as e:
        print(f"[ERROR] LLM subcategory/item selection failed: {e}")
        # CRITICAL FALLBACK: If this is a WiFi/network issue, force correct classification
        problem_lower = problem_description.lower()
        if any(word in problem_lower for word in ["wifi", "wi-fi", "wireless", "network", "internet", "connect"]) and template_type == "IT Helpdesk":
            if "workstation" in subcat_options and "lan issue" in [i.lower() for i in item_options_dict.get("Workstation", [])]:
                print(f"[DEBUG] WiFi issue detected, forcing Workstation/LAN Issue classification")
                return ("Workstation", "LAN Issue")
        return ("Not Specified", "Not Specified")

def is_ticket_creation_intent_dynamic(msg):
    """Dynamic intent detection using LLM - more conservative approach"""
    if not msg:
        return False
    
    try:
        # Use LLM to analyze intent dynamically with more conservative approach
        prompt = f"""Analyze the following user message and determine if they EXPLICITLY want to create a support ticket.

User message: "{msg}"

IMPORTANT: Only respond with "YES" if the user EXPLICITLY requests ticket creation.
Look for clear, direct requests like:
- "create a ticket"
- "generate a ticket" 
- "open a ticket"
- "I want a ticket"
- "please create a ticket"
- "can you create a ticket"

Do NOT respond "YES" for:
- General problem descriptions
- Questions about issues
- Requests for help or solutions
- Indirect mentions of tickets

Respond with only "YES" if the user EXPLICITLY asks for ticket creation, or "NO" if they don't."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at understanding explicit ticket creation requests. Only respond with YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result == "YES"
        
    except Exception as e:
        print(f"[ERROR] Dynamic intent detection failed: {e}, falling back to keyword detection")
        # Fallback to basic keyword detection - more conservative
        lower = msg.lower()
        explicit_keywords = ['create ticket', 'generate ticket', 'open ticket', 'make ticket', 'i want ticket', 'please create ticket', 'can you create ticket']
        return any(keyword in lower for keyword in explicit_keywords)

def is_confirmation_intent_dynamic(msg):
    """Dynamic confirmation detection using LLM"""
    if not msg:
        return False
    
    try:
        prompt = f"""Analyze the following user message and determine if they are confirming or agreeing to something.

User message: "{msg}"

Consider:
- Direct confirmations (yes, yep, yeah, sure, etc.)
- Indirect confirmations (go ahead, please do, ok, etc.)
- Context-specific confirmations
- Various ways users might express agreement

Respond with only "YES" if the user is confirming/agreeing, or "NO" if they're not.
Be liberal in your interpretation."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at understanding user confirmation intent. Only respond with YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result == "YES"
        
    except Exception as e:
        print(f"[ERROR] Dynamic confirmation detection failed: {e}, falling back to keyword detection")
        # Fallback to basic keyword detection
        lower = msg.lower()
        confirmation_words = ['yes', 'yep', 'yeah', 'please do', 'sure', 'go ahead', 'ok', 'okay', 'confirm']
        return any(word in lower for word in confirmation_words)

def generate_dynamic_response(context, intent, issue_type=None):
    """Generate dynamic responses using LLM instead of hardcoded templates"""
    try:
        # Build context for LLM
        context_info = ""
        if context.get('employee_name'):
            context_info += f"Employee: {context['employee_name']}\n"
        if context.get('problem_description'):
            context_info += f"Problem: {context['problem_description']}\n"
        if issue_type:
            context_info += f"Issue Type: {issue_type}\n"
        
        # Generate different prompts based on intent
        if intent == "qa":
            prompt = f"""Generate a helpful response for an IT support chatbot.

Context:
{context_info}

Generate a response that:
- Acknowledges the user's issue
- Provides helpful troubleshooting steps or solutions
- Is professional and supportive
- Offers relevant advice based on the problem type
- Is conversational and natural

Focus on providing immediate help and solutions."""

        elif intent == "ticket_offer":
            prompt = f"""Generate a ticket creation offer for an IT support chatbot.

Context:
{context_info}

Generate a response that:
- Offers to create a support ticket
- Explains the benefits of creating a ticket
- Is polite and professional
- Encourages the user to proceed with ticket creation
- Is conversational and helpful

Example: 'Would you like me to create a support ticket for this issue so our team can assist you further?'"""

        elif intent == "awaiting_confirmation":
            prompt = f"""Generate a confirmation request for ticket creation.

Context:
{context_info}

Generate a response that:
- Asks if the user wants to create a ticket
- Is clear and direct
- Is polite and professional
- Makes it easy for the user to say yes or no

Example: 'Would you like me to create a support ticket for this issue?'"""

        elif intent == "awaiting_details":
            prompt = f"""Generate a request for more details.

Context:
{context_info}

Generate a response that:
- Asks for additional information needed for ticket creation
- Is helpful and specific about what information is needed
- Is polite and professional
- Makes it easy for the user to provide the information

Example: 'I can help create a ticket, but I need a few more details. Could you please confirm your name and describe the problem?'"""

        elif intent == "create_ticket":
            prompt = f"""Generate a ticket creation confirmation response.

Context:
{context_info}

Generate a response that:
- Confirms the ticket is being created
- Provides ticket details
- Is professional and reassuring
- Explains next steps
- Is helpful and informative

Focus on confirming the action and providing next steps."""

        else:
            prompt = f"""Generate a natural, helpful response for an IT support chatbot.

Context:
{context_info}
Intent: {intent}

Generate a response that:
- Is professional and helpful
- Acknowledges the user's issue
- Offers appropriate next steps
- Is conversational and natural
- Matches the intent appropriately

Respond with only the response text, no additional formatting."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful IT support assistant. Generate natural, professional responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[ERROR] Dynamic response generation failed: {e}, using fallback")
        # Fallback responses
        if intent == "create_ticket":
            return "I'll help you create a support ticket for this issue."
        elif intent == "awaiting_confirmation":
            return "Would you like me to create a support ticket for this issue?"
        elif intent == "ticket_offer":
            return "Would you like me to create a support ticket for this issue so our team can assist you further?"
        elif intent == "awaiting_details":
            return "I can help create a ticket, but I need a few more details. Could you please confirm your name and describe the problem?"
        else:
            return "I'm here to help with your IT support needs."

def generate_dynamic_error_message(error_type, context=None):
    """Generate dynamic error messages using LLM"""
    try:
        context_info = ""
        if context:
            context_info = f"Context: {str(context)}\n"
        
        prompt = f"""Generate a helpful error message for an IT support chatbot.

Error Type: {error_type}
{context_info}

Generate a message that:
- Acknowledges the error occurred
- Is apologetic but professional
- Suggests alternative solutions
- Maintains user confidence
- Is conversational and helpful

Respond with only the error message text."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful IT support assistant. Generate professional error messages."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[ERROR] Dynamic error message generation failed: {e}, using fallback")
        return "I apologize, but I encountered an error. Please try again or contact IT support directly."

def validate_field_dynamically(field_name, field_value, field_type, options=None):
    """Dynamic field validation using LLM instead of hardcoded rules"""
    try:
        options_info = ""
        if options:
            options_info = f"Valid options: {', '.join(options)}\n"
        
        prompt = f"""Validate a form field value for an IT support ticket.

Field Name: {field_name}
Field Value: {field_value}
Field Type: {field_type}
{options_info}

Determine if this value is valid and appropriate for the field.
Consider:
- Required vs optional fields
- Data format requirements
- Business logic constraints
- User intent and context

Respond with only "VALID" if the value is acceptable, or "INVALID" if it's not."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at validating form field values. Only respond with VALID or INVALID."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result == "VALID"
        
    except Exception as e:
        print(f"[ERROR] Dynamic field validation failed: {e}, using fallback")
        # Fallback validation
        if field_type == "required" and not field_value:
            return False
        if options and field_value not in options:
            return False
        return True

def determine_routing_dynamically(state, context):
    """Dynamic routing decisions using LLM instead of hardcoded logic"""
    try:
        # Build state summary for LLM
        state_info = f"""
User Message: {state.get('user_message', '')}
Intent: {state.get('intent', '')}
Agent Type: {state.get('agent_type', '')}
Context Keys: {list(context.keys())}
"""
        
        prompt = f"""Determine the next step in an IT support workflow.

Current State:
{state_info}

Available routes:
- ticket_creation: User wants to create a ticket
- end: Conversation should end
- awaiting_confirmation: Waiting for user confirmation
- awaiting_details: Need more information from user

Based on the current state, determine the appropriate next step.
Consider the user's intent, available information, and workflow logic.

Respond with only the route name: ticket_creation, end, awaiting_confirmation, or awaiting_details."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at routing IT support workflows. Only respond with the route name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20
        )
        
        route = response.choices[0].message.content.strip().lower()
        return route
        
    except Exception as e:
        print(f"[ERROR] Dynamic routing failed: {e}, using fallback")
        # Fallback routing logic
        intent = state.get('intent', '')
        if intent == 'create_ticket':
            return 'ticket_creation'
        elif intent in ['awaiting_confirmation', 'awaiting_details']:
            return 'end'
        else:
            return 'end'

def generate_dynamic_tool_description(tool_name, context=None):
    """Generate dynamic tool descriptions using LLM"""
    try:
        context_info = ""
        if context:
            context_info = f"Context: {str(context)}\n"
        
        prompt = f"""Generate a helpful description for an MCP tool.

Tool Name: {tool_name}
{context_info}

Generate a description that:
- Explains what the tool does
- Is clear and concise
- Helps users understand when to use it
- Is professional and technical

Respond with only the description text."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at describing technical tools. Generate clear, helpful descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[ERROR] Dynamic tool description generation failed: {e}, using fallback")
        # Fallback descriptions
        descriptions = {
            'search_policies': 'Search company policies and solutions',
            'extract_user_info': 'Extract employee information from conversation',
            'classify_issue_type': 'Classify if an issue is IT or Electric-related',
            'fill_ticket_fields': 'Fill ticket fields with intelligent matching'
        }
        return descriptions.get(tool_name, f'Tool: {tool_name}')

def generate_dynamic_system_prompt(context=None):
    """Generate dynamic system prompts using LLM"""
    try:
        context_info = ""
        if context:
            context_info = f"Context: {str(context)}\n"
        
        prompt = f"""Generate a system prompt for an IT support chatbot.

{context_info}

The prompt should:
- Define the assistant's role and capabilities
- Include available tools and their purposes
- Set appropriate tone and behavior guidelines
- Be professional and helpful
- Adapt to the current context if provided

Respond with only the system prompt text."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at creating system prompts for AI assistants. Generate clear, effective prompts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[ERROR] Dynamic system prompt generation failed: {e}, using fallback")
        # Fallback system prompt
        return """You are a helpful IT support assistant for Systems Limited. You can help with IT issues, answer questions about policies, and create support tickets when needed. Be professional, helpful, and concise.""" 