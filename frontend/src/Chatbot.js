import React, { useState, useRef, useEffect } from "react";
import "./App.css";
// Remove use-mcp import and replace with custom MCP client
import { getFirestore, doc, getDoc } from "firebase/firestore";
import ReactMarkdown from 'react-markdown';

const BOT_AVATAR  = "https://api.dicebear.com/6.x/thumbs/svg?seed=User";
const USER_AVATAR = "https://api.dicebear.com/6.x/thumbs/svg?seed=Bot";
const API_URL = "http://localhost:8000";

function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Custom MCP client that uses FastAPI proxy
const useCustomMcp = () => {
  const [state, setState] = useState('connecting');
  const [tools, setTools] = useState([]);
  const [log, setLog] = useState([]);

  const addLog = (message) => {
    setLog(prev => [...prev, { timestamp: new Date().toISOString(), message }]);
  };

  const clearStorage = () => {
    setState('connecting');
    setTools([]);
    setLog([]);
  };

  const callTool = async (toolName, params) => {
    try {
      addLog(`Calling tool: ${toolName} with params: ${JSON.stringify(params)}`);
      
      const response = await fetch(`${API_URL}/api/mcp/proxy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          method: 'tools/call',
          params: {
            name: toolName,
            arguments: params
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      addLog(`Tool call result: ${JSON.stringify(result)}`);
      
      // Extract result from MCP response format
      if (result.result) {
        return result.result;
      }
      return result;
    } catch (error) {
      addLog(`Tool call error: ${error.message}`);
      throw error;
    }
  };

  // Check MCP server status and get tools on mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        // Check if MCP server is running
        const statusResponse = await fetch(`${API_URL}/api/mcp/status`);
        const status = await statusResponse.json();
        
        if (status.status === 'connected') {
          setState('connected');
          addLog('MCP server connected via proxy');
          
          // Get available tools from MCP server
          try {
            const toolsResponse = await fetch(`${API_URL}/api/mcp/tools`);
            const toolsData = await toolsResponse.json();
            
            if (toolsData.tools && Array.isArray(toolsData.tools)) {
              setTools(toolsData.tools);
              addLog(`Discovered ${toolsData.tools.length} MCP tools`);
            } else {
              // Fallback to known tools
              setTools([
                { name: 'main_rag_chat', description: 'Main RAG chat tool' },
                { name: 'create_ticket', description: 'Create support ticket' },
                { name: 'complete_ticket', description: 'Complete ticket' },
                { name: 'a2a_task', description: 'Agent-to-agent task' },
                { name: 'search_policies', description: 'Search policies' },
                { name: 'extract_user_info', description: 'Extract user info' },
                { name: 'classify_issue_type', description: 'Classify issue type' }
              ]);
              addLog('Using fallback tool list');
            }
          } catch (toolsError) {
            addLog(`Failed to get tools: ${toolsError.message}`);
            // Use fallback tools
            setTools([
              { name: 'main_rag_chat', description: 'Main RAG chat tool' },
              { name: 'create_ticket', description: 'Create support ticket' },
              { name: 'complete_ticket', description: 'Complete ticket' },
              { name: 'a2a_task', description: 'Agent-to-agent task' }
            ]);
          }
        } else {
          setState('error');
          addLog('MCP server not available');
        }
      } catch (error) {
        setState('error');
        addLog(`Status check failed: ${error.message}`);
      }
    };

    checkStatus();
  }, []);

  return { state, tools, callTool, log, clearStorage };
};

function Chatbot({ username, userUid, onLogout }) {
  const [messages, setMessages] = useState([
    { 
      sender: "bot", 
      text: `Hi, welcome to the SL IT Chatbot!

I can help you with:

- Explaining SL policies
- Providing troubleshooting tips for your IT issues
- Creating support tickets for you

How can I assist you today?`
    }
  ]);
  const [input, setInput] = useState("");
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [theme] = useState("dark");
  const [ticket, setTicket] = useState(null);
  const [helpdeskTemplate, setHelpdeskTemplate] = useState(null);
  const [electricTemplate, setElectricTemplate] = useState(null);
  const [templateFields, setTemplateFields] = useState([]);
  const [sessionId] = useState(generateSessionId());
  const [agentType, setAgentType] = useState('helpdesk');
  const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);
  const [showMcpPanel, setShowMcpPanel] = useState(false);
  const [activeTool, setActiveTool] = useState(null);
  const chatEndRef = useRef(null);
  const [templatesLoading, setTemplatesLoading] = useState(true);
  const [templatesError, setTemplatesError] = useState(null);
  const [userData, setUserData] = useState(null);
  const [userDataLoading, setUserDataLoading] = useState(false);
  const [userDataError, setUserDataError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [awaitingTicketConfirmation, setAwaitingTicketConfirmation] = useState(false);
  const [pendingTicketContext, setPendingTicketContext] = useState(null);
  const [mcpConnectionStatus, setMcpConnectionStatus] = useState('connecting');
  // Add a state to track if a ticket is generated and awaiting confirmation or new chat
  const [ticketGenerated, setTicketGenerated] = useState(false);
  // Add state for the popup
  const [showTicketPopup, setShowTicketPopup] = useState(false);
  // Add state for attachments
  const [attachments, setAttachments] = useState([]);
  // Add state for loading and error during ticket submission
  const [submittingTicket, setSubmittingTicket] = useState(false);
  const [submitError, setSubmitError] = useState("");
  // Add state for preview modal
  const [previewFile, setPreviewFile] = useState(null);
  const [previewType, setPreviewType] = useState(null);
  const fileInputRef = useRef();
  // Add state for file error
  const [fileError, setFileError] = useState("");

  // Allowed file types/extensions
  const allowedTypes = [
    'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 'image/svg+xml',
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
    'application/vnd.ms-excel', // .xls
  ];
  const allowedExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.pdf', '.docx', '.xlsx', '.xls'];
  const maxFileSize = 10 * 1024 * 1024; // 10MB

  // Use custom MCP client instead of use-mcp
  const { state, tools, callTool, log, clearStorage } = useCustomMcp();

  // Monitor MCP connection status
  useEffect(() => {
    if (state === 'connected') {
      setMcpConnectionStatus('connected');
      console.log('MCP connected successfully via proxy');
    } else if (state === 'error') {
      setMcpConnectionStatus('error');
      console.error('MCP connection error via proxy');
    } else if (state === 'connecting') {
      setMcpConnectionStatus('connecting');
      console.log('MCP connecting via proxy...');
    }
  }, [state]);

  useEffect(() => {
    document.body.setAttribute("data-theme", theme);
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isBotTyping, theme]);

  // Fetch templates at runtime from backend static endpoint
  useEffect(() => {
    setTemplatesLoading(true);
    setTemplatesError(null);
    Promise.all([
      fetch('http://localhost:8000/static/templates/SL-IT_Helpdesk_Issues.jsonl')
        .then(res => {
          if (!res.ok) throw new Error('Helpdesk template not found');
          return res.text();
        })
        .then(text => {
          const template = JSON.parse(text.split('\n')[0]);
          setHelpdeskTemplate(template);
          if (agentType === 'helpdesk') setTemplateFields(template.fields);
        }),
      fetch('http://localhost:8000/static/templates/SL%20-%20Electric%20Issues.jsonl')
        .then(res => {
          if (!res.ok) throw new Error('Electric template not found');
          return res.text();
        })
        .then(text => {
          const template = JSON.parse(text.split('\n')[0]);
          setElectricTemplate(template);
          if (agentType === 'electric') setTemplateFields(template.fields);
        })
    ]).catch(err => {
      setTemplatesError('Failed to load ticket templates. Please contact support.');
    }).finally(() => {
      setTemplatesLoading(false);
    });
  }, [agentType]);

  // Fetch the helpdesk template from the backend when a ticket is generated
  useEffect(() => {
    if (ticket && !helpdeskTemplate) {
      fetch('http://localhost:8000/tools/get_template/it')
        .then(res => res.json())
        .then(data => {
          if (data && data.success && data.template) {
            setHelpdeskTemplate(data.template);
            setTemplateFields(data.template.fields || []);
          }
        })
        .catch(() => {});
    }
  }, [ticket, helpdeskTemplate]);

  useEffect(() => {
    if (!userUid) return;
    setUserDataLoading(true);
    setUserDataError("");
    const db = getFirestore();
    const docRef = doc(db, "userData", userUid);
    getDoc(docRef)
      .then((docSnap) => {
        if (docSnap.exists()) {
          setUserData(docSnap.data());
        } else {
          setUserDataError("No user data found.");
        }
      })
      .catch((err) => {
        setUserDataError("Failed to fetch user data.");
      })
      .finally(() => setUserDataLoading(false));
  }, [userUid]);

  // Utility to robustly extract ticket from MCP response
  function extractTicketFromMcpResponse(result) {
    console.log("[DEBUG] extractTicketFromMcpResponse called with:", result);
    if (!result) {
      console.log("[DEBUG] No result provided");
      return null;
    }
    // If result is a stringified JSON, parse it
    if (typeof result === 'string') {
      try {
        result = JSON.parse(result);
        console.log("[DEBUG] Parsed string result:", result);
      } catch (e) {
        console.log("[DEBUG] Failed to parse string result:", e);
        return null;
      }
    }
    // If result has a .content[0].text (MCP tool call), parse it
    if (result.content && result.content[0] && result.content[0].text) {
      try {
        result = JSON.parse(result.content[0].text);
        console.log("[DEBUG] Parsed content result:", result);
      } catch (e) {
        console.log("[DEBUG] Failed to parse content result:", e);
        return null;
      }
    }
    // Prefer 'ticket', fallback to 'artifact'
    if (result.ticket) {
      // If ticket is an object with all fields 'Not Specified', treat as empty
      if (typeof result.ticket === 'object' && result.ticket !== null) {
        const values = Object.values(result.ticket);
        if (values.length === 0) return {};
        if (values.every(v => v === 'Not Specified')) return {};
      }
      console.log("[DEBUG] Found ticket in result:", result.ticket);
      return result.ticket;
    }
    if (result.artifact) {
      console.log("[DEBUG] Found artifact in result:", result.artifact);
      return result.artifact;
    }
    if (result.result && (result.result.ticket || result.result.artifact)) {
      const ticket = result.result.ticket || result.result.artifact;
      // If ticket is an object with all fields 'Not Specified', treat as empty
      if (typeof ticket === 'object' && ticket !== null) {
        const values = Object.values(ticket);
        if (values.length === 0) return {};
        if (values.every(v => v === 'Not Specified')) return {};
      }
      console.log("[DEBUG] Found ticket in result.result:", ticket);
      return ticket;
    }
    console.log("[DEBUG] No ticket found in result");
    return null;
  }

  // Helper to detect agent type from ticket fields
  function detectAgentTypeFromTicket(ticket) {
    if (!ticket) return 'helpdesk';
    // If the ticket has a Category(Required) field with value 'SL - Electric', it's electric
    if (ticket['Category(Required)'] === 'SL - Electric' || ticket['Category'] === 'SL - Electric') return 'electric';
    // If the ticket has fields unique to electric template
    if (ticket['Subcategory'] && ticket['Item']) return 'electric';
    // Default to helpdesk
    return 'helpdesk';
  }

  // Utility: detect if user message is a ticket creation intent
  function isTicketCreationIntent(message) {
    if (!message) return false;
    const lower = message.toLowerCase();
    return [
      'create ticket',
      'generate ticket',
      'open ticket',
      'log ticket',
      'file ticket',
      'submit ticket',
      'raise ticket',
      'make ticket',
      'support ticket',
      'can you generate a ticket',
      'can you create a ticket',
      'can you open a ticket',
      'can you log a ticket',
      'can you file a ticket',
      'can you submit a ticket',
      'can you raise a ticket',
      'can you make a ticket',
      'can you help me with a ticket',
      'i want to create a ticket',
      'i want to open a ticket',
      'i want to log a ticket',
      'i want to file a ticket',
      'i want to submit a ticket',
      'i want to raise a ticket',
      'i want to make a ticket',
      'ticket for',
      'ticket about',
      'ticket regarding',
      'ticket issue',
      'ticket request',
      'ticket problem',
      'ticket help',
      'ticket support',
      'ticket needed',
      'ticket required',
      'ticket please',
      'ticket now',
      'ticket',
    ].some(phrase => lower.includes(phrase));
  }

  // Utility: detect yes/confirmation
  function isYesIntent(message) {
    if (!message) return false;
    const lower = message.toLowerCase();
    return [
      'yes', 'yep', 'yeah', 'please do', 'sure', 'go ahead', 'ok', 'okay', 'confirm', 'create it', 'generate it', 'open it', 'file it', 'submit it', 'raise it', 'make it', 'do it', 'ticket yes', 'yes please', 'yes create', 'yes generate', 'yes open', 'yes file', 'yes submit', 'yes raise', 'yes make', 'yes do',
    ].some(phrase => lower.includes(phrase));
  }

  // Handler for file input change (reset input after selection)
  const handleAttachmentChange = (e) => {
    setFileError("");
    const files = Array.from(e.target.files);
    setAttachments(prev => {
      const existing = prev.map(f => f.name + f.size);
      const newFiles = files.filter(f => !existing.includes(f.name + f.size));
      const validFiles = [];
      for (const file of newFiles) {
        const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(ext)) {
          setFileError(`File '${file.name}' is not a supported type.`);
          continue;
        }
        if (file.size > maxFileSize) {
          setFileError(`File '${file.name}' exceeds the 10MB size limit.`);
          continue;
        }
        validFiles.push(file);
      }
      return [...prev, ...validFiles];
    });
    e.target.value = null; // reset input so same file can be re-added
  };

  // Handler to remove an attachment
  const removeAttachment = (idx) => {
    setAttachments(prev => prev.filter((_, i) => i !== idx));
    if (fileInputRef.current) fileInputRef.current.value = null;
  };

  // Handler to preview file
  const handlePreviewFile = (file, idx) => {
    if (file.type.startsWith('image/')) {
      // Revoke previous previewFile URL if any
      if (previewFile) URL.revokeObjectURL(previewFile);
      const url = URL.createObjectURL(file);
      setPreviewFile(url);
      setPreviewType('image');
    } else if (file.type === 'application/pdf') {
      const url = URL.createObjectURL(file);
      window.open(url, '_blank'); // open PDF in new tab
    } else {
      const url = URL.createObjectURL(file);
      window.open(url, '_blank');
    }
  };

  // Handler to close image preview modal
  const closePreviewModal = () => {
    if (previewFile) URL.revokeObjectURL(previewFile);
    setPreviewFile(null);
    setPreviewType(null);
  };

  // Helper to handle MCP tool calls
  const handleMcpToolCall = async (toolName, params) => {
    try {
      // Check MCP connection status first
      if (mcpConnectionStatus !== 'connected') {
        throw new Error(`MCP server not connected. Status: ${mcpConnectionStatus}. Please ensure the MCP server is running on port 8001.`);
      }
      
      const result = await callTool(toolName, params);
      const ticketObj = extractTicketFromMcpResponse(result);
      console.log('[DEBUG] handleMcpToolCall ticketObj:', ticketObj);
      console.log('[DEBUG] handleMcpToolCall result:', result);
      if (ticketObj && Object.keys(ticketObj).length > 0) {
        // Detect agent type from ticket fields
        const detectedType = detectAgentTypeFromTicket(ticketObj);
        setAgentType(detectedType);
        // Set template fields based on detected type
        let previewString = '';
        if (detectedType === 'electric' && electricTemplate) {
          setTemplateFields(electricTemplate.fields);
          previewString = formatTicketPreviewString(ticketObj, electricTemplate);
        } else if (helpdeskTemplate) {
          setTemplateFields(helpdeskTemplate.fields);
          previewString = formatTicketPreviewString(ticketObj, helpdeskTemplate);
        } else {
          previewString = formatTicketPreviewString(ticketObj, null);
        }
        // Ensure previewString is always plain text
        if (typeof previewString !== 'string') {
          try {
            previewString = Object.entries(ticketObj)
              .map(([key, value]) => `${key}: ${value}`)
              .join('\n');
          } catch {
            previewString = '[Invalid Ticket Preview]';
          }
        }
        setMessages(prev => [...prev, { sender: 'bot', text: previewString, isTicketPreview: true }]);
        setTicket(ticketObj);
        setTicketGenerated(true);
      } else {
        setTicket(null);
        setTicketGenerated(false);
      }
      // Always add the bot's response to the chat if it exists
      if (result && result.response) {
        setMessages(prev => [...prev, { sender: 'bot', text: result.response }]);
      }
      console.log('[DEBUG] handleMcpToolCall messages after ticket logic:', messages);
      // Optionally return the whole parsed result for further use
      return result && result.content && result.content[0] && result.content[0].text
        ? JSON.parse(result.content[0].text)
        : result;
    } catch (error) {
      console.error('MCP Tool Call Error:', error);
      let errorMessage = 'Failed to process your request';
      
      if (error.message.includes('MCP server not connected')) {
        errorMessage = 'MCP server is not connected. Please ensure the MCP server is running on port 8001.';
      } else if (error.message.includes('fetch')) {
        errorMessage = 'Unable to connect to the MCP server. Please check if the server is running.';
      } else if (error.message.includes('timeout')) {
        errorMessage = 'Request timed out. Please try again.';
      } else {
        errorMessage = `Error: ${error.message || 'Unknown error occurred'}`;
      }
      
      setMessages(prev => [...prev, {
        sender: 'bot',
        text: errorMessage,
        isError: true
      }]);
      return null;
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMessage = input.trim();
    setInput("");
    setMessages(prev => [...prev, { sender: 'user', text: userMessage }]);
    setIsBotTyping(true);
    setAwaitingConfirmation(false);
    try {
      const conversationHistoryNew = [
        ...messages.map(msg => ({
          role: msg.sender === 'bot' ? 'assistant' : 'user',
          content: msg.text
        })),
        { role: 'user', content: userMessage }
      ];
      // setConversationHistory(conversationHistoryNew); // This line was removed as per the edit hint
      // Prepare user_info if available
      let user_info = undefined;
      console.log("[DEBUG] userData:", userData);
      if (userData) {
        user_info = {
          employee_name: userData.employee_name || '',
          SL_competency: userData.SL_competency || '',
          floor_information: userData.floor_information || '',
          employee_id: userData.employee_id || ''
        };
        console.log("[DEBUG] Prepared user_info:", user_info);
      } else {
        console.log("[DEBUG] No userData available");
      }
      if (isTicketCreationIntent(userMessage)) {
        // Immediately create the ticket, skip confirmation
        await handleMcpToolCall('main_rag_chat', {
          message: userMessage,
          conversation_history: conversationHistoryNew,
          session_id: sessionId,
          ...(user_info ? { user_info } : {}),
          force_create_ticket: true,
        });
        return;
      }
      if (awaitingTicketConfirmation) {
        // User is responding to ticket offer
        if (isYesIntent(userMessage)) {
          setAwaitingTicketConfirmation(false);
          // Actually create the ticket now, using pending context
          const context = pendingTicketContext || {};
          await handleMcpToolCall('main_rag_chat', {
            message: context.problem_description || context.user_message || userMessage,
            conversation_history: conversationHistoryNew,
            session_id: sessionId,
            ...(user_info ? { user_info } : {}),
            force_create_ticket: true,
            context_summary: context.context_summary || '',
            policy_snippets: context.policy_snippets || '',
            solution_snippets: context.solution_snippets || '',
          });
          setPendingTicketContext(null);
          return;
        } else {
          setAwaitingTicketConfirmation(false);
          setPendingTicketContext(null);
          setMessages(prev => [...prev, { sender: 'bot', text: 'Okay, let me know if you need anything else!' }]);
          return;
        }
      }
      const result = await handleMcpToolCall('main_rag_chat', {
        message: userMessage,
        conversation_history: conversationHistoryNew,
        session_id: sessionId,
        ...(user_info ? { user_info } : {})
      });
      console.log('[DEBUG] handleSendMessage result:', result);
      if (result) {
        if (result.citations && Array.isArray(result.citations) && result.citations.length > 0) {
          setMessages(prev => [...prev, {
            sender: 'bot',
            text: 'References:\n' + result.citations.map((c, i) => `[${i+1}] ${c.source}`).join('\n')
          }]);
        }
        // Always set the ticket and fetch the template
        const ticketObj = extractTicketFromMcpResponse(result);
        console.log('[DEBUG] handleSendMessage ticketObj:', ticketObj);
        if (ticketObj && Object.keys(ticketObj).length > 0) {
          // Detect agent type from ticket fields
          const detectedType = detectAgentTypeFromTicket(ticketObj);
          setAgentType(detectedType);
          // Set template fields based on detected type
          let previewString = '';
          if (detectedType === 'electric' && electricTemplate) {
            setTemplateFields(electricTemplate.fields);
            previewString = formatTicketPreviewString(ticketObj, electricTemplate);
          } else if (helpdeskTemplate) {
            setTemplateFields(helpdeskTemplate.fields);
            previewString = formatTicketPreviewString(ticketObj, helpdeskTemplate);
          } else {
            previewString = formatTicketPreviewString(ticketObj, null);
          }
          // Ensure previewString is always plain text
          if (typeof previewString !== 'string') {
            try {
              previewString = Object.entries(ticketObj)
                .map(([key, value]) => `${key}: ${value}`)
                .join('\n');
            } catch {
              previewString = '[Invalid Ticket Preview]';
            }
          }
          setMessages(prev => [...prev, { sender: 'bot', text: previewString, isTicketPreview: true }]);
          setTicket(ticketObj);
        } else {
          setTicket(null);
        }
        setTimeout(() => {
          console.log('[DEBUG] handleSendMessage messages after ticket logic:', messages);
        }, 0);
        if (result.context) {
          // Optionally store or display context
        }
        // After getting a bot response (result.response), if it offers to create a ticket, set awaitingTicketConfirmation and store context
        if (result && result.response && /would you like to (create|generate|open|file|submit|raise|make) a ticket/i.test(result.response)) {
          setAwaitingTicketConfirmation(true);
          setPendingTicketContext({
            problem_description: result.context?.problem_description || '',
            user_message: userMessage,
            context_summary: result.context?.context_summary || '',
            policy_snippets: result.citations?.join('\n') || '',
            solution_snippets: '', // add if available
          });
        }
      }
    } finally {
      setIsBotTyping(false);
    }
  };

  const handleConfirmTicket = () => {
    setShowTicketPopup(true);
    setAwaitingConfirmation(false);
    // Do NOT clear ticket/template fields here
  };

  const handleStartNewChat = () => {
    setMessages([
      { 
        sender: "bot", 
        text: `Hi, welcome to the SL IT Chatbot!\n\nI can help you with:\n\n- Explaining SL policies\n- Providing troubleshooting tips for your IT issues\n- Creating support tickets for you\n\nHow can I assist you today?`
      }
    ]);
    setTicket(null);
    setTemplateFields([]);
    setAgentType('helpdesk');
    setAwaitingConfirmation(false);
    setTicketGenerated(false);
    setPendingTicketContext(null);
    // Optionally reset other states if needed
  };

  const formatTicketPreviewString = (ticket, template) => {
    if (!ticket) return "";
    if (!template || !Array.isArray(template.fields)) {
      // fallback: show all fields
      return Object.entries(ticket)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\n');
    }
    // Use template field order, always show all fields
    return template.fields.map(field => {
      const value = (ticket && ticket[field.name] !== undefined && ticket[field.name] !== null)
        ? (ticket[field.name] === '' ? '' : ticket[field.name])
        : getTicketValue(field.name);
      return `${field.label || field.name}: ${value}`;
    }).join('\n');
  };

  const renderMessage = (message, index) => {
    const isBot = message.sender === 'bot';
    const isError = message.isError;
    const avatar = isBot
      ? BOT_AVATAR
      : USER_AVATAR;

    return (
      <div key={index} className={`message ${isBot ? 'bot' : 'user'}${isError ? ' error' : ''}`}>
        <img src={avatar} alt="avatar" className="avatar" />
        <div className={`message-content${message.isTicketPreview ? ' ticket-preview' : ''}`}>
          {isBot ? (
            <ReactMarkdown>{message.text}</ReactMarkdown>
          ) : (
            message.text
          )}
        </div>
      </div>
    );
  };

  // Helper to check if a field is required
  const isRequired = (name) => /\(Required\)/i.test(name);

  // Helper to get field label without (Required)
  const cleanLabel = (label) => label.replace(/\(Required\)/gi, '').replace(/:/g, '').trim();

  // Helper to get field value from ticket or empty string
  const getTicketValue = (fieldName) => {
    console.log(`[DEBUG] getTicketValue called for field: "${fieldName}"`);
    console.log(`[DEBUG] Current ticket data:`, ticket);
    
    if (!ticket) {
      console.log("[DEBUG] No ticket data available");
      return 'Not Specified';
    }
    
    // First try exact match
    if (ticket[fieldName] !== undefined && ticket[fieldName] !== '') {
      console.log(`[DEBUG] Exact match found for "${fieldName}": ${ticket[fieldName]}`);
      return ticket[fieldName];
    }
    
    // Try without (Required) - this is the main fix
    const altName = fieldName.replace(/\(Required\)/gi, '').trim();
    if (ticket[altName] !== undefined && ticket[altName] !== '') {
      console.log(`[DEBUG] Match without (Required) found for "${fieldName}" -> "${altName}": ${ticket[altName]}`);
      return ticket[altName];
    }
    
    // Try with (Required) added if not present
    if (!fieldName.includes('(Required)')) {
      const withRequired = `${fieldName}(Required)`;
      if (ticket[withRequired] !== undefined && ticket[withRequired] !== '') {
        console.log(`[DEBUG] Match with (Required) added found for "${fieldName}" -> "${withRequired}": ${ticket[withRequired]}`);
        return ticket[withRequired];
      }
    }
    
    // Try with/without spaces
    const altName2 = fieldName.replace(/\s+/g, '');
    if (ticket[altName2] !== undefined && ticket[altName2] !== '') {
      console.log(`[DEBUG] Match without spaces found for "${fieldName}" -> "${altName2}": ${ticket[altName2]}`);
      return ticket[altName2];
    }
    
    // Try lowercased
    const lowerName = fieldName.toLowerCase();
    for (const key of Object.keys(ticket)) {
      if (key.toLowerCase() === lowerName && ticket[key] !== '') {
        console.log(`[DEBUG] Lowercase match found for "${fieldName}" -> "${key}": ${ticket[key]}`);
        return ticket[key];
      }
    }
    
    // Try partial matches for common field types
    const fieldLower = fieldName.toLowerCase();
    for (const key of Object.keys(ticket)) {
      const keyLower = key.toLowerCase();
      if ((keyLower.includes('requester') && fieldLower.includes('requester')) ||
          (keyLower.includes('category') && fieldLower.includes('category')) ||
          (keyLower.includes('subcategory') && fieldLower.includes('subcategory')) ||
          (keyLower.includes('item') && fieldLower.includes('item')) ||
          (keyLower.includes('competency') && fieldLower.includes('competency')) ||
          (keyLower.includes('floor') && fieldLower.includes('floor')) ||
          (keyLower.includes('machine') && fieldLower.includes('machine')) ||
          (keyLower.includes('subject') && fieldLower.includes('subject')) ||
          (keyLower.includes('description') && fieldLower.includes('description'))) {
        if (ticket[key] !== undefined && ticket[key] !== '') {
          console.log(`[DEBUG] Partial match found for "${fieldName}" -> "${key}": ${ticket[key]}`);
          return ticket[key];
        }
      }
    }
    
    console.log(`[DEBUG] No match found for "${fieldName}", returning 'Not Specified'`);
    return 'Not Specified';
  };

  // Function to submit ticket with attachments
  const submitTicketWithAttachments = async () => {
    if (!ticket) return;
    setSubmittingTicket(true);
    setSubmitError("");
    try {
      const formData = new FormData();
      formData.append('ticket', JSON.stringify(ticket));
      attachments.forEach((file) => {
        formData.append('attachments', file);
      });
      // Replace with your backend endpoint for ticket+attachments
      const response = await fetch('http://localhost:8000/api/ticket_with_attachments', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Failed to submit ticket');
      setShowTicketPopup(true);
      setAttachments([]);
      setTicket(null);
      setTemplateFields([]);
      setTicketGenerated(false);
      setPendingTicketContext(null);
      setMessages([
        { 
          sender: "bot", 
          text: `Hi, welcome to the SL IT Chatbot!\n\nI can help you with:\n\n- Explaining SL policies\n- Providing troubleshooting tips for your IT issues\n- Creating support tickets for you\n\nHow can I assist you today?` }
      ]);
      setInput("");
    } catch (err) {
      setSubmitError('Error submitting ticket: ' + err.message);
    } finally {
      setSubmittingTicket(false);
    }
  };

  // Sidebar close on Esc
  useEffect(() => {
    if (!sidebarOpen) return;
    const handleKey = (e) => { if (e.key === 'Escape') setSidebarOpen(false); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [sidebarOpen]);

  // --- UI ---
  return (
    <>
      {/* MCP panel overlay and panel OUTSIDE chatbot-container */}
      {showMcpPanel && <div className="mcp-panel-overlay" onClick={() => setShowMcpPanel(false)} />}
      {showMcpPanel && (
        <div className={`mcp-panel open`} tabIndex={-1} style={{ zIndex: 31 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3>MCP Tools & Debug</h3>
            <button onClick={() => setShowMcpPanel(false)} style={{ fontSize: 18, background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', zIndex: 32 }}>✖</button>
          </div>
          <div style={{ marginBottom: 12 }}>
            <b>Connection:</b> <span>{mcpConnectionStatus}</span>
            {mcpConnectionStatus === 'error' && (
              <button onClick={clearStorage} style={{ marginLeft: 12, fontSize: 12 }}>Logout/Clear Auth</button>
            )}
          </div>
          <div>
            <b>Available Tools:</b>
            <ul className="mcp-tool-list">
              {tools && tools.length > 0 ? tools.map(tool => {
                // Icon mapping for tool names
                let icon = '🛠️';
                if (tool.name === 'chat') icon = '💬';
                else if (tool.name === 'main_rag_chat') icon = '🤖';
                else if (tool.name === 'create_ticket') icon = '📝';
                else if (tool.name === 'complete_ticket') icon = '✅';
                else if (tool.name === 'a2a_task') icon = '🔄';
                else if (tool.name === 'search_policies') icon = '📚';
                else if (tool.name === 'extract_user_info') icon = '🧑‍💼';
                else if (tool.name === 'classify_issue_type') icon = '🏷️';
                else if (tool.name === 'list_available_tools') icon = '🗂️';
                else if (tool.name === 'health_check') icon = '❤️';
                // Short, user-friendly label
                let label = tool.name
                  .replace(/_/g, ' ')
                  .replace(/\bchat\b/i, 'Chat')
                  .replace(/main rag chat/i, 'Main RAG Chat')
                  .replace(/create ticket/i, 'Create Ticket')
                  .replace(/complete ticket/i, 'Complete Ticket')
                  .replace(/a2a task/i, 'A2A Task');
                // Friendly, more descriptive description
                return (
                  <li key={tool.name}>
                    <button
                      className="mcp-tool-btn"
                      tabIndex={0}
                      onMouseEnter={() => setActiveTool(tool.name)}
                      onFocus={() => setActiveTool(tool.name)}
                      onMouseLeave={() => setActiveTool(null)}
                      onBlur={() => setActiveTool(null)}
                    >
                      <span className="mcp-tool-icon">{icon}</span>
                      <span>{label}</span>
                    </button>
                  </li>
                );
              }) : <li>No tools discovered yet.</li>}
            </ul>
            {/* Dedicated description area below tool list */}
            {activeTool && tools && tools.length > 0 && (() => {
              const tool = tools.find(t => t.name === activeTool);
              if (!tool) return null;
              if (tool.name === 'chat') return (
                <div className="mcp-tool-desc-area" style={{
                  background: 'rgba(35,42,54,0.92)',
                  color: '#fff',
                  borderRadius: 8,
                  padding: '1rem 1.2rem',
                  margin: '1rem 0 0.5rem 0',
                  fontSize: '1.08rem',
                  boxShadow: '0 2px 12px #0005',
                  minHeight: 48,
                  maxWidth: '100%',
                  wordBreak: 'break-word',
                }}>
                  Start a conversation with the IT support agent. Use this for general help and troubleshooting.
                </div>
              );
              if (tool.name === 'main_rag_chat') return (
                <div className="mcp-tool-desc-area" style={{
                  background: 'rgba(35,42,54,0.92)',
                  color: '#fff',
                  borderRadius: 8,
                  padding: '1rem 1.2rem',
                  margin: '1rem 0 0.5rem 0',
                  fontSize: '1.08rem',
                  boxShadow: '0 2px 12px #0005',
                  minHeight: 48,
                  maxWidth: '100%',
                  wordBreak: 'break-word',
                }}>
                  Interact with the main Retrieval-Augmented Generation (RAG) agent for advanced chat and knowledge queries.
                </div>
              );
              if (tool.name === 'create_ticket') return (
                <div className="mcp-tool-desc-area" style={{
                  background: 'rgba(35,42,54,0.92)',
                  color: '#fff',
                  borderRadius: 8,
                  padding: '1rem 1.2rem',
                  margin: '1rem 0 0.5rem 0',
                  fontSize: '1.08rem',
                  boxShadow: '0 2px 12px #0005',
                  minHeight: 48,
                  maxWidth: '100%',
                  wordBreak: 'break-word',
                }}>
                  Create a new support ticket based on your message and conversation history. Use this to report a new issue.
                </div>
              );
              if (tool.name === 'complete_ticket') return (
                <div className="mcp-tool-desc-area" style={{
                  background: 'rgba(35,42,54,0.92)',
                  color: '#fff',
                  borderRadius: 8,
                  padding: '1rem 1.2rem',
                  margin: '1rem 0 0.5rem 0',
                  fontSize: '1.08rem',
                  boxShadow: '0 2px 12px #0005',
                  minHeight: 48,
                  maxWidth: '100%',
                  wordBreak: 'break-word',
                }}>
                  Complete an existing ticket by filling in any missing required information.
                </div>
              );
              if (tool.name === 'a2a_task') return (
                <div className="mcp-tool-desc-area" style={{
                  background: 'rgba(35,42,54,0.92)',
                  color: '#fff',
                  borderRadius: 8,
                  padding: '1rem 1.2rem',
                  margin: '1rem 0 0.5rem 0',
                  fontSize: '1.08rem',
                  boxShadow: '0 2px 12px #0005',
                  minHeight: 48,
                  maxWidth: '100%',
                  wordBreak: 'break-word',
                }}>
                  Perform agent-to-agent (A2A) tasks using the JSON-RPC endpoint. Use for advanced integrations.
                </div>
              );
              // Default: show tool.description
              return (
                <div className="mcp-tool-desc-area" style={{
                  background: 'rgba(35,42,54,0.92)',
                  color: '#fff',
                  borderRadius: 8,
                  padding: '1rem 1.2rem',
                  margin: '1rem 0 0.5rem 0',
                  fontSize: '1.08rem',
                  boxShadow: '0 2px 12px #0005',
                  minHeight: 48,
                  maxWidth: '100%',
                  wordBreak: 'break-word',
                }}>
                  {tool.description}
                </div>
              );
            })()}
          </div>
          <hr className="mcp-panel-divider" />
          <div className="debug-log-section">
            <b>Debug Log:</b>
            <pre>
              {log && log.length > 0 ? log.map((entry, i) => (
                <div key={i}>{entry.timestamp} [{entry.level}] {entry.message}</div>
              )) : 'No log entries.'}
            </pre>
          </div>
        </div>
      )}
      <div className="chatbot-container">
        {/* Gear icon for sidebar trigger, only show when sidebar is closed */}
        {!sidebarOpen && (
          <button
            className="settings-gear-btn"
            style={{ position: 'absolute', top: 24, right: 24, zIndex: 30 }}
            onClick={() => setSidebarOpen(true)}
            title="Settings"
            aria-label="Open settings sidebar"
          >
            <span style={{ fontSize: 28, color: '#fff' }}>&#9881;</span>
          </button>
        )}
        {/* MCP panel toggle */}
        <button
          className="mcp-panel-toggle"
          onClick={() => setShowMcpPanel(v => !v)}
          title="Show MCP Tools & Debug"
          style={{ position: 'absolute', top: 24, left: 24, zIndex: 20 }}
        >
          🛠️
        </button>
        
        {/* MCP Connection Status Indicator */}
        <div 
          className="mcp-connection-status"
          style={{ 
            position: 'absolute', 
            top: 24, 
            left: 80, 
            zIndex: 20,
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: 'bold',
            backgroundColor: mcpConnectionStatus === 'connected' ? '#4CAF50' : 
                           mcpConnectionStatus === 'connecting' ? '#FF9800' : '#F44336',
            color: 'white'
          }}
          title={`MCP Connection: ${mcpConnectionStatus}`}
        >
          MCP: {mcpConnectionStatus === 'connected' ? '✅' : 
                mcpConnectionStatus === 'connecting' ? '⏳' : '❌'}
        </div>
        {/* Main content: ticket panel + chat area */}
        <div className="chat-main-row">
          {/* Ticket panel as a form preview */}
          <div className="ticket-panel" style={{ minWidth: 320, maxWidth: 400, flex: '0 0 350px', marginRight: 24, background: 'rgba(255,255,255,0.07)', borderRadius: 12, padding: 20, height: '80vh', maxHeight: '80vh', overflowY: 'auto' }}>
            <h3 style={{ marginTop: 0 }}>Ticket Preview</h3>
            {templatesLoading ? (
              <div style={{ color: '#aaa', fontStyle: 'italic' }}>Loading ticket templates...</div>
            ) : templatesError ? (
              <div style={{ color: 'red', fontStyle: 'italic' }}>{templatesError}</div>
            ) : ticket && Object.keys(ticket).length > 0 && templateFields && templateFields.length > 0 ? (
              <>
                {templateFields.map(field => {
                  const fieldName = field.name;
                  const value = getTicketValue(fieldName);
                  const required = isRequired(fieldName);
                  const fieldType = field.type;
                  const isAutofill = fieldType === 'autofill';
                  // Dependent dropdown logic for 'Item'
                  if (fieldType === 'dependent_dropdown') {
                    const depField = field.dependency;
                    const depValue = ticket[depField] || Object.keys(field.options)[0];
                    const options = field.options[depValue] || ["Not Specified"];
                    return (
                      <div key={fieldName} className="ticket-field" style={{ marginBottom: 12 }}>
                        <label htmlFor={fieldName} style={{ fontWeight: 500 }}>
                          {cleanLabel(field.label || fieldName)}
                          {required && <span style={{ color: 'red', marginLeft: 4 }}>*</span>}
                        </label>
                        <select
                          id={fieldName}
                          name={fieldName}
                          value={value}
                          disabled={isAutofill}
                          style={{
                            marginLeft: 8,
                            padding: 6,
                            borderRadius: 6,
                            border: '1px solid #333',
                            background: isAutofill ? 'rgba(0,0,0,0.05)' : 'rgba(0,0,0,0.1)',
                            color: 'inherit',
                            width: '100%'
                          }}
                          onChange={!isAutofill ? e => setTicket(prev => ({ ...prev, [fieldName]: e.target.value })) : undefined}
                          required={required}
                        >
                          <option value="" disabled>Select...</option>
                          {options.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      </div>
                    );
                  }
                  // Dropdown logic for normal dropdown fields
                  if (fieldType === 'dropdown') {
                    const options = field.options || ["Not Specified"];
                    return (
                      <div key={fieldName} className="ticket-field" style={{ marginBottom: 12 }}>
                        <label htmlFor={fieldName} style={{ fontWeight: 500 }}>
                          {cleanLabel(field.label || fieldName)}
                          {required && <span style={{ color: 'red', marginLeft: 4 }}>*</span>}
                        </label>
                        <select
                          id={fieldName}
                          name={fieldName}
                          value={value}
                          disabled={isAutofill}
                          style={{
                            marginLeft: 8,
                            padding: 6,
                            borderRadius: 6,
                            border: '1px solid #333',
                            background: isAutofill ? 'rgba(0,0,0,0.05)' : 'rgba(0,0,0,0.1)',
                            color: 'inherit',
                            width: '100%'
                          }}
                          onChange={!isAutofill ? e => setTicket(prev => ({ ...prev, [fieldName]: e.target.value })) : undefined}
                          required={required}
                        >
                          <option value="" disabled>Select...</option>
                          {options.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      </div>
                    );
                  }
                  // Textarea for description
                  if (fieldType === 'textarea') {
                    return (
                      <div key={fieldName} className="ticket-field" style={{ marginBottom: 12 }}>
                        <label htmlFor={fieldName} style={{ fontWeight: 500 }}>
                          {cleanLabel(field.label || fieldName)}
                          {required && <span style={{ color: 'red', marginLeft: 4 }}>*</span>}
                        </label>
                        <textarea
                          id={fieldName}
                          name={fieldName}
                          value={value}
                          disabled={isAutofill}
                          style={{
                            marginLeft: 8,
                            padding: 6,
                            borderRadius: 6,
                            border: '1px solid #333',
                            background: isAutofill ? 'rgba(0,0,0,0.05)' : 'rgba(0,0,0,0.1)',
                            color: 'inherit',
                            width: '100%'
                          }}
                          onChange={!isAutofill ? e => setTicket(prev => ({ ...prev, [fieldName]: e.target.value })) : undefined}
                          rows={3}
                        />
                      </div>
                    );
                  }
                  // Default to text input
                  return (
                    <div key={fieldName} className="ticket-field" style={{ marginBottom: 12 }}>
                      <label htmlFor={fieldName} style={{ fontWeight: 500 }}>
                        {cleanLabel(field.label || fieldName)}
                        {required && <span style={{ color: 'red', marginLeft: 4 }}>*</span>}
                      </label>
                      <input
                        id={fieldName}
                        name={fieldName}
                        type="text"
                        value={value}
                        disabled={isAutofill}
                        style={{
                          marginLeft: 8,
                          padding: 6,
                          borderRadius: 6,
                          border: '1px solid #333',
                          background: isAutofill ? 'rgba(0,0,0,0.05)' : 'rgba(0,0,0,0.1)',
                          color: 'inherit',
                          width: '100%'
                        }}
                        onChange={!isAutofill ? e => setTicket(prev => ({ ...prev, [fieldName]: e.target.value })) : undefined}
                        required={required}
                      />
                    </div>
                  );
                })}
                {/* Always render attachment UI after all fields */}
                <div className="ticket-field" style={{ marginTop: 0, marginBottom: 12 }}>
                  <label style={{ fontWeight: 500, display: 'block', marginBottom: 6 }}>
                    Attachments:
                  </label>
                  <input
                    type="file"
                    multiple
                    accept=".jpg,.jpeg,.png,.gif,.bmp,.webp,.svg,.pdf,.docx,.xlsx,.xls,image/*,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
                    onChange={handleAttachmentChange}
                    className="file-upload"
                    style={{ display: 'none' }}
                    ref={fileInputRef}
                    id="custom-attachment-input"
                  />
                  <button
                    type="button"
                    style={{
                      padding: '6px 16px',
                      borderRadius: 6,
                      border: '1px solid #333',
                      background: '#23272f',
                      color: '#fff',
                      fontWeight: 500,
                      fontSize: 14,
                      cursor: 'pointer',
                      marginBottom: 8,
                      marginRight: 8
                    }}
                    onClick={() => fileInputRef.current && fileInputRef.current.click()}
                  >
                    Choose Files
                  </button>
                  {attachments.length === 0 && <span style={{ color: '#aaa', fontSize: 13 }}>No files selected</span>}
                  {attachments.length > 0 && (
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {attachments.map((file, idx) => (
                        <li key={idx} style={{ display: 'flex', alignItems: 'center', marginBottom: 2, background: 'rgba(0,0,0,0.08)', borderRadius: 6, padding: '4px 8px', maxWidth: 260 }}>
                          {/* Preview thumbnail or icon */}
                          {file.type.startsWith('image/') ? (
                            <img
                              src={URL.createObjectURL(file)}
                              alt={file.name}
                              style={{ width: 32, height: 32, objectFit: 'cover', borderRadius: 4, marginRight: 8, cursor: 'pointer', border: '1px solid #333' }}
                              onClick={() => handlePreviewFile(file, idx)}
                            />
                          ) : file.type === 'application/pdf' ? (
                            <span
                              style={{ fontSize: 24, marginRight: 8, cursor: 'pointer' }}
                              title="Preview PDF"
                              onClick={() => handlePreviewFile(file)}
                            >📄</span>
                          ) : (
                            <span style={{ width: 32, height: 32, marginRight: 8, display: 'inline-block', textAlign: 'center', lineHeight: '32px', background: '#eee', borderRadius: 4, fontSize: 18, cursor: 'pointer' }} onClick={() => handlePreviewFile(file)}>
                              📎
                            </span>
                          )}
                          {/* File name (clickable for preview) */}
                          <span
                            style={{ flex: 1, marginRight: 8, cursor: 'pointer', textDecoration: 'underline', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 120 }}
                            title={file.name}
                            onClick={() => handlePreviewFile(file, idx)}
                          >
                            {file.name}
                          </span>
                          <button type="button" onClick={() => removeAttachment(idx)} style={{ color: '#e53e3e', background: 'none', border: 'none', cursor: 'pointer', fontSize: 14, marginLeft: 4 }}>Remove</button>
                        </li>
                      ))}
                    </ul>
                  )}
                  {fileError && <div style={{ color: '#e53e3e', fontSize: 13, marginTop: 4 }}>{fileError}</div>}
                </div>
              </>
            ) : (
              <div style={{ color: '#aaa', fontStyle: 'italic' }}>No ticket data yet.</div>
            )}
          </div>
          {/* Chat area */}
          <div className="chat-area">
            <div className="chat-window">
              {messages.map(renderMessage)}
              {isBotTyping && (
                <div className="message bot">
                  <img src={BOT_AVATAR} alt="avatar" className="avatar" />
                  <div className="message-content typing-indicator">
                    <span className="dot"></span>
                    <span className="dot"></span>
                    <span className="dot"></span>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            {/* Ticket confirmation */}
            {awaitingConfirmation && (
              <div className="ticket-confirm-panel" style={{ margin: '1rem 0', background: 'rgba(255,255,255,0.07)', borderRadius: 8, padding: 16, textAlign: 'center' }}>
                <div style={{ marginBottom: 8 }}>All required fields are filled. Please confirm your ticket.</div>
                <button className="send-btn" onClick={handleConfirmTicket}>Confirm Ticket</button>
              </div>
            )}
            {/* Input row */}
            {ticketGenerated && !awaitingConfirmation ? (
              <div style={{ height: '70px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(35,42,54,0.92)', borderBottomLeftRadius: 12, borderBottomRightRadius: 12 }}>
                <div className="ticket-action-buttons" style={{ display: 'flex', gap: '1rem' }}>
                  <button className="ticket-action-btn" onClick={submitTicketWithAttachments} disabled={submittingTicket}>
                    {submittingTicket ? 'Submitting...' : 'Confirm Ticket'}
                  </button>
                  <button className="ticket-action-btn" onClick={handleStartNewChat}>Start New Chat</button>
                </div>
              </div>
            ) : (
              <form className="chat-input-form" onSubmit={handleSendMessage} autoComplete="off">
                <input
                  className="chat-input"
                  type="text"
                  placeholder="Type your message..."
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  disabled={isBotTyping}
                  autoFocus
                />
                <button className="send-btn" type="submit" disabled={isBotTyping || !input.trim()}>
                  <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="11" cy="11" r="11" fill="none" />
                    <polygon points="7,5 17,11 7,17" fill="#bfc9d1" />
                  </svg>
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
      {/* Sidebar overlay and sidebar OUTSIDE chatbot-container */}
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}
      <aside className={`settings-sidebar${sidebarOpen ? ' open' : ''}`} tabIndex={-1}>
        <button
          className="sidebar-logout-btn"
          onClick={onLogout}
          style={{ marginBottom: 24 }}
        >
          Logout
        </button>
        <button
          className="sidebar-close-btn"
          onClick={() => setSidebarOpen(false)}
          title="Close sidebar"
          aria-label="Close settings sidebar"
        >
          ×
        </button>
        {/* User Info Group */}
        <div className="sidebar-section">
          <div className="sidebar-label">User Info</div>
          {userDataLoading ? (
            <div className="sidebar-value">Loading user info...</div>
          ) : userDataError ? (
            <div className="sidebar-value sidebar-error">{userDataError}</div>
          ) : userData ? (
            <>
              <div className="sidebar-value"><span role="img" aria-label="user">👤</span> {userData.employee_name || 'N/A'}</div>
              <div className="sidebar-value"><span role="img" aria-label="email">✉️</span> {username}</div>
              <div className="sidebar-value"><span role="img" aria-label="id card">🪪</span> {userData.employee_id || 'N/A'}</div>
            </>
          ) : null}
        </div>
        {/* Work Info Group */}
        <div className="sidebar-section">
          <div className="sidebar-label">Work Info</div>
          {userDataLoading ? null : userDataError ? null : userData ? (
            <>
              <div className="sidebar-value"><span role="img" aria-label="department">🏢</span> {userData.SL_competency || 'N/A'}</div>
              <div className="sidebar-value"><span role="img" aria-label="floor">🏬</span> {userData.floor_information || 'N/A'}</div>
            </>
          ) : null}
        </div>
        {/* Add more sidebar sections here if needed */}
      </aside>
      {showTicketPopup && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(24, 24, 24, 0.65)',
          backdropFilter: 'blur(4px)',
          WebkitBackdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: '#23272f',
            borderRadius: '14px',
            boxShadow: '0 4px 24px rgba(0,0,0,0.28)',
            padding: '32px 24px 24px 24px',
            minWidth: '320px',
            maxWidth: '90vw',
            textAlign: 'center',
            color: '#f5f7fa',
            border: '1.5px solid #333'
          }}>
            <h2 style={{color: '#60a5fa', marginTop: 0, fontWeight: 700}}>Ticket Submitted!</h2>
            <p style={{color: '#e2e8f0', fontSize: '1.1rem', marginBottom: 24}}>Your ticket has been submitted. Our team will get back to you soon.</p>
            <button style={{
              background: '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: '12px',
              padding: '10px 32px',
              fontSize: '1rem',
              fontWeight: 600,
              cursor: 'pointer',
              marginTop: '8px',
              boxShadow: '0 2px 8px 0 rgba(37,99,235,0.10)'
            }} onClick={() => {
              setShowTicketPopup(false);
              setTicket(null);
              setTemplateFields([]);
              setTicketGenerated(false);
              setPendingTicketContext(null);
              setMessages([
                { 
                  sender: "bot", 
                  text: `Hi, welcome to the SL IT Chatbot!\n\nI can help you with:\n\n- Explaining SL policies\n- Providing troubleshooting tips for your IT issues\n- Creating support tickets for you\n\nHow can I assist you today?` }
              ]);
              setInput("");
            }}>OK</button>
          </div>
        </div>
      )}
      {previewFile && previewType === 'image' && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(24,24,24,0.7)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={closePreviewModal}>
          <div style={{ position: 'relative', background: 'rgba(30,32,36,0.98)', borderRadius: 12, boxShadow: '0 4px 24px #0008', padding: 16 }} onClick={e => e.stopPropagation()}>
            <img src={previewFile} alt="Preview" style={{ maxWidth: '80vw', maxHeight: '80vh', borderRadius: 8 }} />
            <button style={{ position: 'absolute', top: 8, right: 8, fontSize: 28, background: 'none', border: 'none', color: '#fff', cursor: 'pointer', zIndex: 2001 }} onClick={closePreviewModal}>&times;</button>
          </div>
        </div>
      )}
    </>
  );
}

export default Chatbot;