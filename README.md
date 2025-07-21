# SL-IT-AI: Advanced Multi-Agent IT Helpdesk System

<div align="center">

![SL-IT-AI Logo](https://img.shields.io/badge/SL--IT--AI-Advanced%20AI%20Helpdesk-blue?style=for-the-badge&logo=openai)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![LangGraph](https://img.shields.io/badge/LangGraph-00FF00?style=for-the-badge&logo=graphql&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
<!-- ![CI](https://img.shields.io/github/actions/workflow/status/your-repo/ci.yml?style=for-the-badge) -->

**An intelligent multi-agent IT helpdesk system powered by [FastAPI](https://fastapi.tiangolo.com/), [LangGraph](https://github.com/langchain-ai/langgraph), [MCP (Model Context Protocol)](https://meghashyamthiruveedula.medium.com/model-context-protocol-a-comprehensive-guide-41c8b56a61f3), and [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service). Provides automated ticket creation, policy-based solutions, and intelligent agent-to-agent (A2A) communication for Systems Limited.**

[🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#️-architecture) • [🔧 Features](#-features) • [📁 Project Structure](#-project-structure) • [🧪 Testing](#-testing)

</div>

---

## 🎯 **Project Overview**

SL-IT-AI is a comprehensive IT helpdesk solution that combines the power of AI with traditional support workflows. The system automatically classifies issues, provides immediate solutions, and creates structured support tickets with minimal user input.

### ✨ **Key Features**

- 🤖 **AI-Powered Chatbot** with natural language understanding
- 📋 **Automated Ticket Creation** with smart field extraction
- 🔍 **Policy-Based Solutions** using RAG (Retrieval-Augmented Generation)
- 🏢 **Employee Database Integration** for automatic user information
- 🔄 **Multi-Agent Communication** (A2A) for specialized issue handling
- 📱 **Modern Web Interface** with real-time chat capabilities
- 🔐 **Secure Authentication** via Firebase
- 📊 **Comprehensive Issue Classification** (IT vs Electric issues)

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.11+
- Node.js 16+
- Azure OpenAI API key
- Firebase project (for authentication)

### **Installation**

1. **Clone the repository**
    ```bash
    git clone <repository-url>
    cd SL-IT-AI
    ```

2. **Install backend dependencies**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

3. **Install frontend dependencies**
    ```bash
    cd ../frontend
    npm install
    ```

4. **Configure environment variables**
    ```bash
    # Backend/.env
    AZURE_OPENAI_API_KEY=your_azure_openai_key
    AZURE_OPENAI_API_BASE=your_azure_endpoint
    AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

    # Frontend environment variables (if needed)
    REACT_APP_API_URL=http://localhost:8000
    ```

### **Running the Application**

**Option 1: Use the Python startup script (Recommended)**
```bash
# Start both backend servers (FastAPI + MCP)
cd backend
python start_servers.py
```

**Option 2: Manual startup**
```bash
# Terminal 1: Start MCP Server (Port 8001)
cd backend
python mcp_tools.py

# Terminal 2: Start FastAPI Server (Port 8000)
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Start React Frontend (Port 3000)
cd frontend
npm start
```

### **Access Points**
- 🌐 **Frontend Application**: http://localhost:3000
- 🔧 **FastAPI Backend**: http://localhost:8000
- 📚 **API Documentation**: http://localhost:8000/docs
- 🛠️ **MCP Server**: http://localhost:8001/mcp

---

## 🏗️ **Architecture**

### **System Components**

```mermaid
graph TB
    A[React Frontend] --> B[FastAPI Backend]
    B --> C[LangGraph Workflow]
    
    subgraph "LangGraph Workflow"
        C --> D[RAG Node]
        C --> E[Intent Detection Node]
        C --> F[Ticket Creation Node]
    end
    
    D --> G[Policy RAG Agent]
    G --> H[ChromaDB Vector Store]
    H --> I[Policy Documents]
    
    F --> J[Agent Communication]
    J --> K[IT Helpdesk Agent]
    J --> L[Electric Issues Agent]
    
    K --> M[IT Template]
    L --> N[Electric Template]
    
    subgraph "MCP Tools"
        O[FastMCP Server]
        O --> P[main_rag_chat]
        O --> Q[create_ticket]
        O --> R[search_policies]
        O --> S[a2a_task]
    end
    
    B --> O
    
    subgraph "Data Sources"
        T[Employee Database]
        U[Policy Documents]
        V[Ticket Templates]
    end
    
    G --> U
    K --> V
    L --> V
    M --> T
    N --> T
```

> **Tip:** For more on workflow orchestration and modular tool protocols, see [LangGraph](https://github.com/langchain-ai/langgraph) and [MCP](https://meghashyamthiruveedula.medium.com/model-context-protocol-a-comprehensive-guide-41c8b56a61f3) documentation.

---

## 🗺️ Enhanced System Workflow Diagram

Below is a detailed workflow diagram covering every major component, file, and flow in the SL-IT-AI system. This diagram provides a comprehensive technical overview of the frontend, backend, workflow, agent logic, MCP tools, RAG engine, and all data sources.

```mermaid
flowchart LR

%% Frontend
subgraph "Frontend React"
    FE1[App.js]
    FE2[Chatbot.js]
    FE3[Login.js]
    FE4[firebaseConfig.js]
end

%% Backend API Layer
subgraph "Backend FastAPI"
    BE1[main.py]
    BE2[api_routes.py]
    BE3[config.py]
    BE4[models.py]
end

%% LangGraph Workflow
subgraph "LangGraph Workflow"
    LG1[langgraph_workflow.py]
    LG2[RAG Node]
    LG3[Intent Detection Node]
    LG4[Ticket Creation Node]
end

%% Agent Logic
subgraph "Agent Logic"
    AG1[agents.py]
    AG2[agent_communication.py]
end

%% MCP Tool Server
subgraph "MCP Tool Server"
    MCP1[mcp_tools.py]
    MCP2[mcp.json]
end

%% RAG Engine
subgraph "Policy RAG"
    PR1[policy_rag.py]
    PR2[ChromaDB: policies_index]
    PR3[Policy Documents: policies]
end

%% Data
subgraph "Data"
    D1[Employee Data: employees.jsonl]
    D2[Ticket Templates: Templates]
end

%% Flows
FE2 -- "User Message/API Call" --> BE1
FE3 -- "Auth" --> FE4
FE1 --> FE2
FE1 --> FE3

BE1 --> BE2
BE2 --> BE3
BE2 --> BE4
BE2 -- "Invoke Workflow" --> LG1

LG1 --> LG2
LG1 --> LG3
LG1 --> LG4

LG2 -- "Policy Query" --> PR1
PR1 -- "Vector Search" --> PR2
PR2 -- "Load Docs" --> PR3

LG3 -- "Intent Classification" --> AG1
LG4 -- "Ticket Creation" --> AG1
LG4 -- "A2A Routing" --> AG2

AG1 -- "Field Extraction" --> D1
AG1 -- "Template Fill" --> D2

AG2 -- "Specialized Routing" --> AG1
AG2 -- "A2A Task" --> MCP1

BE1 -- "MCP Proxy/API" --> MCP1
MCP1 -- "Tool Calls" --> MCP2

MCP1 -- "main_rag_chat" --> LG1
MCP1 -- "create_ticket" --> LG4
MCP1 -- "search_policies" --> PR1
MCP1 -- "extract_user_info" --> AG1

%% Data Flows
PR1 -- "Policy Index" --> PR2
PR2 -- "Policy Docs" --> PR3
AG1 -- "Employee Info" --> D1
AG1 -- "Templates" --> D2

%% Specialized Agents
AG2 -- "IT Helpdesk Agent" --> D2
AG2 -- "Electric Agent" --> D2

%% Session/Context
BE3 -- "Session Storage" --> BE2

%% Auth
FE4 -- "Token" --> BE1

%% Legend
classDef file fill:#fff,stroke:#333,stroke-width:1px;
class FE1,FE2,FE3,FE4,BE1,BE2,BE3,BE4,LG1,LG2,LG3,LG4,AG1,AG2,MCP1,MCP2,PR1,PR2,PR3,D1,D2 file;
```

---

## 🔧 **Features**

### **🎯 Intelligent Issue Classification**
- **IT Helpdesk Issues:** Printer, software, VPN, web access, workstation, VOIP
- **Electric Issues:** Networking, power, services

### **📋 Smart Ticket Creation**
- LLM-based field extraction, employee data integration, template-based forms, fuzzy matching, validation

### **🔍 Policy-Based Solutions**
- RAG-powered, real-time search, context-aware, citation system

### **👥 Employee Management**
- Auto-fill, department mapping, floor information

---

## 📁 **Project Structure**

```
SL-IT-AI/
├── backend/
│   ├── main.py
│   ├── mcp_tools.py
│   ├── langgraph_workflow.py
│   ├── agents.py
│   ├── policy_rag.py
│   ├── config.py
│   ├── models.py
│   ├── api_routes.py
│   ├── agent_communication.py
│   ├── start_servers.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── Chatbot.js
│   │   ├── Login.js
│   │   └── App.css
│   ├── package.json
│   └── public/
├── policies/
│   ├── solutions.txt
│   └── Systems_Limited_Policies.pdf
├── Templates/
│   ├── SL-IT_Helpdesk_Issues.jsonl
│   └── SL - Electric Issues.jsonl
├── Employee Data/
│   └── employees.jsonl
└── README.md
```

---

## 🧪 **Testing**

### **API Testing**
```bash
curl -X POST "http://localhost:8000/main_rag_chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "My printer cartridge is empty", "conversation_history": [], "session_id": "test_session"}'
```
```bash
curl -X POST "http://localhost:8000/create_ticket" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Create a ticket for printer issue", "session_id": "test_session"}'
```

### **MCP Integration Testing**
```bash
curl http://localhost:8000/api/mcp/status
```
```bash
curl -X POST "http://localhost:8000/api/mcp/proxy" \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "search_policies", "arguments": {"query": "printer troubleshooting"}}}'
```

### **Frontend Testing**
```bash
cd frontend
npm test
npm run lint
```

> **Note:** We recommend [pytest](https://docs.pytest.org/) for backend and [Jest](https://jestjs.io/) for frontend testing.

---

## 🔍 **Troubleshooting**

- **Port Conflicts:** Check and kill processes on 8000, 8001, 3000 if needed.
- **MCP Server Issues:** Ensure MCP server is running and dependencies are installed.
- **Azure OpenAI Issues:** Verify API key, endpoint, and deployment name.
- **Logs:** Enable debug logging and check backend/logs.

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_BASE=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-21
MCP_PORT=8001
MCP_HOST=127.0.0.1
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## 📚 **API Documentation**

| Endpoint                  | Method | Description                                 |
|---------------------------|--------|---------------------------------------------|
| `/main_rag_chat`          | POST   | Main chat endpoint with RAG integration     |
| `/create_ticket`          | POST   | Create support ticket                       |
| `/complete_ticket`        | POST   | Complete ticket with missing info           |
| `/a2a/task`               | POST   | Agent-to-agent communication                |
| `/mcp/tools`              | GET    | List available MCP tools                    |
| `/api/mcp/proxy`          | POST   | MCP tool invocation proxy                   |
| `/session/{session_id}`   | GET    | Get session info                            |
| `/clear_session/{session_id}` | GET | Clear session data                          |

> **See** `models.py` for request/response schemas.
> **Note:** Some endpoints may require authentication via Firebase.

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

**Development Guidelines:**
- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Use [ESLint](https://eslint.org/) for JavaScript/React code
- Write comprehensive tests for new features
- Update documentation for API changes
- See [CONTRIBUTING.md](CONTRIBUTING.md) for more details

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Systems Limited** for the comprehensive policy database
- **Azure OpenAI** for powerful language model capabilities
- **LangGraph** for workflow orchestration
- **FastAPI** for high-performance API framework
- **React** for modern frontend development

---

## 📖 **Glossary**

- **RAG (Retrieval-Augmented Generation):** Combines information retrieval with generative AI to provide context-aware answers.
- **MCP (Model Context Protocol):** A protocol for modular, tool-based AI workflows and agent communication.
- **A2A (Agent-to-Agent):** Communication between specialized agents for task delegation and workflow orchestration.
- **ChromaDB:** Vector database for fast similarity search and policy retrieval.
- **LangGraph:** Workflow orchestration library for LLM-powered applications.

---

<div align="center">

**Made with ❤️ for Systems Limited IT Support**

[🔗 Documentation](https://github.com/your-repo/docs) • [🐛 Report Issues](https://github.com/your-repo/issues) • [💡 Feature Requests](https://github.com/your-repo/issues)

</div>
