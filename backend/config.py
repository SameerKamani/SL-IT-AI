"""
SL-IT-AI Configuration and Setup
- Environment variables, path definitions, client initializations
- Global constants and session storage
"""
import os
import json
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import SecretStr
from langchain_openai import AzureOpenAIEmbeddings
# Always load the .env from the backend directory FIRST, before any other imports
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Define all paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(ROOT_DIR, 'Templates')
EMPLOYEE_DATA_DIR = os.path.join(ROOT_DIR, 'Employee Data')
TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, 'SL-IT_Helpdesk_Issues.jsonl')
EMPLOYEE_PATH = os.path.join(EMPLOYEE_DATA_DIR, 'employees.jsonl')
ELECTRIC_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, 'SL - Electric Issues.jsonl')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace Redis with in-memory storage
session_storage = {}  # type: dict

# Azure OpenAI config
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "gpt-4o")  # Fallback to chat model
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE", "")
AZURE_OPENAI_API_VERSION = "2024-10-21"  # Use the latest stable GA version

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_API_BASE,
    api_version=AZURE_OPENAI_API_VERSION
)

# RAG agent - initialize lazily to avoid circular imports
rag_agent = None

def get_rag_agent():
    global rag_agent
    if rag_agent is None:
        from policy_rag import PolicyRAGAgent
        rag_agent = PolicyRAGAgent()
    return rag_agent

# Initialize tools and clients
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_API_BASE
)

# Load employee data for autofill
try:
    with open(EMPLOYEE_PATH, 'r', encoding='utf-8') as f:
        EMPLOYEE_DATA = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    logger.warning("Employee data file not found, using empty list")
    EMPLOYEE_DATA = []

# Mapping from template autofill field names to employee data keys
AUTOFILL_FIELD_MAP = {
    "Requester(Required)": "employee_name",
    "Requester ID": "employee_id",
    "SL Competency(Required)": "SL_competency",
    "Floor Information(Required)": "floor_information",
    # Add more as needed
} 