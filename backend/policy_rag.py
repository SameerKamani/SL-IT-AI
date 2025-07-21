"""
Policy RAG Agent for SL-IT-AI
- Lightweight embedding solution without external dependencies
- Provides policy-based solutions and recommendations
"""
import os
import json
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
# Add import for Hugging Face sentence-transformers
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
# Remove or comment out SimpleEmbeddings and its initialization
# class SimpleEmbeddings:
#     """Simple embedding function that doesn't require external ML libraries"""
    
#     def __init__(self, dimension=384):
#         self.dimension = dimension
    
#     def encode(self, texts):
#         """Generate simple embeddings based on text hash"""
#         if isinstance(texts, str):
#             texts = [texts]
        
#         embeddings = []
#         for text in texts:
#             # Create a deterministic embedding based on text hash
#             text_hash = hashlib.md5(text.encode()).hexdigest()
#             # Convert hash to numbers
#             seed = int(text_hash[:8], 16)
#             np.random.seed(seed)
#             embedding = np.random.rand(self.dimension).tolist()
#             embeddings.append(embedding)
        
#         return np.array(embeddings)
    
#     def embed_query(self, text):
#         """Embed a single query text"""
#         return self.encode(text)[0]
    
#     def embed_documents(self, texts):
#         """Embed multiple documents"""
#         return self.encode(texts)

# Add Hugging Face Embeddings class
class HFEmbeddings:
    def __init__(self, model_path='sentence_transformer/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer, models
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        return self.encode(text)[0]

    def embed_documents(self, texts):
        return self.encode(texts)

# Initialize simple embedding model
# print("[RAG] Initializing simple embedding model...")
# try:
#     embedding_model = SimpleEmbeddings()
#     print("[RAG] Successfully loaded simple embedding model")
# except Exception as e:
#     print(f"[RAG] Error loading simple embedding model: {e}")
#     embedding_model = None

def load_policy_documents(policies_dir: str = "../policies") -> List[Document]:
    """Load policy documents from the policies directory"""
    documents = []
    policies_path = Path(policies_dir)
    
    if not policies_path.exists():
        print(f"[RAG] Policies directory not found at {policies_path}")
        return documents
    
    print(f"[RAG] Loading documents from {policies_path}")
    
    # Load text files
    for file_path in policies_path.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": str(file_path), "type": "text"}
                )
                documents.append(doc)
                print(f"[RAG] Loaded text document: {file_path.name}")
        except Exception as e:
            print(f"[RAG] Error loading {file_path}: {e}")
    
    # Load PDF files (basic text extraction)
    for file_path in policies_path.glob("*.pdf"):
        try:
            # Simple PDF text extraction (you might want to use PyPDF2 or similar for better extraction)
            import subprocess
            import tempfile
            
            # Try to extract text using pdftotext if available
            try:
                result = subprocess.run(['pdftotext', str(file_path), '-'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    content = result.stdout
                else:
                    # Fallback: just use filename as content
                    content = f"PDF document: {file_path.name}"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fallback: just use filename as content
                content = f"PDF document: {file_path.name}"
            
            doc = Document(
                page_content=content,
                metadata={"source": str(file_path), "type": "pdf"}
            )
            documents.append(doc)
            print(f"[RAG] Loaded PDF document: {file_path.name}")
        except Exception as e:
            print(f"[RAG] Error loading {file_path}: {e}")
    
    print(f"[RAG] Total documents loaded: {len(documents)}")
    return documents

class PolicyRAGAgent:
    def __init__(self, embedding_model=None):
        if embedding_model is None:
            # Use Hugging Face all-MiniLM-L6-v2 as the primary embedding model
            try:
                embedding_model = HFEmbeddings('sentence_transformer/all-MiniLM-L6-v2')
                print("[RAG] Using local all-MiniLM-L6-v2 embeddings from sentence_transformer/all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[RAG] Error loading Hugging Face embeddings: {e}")
                embedding_model = None
        
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.documents = []
        self.policies_index_path = Path("policies_index")
        
        # Initialize the RAG system
        self._initialize_rag()
        
        # Load policy documents if index is empty
        self._load_policy_documents_if_needed()
    
    def _initialize_rag(self):
        """Initialize the RAG system with simple embeddings"""
        try:
            if not self.embedding_model:
                print("[RAG] No embedding model available, skipping initialization")
                return
            
            print("[RAG] Loading Chroma index from", self.policies_index_path)
            
            # Initialize Chroma with the embedding model object directly
            self.vectorstore = Chroma(
                persist_directory=str(self.policies_index_path),
                embedding_function=self.embedding_model,  # Pass the object directly
                client_settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            print("[RAG] Successfully initialized Chroma with simple embeddings")
            
        except Exception as e:
            print(f"[RAG] Error initializing RAG system: {e}")
            self.vectorstore = None
    
    def _load_policy_documents_if_needed(self):
        """Load policy documents if the index is empty"""
        try:
            if not self.vectorstore:
                print("[RAG] Vectorstore not initialized, skipping document loading")
                return
            
            # Check if index has documents
            collection = self.vectorstore._collection
            if collection and collection.count() > 0:
                print(f"[RAG] Index already has {collection.count()} documents, skipping loading")
                return
            
            print("[RAG] Index is empty, loading policy documents...")
            
            # Load documents from policies directory
            documents = load_policy_documents()
            
            if documents:
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                
                split_docs = []
                for doc in documents:
                    chunks = text_splitter.split_documents([doc])
                    split_docs.extend(chunks)
                
                print(f"[RAG] Split {len(documents)} documents into {len(split_docs)} chunks")
                
                # Add documents to vectorstore
                self.vectorstore.add_documents(split_docs)
                print(f"[RAG] Successfully added {len(split_docs)} document chunks to index")
            else:
                print("[RAG] No documents found to load")
                
        except Exception as e:
            print(f"[RAG] Error loading policy documents: {e}")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant policies using simple embeddings"""
        try:
            if not self.vectorstore:
                print("[RAG] Vectorstore not initialized, returning empty results")
                return []
            
            print(f"[RAG] search called with query: {query}")
            print(f"[RAG] Embedding model: {type(self.embedding_model)}")
            print(f"[RAG] Vectorstore: {type(self.vectorstore)}")
            
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            print(f"[RAG] Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            print(f"[RAG] similarity_search_with_score error: {e}")
            print(f"[RAG_METRICS] Query: '{query}', Error: {e}")
            return []

    def add_documents(self, documents: List[Document]):
        """Add documents to the vectorstore"""
        try:
            if not self.vectorstore:
                print("[RAG] Vectorstore not initialized, cannot add documents")
                return
            
            self.vectorstore.add_documents(documents)
            print(f"[RAG] Added {len(documents)} documents to vectorstore")
            
        except Exception as e:
            print(f"[RAG] Error adding documents: {e}")
    
    def get_relevant_policies(self, query: str, k: int = 3) -> List[str]:
        """Get relevant policy content as strings"""
        results = self.search(query, k)
        return [result["content"] for result in results]
    
    def get_policy_citations(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get policy citations with metadata"""
        results = self.search(query, k)
        citations = []
        
        for i, result in enumerate(results):
            citation = {
                "content": result["content"],
                "metadata": result["metadata"],
                "score": result["score"],
                "citation_id": f"policy_{i+1}"
            }
            citations.append(citation)
        
        return citations

# Global RAG agent instance
_rag_agent = None

def get_rag_agent() -> PolicyRAGAgent:
    """Get or create the global RAG agent instance"""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = PolicyRAGAgent()
    return _rag_agent

# Initialize RAG agent on module import
try:
    _rag_agent = PolicyRAGAgent()
    print("[RAG] Global RAG agent initialized successfully")
except Exception as e:
    print(f"[RAG] Error initializing global RAG agent: {e}")
    _rag_agent = None 