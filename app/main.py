import os
import logging
from dotenv import load_dotenv

# FastMCP imports - updated to use the correct imports
from fastmcp import FastMCP as MCPApp
from fastmcp import Service, Tool

# Langchain imports
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PROMPT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompt.txt")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Initialize MCP app
mcp_app = MCPApp(title="PDF RAG Application")

# Create a service
pdf_query_service = Service(
    name="pdf_query_service",
    title="PDF Query Service",
    description="Service to query PDF documents using RAG"
)

# Add service to app
mcp_app.add_service(pdf_query_service)

# Load prompt template
def load_prompt_template():
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt template: {e}")
        return None

# Load and process documents
def load_documents():
    documents = []
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory at {DATA_DIR}")
        return documents
    
    # Get all PDF files in the data directory
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.warning("No PDF files found in the data directory.")
        return documents
    
    # Load each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            logger.info(f"Loaded {pdf_file}")
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {e}")
    
    return documents

# Split documents into chunks
def split_documents(documents):
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    return text_splitter.split_documents(documents)

# Initialize embeddings and LLM
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

# Initialize RAG pipeline
documents = load_documents()
chunks = split_documents(documents)
vector_store = None
qa_chain = None

if chunks:
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Load prompt template
        prompt_template = load_prompt_template()
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")

# Define query tool
@pdf_query_service.tool
def query_documents(question: str):
    """
    Query the PDF documents with a question.
    
    Args:
        question: The question to ask about the documents
        
    Returns:
        A dictionary containing the answer and source documents
    """
    if not qa_chain:
        return {
            "answer": "No documents have been loaded. Please add PDF files to the data directory.",
            "sources": []
        }
    
    try:
        result = qa_chain({"query": question})
        
        # Extract source information
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source = {
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown")
                }
                sources.append(source)
        
        return {
            "answer": result["result"],
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "sources": []
        }

# Print startup information
logger.info(f"PDF RAG Application is running")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Loaded {len(documents)} documents")
logger.info(f"Created {len(chunks) if chunks else 0} chunks")
logger.info(f"Using Ollama model: {OLLAMA_MODEL}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:mcp_app", host="0.0.0.0", port=8765)