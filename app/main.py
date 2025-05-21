import os
import logging
from dotenv import load_dotenv

from fastmcp import FastMCP

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
load_dotenv()

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PROMPT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompt.txt")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# FastMCP App Init
mcp_app = FastMCP(title="PDF RAG Application")

# Load Prompt Template
def load_prompt_template():
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt template: {e}")
        return None

# Load Documents
def load_documents():
    documents = []

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory at {DATA_DIR}")
        return documents

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]

    if not pdf_files:
        logger.warning("No PDF files found in the data directory.")
        return documents

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            logger.info(f"Loaded {pdf_file}")
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {e}")

    return documents

# Split Documents
def split_documents(documents):
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Initialize
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

documents = load_documents()
chunks = split_documents(documents)

vector_store = None
qa_chain = None

if chunks:
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        prompt_template = load_prompt_template()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")

# Tool definition
@mcp_app.tool
def query_documents(question: str):
    """
    Query the PDF documents with a question.
    """
    if not qa_chain:
        return {
            "answer": "No documents have been loaded. Please add PDF files to the data directory.",
            "sources": []
        }

    try:
        result = qa_chain({"query": question})

        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown")
                })

        return {
            "answer": result["result"],
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "sources": []
        }

# Info
logger.info("PDF RAG Application is running")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Loaded {len(documents)} documents")
logger.info(f"Created {len(chunks)} chunks")
logger.info(f"Using Ollama model: {OLLAMA_MODEL}")

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:mcp_app", host="0.0.0.0", port=8765)
