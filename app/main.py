import os
import logging
from dotenv import load_dotenv

# FastMCP
from fastmcp import FastMCP, Service

# LangChain
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()

# Paths & Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PROMPT_FILE_PATH = os.path.join(BASE_DIR, "..", "prompt.txt")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Init FastMCP
mcp_app = FastMCP(title="PDF RAG Application")
pdf_query_service = Service(
    name="pdf_query_service",
    title="PDF Query Service",
    description="Service to query PDF documents using RAG"
)
mcp_app.add_service(pdf_query_service)


def load_prompt_template() -> str | None:
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt template: {e}")
        return None


def load_documents() -> list:
    documents = []

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory at {DATA_DIR}")
        return documents

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]

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


def split_documents(documents: list) -> list:
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


# Embeddings & LLM
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

# Load & Split
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


@pdf_query_service.tool
def query_documents(question: str) -> dict:
    """
    Query the PDF documents with a question.
    Returns a dictionary with the answer and sources.
    """
    if not qa_chain:
        return {
            "answer": "No documents have been loaded. Please add PDF files to the data directory.",
            "sources": []
        }

    try:
        result = qa_chain({"query": question})
        sources = []

        for doc in result.get("source_documents", []):
            sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "source": doc.metadata.get("source", "Unknown")
            })

        return {
            "answer": result.get("result", "No answer generated."),
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "sources": []
        }


# Startup Info
logger.info("PDF RAG Application is running")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Loaded {len(documents)} documents")
logger.info(f"Created {len(chunks)} chunks")
logger.info(f"Using Ollama model: {OLLAMA_MODEL}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:mcp_app", host="0.0.0.0", port=8765)
