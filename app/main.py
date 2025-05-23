import os
import logging
from pathlib import Path
import time
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from fastmcp import FastMCP
import requests
from dotenv import load_dotenv

# Step 1: Initialize FastMCP application
# - Creates a FastMCP instance to organize tools for querying PDFs and the Financial Times API
mcp_app = FastMCP("PDF RAG App")

# Step 2: Configure logging (optional)
# - Sets up logging to capture info, warnings, and errors for debugging and monitoring
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 3: Load environment variables and set configuration
# - Loads variables from .env file (e.g., FT_COOKIE, OLLAMA_MODEL)
# - Defines paths for data (PDFs) and vector store, and sets default model and question
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vector_store")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
# Updated to 2020 to match PDF data
DEFAULT_QUESTION = "What is the total assets value of Barclays PLC in 2020?"
FT_API_URL = "https://markets.ft.com/research/webservices/companies/v1/profile"
FT_COOKIE = os.getenv("FT_COOKIE", "")

# Step 4: Initialize global variables for RAG system
# - Vector store holds document embeddings; QA chain handles question-answering
vector_store = None
qa_chain = None

# Step 5: Define function to fetch data from Financial Times API
# - Queries the FT API for financial metrics (e.g., for Barclays) using a provided symbol


def fetch_from_ft_api(symbol="BARC:LSE"):
    logger.info(f"Querying FT API for symbol: {symbol}")
    if not FT_COOKIE:
        logger.warning("FT_COOKIE not found in environment variables")
        return {"error": "FT_COOKIE not configured"}
    try:
        timestamp = int(time.time() * 1000)  # Milliseconds timestamp
        # Added timestamp parameter
        params = {"symbols": symbol, "_": timestamp}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://markets.ft.com/research/webservices/companies/v1/docs",
            "X-FT-Source": "f749296b753bb19e",  # Add X-FT-Source
            "X-Requested-With": "XMLHttpRequest",  # Add X-Requested-With
        }
        cookie_dict = {cookie.split('=')[0]: cookie.split('=')[1]
                       for cookie in FT_COOKIE.strip().split('; ') if '=' in cookie}
        response = requests.get(
            FT_API_URL, params=params, headers=headers, cookies=cookie_dict)  # Use params
        response.raise_for_status()
        data = response.json()
        logger.info(
            f"Successfully retrieved data from FT API for {symbol}")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from FT API: {str(e)}")
        return {"error": f"Failed to retrieve data: {str(e)}"}

# Step 6: Define function to load and process PDF
# - Loads a PDF file, splits it into chunks for processing


def load_and_process_pdf(pdf_name="Barclays-PLC-Annual-Report-2020.pdf"):
    logger.info(f"Loading PDF from {DATA_DIR}")
    pdf_path = os.path.join(DATA_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found at {pdf_path}")
        return []
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return []

# Step 7: Initialize the RAG system
# - Sets up embeddings, LLM, vector store, and QA chain for document querying


def initialize_rag_system():
    global vector_store, qa_chain
    logger.info("Initializing embeddings and LLM")
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    llm = ChatOllama(model=OLLAMA_MODEL)
    if os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH) and len(os.listdir(VECTOR_STORE_PATH)) > 0:
        try:
            logger.info(
                f"Loading existing vector store from {VECTOR_STORE_PATH}")
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            vector_store = None
    if vector_store is None:
        chunks = load_and_process_pdf()
        if chunks:
            try:
                logger.info("Creating new FAISS vector store")
                vector_store = FAISS.from_documents(chunks, embeddings)
                Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
                vector_store.save_local(VECTOR_STORE_PATH)
                logger.info(f"Vector store saved to {VECTOR_STORE_PATH}")
            except Exception as e:
                logger.error(f"Error creating vector store: {str(e)}")
                return False
        else:
            logger.warning("No chunks available to create vector store")
            return False
    if vector_store:
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                return_source_documents=True
            )
            logger.info("RAG pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            return False
    return False

# Step 8: Extract financial information from FT API response
# - Parses the FT API response to extract key financial metrics


logger = logging.getLogger(__name__)


def extract_financial_info(api_response):
    """
    Extracts key financial information from the FT API response.

    Args:
        api_response (dict): The JSON response from the FT API.

    Returns:
        dict: A dictionary containing the extracted financial data, or None if an error occurs or no data is found.
    """
    if "error" in api_response:
        # Log the error
        logger.warning(
            f"Error present in API response: {api_response['error']}")
        return None

    try:
        items = api_response.get("data", {}).get(
            "items", [])  # Access items within "data"
        if not items:
            logger.warning("No 'items' found in API response.")
            return None

        item = items[0]
        basic_info = item.get("basic", {})
        profile_info = item.get("profile", {})

        financial_data = {
            "company": basic_info.get("name", ""),
            "symbol": basic_info.get("symbol", ""),
            "exchange": basic_info.get("exchange", ""),
            "currency": basic_info.get("currency", ""),
            "employees": profile_info.get("employees", ""),
            "incorporatedYear": profile_info.get("incorporatedYear", ""),
            "netIncomeMRFY": profile_info.get("netIncomeMRFY", ""),
            "reportingCurrency": profile_info.get("reportingCurrency", ""),
            "description": profile_info.get("description", ""),
            "address": profile_info.get("contact", {}).get("physicalAddress", {}).get("line1", ""),
            "phone": profile_info.get("contact", {}).get("phone", {}).get("number", ""),
            "website": profile_info.get("contact", {}).get("webURL", ""),
        }

        return financial_data

    except Exception as e:
        logger.error(f"Error extracting financial information: {str(e)}")
        return None


# Step 9: Define FastMCP tool to query documents
# - Queries the PDF documents using the RAG system, falls back to FT API if needed


@mcp_app.tool()
def query_documents(question=DEFAULT_QUESTION):
    logger.info(f"Received query: {question}")
    if qa_chain:
        try:
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = [{
                "source": doc.metadata.get("source", "Unknown")}
                for doc in result.get("source_documents", [])]
            if not any(phrase in answer.lower() for phrase in [
                "i don't know", "i don't have", "cannot find",
                "no information", "not mentioned", "not provided"
            ]):
                logger.info(
                    f"Found answer in PDF documents. Sources: {len(sources)}")
                return {"answer": answer, "sources": sources, "source_type": "document"}
            else:
                logger.info(
                    "No definitive answer found in PDF. Trying Financial Times API...")
        except Exception as e:
            logger.error(f"Error processing RAG query: {str(e)}")
    logger.info("Falling back to Financial Times API")
    try:
        # Assuming fetch_from_ft_api is defined elsewhere
        symbol = "BARC:LSE"
        ft_data = fetch_from_ft_api(symbol)
        financial_info = extract_financial_info(ft_data)

        if financial_info:
            company = financial_info["company"]
            # Get employees or "N/A" if not found
            employees = financial_info.get("employees", "N/A")

            answer = f"According to Financial Times data for {company} ({financial_info['symbol']}):\n"
            # Include employee count in the answer
            answer += f"Number of Employees: {employees}\n"

            logger.info(f"Found answer in Financial Times API for {company}")
            return {
                "answer": answer,
                "sources": [{"source": "Financial Times API", "symbol": financial_info["symbol"]}],
                "source_type": "financial_times"
            }
        else:
            # More specific warning
            logger.warning(f"No financial information found for {symbol}")
            return {  # Return a not found message instead of a generic error
                "answer": f"Could not find financial information for {symbol}.",
                "sources": [],
                "source_type": "none"
            }

    except Exception as e:
        logger.error(f"Error processing FT API query: {str(e)}")
        return {
            "answer": "Could not find information in documents or financial data sources.",
            "sources": [],
            "source_type": "none"
        }


# Step 10: Main execution
# - Initializes the RAG system and processes the default question
if __name__ == "__main__":
    # Initialize the RAG system
    rag_initialized = initialize_rag_system()

    # Process the default question
    if rag_initialized:
        logger.info(f"Processing default question: {DEFAULT_QUESTION}")
        result = query_documents(DEFAULT_QUESTION)
        print(f"Question: {DEFAULT_QUESTION}")
        print(f"Answer: {result['answer']}")
        # print(f"Sources: {result['sources']}")
        print(f"Source Type: {result['source_type']}")
    else:
        logger.error("Failed to initialize RAG pipeline")
        print("Error: RAG pipeline could not be initialized")
