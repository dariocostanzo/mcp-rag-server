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


# Configuring logging for debugging and monitoring
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Loading environment variables and setting paths
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vector_store")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
FT_API_URL = "https://markets.ft.com/research/webservices/companies/v1/profile"
FT_COOKIE = os.getenv("FT_COOKIE", "")
DEFAULT_QUESTION = "What is the total assets value of Barclays PLC in 2029?"


# Initializing FastMCP application
mcp = FastMCP("PDF RAG Financial Data Extractor")


# Setting up global variables for RAG system
vector_store = None
qa_chain = None
system_prompt = ""


# Loading prompt from file for LLM
def load_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompt.txt")
    if not os.path.exists(prompt_path):
        logger.error(f"Prompt file not found at {prompt_path}")
        return "", DEFAULT_QUESTION
    try:
        with open(prompt_path, 'r') as file:
            prompt = file.read().strip()
        logger.info("Prompt loaded successfully from prompt.txt")
        # Extract question from prompt (assuming it ends with "Question: {question}")
        if "Question: " in prompt:
            question = prompt.split("Question: ")[-1].strip()
            system_prompt = prompt.split("Question: ")[0].strip()
        else:
            question = DEFAULT_QUESTION
            system_prompt = prompt
        return system_prompt, question
    except Exception as e:
        logger.error(f"Error loading prompt: {str(e)}")
        return "", DEFAULT_QUESTION


# Loading and processing PDF documents
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


# Initializing RAG system with embeddings, LLM, and QA chain
def initialize_rag_system():
    global vector_store, qa_chain, system_prompt
    logger.info("Initializing embeddings and LLM")
    system_prompt, _ = load_prompt()

    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        llm = ChatOllama(model=OLLAMA_MODEL, system=system_prompt)

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
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False


# MCP Tool: Query documents using RAG system
@mcp.tool()
def query_documents(question: str) -> dict:
    """
    Query the vector database for relevant information from annual reports.

    Args:
        question: The question to search for in the documents

    Returns:
        dict: Contains answer, sources, and source_type
    """
    logger.info(f"Received document query: {question}")

    if qa_chain is None:
        logger.warning("QA chain not initialized")
        return {
            "answer": "RAG system not initialized.",
            "sources": [],
            "source_type": "none"
        }

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = [{"source": doc.metadata.get("source", "Unknown")}
                   for doc in result.get("source_documents", [])]

        # Check if answer contains uncertainty phrases
        uncertainty_phrases = [
            "i don't know", "i don't have", "cannot find",
            "no information", "not mentioned", "not provided",
            "unable to find", "not available"
        ]

        if not any(phrase in answer.lower() for phrase in uncertainty_phrases):
            logger.info(
                f"Found answer in PDF documents. Sources: {len(sources)}")
            return {
                "answer": answer,
                "sources": sources,
                "source_type": "document"
            }
        else:
            logger.info("No definitive answer found in PDF.")
            return {
                "answer": "Could not find definitive answer in documents.",
                "sources": sources,
                "source_type": "document"
            }
    except Exception as e:
        logger.error(f"Error processing RAG query: {str(e)}")
        return {
            "answer": f"Error processing document query: {str(e)}",
            "sources": [],
            "source_type": "none"
        }


# MCP Tool: Query Financial Times API
@mcp.tool()
def query_ft_api(symbol: str = "BARC:LSE") -> dict:
    """
    Query the Financial Times API for company financial metrics.

    Args:
        symbol: Stock symbol to query (default: "BARC:LSE")

    Returns:
        dict: Financial data from FT API or error information
    """
    logger.info(f"Querying FT API for symbol: {symbol}")

    if not FT_COOKIE:
        logger.warning("FT_COOKIE not found in environment variables")
        return {"error": "FT_COOKIE not configured"}

    try:
        timestamp = int(time.time() * 1000)
        params = {"symbols": symbol, "_": timestamp}
        headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Referer": "https://markets.ft.com/research/webservices/companies/v1/docs",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "X-FT-Source": "296838f60ae2ae12",
            "X-Requested-With": "XMLHttpRequest",
            "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
        }

        cookie_dict = {
            cookie.split("=")[0]: cookie.split("=")[1]
            for cookie in FT_COOKIE.strip().split("; ")
            if "=" in cookie
        }

        response = requests.get(
            FT_API_URL, params=params, headers=headers, cookies=cookie_dict
        )
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully retrieved data from FT API for {symbol}")
        return data

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from FT API: {str(e)}")
        return {"error": f"Failed to retrieve data: {str(e)}"}


# MCP Tool: Extract financial information from FT API response
@mcp.tool()
def extract_financial_info(api_response: dict) -> dict:
    """
    Extract financial information from FT API response.

    Args:
        api_response: Raw response from FT API

    Returns:
        dict: Extracted financial data or None if error
    """
    if "error" in api_response:
        logger.warning(
            f"Error present in API response: {api_response['error']}")
        return {"error": api_response["error"]}

    try:
        items = api_response.get("data", {}).get("items", [])
        if not items:
            logger.warning("No 'items' found in API response.")
            return {"error": "No items found in API response"}

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
        return {"error": f"Error extracting financial information: {str(e)}"}


# MCP Tool: Process query with fallback to FT API
@mcp.tool()
def process_financial_query(question: str, symbol: str = "BARC:LSE") -> dict:
    """
    Process a financial query using RAG first, then fallback to FT API if needed.

    Args:
        question: The financial question to answer
        symbol: Stock symbol for FT API fallback (default: "BARC:LSE")

    Returns:
        dict: Complete response with answer, sources, and source_type
    """
    logger.info(f"Processing financial query: {question}")

    # Step 1: Try querying documents
    doc_result = query_documents(question)

    if (doc_result["source_type"] == "document" and
        "Could not find definitive answer" not in doc_result["answer"] and
            "Error processing" not in doc_result["answer"]):
        logger.info("Found answer in documents")
        return doc_result

    # Step 2: Fallback to FT API
    logger.info("Falling back to Financial Times API")
    try:
        ft_data = query_ft_api(symbol)
        financial_info = extract_financial_info(ft_data)

        if "error" not in financial_info:
            company = financial_info["company"]
            employees = financial_info.get("employees", "N/A")
            answer = f"According to Financial Times data for {company} ({financial_info['symbol']}):\n"
            answer += f"Number of Employees: {employees}\n"
            logger.info(f"Found answer in Financial Times API for {company}")
            return {
                "answer": answer,
                "sources": [{"source": "Financial Times API", "symbol": financial_info["symbol"]}],
                "source_type": "financial_times"
            }
        else:
            logger.warning(f"No financial information found for {symbol}")
            return {
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


# Main execution (for testing purposes)
# if __name__ == "__main__":
#     # Initialize the RAG system
#     if initialize_rag_system():
#         logger.info("RAG system initialized successfully")

#         # Load the default question
#         system_prompt, question = load_prompt()
#         if question:
#             logger.info(f"Processing default question: {question}")

#             # Process the query using the MCP tool
#             result = process_financial_query(question)

#             print(f"Question: {question}")
#             print(f"Answer: {result['answer']}")
#             print(f"Source Type: {result['source_type']}")
#             if result.get('sources'):
#                 print(f"Sources: {result['sources']}")
#         else:
#             logger.error("No question loaded")
#             print("Error: Could not load question from prompt.txt")
#     else:
#         logger.error("Failed to initialize RAG system")
#         print("Error: RAG pipeline could not be initialized")
if __name__ == "__main__":
    # Initialize the RAG system
    if initialize_rag_system():
        # Load the prompt and question
        system_prompt, question = load_prompt()

        if question:
            # Process the query
            result = process_financial_query(question)

            # Print the results
            print(f"Question: {question}")
            print(f"Answer: {result['answer']}")
            print(f"Source Type: {result['source_type']}")
            if result.get('sources'):
                print(f"Sources: {result['sources']}")

        else:
            logger.error("No question loaded")
            print("Error: Could not load question from prompt.txt")

    else:
        logger.error("Failed to initialize RAG system")
        print("Error: RAG pipeline could not be initialized")
