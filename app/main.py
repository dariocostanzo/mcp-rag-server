import os
import logging
from pathlib import Path
import time
import argparse
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from fastmcp import FastMCP
import requests
from dotenv import load_dotenv

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables and set paths
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vector_store")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
FT_API_BASE_URL = "https://markets.ft.com/research/webservices/companies/v1/profile"
FT_COOKIE = os.getenv("FT_COOKIE", "")

# Initialize FastMCP application - THIS IS THE MCP SERVER
mcp = FastMCP("PDF RAG Financial Data Extractor")

# Global variables for RAG system
vector_store = None
qa_chain = None
system_prompt = ""


def check_ollama_connection():
    """Check if Ollama is running and model is available"""
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        test_text = ["test connection"]
        embeddings.embed_documents(test_text)
        logger.info(
            f"✅ Ollama connection successful with model: {OLLAMA_MODEL}")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot connect to Ollama: {str(e)}")
        logger.error("Make sure Ollama is running and the model is available")
        return False


def load_prompt():
    """Load system prompt from file"""
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompt.txt")
    default_prompt = """You are a financial assistant. Answer questions using the provided context from documents.
If you cannot find the answer in the context, say "I cannot find this information in the provided documents."
Be specific and accurate in your responses."""

    if not os.path.exists(prompt_path):
        logger.warning(
            f"Prompt file not found at {prompt_path}, using default")
        return default_prompt

    try:
        with open(prompt_path, 'r') as file:
            prompt = file.read().strip()
        logger.info("✅ Prompt loaded successfully from prompt.txt")
        return prompt if prompt else default_prompt
    except Exception as e:
        logger.error(f"Error loading prompt: {str(e)}")
        return default_prompt


def load_and_process_pdf(pdf_name="Barclays-PLC-Annual-Report-2020.pdf"):
    """Load and process PDF documents"""
    logger.info(f"📄 Loading PDF from {DATA_DIR}")
    pdf_path = os.path.join(DATA_DIR, pdf_name)

    if not os.path.exists(pdf_path):
        logger.error(f"❌ PDF not found at {pdf_path}")
        # List available files for debugging
        if os.path.exists(DATA_DIR):
            available_files = [f for f in os.listdir(
                DATA_DIR) if f.endswith('.pdf')]
            logger.info(f"Available PDF files: {available_files}")
        return []

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"✅ Loaded {len(documents)} pages from PDF")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,  # Better overlap for context
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"✅ Created {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        logger.error(f"❌ Error processing PDF: {str(e)}")
        return []


def initialize_rag_system():
    """Initialize the RAG system with embeddings, LLM, and QA chain"""
    global vector_store, qa_chain, system_prompt

    logger.info("🚀 Starting RAG system initialization...")

    # Check Ollama connection first
    if not check_ollama_connection():
        return False

    # Load system prompt
    system_prompt = load_prompt()

    try:
        # Initialize embeddings and LLM
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            system=system_prompt,
            temperature=0.1  # Lower temperature for consistent answers
        )
        logger.info("✅ Embeddings and LLM initialized")

        # Load or create vector store
        if (os.path.exists(VECTOR_STORE_PATH) and
            os.path.isdir(VECTOR_STORE_PATH) and
                len(os.listdir(VECTOR_STORE_PATH)) > 0):
            try:
                logger.info(
                    f"📂 Loading existing vector store from {VECTOR_STORE_PATH}")
                vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Vector store loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading vector store: {str(e)}")
                logger.info("Will create new vector store...")
                vector_store = None

        # Create new vector store if needed
        if vector_store is None:
            chunks = load_and_process_pdf()
            if not chunks:
                logger.error("❌ No document chunks available")
                return False

            try:
                logger.info("🔨 Creating new FAISS vector store...")
                vector_store = FAISS.from_documents(chunks, embeddings)
                Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
                vector_store.save_local(VECTOR_STORE_PATH)
                logger.info(f"✅ Vector store saved to {VECTOR_STORE_PATH}")
            except Exception as e:
                logger.error(f"❌ Error creating vector store: {str(e)}")
                return False

        # Initialize QA chain
        if vector_store:
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(
                        search_kwargs={"k": 5}),
                    return_source_documents=True
                )
                logger.info("✅ RAG pipeline initialized successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Error initializing QA chain: {str(e)}")
                return False

        return False

    except Exception as e:
        logger.error(f"❌ Error initializing RAG system: {str(e)}")
        return False

# MCP TOOL 1: Query documents using RAG system


@mcp.tool()
def query_documents(question: str) -> dict:
    """
    Query the vector database for relevant information from PDF documents.

    This is an MCP tool that can be called by MCP clients like Claude Desktop.

    Args:
        question: The question to search for in the documents

    Returns:
        dict: Contains answer, sources, and source_type
    """
    logger.info(f"🔍 RAG Query: {question}")

    if qa_chain is None:
        logger.warning("❌ QA chain not initialized")
        return {
            "answer": "RAG system not initialized.",
            "sources": [],
            "source_type": "none",
            "success": False
        }

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            }
            for doc in result.get("source_documents", [])
        ]

        # Check if answer contains uncertainty phrases
        uncertainty_phrases = [
            "i don't know", "i don't have", "cannot find",
            "no information", "not mentioned", "not provided",
            "unable to find", "not available", "i cannot find this information"
        ]

        is_helpful = not any(phrase in answer.lower()
                             for phrase in uncertainty_phrases)

        if is_helpful:
            logger.info(
                f"✅ Found answer in PDF documents. Sources: {len(sources)}")
            return {
                "answer": answer,
                "sources": sources,
                "source_type": "document",
                "success": True
            }
        else:
            logger.info("❌ No definitive answer found in PDF")
            return {
                "answer": "Could not find definitive answer in documents.",
                "sources": sources,
                "source_type": "document",
                "success": False
            }

    except Exception as e:
        logger.error(f"❌ Error processing RAG query: {str(e)}")
        return {
            "answer": f"Error processing document query: {str(e)}",
            "sources": [],
            "source_type": "none",
            "success": False
        }

# MCP TOOL 2: Query Financial Times API


@mcp.tool()
def query_ft_api(symbol: str = "BARC:LSE", endpoint: str = "company-overview") -> dict:
    """
    Query the Financial Times API for company financial metrics.

    Args:
        symbol: Stock symbol to query (default: "BARC:LSE").
        endpoint: The specific API endpoint to query (e.g., "company-overview", "financials").

    Returns:
        dict: Financial data from FT API or error information.
    """

    logger.info(f"📊 FT API Query: {symbol} (Endpoint: {endpoint})")

    if not FT_COOKIE:
        logger.warning(
            "❌ FT_COOKIE not configured. Please set the environment variable.")
        return {"error": "FT_COOKIE not configured", "success": False}

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

        # Parse cookies safely
        cookie_dict = {}
        for cookie in FT_COOKIE.strip().split(";"):  # Removed extra space after semicolon
            if "=" in cookie:
                key, value = cookie.split("=", 1)
                # Added .strip() to remove whitespace
                cookie_dict[key] = value.strip()

        # Construct the full URL dynamically

        response = requests.get(
            FT_API_BASE_URL, params=params, headers=headers, cookies=cookie_dict
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        logger.info(f"✅ Successfully retrieved data from FT API for {symbol}")
        return {"data": data, "success": True}

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error fetching data from FT API: {str(e)}")
        return {"error": f"Failed to retrieve data: {str(e)}", "success": False}

    except Exception as e:  # Catch any other potential errors
        # Log the full traceback
        logger.exception(f"❌ An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}", "success": False}

# MCP TOOL 3: Extract financial information from FT API response


@mcp.tool()
def extract_financial_info(api_response: dict) -> dict:
    """
    Extract and format financial information from FT API response.

    Args:
        api_response: Raw response from query_ft_api tool

    Returns:
        dict: Extracted and formatted financial data
    """
    if not api_response.get("success", False) or "error" in api_response:
        return {
            "error": api_response.get("error", "Invalid API response"),
            "success": False
        }

    try:
        data = api_response.get("data", {})
        items = data.get("data", {}).get("items", [])

        if not items:
            logger.warning("❌ No items found in FT API response")
            return {
                "error": "No financial data found",
                "success": False
            }

        item = items[0]
        basic_info = item.get("basic", {})
        profile_info = item.get("profile", {})

        financial_data = {
            "company": basic_info.get("name", "Unknown"),
            "symbol": basic_info.get("symbol", "Unknown"),
            "exchange": basic_info.get("exchange", "Unknown"),
            "currency": basic_info.get("currency", "Unknown"),
            "price": basic_info.get("last", "N/A"),
            "employees": profile_info.get("employees", "N/A"),
            "incorporated_year": profile_info.get("incorporatedYear", "N/A"),
            "net_income": profile_info.get("netIncomeMRFY", "N/A"),
            "total_revenue": profile_info.get("totalRevenueMRFY", "N/A"),
            "market_cap": profile_info.get("marketCapitalization", "N/A"),
            "description": profile_info.get("description", "No description available"),
            "address": profile_info.get("contact", {}).get("physicalAddress", {}).get("line1", "N/A"),
            "phone": profile_info.get("contact", {}).get("phone", {}).get("number", "N/A"),
            "website": profile_info.get("contact", {}).get("webURL", "N/A"),
            "success": True
        }

        logger.info(
            f"✅ Extracted financial info for {financial_data['company']}")
        return financial_data

    except Exception as e:
        logger.error(f"❌ Error extracting financial information: {str(e)}")
        return {
            "error": f"Error extracting financial information: {str(e)}",
            "success": False
        }

# MCP TOOL 4: Main financial query processor with RAG → FT API fallback


@mcp.tool()
def financial_query(question: str, symbol: str = "BARC:LSE") -> dict:
    """
    Process a financial query using RAG first, then fallback to FT API if needed.

    This is the main MCP tool that orchestrates the RAG → API fallback logic.

    Args:
        question: The financial question to answer
        symbol: Stock symbol for FT API fallback (default: "BARC:LSE")

    Returns:
        dict: Complete response with answer, sources, and source_type
    """
    logger.info(f"🎯 Financial Query: {question}")

    # Step 1: Try RAG system first
    logger.info("📚 Step 1: Trying RAG system...")
    doc_result = query_documents(question)

    if doc_result.get("success", False):
        logger.info("✅ Found answer in documents!")
        return {
            "question": question,
            "answer": doc_result["answer"],
            "sources": doc_result["sources"],
            "source_type": "document",
            "success": True
        }

    # Step 2: Fallback to FT API
    logger.info("🌐 Step 2: Falling back to Financial Times API...")
    try:
        # Query FT API
        api_result = query_ft_api(symbol)
        if not api_result.get("success", False):
            logger.warning(
                f"❌ FT API query failed: {api_result.get('error', 'Unknown error')}")
            return {
                "question": question,
                "answer": f"Could not find information in documents or FT API. Error: {api_result.get('error', 'Unknown error')}",
                "sources": [],
                "source_type": "none",
                "success": False
            }

        # Extract financial information
        financial_info = extract_financial_info(api_result)
        if not financial_info.get("success", False):
            logger.warning(
                f"❌ Failed to extract financial info: {financial_info.get('error', 'Unknown error')}")
            return {
                "question": question,
                "answer": f"Could not extract financial information for {symbol}.",
                "sources": [],
                "source_type": "none",
                "success": False
            }

        # Format comprehensive answer
        company = financial_info["company"]
        answer_parts = [
            f"📊 Financial Information for {company} ({financial_info['symbol']}):",
            f"💰 Current Price: {financial_info['price']} {financial_info['currency']}",
            f"🏢 Market Cap: {financial_info['market_cap']}",
            f"💵 Net Income (MRFY): {financial_info['net_income']}",
            f"📈 Total Revenue (MRFY): {financial_info['total_revenue']}",
            f"👥 Employees: {financial_info['employees']}",
            f"📅 Incorporated: {financial_info['incorporated_year']}",
            f"🌐 Website: {financial_info['website']}",
            f"📍 Address: {financial_info['address']}",
        ]

        # Add description if available and not too long
        if financial_info['description'] != "No description available":
            desc = financial_info['description']
            if len(desc) > 200:
                desc = desc[:200] + "..."
            answer_parts.append(f"ℹ️ Description: {desc}")

        logger.info(f"✅ Found comprehensive answer in FT API for {company}")
        return {
            "question": question,
            "answer": "\n".join(answer_parts),
            "sources": [{"source": "Financial Times API", "symbol": financial_info["symbol"]}],
            "source_type": "api",
            "success": True
        }

    except Exception as e:
        logger.error(f"❌ Error processing FT API query: {str(e)}")
        return {
            "question": question,
            "answer": f"Error processing FT API query: {str(e)}",
            "sources": [],
            "source_type": "none",
            "success": False
        }

# MCP TOOL 5: System status checker


@mcp.tool()
def system_status() -> dict:
    """
    Check the status of all system components.

    Returns:
        dict: Status of RAG system, Ollama, vector store, and FT API configuration
    """
    status = {
        "rag_system": {
            "initialized": qa_chain is not None,
            "vector_store": vector_store is not None,
        },
        "ollama": {
            "connected": check_ollama_connection(),
            "model": OLLAMA_MODEL
        },
        "ft_api": {
            "configured": bool(FT_COOKIE),
        },
        "data_sources": {
            "pdf_files": [],
            "data_dir": DATA_DIR
        }
    }

    # Check for PDF files
    if os.path.exists(DATA_DIR):
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        status["data_sources"]["pdf_files"] = pdf_files

    return status


def run_test_queries():
    """Run automated test queries to verify the MCP tools work correctly"""
    logger.info("🧪 Running automated test queries...")

    test_questions = [
        "What was the total assets value in 2020?",
        "What was the total assets value in 2024?",
    ]

    print("\n" + "="*80)
    print("🧪 AUTOMATED MCP TOOL TESTING")
    print("="*80)

    # Test system status first
    print("\n📊 SYSTEM STATUS:")
    print("-" * 40)
    status = system_status()
    for component, details in status.items():
        print(f"{component}: {details}")

    # Test each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 TEST {i}/5: {question}")
        print("-" * 60)

        try:
            # Test the main financial_query tool
            result = financial_query(question)

            print(f"✅ Success: {result.get('success', False)}")
            print(f"📝 Answer: {result.get('answer', 'No answer')}")
            print(f"📚 Source Type: {result.get('source_type', 'unknown')}")

            sources = result.get('sources', [])
            if sources:
                print(f"📖 Sources ({len(sources)}):")
                for source in sources[:3]:  # Show first 3 sources
                    print(f"   - {source}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        print()  # Add spacing
        time.sleep(1)  # Brief pause between queries

    print("\n" + "="*80)
    print("🏁 Test completed! Check the results above.")
    print("="*80)


def main():
    """Main function that initializes RAG and starts the MCP server or runs tests"""
    parser = argparse.ArgumentParser(
        description='MCP Server with RAG and Financial Data')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (execute sample queries automatically)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive test mode (ask questions manually)')

    args = parser.parse_args()

    logger.info("🚀 Starting MCP Server with RAG...")

    # Initialize RAG system
    if not initialize_rag_system():
        logger.error("❌ Failed to initialize RAG system")
        print("Error: Could not initialize RAG system. Please check:")
        print("1. Ollama is running")
        print(f"2. Model '{OLLAMA_MODEL}' is available")
        print("3. PDF documents exist in the data directory")
        print("4. All required dependencies are installed")
        return False

    logger.info("✅ RAG system initialized successfully")

    # Handle different modes
    if args.test:
        # Run automated tests
        run_test_queries()
        return True
    elif args.interactive:
        # Run interactive test mode
        run_interactive_tests()
        return True
    else:
        # Start the MCP server normally
        logger.info("🌐 Starting FastMCP server...")
        logger.info("🔧 Available MCP tools:")
        logger.info("   - query_documents: Query PDF documents using RAG")
        logger.info("   - query_ft_api: Query Financial Times API")
        logger.info(
            "   - extract_financial_info: Extract data from FT API response")
        logger.info("   - financial_query: Main query with RAG → API fallback")
        logger.info("   - system_status: Check system component status")
        logger.info("📡 MCP server will run until interrupted (Ctrl+C)")
        logger.info("")
        logger.info(
            "💡 To test the server, run with --test or --interactive flags:")
        logger.info("   python your_script.py --test")
        logger.info("   python your_script.py --interactive")

        try:
            mcp.run()  # This starts the MCP server and keeps it running
        except KeyboardInterrupt:
            logger.info("🛑 MCP server stopped by user")
        except Exception as e:
            logger.error(f"❌ MCP server error: {str(e)}")
            return False

    return True


def run_interactive_tests():
    """Run interactive test mode where user can ask questions"""
    print("\n" + "="*80)
    print("🎮 INTERACTIVE MCP TOOL TESTING")
    print("="*80)
    print("Type your questions and press Enter. Type 'quit' to exit.")
    print("Example questions:")
    print("  - What is the company's revenue?")
    print("  - How many employees work there?")
    print("  - What is the current stock price?")
    print("-" * 80)

    while True:
        try:
            question = input("\n🔍 Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not question:
                continue

            print(f"\n⏳ Processing: '{question}'...")

            # Test the main financial_query tool
            result = financial_query(question)

            print(f"\n✅ Success: {result.get('success', False)}")
            print(f"📝 Answer:\n{result.get('answer', 'No answer')}")
            print(f"📚 Source Type: {result.get('source_type', 'unknown')}")

            sources = result.get('sources', [])
            if sources:
                print(f"📖 Sources ({len(sources)}):")
                for source in sources[:3]:  # Show first 3 sources
                    print(f"   - {source}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


# Entry point - this is what runs when you execute the script
if __name__ == "__main__":
    main()
