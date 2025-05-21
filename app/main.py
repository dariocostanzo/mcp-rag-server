import os
import logging
from dotenv import load_dotenv

# FastMCP imports
from fastmcp.server import MCPApplication
from fastmcp.tool import ToolContext

# Langchain imports
from langchain_community.llms import Ollama as LangchainOllama # Renamed to avoid conflict
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Basic Logging Setup ---
# This helps us see what the application is doing.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
# This loads the variables from your .env file (OLLAMA_BASE_URL, OLLAMA_MODEL)
load_dotenv()

# --- Configuration ---
# These are settings for our application.
PDF_DATA_DIRECTORY = "data"  # Folder where your PDF files are stored
PROMPT_FILE_PATH = "prompt.txt" # Path to your prompt template file
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL")

# --- Global Variables ---
# We'll store our initialized RAG chain here so it's created only once.
rag_chain_instance = None
is_rag_pipeline_ready = False

# --- FastMCP Application Definition ---
# This creates our FastMCP application instance.
# The name "pdf_query_service" will be part of how other services might find our tool.
mcp_app = MCPApplication(
    name="pdf_query_service",
    title="PDF Document Query Service (RAG + Ollama)",
    description="A service to ask questions about PDF documents using a RAG pipeline with Ollama."
)

# --- Helper Function: Load Prompt Template ---
def load_prompt_from_file(file_path: str) -> PromptTemplate | None:
    """Loads the prompt template from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        return PromptTemplate(template=prompt_content, input_variables=["context", "question"])
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading prompt template: {e}")
        return None

# --- Core Function: Initialize the RAG Pipeline ---
def initialize_rag_pipeline():
    """
    Sets up the entire RAG (Retrieval Augmented Generation) pipeline.
    1. Loads PDFs from the 'data' directory.
    2. Splits them into smaller chunks.
    3. Creates embeddings (numerical representations) for these chunks using Ollama.
    4. Stores these embeddings in a FAISS vector store for quick searching.
    5. Sets up the Langchain QA chain with Ollama as the LLM.
    """
    global rag_chain_instance, is_rag_pipeline_ready

    if is_rag_pipeline_ready:
        logger.info("RAG pipeline is already initialized.")
        return True

    logger.info("Starting RAG pipeline initialization...")

    # Check if Ollama settings are available
    if not OLLAMA_BASE_URL or not OLLAMA_MODEL_NAME:
        logger.error("Ollama base URL or model name not configured in .env file. Halting RAG initialization.")
        return False

    # 1. Load PDF documents
    if not os.path.exists(PDF_DATA_DIRECTORY) or not os.path.isdir(PDF_DATA_DIRECTORY):
        logger.error(f"PDF data directory '{PDF_DATA_DIRECTORY}' not found. Please create it and add PDF files.")
        return False

    pdf_files = [f for f in os.listdir(PDF_DATA_DIRECTORY) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{PDF_DATA_DIRECTORY}'. RAG pipeline will be initialized without documents.")
        # We can choose to proceed or return False. For a beginner, let's allow it to proceed
        # but the tool won't be very useful.
        # return False # Uncomment this if you want to stop if no PDFs are found

    all_doc_chunks = []
    for pdf_file_name in pdf_files:
        pdf_path = os.path.join(PDF_DATA_DIRECTORY, pdf_file_name)
        try:
            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 2. Split documents into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            all_doc_chunks.extend(chunks)
            logger.info(f"Loaded and split {pdf_file_name} into {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to load or split {pdf_file_name}: {e}")
            continue # Skip to the next PDF if one fails

    if not all_doc_chunks:
        logger.error("No text chunks were extracted from any PDF files. Cannot build vector store.")
        return False

    try:
        # 3. Initialize Ollama embeddings
        logger.info(f"Initializing Ollama embeddings (Model: {OLLAMA_MODEL_NAME}, URL: {OLLAMA_BASE_URL})")
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL)

        # 4. Create FAISS vector store from document chunks
        logger.info(f"Creating FAISS vector store from {len(all_doc_chunks)} chunks...")
        vector_store = FAISS.from_documents(all_doc_chunks, embeddings)
        retriever = vector_store.as_retriever() # This will be used to fetch relevant chunks
        logger.info("FAISS vector store created successfully.")

        # 5. Load prompt template
        prompt_template = load_prompt_from_file(PROMPT_FILE_PATH)
        if not prompt_template:
            logger.error("Failed to load prompt template. Cannot create QA chain.")
            return False

        # 6. Initialize Ollama LLM for Langchain
        logger.info(f"Initializing Ollama LLM for Langchain (Model: {OLLAMA_MODEL_NAME}, URL: {OLLAMA_BASE_URL})")
        llm = LangchainOllama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL)

        # 7. Create RetrievalQA chain
        # This chain combines the retriever (to get relevant text) and the LLM (to generate an answer)
        logger.info("Creating RetrievalQA chain...")
        rag_chain_instance = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" puts all retrieved chunks into the prompt
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True # So we can see which chunks were used
        )
        is_rag_pipeline_ready = True
        logger.info("RAG pipeline initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"An error occurred during RAG pipeline initialization: {e}", exc_info=True)
        is_rag_pipeline_ready = False
        return False

# --- FastMCP Tool Definition ---
@mcp_app.tool(
    name="query_documents",
    description="Ask a question about the content of the loaded PDF documents. Returns an answer and cited sources.",
    # We can define inputs and outputs for better clarity and validation if needed.
    # For a beginner, a simple string input is fine. FastMCP infers this.
)
async def query_pdf_documents_tool(context: ToolContext, question: str) -> dict:
    """
    This is the function that FastMCP will call when the 'query_documents' tool is used.
    It takes a 'question' string and returns a dictionary with the answer.
    """
    logger.info(f"MCP Tool 'query_documents' received question: \"{question}\"")

    if not is_rag_pipeline_ready:
        logger.warning("RAG pipeline is not ready. Attempting to initialize now...")
        if not initialize_rag_pipeline(): # Try to initialize if it failed earlier or wasn't run
            return {"error": "RAG pipeline could not be initialized. Please check server logs and PDF files."}
    
    if not rag_chain_instance: # Should be caught by the above, but as a double check
         return {"error": "RAG chain is not available. Initialization might have failed."}

    try:
        # Langchain's .invoke is synchronous. FastMCP tools are async.
        # We use run_in_executor to run the synchronous Langchain call in a separate thread
        # so it doesn't block FastMCP's event loop.
        logger.info("Invoking RAG chain...")
        result = await mcp_app.run_in_executor(None, rag_chain_instance.invoke, {"query": question})
        
        answer = result.get("result", "No answer could be generated.")
        source_docs = result.get("source_documents", [])
        
        # Format source documents for the response
        sources_info = []
        for doc in source_docs:
            sources_info.append({
                "content_snippet": doc.page_content[:250] + "...", # Show a snippet
                "metadata": doc.metadata # e.g., {"source": "data/my_doc.pdf", "page": 3}
            })

        logger.info(f"Answer generated: \"{answer[:100]}...\"")
        return {
            "question": question,
            "answer": answer,
            "sources": sources_info
        }
    except Exception as e:
        logger.error(f"Error processing question in 'query_documents' tool: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Main Execution Block ---
# This code runs when you execute `python app/main.py`
if __name__ == "__main__":
    logger.info("Application starting...")

    # Attempt to initialize the RAG pipeline when the application starts.
    # This is "eager" initialization.
    initialize_rag_pipeline()

    # If initialization failed, the tool will try again when called,
    # but it's good to know at startup.
    if not is_rag_pipeline_ready:
        logger.warning("RAG pipeline failed to initialize on startup. The tool might not work correctly until PDFs are available and Ollama is running.")

    # Start the FastMCP server
    # This makes your tool available over the network.
    # Default host is 'localhost' and port is often 8765 for MCP.
    # You can specify host="0.0.0.0" to make it accessible from other machines on your network.
    try:
        logger.info("Starting FastMCP server on http://localhost:8765")
        logger.info("The tool 'query_documents' will be available at /pdf_query_service/tools/query_documents")
        mcp_app.run(host="localhost", port=8765) # FastMCP's built-in way to run the server
    except Exception as e:
        logger.error(f"Failed to start FastMCP server: {e}", exc_info=True)
        logger.info("If the above failed, try running with: poetry run fastmcp serve app.main:mcp_app --port 8765")