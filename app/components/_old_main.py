# import os
# import logging
# from pathlib import Path
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from fastmcp import FastMCP
# import uvicorn


# # Basic logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# # Configuration
# DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
# VECTOR_STORE_PATH = os.path.join(
#     os.path.dirname(__file__), "..", "vector_store")
# OLLAMA_MODEL = "llama3"
# DEFAULT_QUESTION = "What is the total assets value of Barclays PLC in 2020?"


# # Initialize FastMCP
# mcp_app = FastMCP("PDF RAG App")
# logger.info("FastMCP initialized")


# # Load and process PDF
# def load_and_process_pdf(pdf_name="Barclays-PLC-Annual-Report-2020.pdf"):
#     logger.info(f"Loading PDF from {DATA_DIR}")
#     pdf_path = os.path.join(DATA_DIR, pdf_name)

#     if not os.path.exists(pdf_path):
#         logger.error(f"PDF not found at {pdf_path}")
#         return []

#     try:
#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()
#         logger.info(f"Loaded {len(documents)} pages from PDF")

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=100)
#         chunks = splitter.split_documents(documents)
#         logger.info(f"Created {len(chunks)} chunks")
#         return chunks
#     except Exception as e:
#         logger.error(f"Error processing PDF: {str(e)}")
#         return []


# # Initialize embeddings and LLM
# logger.info("Initializing embeddings and LLM")
# embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
# llm = ChatOllama(model=OLLAMA_MODEL)


# # Initialize or load vector store
# vector_store = None
# qa_chain = None


# def initialize_or_load_vectorstore():
#     global vector_store, qa_chain

#     # Check if vector store exists
#     if os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH) and len(os.listdir(VECTOR_STORE_PATH)) > 0:
#         try:
#             logger.info(
#                 f"Loading existing vector store from {VECTOR_STORE_PATH}")
#             vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
#             logger.info("Vector store loaded successfully")
#         except Exception as e:
#             logger.error(f"Error loading vector store: {str(e)}")
#             vector_store = None

#     # If vector store doesn't exist or couldn't be loaded, create a new one
#     if vector_store is None:
#         chunks = load_and_process_pdf()
#         if chunks:
#             try:
#                 logger.info("Creating new FAISS vector store")
#                 vector_store = FAISS.from_documents(chunks, embeddings)

#                 # Save the vector store
#                 Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
#                 vector_store.save_local(VECTOR_STORE_PATH)
#                 logger.info(f"Vector store saved to {VECTOR_STORE_PATH}")
#             except Exception as e:
#                 logger.error(f"Error creating vector store: {str(e)}")
#                 return False
#         else:
#             logger.warning("No chunks available to create vector store")
#             return False

#     # Initialize QA chain
#     if vector_store:
#         try:
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=vector_store.as_retriever(),
#                 return_source_documents=True
#             )
#             logger.info("RAG pipeline initialized successfully")
#             return True
#         except Exception as e:
#             logger.error(f"Error initializing QA chain: {str(e)}")
#             return False

#     return False


# # Initialize the RAG pipeline
# rag_initialized = initialize_or_load_vectorstore()


# # FastMCP endpoint
# @mcp_app.tool()
# def query_documents(question: str = DEFAULT_QUESTION):
#     """
#     Query the RAG pipeline with a question about the loaded documents.

#     Args:
#         question: The question to ask about the documents

#     Returns:
#         A dictionary containing the answer and source information
#     """
#     logger.info(f"Received query: {question}")

#     if not qa_chain:
#         logger.warning("QA chain not initialized. Attempting to initialize...")
#         if not initialize_or_load_vectorstore():
#             return {"answer": "System is not ready. No documents loaded or indexed.", "sources": []}

#     try:
#         result = qa_chain.invoke({"query": question})
#         sources = [{"page": doc.metadata.get("page", "Unknown"),
#                    "source": doc.metadata.get("source", "Unknown")}
#                    for doc in result.get("source_documents", [])]

#         logger.info(f"Query successful. Found {len(sources)} sources")
#         return {"answer": result["result"], "sources": sources}
#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         return {"answer": f"Error processing your query: {str(e)}", "sources": []}


# # Process default question if RAG is initialized
# if rag_initialized:
#     logger.info(f"Processing default question: {DEFAULT_QUESTION}")
#     result = query_documents(DEFAULT_QUESTION)
#     logger.info(f"Default question answer: {result['answer']}")
#     logger.info(f"Sources: {result['sources']}")
# else:
#     logger.warning(
#         "Skipping default question processing as RAG pipeline is not initialized")


# # Add a health check endpoint
# @mcp_app.tool()
# def health_check():
#     """Return the health status of the RAG application"""
#     return {
#         "status": "healthy" if qa_chain else "not_ready",
#         "vector_store": "loaded" if vector_store else "not_loaded",
#         "model": OLLAMA_MODEL
#     }


# # Run server
# if __name__ == "__main__":
#     logger.info("Starting Uvicorn server on 0.0.0.0:8765")
#     uvicorn.run(mcp_app, host="0.0.0.0", port=8765)
