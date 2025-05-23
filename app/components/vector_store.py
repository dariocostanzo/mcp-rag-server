import logging
from pathlib import Path
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Step 2 (partial): Use logging from main.py
# - Reuses the logger configured in main.py for consistency
logger = logging.getLogger(__name__)

# Step 4 & 6 & 7: Vector Store class
# - Manages PDF loading, chunking, and RAG system initialization


class VectorStore:
    # Step 4: Initialize global variables for RAG system
    # - Vector store holds document embeddings; QA chain handles question-answering
    vector_store = None
    qa_chain = None

    @staticmethod
    def load_and_process_pdf(pdf_name="Barclays-PLC-Annual-Report-2020.pdf", data_dir=None):
        # Step 6: Load and process PDF
        # - Loads a PDF file, splits it into chunks for efficient processing
        logger.info(f"Loading PDF from {data_dir}")
        pdf_path = os.path.join(data_dir, pdf_name)
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

    @classmethod
    def initialize_rag_system(cls, ollama_model=None, vector_store_path=None):
        # Step 7: Initialize the RAG system
        # - Sets up embeddings, LLM, vector store, and QA chain for document querying
        logger.info("Initializing embeddings and LLM")
        embeddings = OllamaEmbeddings(model=ollama_model)
        llm = ChatOllama(model=ollama_model)
        if os.path.exists(vector_store_path) and os.path.isdir(vector_store_path) and len(os.listdir(vector_store_path)) > 0:
            try:
                logger.info(
                    f"Loading existing vector store from {vector_store_path}")
                cls.vector_store = FAISS.load_local(
                    vector_store_path, embeddings, allow_dangerous_deserialization=True)
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                cls.vector_store = None
        if cls.vector_store is None:
            chunks = cls.load_and_process_pdf(
                data_dir=vector_store_path.rsplit(os.sep, 1)[0])
            if chunks:
                try:
                    logger.info("Creating new FAISS vector store")
                    cls.vector_store = FAISS.from_documents(chunks, embeddings)
                    Path(vector_store_path).mkdir(parents=True, exist_ok=True)
                    cls.vector_store.save_local(vector_store_path)
                    logger.info(f"Vector store saved to {vector_store_path}")
                except Exception as e:
                    logger.error(f"Error creating vector store: {str(e)}")
                    return False
            else:
                logger.warning("No chunks available to create vector store")
                return False
        if cls.vector_store:
            try:
                cls.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=cls.vector_store.as_retriever(),
                    return_source_documents=True
                )
                logger.info("RAG pipeline initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Error initializing QA chain: {str(e)}")
                return False
        return False
