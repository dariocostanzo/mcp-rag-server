# PDF RAG Application with FastMCP, Ollama, and LangChain

A simple application for querying PDF documents using Retrieval-Augmented Generation (RAG) with local LLMs via Ollama, with a fallback to the Financial Times API for financial data.

## How it Works

1. **Load PDFs**: The app reads PDF files from a `data/` folder.
2. **Chunking**: Text from PDFs is split into smaller, manageable chunks.
3. **Embedding**: Ollama converts these text chunks into numerical representations (embeddings).
4. **Vector Store**: Embeddings are stored in a FAISS vector store for semantic search.
5. **Querying**: When a question is asked (e.g., the default question), the app:
   - Converts the question to an embedding.
   - Finds the most relevant text chunks from the PDF.
   - Sends these chunks and the question to the Ollama LLM.
   - Returns an answer based on the PDF context.
   - If no answer is found in the PDF, it queries the Financial Times API for relevant data.
6. **Output**: The answer and its sources (PDF pages or FT API) are printed to the console.

## Prerequisites

- **Python 3.11**: [Download Python 3.11](https://www.python.org/downloads/release/python-3110/)
- **Poetry**: For Python package management. [Installation instructions](https://python-poetry.org/docs/#installation)
- **Ollama**: For running local LLMs.
  - Download from [ollama.com/download](https://ollama.com/download)
  - Pull the model: `ollama pull llama3`
- **PDF File**: A PDF file (e.g., `Barclays-PLC-Annual-Report-2020.pdf`) in the `data/` folder.

## Setup Instructions

1. **Clone/Download the Project**:
   - Ensure all files (`main.py`, `pyproject.toml`, `.env.example`) are in the project root.

2. **Install Dependencies**:
   - Open a terminal in the project root and run:
     ```bash
     poetry env use python3.11
     poetry install


