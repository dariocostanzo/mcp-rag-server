# PDF RAG Application with FastMCP, Ollama, and Langchain

A simple application for querying PDF documents using Retrieval Augmented Generation (RAG) with local LLMs via Ollama.

## How it Works

1. **Load PDFs**: The app reads PDF files from a `data/` folder.
2. **Chunking**: Text from PDFs is split into smaller, manageable chunks.
3. **Embedding**: Ollama converts these text chunks into numerical representations (embeddings).
4. **Vector Store**: These embeddings are stored in a FAISS vector store for semantic search.
5. **Querying**: When you ask a question, the app:
   - Converts your question to an embedding
   - Finds the most relevant text chunks
   - Sends these chunks and your question to the Ollama LLM
   - Returns the answer based only on the provided context

## Prerequisites

- **Python 3.11**: [Download Python 3.11](https://www.python.org/downloads/release/python-3110/)
- **Poetry**: For Python package management. [Installation instructions](https://python-poetry.org/docs/#installation)
- **Ollama**: For running local LLMs.
  - Download from [ollama.com/download/windows](https://ollama.com/download/windows)
  - Pull a model: `ollama pull mistral`

## Setup Instructions

1. **Clone/Download the Project**:
   Ensure all files are in the correct structure.

2. **Install Dependencies**:
   Open your command prompt, navigate to the project root, and run:

   ```bash
   poetry env use python3.11
   poetry install
   ```

   3. Configure Environment Variables :

   - Copy the .env.example file to a new file named .env :
     ```
     copy .env.example .env
     ```
   - Edit the .env file:
     - OLLAMA_BASE_URL : Should be http://localhost:11434 if Ollama is running locally.
     - OLLAMA_MODEL : Specify the Ollama model (e.g., mistral ).

3. Add Your PDF Files :

   - Create a folder named data in the project root if it doesn't exist.
   - Place any PDF files you want to query into this data folder.

## Running the Application

1. Ensure Ollama is Running :
   Start your Ollama application and make sure the model specified in your .env file is available.
2. Start the FastMCP Server :
   Open your terminal in the project root and run:

   ```
   poetry run python app/main.
   py
   ```

   Alternatively, you can use the FastMCP CLI:

   ```
   poetry run fastmcp serve 
   app.main:mcp_app --port 
   8765
   ```

   The server should start on http://localhost:8765 .

## Using the query_documents Tool

Once the server is running, the query_documents tool is available. You can test it with curl :

- Tool Endpoint : http://localhost:8765/pdf_query_service/tools/query_documents
- Method : POST
- Request Body (JSON) :
  ```
  {
      "question": "Your 
      question about the 
      PDFs here"
  }
  ```
  Example using curl :

```
curl -X POST -H 
"Content-Type: application/
json" -d "{\"question\": 
\"What is the main topic 
discussed in the documents?
\"}" http://localhost:8765/
pdf_query_service/tools/
query_documents
```

## Troubleshooting

1. Ollama Connection Issues : Make sure Ollama is running.
2. Python Version Issues : Ensure you're using Python 3.11 with python --version .
3. FAISS Installation Issues : If you encounter issues with FAISS:
   ```
   poetry run pip install 
   --no-deps faiss-cpu
   ```
4. Path Issues : Make sure all paths are correctly set for Windows.
5. Permission Issues : Run your command prompt as administrator if needed.

## Project Structure

- app/main.py : The main application code
- data/ : Directory for PDF files
- prompt.txt : The prompt template for the LLM
- pyproject.toml : Poetry configuration file
- .env : Environment variables (create from .env.example )

````

## Create .env.example file

If you don't have an .env.
example file yet, here's one:

```bash:c%3A%5CUsers%5Cdario%
5Cprojects%5Cmcp-server-rag-o
llama-langchain%5C.env.
example
OLLAMA_BASE_URL=http://
localhost:11434
OLLAMA_MODEL=mistral
````

These updates should make your application work properly with Python 3.11 on Windows. The main changes include:

1. Simplified the main.py file for better readability and maintainability
2. Updated the pyproject.toml to use Python 3.11
3. Added uvicorn as a dependency for running the server
4. Updated the README with clear instructions for Windows users
5. Made sure all paths use proper Windows path separators
