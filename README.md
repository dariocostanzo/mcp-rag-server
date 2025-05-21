# Basic PDF RAG Query App with FastMCP, Ollama, and Langchain

This is a beginner-friendly project demonstrating how to build a simple Question-Answering application over your PDF documents using Retrieval Augmented Generation (RAG).

It uses:
-   **FastMCP**: To create a "Tool" (like an API endpoint) that can be called to ask questions.
-   **Ollama**: To run a local Large Language Model (LLM) like Mistral or Llama for generating answers and creating text embeddings.
-   **Langchain**: A framework to chain together components for building LLM applications (loading PDFs, splitting text, vector stores, interacting with LLMs).
-   **Poetry**: For managing Python dependencies.
-   **FAISS**: For creating a local vector store to enable semantic search over PDF content.

## How it Works (Simplified)

1.  **Load PDFs**: The app reads PDF files from a `data/` folder.
2.  **Chunking**: Text from PDFs is split into smaller, manageable chunks.
3.  **Embedding**: Ollama (via Langchain) converts these text chunks into numerical representations (embeddings) that capture their meaning.
4.  **Vector Store**: These embeddings are stored in a FAISS vector store. This allows for fast semantic search (finding chunks смыслово_похожие to your question).
5.  **Querying**:
    a.  When you ask a question, it's also converted into an embedding.
    b.  The vector store finds the most relevant text chunks from your PDFs based on embedding similarity.
    c.  These relevant chunks (the "context") and your original question are given to the Ollama LLM using a predefined prompt (from `prompt.txt`).
    d.  The LLM generates an answer based *only* on the provided context and question.
6.  **FastMCP Tool**: All this logic is wrapped in a FastMCP tool named `query_documents` that you can call.

## Prerequisites

-   **Python 3.9+**
-   **Poetry**: For Python package management. (See [Poetry's official website](https://python-poetry.org/docs/) for installation instructions).
-   **Ollama**: Installed and running locally.
    -   Download Ollama from [ollama.com](https://ollama.com/).
    -   Pull a model you want to use (e.g., `ollama pull mistral`). This project defaults to `mistral`.

## Setup Instructions

1.  **Clone/Download the Project:**
    If this were a Git repository: `git clone <repository_url>`
    Otherwise, ensure all files are in the correct structure as outlined. The root folder is `c:\Users\dario\projects\mcp-server-rag-ollama-langchain`.

2.  **Install Dependencies:**
    Open your terminal/command prompt, navigate to the project root (`c:\Users\dario\projects\mcp-server-rag-ollama-langchain`), and run:
    ```bash
    poetry install
    ```

3.  **Configure Environment Variables:**
    -   Copy the `.env.example` file to a new file named `.env` in the project root.
        ```bash
        copy .env.example .env
        ```
    -   Edit the `.env` file:
        -   `OLLAMA_BASE_URL`: Should be `http://localhost:11434` if Ollama is running locally with default settings.
        -   `OLLAMA_MODEL`: Specify the Ollama model you want to use (e.g., `mistral`, `llama2`, `llama3`). Make sure you have pulled this model using `ollama pull <model_name>`.

4.  **Add Your PDF Files:**
    -   Create a folder named `data` in the project root (`c:\Users\dario\projects\mcp-server-rag-ollama-langchain\data\`).
    -   Place any PDF files you want to query into this `data` folder.

## Running the Application

1.  **Ensure Ollama is Running:**
    Start your Ollama application and make sure the model specified in your `.env` file is available.

2.  **Start the FastMCP Server:**
    Open your terminal in the project root and run:
    ```bash
    poetry run python app/main.py
    ```
    Alternatively, you can use the FastMCP CLI (this is often preferred for more complex setups):
    ```bash
    poetry run fastmcp serve app.main:mcp_app --port 8765
    ```
    The server should start, typically on `http://localhost:8765`. You'll see log messages indicating the RAG pipeline initialization. This might take a few moments, especially the first time or with large PDFs, as it processes the documents.

## Using the `query_documents` Tool

Once the server is running, the `query_documents` tool is available. FastMCP tools are usually called by other programs, but you can test it with `curl` or a tool like Postman/Insomnia.

-   **Tool Endpoint**: `http://localhost:8765/pdf_query_service/tools/query_documents`
-   **Method**: `POST`
-   **Request Body (JSON)**:
    ```json
    {
        "question": "Your question about the PDFs here"
    }
    ```

**Example using `curl` (run this in a new terminal):**

```bash
curl -X POST -H "Content-Type: application/json" \
-d "{\"question\": \"What is the main topic discussed in the documents?\"}" \
http://localhost:8765/pdf_query_service/tools/query_documents