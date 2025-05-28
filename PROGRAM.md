# Code Review and Enhancement Suggestions

## What This App Does

This app demonstrates how to combine static documents (PDFs) with live financial data (via the Financial Times API) to answer user questions intelligently. It leverages several key technologies:

* **RAG with PDFs:** Annual reports are loaded, chunked, embedded using Ollama, and queried using LangChain's `RetrievalQA`.
* **LLM (Ollama):** The model generates answers based on retrieved chunks or prompts.
* **Live API Data (FT):** If the answer isn't found in the documents, it queries live company data via the Financial Times API.
* **FastMCP:** A lightweight orchestration layer that registers tools (functions) for programmatic calling, ideal for chaining multiple components.

## How It Works — Step by Step

1. **Load Prompt:** Loads a system prompt and a sample user question from `prompt.txt`.

2. **Load/Process PDF:**
    * Uses `PyPDFLoader` to load a specific PDF (e.g., `Barclays-PLC-Annual-Report-2020.pdf`).
    * Splits the PDF into text chunks using `RecursiveCharacterTextSplitter`.

3. **Embed Chunks & Create Vector Store:**
    * Embeds all text chunks using `OllamaEmbeddings`.
    * Loads a saved FAISS vector store if it exists; otherwise, creates a new one and saves it.

4. **Build RAG Chain:** Uses `RetrievalQA` from LangChain to build a QA system from the FAISS store and Ollama model.

5. **MCP Tools:**
    * `query_documents`: Searches the embedded PDF content for answers.
    * `query_ft_api`: Retrieves real-time data from FT Markets.
    * `extract_financial_info`: Cleans and extracts specific information from the FT API response.


Okay, here's the step-by-step execution analysis formatted in Markdown:

1. **`if __name__ == "__main__":` block:** This is the entry point of the script.

2. **`initialize_rag_system()`:**
 - Logs "Initializing embeddings and LLM."
 - Calls `load_prompt()`:
 - Attempts to open `prompt.txt`. If it exists, reads the content, splits it into `system_prompt` and `question` based on the "Question: " delimiter. If the file doesn't exist or doesn't contain "Question: ", uses the `DEFAULT_QUESTION`.
 - Returns `system_prompt` and `question` to `initialize_rag_system()`.
 - Initializes `OllamaEmbeddings` and `ChatOllama` with the specified `OLLAMA_MODEL` and `system_prompt`.
 - Checks if a vector store already exists at `VECTOR_STORE_PATH`.
 - If it exists, loads the vector store.
 - If it doesn't exist, calls `load_and_process_pdf()`:
 - Attempts to load the specified PDF. If successful, splits it into chunks.
 - Creates a new FAISS vector store from the chunks and saves it to `VECTOR_STORE_PATH`.
 - Returns the chunks to `initialize_rag_system()`.
 - If the vector store is successfully loaded or created, creates the `qa_chain` (RetrievalQA) using the LLM and vector store.
 - Returns `True` if the RAG system is initialized successfully, `False` otherwise.

3. Back in the `if __name__ == "__main__":` block:
 - Checks the return value of `initialize_rag_system()`. If `True`:
 - Logs "RAG system initialized successfully."
 - Calls `load_prompt()` again (this could be optimized by storing the result from the first call).
 - Checks if a `question` was loaded. If so:
 - Logs "Processing default question: {question}".
 - Calls `process_financial_query(question)`:
 - Logs "Processing financial query: {question}".
 - Calls `query_documents(question)`:
 - Logs "Received document query: {question}".
 - If `qa_chain` is initialized, queries the vector store with the `question` and returns the answer, sources, and source type.
 - If no good answer is found, returns a message indicating that.
 - If `query_documents` returns a definitive answer from the documents, returns that result.
 - Otherwise, logs "Falling back to Financial Times API" and calls `query_ft_api()`:
 - Logs "Querying FT API for symbol: BARC:LSE".
 - Makes a request to the FT API.
 - Returns the API response.
 - Calls `extract_financial_info(ft_data)` to extract relevant information from the FT API response.
 - Constructs an answer string using the extracted information.
 - Returns the answer, sources, and source type.
 - Prints the question, answer, source type, and sources to the console.
 - If no `question` was loaded:
 - Logs an error and prints an error message.
 - If `initialize_rag_system()` returned `False`:
 - Logs an error and prints an error message.


This uses basic Markdown formatting for readability. Let me know if you'd like any specific elements emphasized or formatted differently.
