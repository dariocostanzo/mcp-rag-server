import logging
from fastmcp import FastMCP
from .vector_store import VectorStore
from .ft_api import FTAPI
from .config import Config

# Step 1: Initialize FastMCP application
# - Creates a FastMCP instance to organize tools for querying PDFs and the Financial Times API
mcp_app = FastMCP("PDF RAG App")

# Step 2 (partial): Use logging from main.py
# - Reuses the logger configured in main.py for consistency
logger = logging.getLogger(__name__)

# Step 9: FastMCP server class
# - Defines tools for querying documents, FT API, and health checks


class MCPServer:
    @staticmethod
    @mcp_app.tool()
    def query_documents(question=Config().DEFAULT_QUESTION):
        # Step 9: Query documents tool
        # - Queries the PDF documents using the RAG system, falls back to FT API if needed
        logger.info(f"Received query: {question}")
        if VectorStore.qa_chain:
            try:
                result = VectorStore.qa_chain.invoke({"query": question})
                answer = result["result"]
                sources = [{"page": doc.metadata.get("page", "Unknown"),
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
            config = Config()
            symbol = "BARC:LSE"
            ft_data = FTAPI.fetch_from_ft_api(
                symbol, config.FT_API_URL, config.FT_COOKIE)
            financial_info = FTAPI.extract_financial_info(ft_data)
            if financial_info:
                company = financial_info["company"]
                metrics_str = [f"{name}: {details['value']} ({details['period']})"
                               for name, details in financial_info["metrics"].items()]
                answer = f"According to Financial Times data for {company} ({financial_info['symbol']}):\n"
                answer += "\n".join(metrics_str)
                logger.info(
                    f"Found answer in Financial Times API for {company}")
                return {
                    "answer": answer,
                    "sources": [{"source": "Financial Times API", "symbol": financial_info["symbol"]}],
                    "source_type": "financial_times"
                }
        except Exception as e:
            logger.error(f"Error processing FT API query: {str(e)}")
        return {
            "answer": "Could not find information in documents or financial data sources.",
            "sources": [],
            "source_type": "none"
        }

    @staticmethod
    @mcp_app.tool()
    def query_ft_api(symbol="BARC:LSE"):
        # Step 9 (continued): Direct FT API query tool
        # - Directly queries the Financial Times API for financial metrics
        logger.info(f"Direct query to FT API for symbol: {symbol}")
        config = Config()
        if not config.FT_COOKIE:
            return {"error": "FT API cookie not configured in .env file"}
        try:
            data = FTAPI.fetch_from_ft_api(
                symbol, config.FT_API_URL, config.FT_COOKIE)
            financial_info = FTAPI.extract_financial_info(data)
            if financial_info:
                return {
                    "company": financial_info["company"],
                    "symbol": financial_info["symbol"],
                    "metrics": financial_info["metrics"]
                }
            else:
                return {"error": "No data found for the symbol"}
        except Exception as e:
            logger.error(f"Error in direct FT API query: {str(e)}")
            return {"error": f"Failed to query FT API: {str(e)}"}

    @staticmethod
    @mcp_app.tool()
    def health_check():
        # Step 9 (continued): Health check tool
        # - Returns the health status of the RAG application and FT API connection
        config = Config()
        ft_api_status = "configured" if config.FT_COOKIE else "not_configured"
        if config.FT_COOKIE:
            try:
                test_response = FTAPI.fetch_from_ft_api(
                    ft_api_url=config.FT_API_URL, ft_cookie=config.FT_COOKIE)
                if "error" not in test_response:
                    ft_api_status = "connected"
                else:
                    ft_api_status = f"error: {test_response['error']}"
            except Exception as e:
                ft_api_status = f"error: {str(e)}"
        return {
            "status": "healthy" if VectorStore.qa_chain else "not_ready",
            "vector_store": "loaded" if VectorStore.vector_store else "not_loaded",
            "model": config.OLLAMA_MODEL,
            "ft_api": ft_api_status
        }
