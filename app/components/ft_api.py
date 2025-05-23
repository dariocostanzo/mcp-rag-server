import requests
import logging
from .config import FT_API_URL, FT_COOKIE

# Step 2 (partial): Use logging from main.py
# - Reuses the logger configured in main.py for consistency
logger = logging.getLogger(__name__)

# Step 5: Define function to fetch data from Financial Times API
# - Queries the FT API for financial metrics (e.g., for Barclays) using a provided symbol


def fetch_from_ft_api(symbol="BARC:LSE"):
    logger.info(f"Querying FT API for symbol: {symbol}")
    if not FT_COOKIE:
        logger.warning("FT_COOKIE not found in environment variables")
        return {"error": "FT_COOKIE not configured"}
    try:
        params = {"symbols": symbol, "_": "1747839845971"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://markets.ft.com/research/webservices/securities/v1/docs"
        }
        cookie_dict = {cookie.split('=')[0]: cookie.split('=')[1]
                       for cookie in FT_COOKIE.strip().split('; ') if '=' in cookie}
        response = requests.get(FT_API_URL, params=params,
                                headers=headers, cookies=cookie_dict)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully retrieved data from FT API for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data from FT API: {str(e)}")
        return {"error": f"Failed to retrieve data: {str(e)}"}

# Step 8: Extract financial information from FT API response
# - Parses the FT API response to extract key financial metrics


def extract_financial_info(api_response):
    if "error" in api_response:
        return None
    try:
        if not api_response.get("items") or len(api_response["items"]) == 0:
            return None
        item = api_response["items"][0]
        company_name = item.get("basic", {}).get("name", "")
        financial_data = {
            "company": company_name,
            "symbol": item.get("basic", {}).get("symbol", ""),
            "metrics": {}
        }
        for metric in item.get("items", []):
            name = metric.get("name", "")
            value = metric.get("value", "")
            period = metric.get("period", "")
            financial_data["metrics"][name] = {
                "value": value, "period": period}
        return financial_data
    except Exception as e:
        logger.error(f"Error extracting financial information: {str(e)}")
        return None
