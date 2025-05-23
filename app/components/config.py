import os
from pathlib import Path
from dotenv import load_dotenv

# Step 3: Configuration class
# - Loads environment variables and defines paths, model, and default question


class Config:
    def __init__(self):
        load_dotenv()
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        self.VECTOR_STORE_PATH = os.path.join(
            os.path.dirname(__file__), "..", "vector_store")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
        self.DEFAULT_QUESTION = "What is the total assets value of Barclays PLC in 2020?"
        self.FT_API_URL = "https://markets.ft.com/research/webservices/securities/v1/performance-metrics"
        self.FT_COOKIE = os.getenv("FT_COOKIE", "")
