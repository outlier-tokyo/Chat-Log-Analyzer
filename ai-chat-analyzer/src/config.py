import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATA_PATH_RAW = "data/raw"
    DATA_PATH_PROCESSED = "data/processed"
    MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens" # Example for Japanese SBERT