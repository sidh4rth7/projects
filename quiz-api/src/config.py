from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    """Configuration settings for the FastAPI application."""
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "true").lower() in ("true", "1", "t")
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "127.0.0.1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your .env file
    COMPLEXITY_OPTIONS = os.getenv("COMPLEXITY_OPTIONS", "Easy,Medium,Hard").split(",")