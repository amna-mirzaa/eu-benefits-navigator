"""Central configuration, loaded from environment variables / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SCHEMES_DIR = DATA_DIR / "schemes"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.3-70b-versatile")
GROQ_REASONING_MODEL = os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "eu-benefits")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))


def has_groq_key() -> bool:
    return bool(GROQ_API_KEY)


def has_pinecone_key() -> bool:
    return bool(PINECONE_API_KEY)
