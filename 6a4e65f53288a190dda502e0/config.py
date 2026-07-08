from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


def _resolve_path(env_name: str, default: str) -> Path:
    raw_value = os.getenv(env_name, default)
    path = Path(raw_value)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def require_env(env_name: str) -> str:
    value = os.getenv(env_name, "").strip()
    if not value or value.startswith("your_"):
        raise ValueError(
            f"Missing required environment variable: {env_name}. "
            "Update your .env file before running the pipeline."
        )
    return value


PROJECT_ROOT = _resolve_path("PROJECT_ROOT", ".")
DATA_DIR = _resolve_path("DATA_DIR", "./data")
CHROMA_DB_DIR = _resolve_path("CHROMA_DB_DIR", "./chroma_db")
REPORT_PATH = _resolve_path("REPORT_PATH", "./report.md")

LLAMAPARSE_API_KEY = require_env("LLAMAPARSE_API_KEY")
GEMINI_API_KEY = require_env("GEMINI_API_KEY")

LLM = Gemini(
    model="models/gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
)

EMBED_MODEL = GeminiEmbedding(
    model_name="models/text-embedding-004",
    api_key=GEMINI_API_KEY,
)

Settings.llm = LLM
Settings.embed_model = EMBED_MODEL
