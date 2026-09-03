"""
ED Triage Assist — RAG Pipeline Configuration
Set your API keys via environment variables or a .env file (not committed to git).
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    environment: str = "development"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000

    # Set via OPENAI_API_KEY env var or .env file
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-70b-instruct"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    groq_api_key: str = ""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "ed_triage_knowledge"
    child_chunk_size: int = 256
    child_chunk_overlap: int = 30
    parent_chunk_size: int = 1024
    parent_chunk_overlap: int = 100
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    top_k_final: int = 3
    mmr_lambda: float = 0.5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"
    documents_dir: str = "../data"
    processed_dir: str = "../data/processed"
    max_file_size_mb: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()
