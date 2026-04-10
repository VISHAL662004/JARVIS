from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Legal Document AI"
    app_env: str = "dev"
    api_prefix: str = "/api/v1"
    max_upload_size_mb: int = 25
    allowed_extensions: tuple[str, ...] = (".pdf",)
    groq_api_key: str | None = None
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_sec: float = 45.0
    groq_context_chars: int = 12000
    legal_disclaimer: str = (
        "This system provides automated analysis support and is not legal advice. "
        "Consult a licensed legal professional before making legal decisions."
    )

    base_dir: Path = Path(__file__).resolve().parents[2]
    data_raw_dir: Path = base_dir / "data" / "raw_judgments"
    data_processed_dir: Path = base_dir / "data" / "processed"
    checkpoints_dir: Path = base_dir / "models" / "checkpoints"
    cache_dir: Path = base_dir / ".cache"

    retrieval_top_k: int = 5

    retrieval_matrix_path: Path = data_processed_dir / "tfidf_matrix.npz"
    retrieval_vectorizer_path: Path = data_processed_dir / "tfidf_vectorizer.joblib"
    corpus_meta_path: Path = data_processed_dir / "tfidf_corpus.json"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        env_prefix="LEGAL_DOC_",
        extra="ignore",
    )


settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
