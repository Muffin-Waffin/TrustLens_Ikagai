from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Trustlens Backend"
    APP_VERSION: str = "0.1.0"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite:///./data/trustlens.db"

    # Directories
    DATA_DIR: Path = Path("./data")
    UPLOAD_DIR: Path = Path("./data/uploads")
    PROCESSING_DIR: Path = Path("./data/processing")
    EVIDENCE_DIR: Path = Path("./data/evidence")
    REPORT_DIR: Path = Path("./data/reports")

    # Upload limits
    MAX_UPLOAD_MB: int = 250

    # OpenRouter API
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "google/gemini-2.0-flash-001"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Frontend URLs
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        directories = [
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.PROCESSING_DIR,
            self.EVIDENCE_DIR,
            self.REPORT_DIR,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


settings = Settings()