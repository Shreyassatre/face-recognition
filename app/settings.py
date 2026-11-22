from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_secret: str = "unsafe-local-secret-change-me-32chars"
    app_env: str = "production"
    admin_key: str = "local-admin-key-change-me"
    default_client_key: str = "local-client-key-change-me"
    database_path: str = "data/attendance.db"
    supabase_password: str | None = Field(default=None, validation_alias="supasbase_password")
    supabase_database_url: str | None = None
    model_dir: str = "models"
    hosted_origin: str = "http://localhost:5173"
    session_ttl_seconds: int = 300
    match_threshold: float = 0.78
    match_margin: float = 0.04

    @property
    def database_url(self) -> str:
        if self.supabase_database_url:
            return self.supabase_database_url
        if self.supabase_password:
            password = quote_plus(self.supabase_password)
            return f"postgresql+psycopg://postgres:{password}@db.doauyoyfnddnlaezuevx.supabase.co:5432/postgres?sslmode=require"
        return f"sqlite:///{Path(self.database_path).as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
