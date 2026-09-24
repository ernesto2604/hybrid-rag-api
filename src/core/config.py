from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  # Le decimos que lea automáticamente el archivo .env
  model_config = SettingsConfigDict(
      env_file=".env", env_file_encoding="utf-8", extra="ignore"
  )

  # Variables con tipado estricto
  PROJECT_NAME: str = "Hybrid RAG API"
  ENVIRONMENT: str = "development"

  # Cadena de conexión asíncrona requerida
  DATABASE_URL: str = Field(..., description="PostgreSQL async connection string")

  # Dimensión del vector para pgvector
  EMBEDDING_DIMENSION: int = 768


# Creamos una única instancia global para usar en todo el proyecto
settings = Settings()