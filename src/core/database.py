from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from src.core.config import settings

# 1. El Engine: El motor asíncrono que administra el pool de conexiones
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=(
        settings.ENVIRONMENT == "development"
    ),  # Muestra las consultas SQL en consola en desarrollo
    future=True,
    pool_size=10,  # Conexiones fijas abiertas en espera
    max_overflow=20,  # Conexiones extra que puede abrir bajo picos de tráfico
)

# 2. La fábrica de sesiones: Genera sesiones asíncronas independientes para cada petición
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# 3. Base declarativa: De aquí heredarán todas nuestras tablas/modelos de datos
Base = declarative_base()


# 4. Generador de sesiones para FastAPI (Dependency Injection)
async def get_db() -> AsyncGenerator[AsyncSession, None]:
  """Entrega una sesión de base de datos a un endpoint y garantiza su cierre al terminar."""
  async with AsyncSessionLocal() as session:
    try:
      yield session
    finally:
      await session.close()