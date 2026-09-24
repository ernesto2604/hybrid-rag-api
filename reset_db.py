import asyncio
from src.core.database import engine, Base
from src.core.models import DocumentChunk  # Asegura que SQLAlchemy cargue el modelo


async def reset():
  print("Reiniciando tablas en PostgreSQL...")
  async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.drop_all)
    await conn.run_sync(Base.metadata.create_all)
  print("✅ Tablas recreadas exitosamente con la nueva dimensión de vector.")


if __name__ == "__main__":
  asyncio.run(reset())