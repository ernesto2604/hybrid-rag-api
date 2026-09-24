import asyncio
from sqlalchemy import text
from src.core.database import Base, engine
import src.core.models  # Importamos los modelos para que Base los reconozca


async def init_tables():
  print("🛠️ Inicializando base de datos...")
  async with engine.begin() as conn:
    # 1. Aseguramos que la extensión vector existe
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    # 2. Crea las tablas definidas en Base si no existen
    await conn.run_sync(Base.metadata.create_all)
  print("✅ Tablas creadas correctamente en PostgreSQL.")
  await engine.dispose()


if __name__ == "__main__":
  asyncio.run(init_tables())