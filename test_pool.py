import asyncio
from sqlalchemy import text
from src.core.database import engine


async def main():
  print("🔄 Probando conexión a través del Pool de SQLAlchemy...")

  # 'async with' toma prestada la conexión del pool y la devuelve sola al salir
  async with engine.connect() as conn:
    resultado = await conn.execute(text("SELECT 1"))
    valor = resultado.scalar()
    print(f"✅ Respuesta de PostgreSQL: {valor}")

  # Apagamos el motor y cerramos todas las conexiones del pool limpiamente
  await engine.dispose()
  print("🔒 Conexiones cerradas limpiamente.")


if __name__ == "__main__":
  asyncio.run(main())