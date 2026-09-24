import asyncio
from src.core.database import AsyncSessionLocal, engine
from src.services.embeddings import MockEmbeddingService
from src.services.vector_store import VectorStoreService


async def test():
  print("🔎 Probando búsqueda léxica (Full-Text Search) en PostgreSQL...\n")

  embedding_service = MockEmbeddingService(dimension=1536)
  vector_store = VectorStoreService(embedding_service=embedding_service)

  async with AsyncSessionLocal() as session:
    # Probamos buscando palabras clave exactas que existen en el chunk ingresado
    query = "FastAPI Starlette Pydantic"

    resultados = await vector_store.search_fulltext(
        session=session, query=query, limit=2
    )

    print(f"Palabras buscadas: '{query}'")
    print(f"Resultados encontrados: {len(resultados)}\n")

    for i, (chunk, rank) in enumerate(resultados):
      print(f"--- [Top {i+1}] Relevancia Léxica (ts_rank): {rank:.5f} ---")
      print(f"Doc ID: {chunk.document_id} | Chunk #{chunk.chunk_index}")
      print(f"Texto: {chunk.content[:100]}...\n")

  await engine.dispose()


if __name__ == "__main__":
  asyncio.run(test())