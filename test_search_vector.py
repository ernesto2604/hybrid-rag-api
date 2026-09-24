import asyncio
from src.core.database import AsyncSessionLocal, engine
from src.services.embeddings import MockEmbeddingService
from src.services.vector_store import VectorStoreService


async def test():
  print("🔎 Probando búsqueda vectorial en PostgreSQL...\n")

  embedding_service = MockEmbeddingService(dimension=1536)
  vector_store = VectorStoreService(embedding_service=embedding_service)

  async with AsyncSessionLocal() as session:
    # Hacemos una pregunta de prueba
    query = "FastAPI es un framework web moderno y rápido para construir APIs con Python 3.8+ basado en anotaciones de tipo estándar."

    resultados = await vector_store.search_vectorial(
        session=session, query=query, limit=2
    )

    print(f"Pregunta: '{query}'\n")
    print(f"Resultados encontrados: {len(resultados)}\n")

    for i, (chunk, distancia) in enumerate(resultados):
      print(f"--- [Top {i+1}] Distancia Coseno: {distancia:.5f} ---")
      print(f"Doc ID: {chunk.document_id} | Chunk #{chunk.chunk_index}")
      print(f"Texto: {chunk.content}\n")

  await engine.dispose()


if __name__ == "__main__":
  asyncio.run(test())