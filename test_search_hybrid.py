import asyncio
from src.core.database import AsyncSessionLocal, engine
from src.services.embeddings import MockEmbeddingService
from src.services.vector_store import VectorStoreService


async def test():
  print("🔥 Probando Búsqueda Híbrida (Vectorial + Full-Text + RRF)...\n")

  embedding_service = MockEmbeddingService(dimension=1536)
  vector_store = VectorStoreService(embedding_service=embedding_service)

  async with AsyncSessionLocal() as session:
    # Consulta combinada: mezcla semántica con palabras clave exactas
    query = "FastAPI Starlette pgvector arquitectura RAG"

    resultados = await vector_store.search_hybrid(
        session=session, query=query, limit=3
    )

    print(f"Consulta híbrida: '{query}'")
    print(f"Total devuelto: {len(resultados)}\n")

    for i, (chunk, rrf_score) in enumerate(resultados):
      print(f"=== [Top {i+1}] Puntuación RRF: {rrf_score:.5f} ===")
      print(f"Doc ID: {chunk.document_id} | Chunk #{chunk.chunk_index}")
      print(f"Contenido: {chunk.content[:120]}...\n")

  await engine.dispose()


if __name__ == "__main__":
  asyncio.run(test())