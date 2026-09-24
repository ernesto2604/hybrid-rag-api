import asyncio
from sqlalchemy import func, select
from src.core.database import AsyncSessionLocal, engine
from src.core.models import DocumentChunk
from src.services.embeddings import MockEmbeddingService
from src.services.vector_store import VectorStoreService

sample_document = """FastAPI es un framework web moderno y rápido para construir APIs con Python 3.8+ basado en anotaciones de tipo estándar. Su rendimiento es comparable con NodeJS y Go gracias a Starlette y Pydantic.

pgvector es una extensión de código abierto para PostgreSQL que permite almacenar vectores y realizar búsquedas de similitud por vecinos más cercanos (KNN). Soporta distancias L2, producto interno y similitud coseno.

En una arquitectura RAG, el contexto recuperado de pgvector se combina con el prompt del usuario para que el modelo genere respuestas precisas sin alucinaciones."""


async def test():
  print("🚀 Iniciando prueba de ingesta completa...")

  # Inicializamos servicios
  embedding_service = MockEmbeddingService(dimension=1536)
  vector_store = VectorStoreService(embedding_service=embedding_service)

  async with AsyncSessionLocal() as session:
    # 1. Ingestar el documento
    doc_id = "doc_tecnico_001"
    meta = {"source": "manual_arquitectura.pdf", "author": "dev_team"}

    total_guardados = await vector_store.ingest_document(
        session=session,
        document_id=doc_id,
        text=sample_document,
        metadata=meta,
    )
    print(
        f"✅ Ingesta finalizada: {total_guardados} chunks almacenados con éxito."
    )

    # 2. Consultar la base de datos para verificar que realmente están ahí
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
    )
    resultado = await session.execute(stmt)
    chunks_recuperados = resultado.scalars().all()

    print("\n📦 Verificando registros en PostgreSQL:")
    for ch in chunks_recuperados:
      print(
          f" - [ID: {ch.id}] Chunk #{ch.chunk_index} | Longitud vector:"
          f" {len(ch.embedding)} | Metadata: {ch.metadata_}"
      )
      print(f"   Texto: {ch.content[:80]}...\n")

  await engine.dispose()
  print("🔒 Conexión con PostgreSQL cerrada limpiamente.")


if __name__ == "__main__":
  asyncio.run(test())