import asyncio
from pathlib import Path
from src.core.database import AsyncSessionLocal, engine
from src.services.embeddings import MockEmbeddingService
from src.services.vector_store import VectorStoreService


async def main():
  pdf_path = Path("sample.pdf")
  if not pdf_path.exists():
    print(
        "⚠️ No se encontró 'sample.pdf'. Coloca el archivo en la raíz del"
        " proyecto."
    )
    return

  print(f"📄 Abriendo archivo real: {pdf_path.name}...")
  file_bytes = pdf_path.read_bytes()

  embedding_service = MockEmbeddingService(dimension=1536)
  vector_store = VectorStoreService(embedding_service=embedding_service)

  async with AsyncSessionLocal() as session:
    print("⏳ Extrayendo texto, troceando e ingestando en PostgreSQL...")
    total_chunks = await vector_store.ingest_pdf(
        session=session,
        document_id="paper_guerra_fria_01",
        file_bytes=file_bytes,
        filename=pdf_path.name,
    )
    print(
        f"✅ Ingesta finalizada: {total_chunks} chunks procesados y guardados.\n"
    )

    print("=" * 60)
    print("💬 MODO CONSULTA INTERACTIVA (Escribe 'salir' para terminar)")
    print("=" * 60)

    while True:
      query = input("\n🔎 Introduce tu pregunta o términos de búsqueda: ").strip()

      if not query:
        continue

      if query.lower() in ["salir", "exit", "quit"]:
        print("\n👋 Cerrando sesión de consulta.")
        break

      resultados = await vector_store.search_hybrid(
          session=session, query=query, limit=3
      )

      if not resultados:
        print("❌ No se encontraron fragmentos relevantes.")
        continue

      print(f"\n📊 Resultados encontrados para: '{query}'")
      for i, (chunk, score) in enumerate(resultados):
        origen = chunk.metadata_.get("source", "N/A")
        pagina = chunk.metadata_.get("page", "N/A")
        print(f"\n--- [Top {i+1}] Score RRF: {score:.5f} | Pág. {pagina} ({origen}) ---")
        print(chunk.content)

  await engine.dispose()
  print("\n🔒 Conexión con PostgreSQL cerrada.")


if __name__ == "__main__":
  asyncio.run(main())