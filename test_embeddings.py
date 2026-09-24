import asyncio
from src.services.embeddings import MockEmbeddingService


async def test():
  service = MockEmbeddingService(dimension=1536)

  frases = [
      "La arquitectura de Microservicios desacopla el backend.",
      "PostgreSQL con pgvector almacena vectores de alta dimensión.",
  ]

  print("🔄 Generando embeddings...")
  vectores = await service.get_embeddings(frases)

  print(f"📊 Cantidad de vectores devueltos: {len(vectores)}")
  print(f"📏 Dimensión del vector 1: {len(vectores[0])}")
  print(f"📏 Dimensión del vector 2: {len(vectores[1])}")

  # Comprobamos que es determinista (mismo texto -> mismo vector)
  vector_repetido = await service.get_query_embedding(frases[0])
  es_identico = vectores[0] == vector_repetido
  print(f"🎯 ¿Es determinista y reproducible?: {es_identico}")


if __name__ == "__main__":
  asyncio.run(test())