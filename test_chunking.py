from src.services.chunking import RecursiveChunker

# Texto de prueba con varios párrafos
sample_text = """La arquitectura de Microservicios permite desacoplar componentes grandes en servicios pequeños e independientes. Cada servicio suele tener su propia base de datos y se comunica mediante APIs REST o colas de eventos como Kafka o RabbitMQ.

Por otro lado, los sistemas RAG (Retrieval-Augmented Generation) combinan bases de datos vectoriales con modelos de lenguaje. En vez de reentrenar un LLM, se busca la información relevante en tiempo real en una base de datos como PostgreSQL con pgvector y se le inyecta en el prompt.

Esta técnica reduce drásticamente las alucinaciones del modelo y permite a las empresas consultar sus propios documentos privados de forma segura y actualizada."""

# Configuramos chunks pequeños a propósito para ver cómo actúa el corte y el overlap
chunker = RecursiveChunker(chunk_size=220, chunk_overlap=40)
chunks = chunker.split_text(sample_text)

print(f"📊 Total de chunks generados: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
  print(f"--- CHUNK {i} (Longitud: {len(chunk)}) ---")
  print(chunk)
  print()