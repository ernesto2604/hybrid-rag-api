import asyncio
import asyncpg
from pgvector.asyncpg import register_vector

# Cadena de conexión: postgresql://usuario:contraseña@servidor:puerto/base_de_datos
DATABASE_URL = "postgresql://rag_user:rag_password@localhost:5432/rag_db"


async def main():
  print("🔌 Conectando a PostgreSQL...")
  conn = await asyncpg.connect(DATABASE_URL)

  # 1. Habilitamos la extensión de vectores en PostgreSQL (solo se hace una vez)
  await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

  # 2. Le enseñamos al driver asyncpg a interpretar el tipo de dato VECTOR
  await register_vector(conn)

  # 3. Creamos una tabla temporal de prueba con vectores de dimensión 3
  await conn.execute("""
        CREATE TABLE IF NOT EXISTS test_items (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding VECTOR(3)
        );
    """)

  # Limpiamos datos de pruebas anteriores si los hubiera
  await conn.execute("TRUNCATE test_items;")

  # 4. Insertamos dos filas con vectores opuestos:
  # 'Texto de tecnologia' apunta hacia el eje X: [1.0, 0.0, 0.0]
  # 'Texto de cocina' apunta hacia el eje Y: [0.0, 1.0, 0.0]
  print("📝 Insertando datos de prueba...")
  await conn.execute(
      "INSERT INTO test_items (content, embedding) VALUES ($1, $2)",
      "Texto de tecnologia",
      [1.0, 0.0, 0.0],
  )
  await conn.execute(
      "INSERT INTO test_items (content, embedding) VALUES ($1, $2)",
      "Texto de cocina",
      [0.0, 1.0, 0.0],
  )

  # 5. Simulamos una consulta de búsqueda:
  # Preguntamos con un vector que apunta casi todo al eje X: [0.9, 0.1, 0.0]
  query_vector = [0.9, 0.1, 0.0]
  print(f"🔍 Buscando el vector más cercano a {query_vector}...")

  row = await conn.fetchrow(
      """
        SELECT content, embedding <=> $1 AS distance
        FROM test_items
        ORDER BY distance ASC
        LIMIT 1;
    """,
      query_vector,
  )

  print(
      f"✅ Éxito! El resultado más cercano es: '{row['content']}' con distancia {row['distance']:.4f}"
  )
  await conn.close()


if __name__ == "__main__":
  asyncio.run(main())