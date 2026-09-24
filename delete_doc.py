import asyncio
from sqlalchemy import text
from src.core.database import AsyncSessionLocal


async def delete_document():
    async with AsyncSessionLocal() as session:
        # 1. Comprobamos los documentos existentes
        check = await session.execute(
            text(
                "SELECT DISTINCT document_id, metadata->>'source' FROM"
                " document_chunks"
            )
        )
        docs = check.fetchall()
        print("Documentos encontrados en BD:", docs)

        # 2. Eliminamos los chunks de sample3
        result = await session.execute(
            text("""
                DELETE FROM document_chunks 
                WHERE document_id ILIKE '%sample3%' 
                   OR metadata->>'source' ILIKE '%sample3%'
            """)
        )
        await session.commit()
        print(f"Eliminados {result.rowcount} chunks asociados a 'sample3'.")


if __name__ == "__main__":
    asyncio.run(delete_document())