import uuid
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    IngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SearchRequest,
    SearchResponse,
)
from src.core.database import AsyncSessionLocal
from src.core.models import DocumentChunk
from src.services.embeddings import GeminiEmbeddingService
from src.services.rag_generator import RAGGeneratorService
from src.services.vector_store import VectorStoreService

load_dotenv()

app = FastAPI(
    title="Hybrid RAG Engine API",
    version="1.0.0",
    description=(
        "Motor de Búsqueda Híbrida y Generación RAG de alta precisión con citas"
        " verificables."
    ),
)

# Montar carpeta de estáticos para frontend
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Inicialización de servicios
embedding_service = GeminiEmbeddingService()
vector_store = VectorStoreService(embedding_service=embedding_service)
rag_generator = RAGGeneratorService()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Inyector de dependencias para sesión de base de datos."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@app.get("/", include_in_schema=False)
async def serve_index():
    """Ruta raíz para servir la UI web."""
    index_file = Path("static/index.html")
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró static/index.html.",
        )
    return FileResponse(index_file)


@app.get(
    "/api/v1/documents",
    summary="Listar documentos indexados en PostgreSQL",
)
async def list_documents(session: AsyncSession = Depends(get_db_session)):
    """Devuelve los documentos únicos persistidos en PostgreSQL y el total de chunks."""
    try:
        stmt = select(DocumentChunk.document_id, DocumentChunk.metadata_)
        result = await session.execute(stmt)
        rows = result.all()

        docs_map = {}
        for doc_id, meta in rows:
            if doc_id not in docs_map:
                filename = (meta or {}).get("source", "document.pdf")
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": filename,
                    "total_chunks": 0,
                }
            docs_map[doc_id]["total_chunks"] += 1

        return {"documents": list(docs_map.values())}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo documentos: {str(e)}",
        )


@app.post(
    "/api/v1/documents/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subir e ingestar documento PDF",
)
async def upload_pdf(
    file: UploadFile,
    session: AsyncSession = Depends(get_db_session),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se admiten documentos en formato PDF (.pdf).",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo proporcionado está vacío.",
        )

    document_id = f"doc_{uuid.uuid4().hex[:8]}"

    try:
        total_chunks = await vector_store.ingest_pdf(
            session=session,
            document_id=document_id,
            file_bytes=file_bytes,
            filename=file.filename,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento del documento: {str(e)}",
        )

    return IngestResponse(
        document_id=document_id,
        filename=file.filename,
        chunks_stored=total_chunks,
        message="Documento procesado, troceado y vectorizado exitosamente.",
    )


@app.post(
    "/api/v1/search/hybrid",
    response_model=SearchResponse,
    summary="Búsqueda Híbrida (Vectorial + Full-Text + RRF)",
)
async def hybrid_search(
    body: SearchRequest,
    session: AsyncSession = Depends(get_db_session),
):
    try:
        raw_results = await vector_store.search_hybrid(
            session=session,
            query=body.query,
            limit=body.limit,
            document_id=getattr(body, "document_id", None),
        )

        formatted_results = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "metadata": chunk.metadata_,
                "rrf_score": score,
            }
            for chunk, score in raw_results
        ]

        return SearchResponse(
            query=body.query,
            total_results=len(formatted_results),
            results=formatted_results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ejecutando la búsqueda: {str(e)}",
        )


@app.post(
    "/api/v1/rag/ask",
    response_model=RAGQueryResponse,
    summary="Pregunta RAG de extremo a extremo (Retrieval + LLM Generation con Memoria)",
)
async def ask_rag(
    body: RAGQueryRequest,
    session: AsyncSession = Depends(get_db_session),
):
    try:
        # 1. Recuperación híbrida en PostgreSQL (filtrada si se provee document_id)
        retrieved_chunks = await vector_store.search_hybrid(
            session=session,
            query=body.query,
            limit=body.limit,
            document_id=body.document_id,
        )

        # 2. Conversión del historial tipado a diccionarios para el generador
        history_dicts = (
            [msg.model_dump() for msg in body.history] if body.history else []
        )

        # 3. Síntesis generativa con Gemini condicionada al contexto y memoria previa
        result = await rag_generator.generate_answer(
            query=body.query,
            retrieved_chunks=retrieved_chunks,
            history=history_dicts,
        )

        return RAGQueryResponse(
            query=body.query,
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la consulta RAG: {str(e)}",
        )