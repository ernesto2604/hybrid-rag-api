import asyncio
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import DocumentChunk
from src.services.chunking import RecursiveChunker
from src.services.embeddings import BaseEmbeddingService


class VectorStoreService:

    def __init__(
        self,
        embedding_service: BaseEmbeddingService,
        chunker: RecursiveChunker = None,
    ):
        self.embedding_service = embedding_service
        self.chunker = chunker or RecursiveChunker()

    async def ingest_document(
        self,
        session: AsyncSession,
        document_id: str,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> int:
        metadata = metadata or {}
        chunks_text = self.chunker.split_text(text)
        if not chunks_text:
            return 0

        embeddings = await self.embedding_service.get_embeddings(chunks_text)

        db_chunks: List[DocumentChunk] = []
        for idx, (content, vector) in enumerate(zip(chunks_text, embeddings)):
            chunk_record = DocumentChunk(
                document_id=document_id,
                chunk_index=idx,
                content=content,
                metadata_=metadata,
                embedding=vector,
            )
            db_chunks.append(chunk_record)

        session.add_all(db_chunks)
        await session.commit()
        return len(db_chunks)

    async def search_vectorial(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        query_vector = await self.embedding_service.get_query_embedding(query)
        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label(
            "distance"
        )

        stmt = select(DocumentChunk, distance_col)
        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        stmt = stmt.order_by(distance_col.asc()).limit(limit)

        result = await session.execute(stmt)
        return [(row[0], float(row[1])) for row in result.all()]

    async def search_fulltext(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
        language: str = "spanish",
        document_id: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        ts_vector = func.to_tsvector(language, DocumentChunk.content)
        ts_query = func.websearch_to_tsquery(language, query)
        rank_col = func.ts_rank(ts_vector, ts_query).label("rank")

        stmt = select(DocumentChunk, rank_col).where(ts_vector.op("@@")(ts_query))
        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        stmt = stmt.order_by(desc(rank_col)).limit(limit)

        result = await session.execute(stmt)
        return [(row[0], float(row[1])) for row in result.all()]

    async def search_hybrid(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 3,
        rrf_k: int = 60,
        document_id: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Ejecuta búsqueda híbrida combinando Vectorial y Full-Text mediante Reciprocal Rank Fusion (RRF)."""
        candidate_limit = limit * 2

        vector_results = await self.search_vectorial(
            session, query, limit=candidate_limit, document_id=document_id
        )
        fulltext_results = await self.search_fulltext(
            session, query, limit=candidate_limit, document_id=document_id
        )

        scores: Dict[UUID, float] = {}
        chunks_map: Dict[UUID, DocumentChunk] = {}

        for rank, (chunk, _) in enumerate(vector_results, start=1):
            chunks_map[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (chunk, _) in enumerate(fulltext_results, start=1):
            chunks_map[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + (1.0 / (rrf_k + rank))

        sorted_chunks = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )

        return [
            (chunks_map[chunk_id], score)
            for chunk_id, score in sorted_chunks[:limit]
        ]

    async def ingest_pdf(
        self,
        session: AsyncSession,
        document_id: str,
        file_bytes: bytes,
        filename: str = "document.pdf",
    ) -> int:
        from src.services.pdf_loader import PDFLoaderService

        pages_data = PDFLoaderService.extract_text_from_bytes(
            file_bytes, filename
        )
        if not pages_data:
            return 0

        total_chunks = 0
        for page_info in pages_data:
            chunks_count = await self.ingest_document(
                session=session,
                document_id=document_id,
                text=page_info["text"],
                metadata=page_info["metadata"],
            )
            total_chunks += chunks_count

        return total_chunks