import uuid
from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Index, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from src.core.config import settings
from src.core.database import Base


class DocumentChunk(Base):
  __tablename__ = "document_chunks"

  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  document_id = Column(String, nullable=False, index=True)
  chunk_index = Column(Integer, nullable=False)
  content = Column(String, nullable=False)
  # 'metadata' es el nombre en PostgreSQL, 'metadata_' es el nombre en el modelo Python
  metadata_ = Column("metadata", JSONB, nullable=False, default=dict)
  embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)
  created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

  __table_args__ = (
      Index(
          "idx_chunks_embedding_hnsw",
          embedding,
          postgresql_using="hnsw",
          postgresql_with={"m": 16, "ef_construction": 64},
          postgresql_ops={"embedding": "vector_cosine_ops"},
      ),
      Index(
          "idx_chunks_content_tsv",
          func.to_tsvector(text("'spanish'"), content),
          postgresql_using="gin",
      ),
  )