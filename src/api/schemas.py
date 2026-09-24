from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


# ==========================================
# Modelos de Ingesta de Documentos
# ==========================================

class IngestResponse(BaseModel):
    """Respuesta tras procesar y almacenar un PDF en la base de datos."""
    document_id: str = Field(..., description="Identificador único asignado al documento.")
    filename: str = Field(..., description="Nombre del archivo original procesado.")
    chunks_stored: int = Field(..., description="Cantidad total de fragmentos vectorizados y guardados.")
    message: str = Field(..., description="Mensaje de estado de la operación.")


# ==========================================
# Modelos de Búsqueda Híbrida Pura
# ==========================================

class SearchRequest(BaseModel):
    """Petición para el endpoint de búsqueda híbrida directa."""
    query: str = Field(..., min_length=2, description="Texto o consulta a buscar.")
    limit: int = Field(default=5, ge=1, le=20, description="Cantidad máxima de fragmentos a recuperar.")
    document_id: Optional[str] = Field(
        default=None,
        description="ID del documento para acotar la búsqueda (opcional)."
    )


class ChunkResult(BaseModel):
    """Estructura de un fragmento recuperado de PostgreSQL."""
    id: UUID
    document_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    rrf_score: float


class SearchResponse(BaseModel):
    """Respuesta del endpoint de búsqueda híbrida."""
    query: str
    total_results: int
    results: List[ChunkResult]


# ==========================================
# Modelos de Generación RAG y Memoria
# ==========================================

class ChatMessage(BaseModel):
    """Turno individual de conversación para memoria conversacional."""
    role: str = Field(
        ..., 
        description="Rol del emisor: 'user' o 'assistant'."
    )
    content: str = Field(
        ..., 
        description="Texto del mensaje intercambiado."
    )


class RAGQueryRequest(BaseModel):
    """Petición para el endpoint generativo RAG con contexto y memoria."""
    query: str = Field(
        ...,
        min_length=2,
        description="Pregunta actual en lenguaje natural.",
    )
    limit: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Número de chunks de contexto que se le enviarán al LLM.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="ID del documento para acotar la búsqueda (opcional, null para global).",
    )
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Historial previo de la conversación en la sesión actual.",
    )


class SourceCitation(BaseModel):
    """Metadatos de la fuente y fragmento utilizado para fundamentar la respuesta."""
    source: str
    page: Any
    chunk_id: str
    rrf_score: float


class RAGQueryResponse(BaseModel):
    """Respuesta sintetizada por el LLM con citas verificables."""
    query: str
    answer: str
    sources: List[SourceCitation]