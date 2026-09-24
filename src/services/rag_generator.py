import os
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from src.core.models import DocumentChunk


class RAGGeneratorService:

    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _build_context(
        self, chunks: List[Tuple[DocumentChunk, float]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        context_blocks = []
        sources = []

        for i, (chunk, score) in enumerate(chunks, start=1):
            page = chunk.metadata_.get("page", "N/A")
            source_name = chunk.metadata_.get("source", "desconocido")

            block = (
                f"[Fuente #{i} | Documento: {source_name} | Página: {page}]\n"
                f"{chunk.content}"
            )
            context_blocks.append(block)

            sources.append({
                "source": source_name,
                "page": page,
                "chunk_id": str(chunk.id),
                "rrf_score": score,
            })

        full_context = "\n\n---\n\n".join(context_blocks)
        return full_context, sources

    async def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not retrieved_chunks:
            return {
                "answer": (
                    "No se encontraron fragmentos relevantes en los documentos"
                    " almacenados."
                ),
                "sources": [],
            }

        context_text, sources = self._build_context(retrieved_chunks)

        if not self.client or not self.api_key:
            return {
                "answer": (
                    "[Modo Demo / Sin GEMINI_API_KEY configurada]\n"
                    f"Se recuperaron {len(retrieved_chunks)} fragmentos. Fuente"
                    f" principal: {sources[0]['source']} (Pág. {sources[0]['page']})."
                ),
                "context_used": context_text,
                "sources": sources,
            }

        system_instruction = (
            "Eres un asistente de IA corporativo y riguroso.\n"
            "Responde a la pregunta del usuario utilizando EXCLUSIVAMENTE los"
            " fragmentos provistos en el CONTEXTO.\n"
            "Toma en cuenta el historial previo para resolver dudas de seguimiento"
            " o referencias como 'él', 'ella', 'esto' o 'la máquina'.\n"
            "Reglas:\n"
            "1. No inventes información ni asumas hechos que no estén en el"
            " contexto.\n"
            "2. Si el contexto no tiene suficiente información, di: 'La"
            " documentación proporcionada no contiene información suficiente para"
            " responder a esta pregunta'.\n"
            "3. Siempre que des un dato, cita entre corchetes el documento y la"
            " página (ej: [sample.pdf, Pág. 12])."
        )

        history_prompt = ""
        if history:
            formatted_turns = []
            for msg in history[-6:]:
                role = "Usuario" if msg.get("role") == "user" else "Asistente"
                formatted_turns.append(f"{role}: {msg.get('content')}")
            history_prompt = (
                "HISTORIAL PREVIO DE CONVERSACIÓN:\n"
                + "\n".join(formatted_turns)
                + "\n\n"
            )

        user_prompt = (
            f"{history_prompt}"
            f"CONTEXTO RECUPERADO:\n{context_text}\n\n"
            f"PREGUNTA ACTUAL:\n{query}"
        )

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
            ),
        )

        return {
            "answer": response.text,
            "context_used": context_text,
            "sources": sources,
        }