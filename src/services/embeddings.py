from abc import ABC, abstractmethod
from typing import List
import numpy as np
import os
from typing import List
from google import genai
from google.genai import types
import asyncio


class BaseEmbeddingService(ABC):
  """Interfaz abstracta que define el contrato para cualquier proveedor de embeddings."""

  @abstractmethod
  async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
    """Convierte una lista de trozos de texto en una lista de vectores (Batching)."""
    pass

  @abstractmethod
  async def get_query_embedding(self, text: str) -> List[float]:
    """Genera el vector correspondiente a una consulta de búsqueda."""
    pass


class MockEmbeddingService(BaseEmbeddingService):
  """Servicio simulador determinista para desarrollo y tests locales.

  Genera vectores normalizados de 1536 dimensiones sin coste ni necesidad de API Key externa.
  """

  def __init__(self, dimension: int = 1536):
    self.dimension = dimension

  async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
    return [self._generate_pseudo_vector(t) for t in texts]

  async def get_query_embedding(self, text: str) -> List[float]:
    return self._generate_pseudo_vector(text)

  def _generate_pseudo_vector(self, text: str) -> List[float]:
    # Usamos la suma de caracteres como semilla para que el mismo texto genere siempre el mismo vector
    seed = sum(ord(c) for c in text) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(self.dimension)

    # Normalizamos el vector para que su norma L2 sea exactamente 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
      vector = vector / norm
    return vector.tolist()

class GeminiEmbeddingService:
  """Servicio de generación de vectores semánticos usando Gemini."""

  def __init__(
      self, api_key: str = None, model: str = "models/gemini-embedding-001"
  ):
    self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    self.model = model
    self.dimension = 768
    self.client = genai.Client(api_key=self.api_key)

  async def get_embedding(self, text: str) -> List[float]:
    """Genera el embedding de un texto recortado a 768 dimensiones."""
    response = await self.client.aio.models.embed_content(
        model=self.model,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=self.dimension),
    )
    return response.embeddings[0].values

  async def get_query_embedding(self, text: str) -> List[float]:
    """Alias para la búsqueda vectorial en VectorStoreService."""
    return await self.get_embedding(text)

  async def get_embeddings(
      self,
      texts: List[str],
      batch_size: int = 16,
      delay_between_batches: float = 0.5,
  ) -> List[List[float]]:
    """Genera embeddings procesando en lotes pequeños para evitar límites de tasa (429)."""
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
      batch = texts[i : i + batch_size]

      response = await self.client.aio.models.embed_content(
          model=self.model,
          contents=batch,
          config=types.EmbedContentConfig(output_dimensionality=self.dimension),
      )
      all_embeddings.extend([e.values for e in response.embeddings])

      # Pausa breve si aún quedan lotes pendientes por enviar
      if i + batch_size < len(texts):
        await asyncio.sleep(delay_between_batches)

    return all_embeddings