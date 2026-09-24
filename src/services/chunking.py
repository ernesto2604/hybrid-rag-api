from typing import List


class RecursiveChunker:

  def __init__(
      self,
      chunk_size: int = 800,
      chunk_overlap: int = 150,
      separators: List[str] = None,
  ):
    """Inicializa el divisor de texto recursivo.

    :param chunk_size: Tamaño máximo objetivo de cada trozo en caracteres.
    :param chunk_overlap: Caracteres que se repiten entre trozos consecutivos
    para mantener contexto.
    :param separators: Lista ordenada de separadores prioritarios.
    """
    if chunk_overlap >= chunk_size:
      raise ValueError("El overlap no puede ser mayor o igual al chunk_size.")

    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

  def split_text(self, text: str) -> List[str]:
    """Punto de entrada: divide el texto respetando los límites naturales."""
    text = text.strip()
    if not text:
      return []

    return self._split(text, self.separators)

  def _split(self, text: str, separators: List[str]) -> List[str]:
    if len(text) <= self.chunk_size:
      return [text]

    # Determinamos el mejor separador presente en el texto actual
    separator = separators[-1]
    new_separators = []

    for i, sep in enumerate(separators):
      if sep == "":
        separator = ""
        break
      if sep in text:
        separator = sep
        new_separators = separators[i + 1 :]
        break

    splits = text.split(separator) if separator != "" else list(text)

    chunks: List[str] = []
    current_chunk = ""

    for s in splits:
      # Si una sección aislada excede el límite, bajamos de nivel recursivamente
      if len(s) > self.chunk_size and new_separators:
        deeper_chunks = self._split(s, new_separators)
        for dc in deeper_chunks:
          if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""
          chunks.append(dc)
        continue

      candidate = (
          f"{current_chunk}{separator}{s}" if current_chunk else s
      ).strip()

      if len(candidate) <= self.chunk_size:
        current_chunk = candidate
      else:
        if current_chunk:
          chunks.append(current_chunk)
        current_chunk = self._create_overlap_start(current_chunk, s, separator)

    if current_chunk:
      chunks.append(current_chunk)

    return chunks

  def _create_overlap_start(
      self, previous_chunk: str, current_part: str, separator: str
  ) -> str:
    """Construye el inicio del siguiente chunk tomando los últimos caracteres del anterior."""
    if not previous_chunk or self.chunk_overlap <= 0:
      return current_part

    overlap_text = previous_chunk[-self.chunk_overlap :]

    # Intentamos no cortar una palabra por la mitad
    space_idx = overlap_text.find(" ")
    if space_idx != -1 and space_idx < len(overlap_text) - 1:
      overlap_text = overlap_text[space_idx + 1 :]

    return f"{overlap_text}{separator}{current_part}".strip()