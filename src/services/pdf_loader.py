from io import BytesIO
from typing import Any, Dict, List
from pypdf import PdfReader


class PDFLoaderService:
  """Servicio encargado de extraer texto limpio y metadatos de documentos PDF."""

  @staticmethod
  def extract_text_from_bytes(
      file_bytes: bytes, filename: str = "document.pdf"
  ) -> List[Dict[str, Any]]:
    """Lee el binario de un PDF y extrae el texto página a página con sus metadatos.

    :param file_bytes: Los bytes brutos del PDF (tal como llegan de un archivo o
    de una subida HTTP).
    :param filename: El nombre original del archivo para guardar en la fuente.
    :return: Lista de diccionarios con el texto de cada página y su número de
    página.
    """
    # BytesIO permite leer los bytes en memoria RAM como si fueran un archivo físico
    reader = PdfReader(BytesIO(file_bytes))
    pages_data: List[Dict[str, Any]] = []

    for index, page in enumerate(reader.pages):
      page_text = page.extract_text() or ""
      page_text = page_text.strip()

      # Si la página tiene contenido (evita páginas en blanco), la guardamos
      if page_text:
        pages_data.append({
            "text": page_text,
            "metadata": {
                "source": filename,
                "page": index + 1,
                "total_pages": len(reader.pages),
            },
        })

    return pages_data