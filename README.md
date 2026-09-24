# Enterprise Hybrid RAG Engine API

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-blue?style=flat)](https://github.com/pgvector/pgvector)
[![Google GenAI](https://img.shields.io/badge/Google_GenAI-Gemini-8E75C2?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)

Motor de producción para **Retrieval-Augmented Generation (RAG) Híbrido** implementado desde cero sin dependencias de frameworks caja-negra (sin LangChain ni LlamaIndex). Integra búsqueda densa (vectorial) y dispersa (léxica) en PostgreSQL, fusión de resultados mediante **Reciprocal Rank Fusion (RRF)**, aislamiento granular por documento, atribución estricta con citas por página y memoria conversacional multi-turno.

---

## 🎥 Demostración en Video

Demostración del ciclo completo del motor: subida e ingesta concurrente de PDFs, búsqueda híbrida determinista con RRF, citas estrictas por página, aislamiento documental y memoria multi-turno:

<!-- Inserta la URL generada al arrastrar tu archivo .mp4 al editar este archivo en GitHub -->
[https://github.com/user-attachments/assets/TU_VIDEO_ID.mp4
](https://github.com/user-attachments/assets/cbd65417-41b7-42b2-84fe-cb9ecac8c319
)
---

## 🖥️ Interfaz de Usuario

![Enterprise Hybrid RAG Dashboard](docs/images/dashboard.png)

*Funcionalidades destacadas de la interfaz:*
1. **Catálogo de Documentos Persistidos:** Listado reactivo de PDFs indexados en PostgreSQL con desglose exacto de fragmentos (*chunks*).
2. **Selector de Ámbito Dinámico:** Filtrado estricto por documento (`document_id`) o modo de búsqueda transversal global.
3. **Generación con Citas Verificables:** Respuestas sintetizadas con atribución obligatoria `[archivo, Pág. X]` y tarjetas interactivas con el valor exacto de relevancia RRF ($k=60$).
4. **Memoria Conversacional Multi-Turno:** Mantenimiento de contexto previo para preguntas de seguimiento sin pérdida de foco.

---

## 📐 Arquitectura del Sistema

El flujo desacopla la persistencia/recuperación en base de datos de la inferencia generativa mediante un pipeline determinista y asíncrono:

```mermaid
flowchart TD
    subgraph Frontend ["Capa de Presentación"]
        UI["🖥️ Dashboard Web / Cliente HTTP"]
    end

    subgraph Gateway ["Capa de Entrada & Orquestación"]
        API["⚡ FastAPI Gateway<br/>(Endpoints asíncronos / REST)"]
    end

    subgraph Storage ["PostgreSQL 16 + pgvector"]
        direction TB
        subgraph Ingestion ["Ingesta & Indexación"]
            PDF["📄 PDF Loader & Chunker<br/>(pypdf / segmentación por página)"]
            EMB_IN["🧬 gemini-embedding-001<br/>(Lotes con backoff exponencial)"]
            PDF --> EMB_IN
        end

        subgraph SearchEngine ["Recuperación Concurrente"]
            VEC["🔍 Búsqueda Semántica Densa<br/>pgvector (HNSW Cosine, 768d)"]
            FTS["🔎 Búsqueda Léxica Dispersa<br/>GIN Index (tsvector @@ websearch_to_tsquery)"]
        end

        EMB_IN --> VEC
        PDF --> FTS
    end

    subgraph Ranking ["Fusión & Reordenamiento"]
        RRF["⚖️ Reciprocal Rank Fusion (RRF)<br/>Score = Σ 1 / (k + r),  k=60"]
    end

    subgraph Generation ["Capa Generativa"]
        CTX["📦 Ensamblador de Contexto<br/>(Top-K Chunks + Historial multi-turno)"]
        LLM["🤖 Google GenAI / Gemini<br/>(Inferencia determinista temp=0.0)"]
        ANS["📝 Respuesta Sintetizada<br/>Citas verificables [archivo, Pág. X]"]
        CTX --> LLM --> ANS
    end

    %% Conexiones principales
    UI -->|Petición RAG / Documentos| API
    API -->|Ingesta de archivo| PDF
    API -->|Query paralelizada| VEC
    API -->|Query paralelizada| FTS

    VEC -->|Top-N densos| RRF
    FTS -->|Top-N léxicos| RRF

    RRF -->|Top-K fusionados| CTX
    ANS -->|Payload JSON| UI

    %% Estilos de nodos
    classDef client fill:#1e1e2e,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef gateway fill:#181825,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef db fill:#11111b,stroke:#f9e2af,stroke-width:2px,color:#cdd6f4;
    classDef algo fill:#181825,stroke:#fab387,stroke-width:2px,color:#cdd6f4;
    classDef gen fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;

    class UI client;
    class API gateway;
    class VEC,FTS,PDF,EMB_IN db;
    class RRF algo;
    class CTX,LLM,ANS gen;
```

---

## 💡 Decisiones de Diseño e Ingeniería

### 1. Fusión de Resultados: RRF frente a Weighted Sum
La combinación clásica de búsqueda semántica y léxica mediante suma lineal ponderada ($\alpha \cdot S_{vec} + (1-\alpha) \cdot S_{text}$) suele degradar la precisión debido a que las escalas de similitud coseno $[0, 1]$ y las puntuaciones de relevancia de texto (como `ts_rank`) no son comparables ni lineales.

Este motor implementa **Reciprocal Rank Fusion (RRF)**:

$$RRF(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

* $M$: conjunto de métodos de recuperación (Vectorial y Full-Text).
* $r_m(d)$: posición ordinal en el ranking del documento $d$ dentro del método $m$.
* $k = 60$: constante de suavizado estándar que evita que las primeras posiciones monopolicen de forma desproporcionada la puntuación combinada.

### 2. Índices de Base de Datos y Representación Vectorial
* **Vectores densos:** Índice **HNSW** (`vector_cosine_ops`) sobre vectores normalizados de 768 dimensiones generados con `models/gemini-embedding-001`, optimizando la latencia de búsqueda frente a algoritmos basados en clustering como IVFFlat.
* **Texto disperso:** Índice **GIN** sobre columnas generadas `to_tsvector('spanish', content)` para búsquedas exactas con soporte de lematización y diccionario en español.

### 3. Pipeline de Ingesta Masiva Asíncrona (Batching & Backoff)
Para evitar el cuello de botella de realizar llamadas HTTP individuales por cada fragmento (*chunk*):
* Segmentación en memoria basada en páginas físicas del PDF.
* Vectorización por lotes (*batch size* configurable de hasta 100 fragmentos) con ejecución concurrente.
* Manejo de cuotas mediante reintentos automáticos con **backoff exponencial** ante respuestas HTTP 429 y 503, garantizando la persistencia íntegra de documentos extensos.

### 4. Aislamiento Multidocumento y Mitigación de Alucinaciones
* **Filtrado determinista:** Consultas acotadas a nivel de base de datos vía JSONB (`metadata_->>'source'`), impidiendo la contaminación cruzada entre documentos.
* **Citas verificables:** El prompt del sistema fuerza la atribución de cada afirmación con formato `[documento, Pág. X]`, mitigando alucinaciones al rechazar explícitamente responder sobre información no contenida en el corpus recuperado.
* **Memoria conversacional:** Inyección de turnos previos para contextualizar las consultas y permitir el seguimiento del diálogo.

---

## 📂 Estructura del Proyecto

```bash
hybrid-rag-api/
├── docker-compose.yml       # Orquestación de PostgreSQL 16 con extensión pgvector
├── init_db.py               # Inicialización de esquemas e índices HNSW/GIN
├── delete_doc.py            # Utilidad CLI para borrado selectivo de documentos
├── requirements.txt         # Dependencias del proyecto
├── docs/
│   └── images/
│       └── dashboard.png    # Captura de la interfaz para documentación
├── src/
│   ├── api/
│   │   └── schemas.py       # Contratos Pydantic v2 (I/O, RAG, Memoria)
│   ├── core/
│   │   ├── config.py        # Carga de variables de entorno y configuración
│   │   ├── database.py      # Motor y pool asíncrono con SQLAlchemy + asyncpg
│   │   └── models.py        # Modelo DocumentChunk con columnas Vector y tsvector
│   ├── services/
│   │   ├── chunking.py      # Segmentación semántica de texto con solapamiento
│   │   ├── embeddings.py    # Cliente Google GenAI (gemini-embedding-001) con batching
│   │   ├── pdf_loader.py    # Extracción de texto y metadatos por página (pypdf)
│   │   ├── rag_generator.py # Generación LLM con control determinista y citas
│   │   └── vector_store.py  # Algoritmo RRF y consultas combinadas SQL
│   └── main.py              # API FastAPI y montaje de endpoints
└── static/
    └── index.html           # UI interactiva (Tailwind CSS, selector de ámbito y chat)
```

---

## 🔌 Especificación de la API

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/api/v1/documents` | Devuelve el catálogo de documentos ingestados y su desglose de chunks. |
| `POST` | `/api/v1/documents/upload` | Procesa un PDF, extrae texto por páginas, vectoriza por lotes y persiste en BD. |
| `POST` | `/api/v1/search/hybrid` | Búsqueda híbrida pura (HNSW + GIN) fusionada mediante RRF ($k=60$). |
| `POST` | `/api/v1/rag/ask` | Pipeline RAG completo: recuperación híbrida + memoria previa + síntesis generativa. |

---

## 🚀 Despliegue Local

### 1. Clonar el repositorio y configurar variables de entorno
```bash
git clone [https://github.com/ernesto2604/hybrid-rag-api.git](https://github.com/ernesto2604/hybrid-rag-api.git)
cd hybrid-rag-api
cp .env.example .env
```

Configura en tu archivo `.env`:
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db
GEMINI_API_KEY=tu_clave_de_gemini
```

### 2. Iniciar PostgreSQL con pgvector
```bash
docker compose up -d
```

### 3. Instalar dependencias e inicializar esquemas
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
python init_db.py
```

### 4. Ejecutar el servidor
```bash
uvicorn src.main:app --reload
```

* **Dashboard Web UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Documentación Interactiva (Swagger OpenAPI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
