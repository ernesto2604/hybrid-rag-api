# Enterprise Hybrid RAG Engine API

Motor de Búsqueda Híbrida y Generación RAG (Retrieval-Augmented Generation) de alta precisión desarrollado con FastAPI, PostgreSQL (pgvector + Full-Text Search) y Google Gemini.

El sistema implementa aislamiento multi-documento por metadatos, memoria conversacional multi-turno y citas bibliográficas directas por página para mitigar alucinaciones en entornos corporativos.

---

## 🏗️ Arquitectura del Sistema

```
                        [ Cliente / UI Web ]
                                |
                                v
                       [ FastAPI Engine ]
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
   [ Búsqueda Vectorial ]                 [ Búsqueda Full-Text ]
(pgvector / HNSW Cosine Ops)            (tsvector / tsquery GIN)
             |                                     |
             +------------------+------------------+
                                |
                                v
               [ Reciprocal Rank Fusion (RRF) ]
                       k = 60 Scoring
                                |
                                v
                    [ Top-K Chunks Re-ranked ]
                                |
                                v
             [ Gemini 2.5 Flash + Conversational Memory ]
                                |
                                v
               [ Respuesta Verificable con Citas ]
```

---

## ✨ Características Principales

* **Búsqueda Híbrida Balanceada:** Combina similitud semántica densa (embeddings) con coincidencia léxica dispersa (palabras clave y códigos exactos).
* **Fusión por RRF (Reciprocal Rank Fusion):** Normaliza y combina los rankings de ambos motores con constante k=60 sin necesidad de calibrar pesos manuales.
* **Aislamiento Multi-documento:** Permite búsquedas globales en todo el corpus o consultas acotadas a un document_id específico mediante filtrado JSONB en PostgreSQL.
* **Trazabilidad y Citas Verificables:** Respuestas estrictamente condicionadas al contexto provisto, incluyendo fuente documental, página y score de relevancia RRF.
* **Memoria Conversacional:** Gestión de historial de sesión para resolución de anáforas y preguntas de seguimiento encadenadas.
* **Interfaz de Usuario Reactiva:** Dashboard minimalista en Tailwind CSS con panel de métricas, catálogo de documentos activos y control de ámbito.

---

## 🛠️ Stack Tecnológico

* **Backend:** FastAPI, Python 3.11+, Pydantic v2.
* **Base de Datos & Vector Store:** PostgreSQL 16+, extensión pgvector, SQLAlchemy (asyncpg).
* **Embeddings & LLM:** Google Gemini (text-embedding-004 / gemini-2.5-flash) vía Google GenAI SDK.
* **Frontend:** HTML5, Tailwind CSS, FontAwesome.

---

## 🚀 Despliegue y Puesta en Marcha

### 1. Clonar el repositorio y configurar variables de entorno

```bash
git clone [https://github.com/TU_USUARIO/hybrid-rag-api.git](https://github.com/TU_USUARIO/hybrid-rag-api.git)
cd hybrid-rag-api
cp .env.example .env
```

Configura tus credenciales en el archivo `.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db
GEMINI_API_KEY=tu_api_key_de_gemini
```

### 2. Iniciar PostgreSQL con pgvector (Docker Compose)

```bash
docker compose up -d
```

### 3. Instalar dependencias

```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Inicializar esquemas e índices

```bash
python init_db.py
```

### 5. Iniciar la API

```bash
uvicorn src.main:app --reload
```

Accede a:
* **Interfaz Web:** http://127.0.0.1:8000
* **Documentación Interactiva Swagger:** http://127.0.0.1:8000/docs