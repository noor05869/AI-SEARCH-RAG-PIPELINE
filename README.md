# TechVault AI & FastAPI RAG Pipeline

A production-grade AI Search & Retrieval-Augmented Generation (RAG) backend engineered with **FastAPI**, **LangChain**, **Google Gemini**, and **Qdrant Vector Database**.

This project provides an intelligent, domain-adapted e-commerce product search and conversational sales assistant tailored for tech gadgets and electronics. It parses natural-language user queries into structured filters, executes filtered vector search on Qdrant, and synthesizes grounded answers with price and stock availability.

---

## 🚀 Key Features

- **⚡ Fast & Async REST API**: Built on FastAPI with automatic interactive OpenAPI / Swagger docs at `/docs`.
- **🧠 Zero-Shot Intent Extraction**: Converts unstructured natural-language buyer queries into strict Pydantic search filters (`category`, `brand`, `max_price_pkr`, `in_stock_only`, `search_query`) using Google Gemini native schema extraction.
- **🏷️ Zero-Cost Local Embeddings**: Uses HuggingFace `BAAI/bge-small-en-v1.5` (384-dimensional cosine embeddings) running locally on CPU. No external embedding API rate limits or costs.
- **🎯 Filtered Vector Retrieval**: Connects with Qdrant vector database with hybrid pre-filtering (price ceiling, brand, category) to retrieve the top-$k$ relevant catalog items.
- **💬 Dual RAG Response Modes**:
  - **Conversational Chat (`/api/ai-search/chat`)**: Human-like sales assistant providing natural language advice strictly grounded in catalog data.
  - **Structured JSON RAG (`/api/ai-search/chat-structured`)**: Returns a typed Pydantic response containing both a natural language summary and an array of typed product objects (`title`, `price_pkr`, `stock_status`).
- **🛡️ Quota Limit Resilience**: Built-in multi-model fallback chain (`gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-flash-latest`) to withstand API rate limits or quota throttling.
- **🔄 Incremental Checkpointed Ingestion**: Ingestion script (`scripts/embed_catalog.py`) tracks existing indexed IDs in Qdrant to resume catalog embeddings without re-processing items.
- **🐳 Containerized Infrastructure**: Single-command Docker Compose stack providing PostgreSQL, Qdrant, and Redis.

---

## 🏗️ Architecture Flow

```mermaid
flowchart TD
    User([User / Client Application]) -->|1. Search Query| API["FastAPI Gateway (/api/ai-search/*)"]
    
    subgraph Intent & Retrieval
        API -->|2. Raw Query| Extractor["Intent Extractor (Gemini LLM)"]
        Extractor -->|3. TechVaultSearchFilter| SearchEngine["Vector Retriever"]
        SearchEngine -->|4. Query Embedding (BAAI/bge-small-en-v1.5)| Embedder["Local HuggingFace Embedder"]
        SearchEngine -->|5. Vector Search + Payload Filters| Qdrant[("Qdrant Vector DB (384-dim COSINE)")]
        Qdrant -->|6. Top-K Catalog Matches| SearchEngine
    end
    
    subgraph Augmentation & Generation
        SearchEngine -->|7. Grounded Context| Chain["LCEL RAG Chain"]
        API -->|User Prompt| Chain
        Chain -->|8. Context + Prompt| Gemini["Google Gemini (Flash Lite / 3.8 / Latest)"]
        Gemini -->|9. Grounded Answer / JSON| Formatter["Response Parser (Str or Structured)"]
    end
    
    Formatter -->|10. Final Response| User
```

---

## 📁 Project Structure

```
fastapi_blog/
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore configuration
├── .python-version               # Python version specification (3.12)
├── 01_intent_extractor_gemini.py # Standalone CLI testing script for intent extraction
├── docker-compose.yml            # Docker stack (PostgreSQL, Qdrant, Redis)
├── pyproject.toml                # Project metadata & dependency definitions
├── README.md                     # Project documentation
├── ROADMAP_AND_SPEC.md           # Engineering roadmap & architectural spec
├── uv.lock                       # Deterministic dependency lockfile
├── scripts/
│   ├── embed_catalog.py          # SQL-to-Qdrant batch catalog embedding script
│   └── seed_db.py                # Database seeder with sample electronics catalog
└── src/
    └── fastapi_blog/
        ├── __init__.py
        ├── database.py           # SQLModel database engine & session dependency
        ├── main.py               # FastAPI application setup and API endpoints
        ├── models.py             # SQLModel data models (Listing / Product)
        ├── rag_chain.py          # Intent extraction, Qdrant retriever, and LCEL chains
        └── schemas.py            # Pydantic request/response schemas and filters
```

---

## 🧰 Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic v2](https://docs.pydantic.dev/)
- **LLM & RAG Orchestration**: [LangChain](https://www.langchain.com/), [langchain-google-genai](https://github.com/langchain-ai/langchain-google), [langchain-qdrant](https://github.com/langchain-ai/langchain-qdrant), [langchain-huggingface](https://github.com/langchain-ai/langchain-huggingface)
- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **Relational Database & ORM**: [SQLModel](https://sqlmodel.tiangolo.com/), [SQLAlchemy](https://www.sqlalchemy.org/) (PostgreSQL / MySQL)
- **Embeddings Model**: `BAAI/bge-small-en-v1.5` (via `sentence-transformers` & PyTorch)
- **Dependency Management**: [uv](https://github.com/astral-sh/uv) / Python 3.12

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.12 (`python --version`)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant, PostgreSQL, Redis)
- Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Clone and Setup Environment

Using `uv` (recommended):
```bash
# Clone the repository
git clone <repository-url>
cd fastapi_blog

# Create virtual environment & sync dependencies
uv sync
```

Alternatively, using standard `python -m venv`:
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### 3. Configure Environment Variables
Copy `.env.example` to create your `.env`:
```bash
cp .env.example .env
```

Edit `.env` and set your credentials:
```ini
# Google Gemini API Key
GOOGLE_API_KEY=your_actual_gemini_api_key

# Relational Database URL (PostgreSQL default or MySQL)
DATABASE_URL=postgresql://techvault:techvault123@localhost:5432/techvault_db

# Qdrant Vector DB Settings
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis Cache Settings
REDIS_URL=redis://localhost:6379
```

### 4. Start Infrastructure Containers
Launch PostgreSQL, Qdrant, and Redis using Docker Compose:
```bash
docker compose up -d
```

Verify services:
- **Qdrant Web Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **PostgreSQL**: Port `5432`
- **Redis**: Port `6379`

### 5. Seed Database & Ingest Catalog

1. **Seed the Relational Database**:
   ```bash
   python scripts/seed_db.py
   ```
   *Creates the `listings` table and seeds sample electronics (Samsung, Apple, Dell, Soundpeats, Redmi, Anker).*

2. **Embed and Index into Qdrant**:
   ```bash
   python scripts/embed_catalog.py
   ```
   *Fetches active listings from the database, converts them to documents, generates 384-dimensional embeddings locally, and populates the `techvault_products` collection.*

---

## 🏃 Running the Application

Start the FastAPI development server:
```bash
# Using uvicorn directly
uvicorn fastapi_blog.main:app --host 127.0.0.1 --port 8000 --reload
```
or run via the script:
```bash
python src/fastapi_blog/main.py
```

The application will be live at:
- **Root Endpoint**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Alternative**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Reference

### 1. Intent Extraction
Extracts structured search criteria from a natural-language query without running vector search.

- **Endpoint**: `POST /api/ai-search/extract-intent?query=...`
- **Example Request**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/ai-search/extract-intent?query=Show%20me%20Dell%20laptops%20under%20200000%20PKR"
  ```
- **Example Response**:
  ```json
  {
    "category": "laptop",
    "brand": "Dell",
    "max_price_pkr": 200000.0,
    "in_stock_only": true,
    "search_query": "Dell laptop"
  }
  ```

---

### 2. Conversational RAG Chat
Performs intent extraction, vector search, and returns a grounded conversational sales response.

- **Endpoint**: `POST /api/ai-search/chat`
- **Request Body**:
  ```json
  {
    "session_id": "user-session-101",
    "message": "I need wireless earbuds from Soundpeats under 15k PKR"
  }
  ```
- **Example Response**:
  ```json
  {
    "session_id": "user-session-101",
    "reply": "We have the **Soundpeats Engine 4 Wireless Earbuds** currently in stock! They feature dual coaxial dynamic drivers and LDAC support, priced at **12,500 PKR** (50 units available).",
    "extracted_filters": {
      "category": "earbuds",
      "brand": "Soundpeats",
      "max_price_pkr": 15000.0,
      "in_stock_only": true,
      "search_query": "Soundpeats wireless earbuds"
    }
  }
  ```

---

### 3. Structured JSON RAG
Returns both a conversational summary and a typed JSON list of recommended products for frontend card rendering.

- **Endpoint**: `POST /api/ai-search/chat-structured`
- **Request Body**:
  ```json
  {
    "session_id": "user-session-102",
    "message": "Recommend a Samsung smartphone below 150k"
  }
  ```
- **Example Response**:
  ```json
  {
    "session_id": "user-session-102",
    "reply_data": {
      "message": "Here is the best match from our inventory matching your budget:",
      "products": [
        {
          "title": "Samsung Galaxy A55 5G",
          "price_pkr": 134999.0,
          "stock_status": "In Stock (25 available)"
        }
      ]
    },
    "extracted_filters": {
      "category": "mobile",
      "brand": "Samsung",
      "max_price_pkr": 150000.0,
      "in_stock_only": true,
      "search_query": "Samsung smartphone"
    }
  }
  ```

---

### 4. Blog Mock Endpoints
- `GET /api/posts` — Retrieve all posts.
- `GET /api/posts/{id}` — Retrieve a single post by ID.

---

## 🧪 Testing Intent Extraction in CLI

To quickly test the Gemini structured intent extraction pipeline without running the server or Qdrant:

```bash
python 01_intent_extractor_gemini.py
```

This runs sample queries through Gemini Flash and verifies Pydantic schema validation.

---

## 🗺️ Roadmap & Next Steps

As outlined in [ROADMAP_AND_SPEC.md](ROADMAP_AND_SPEC.md):
- [x] Pydantic Intent Extractor with Gemini native JSON schema
- [x] Local HuggingFace embedding ingestion into Qdrant (384-dim COSINE)
- [x] LCEL RAG chain for text & structured JSON endpoints
- [ ] Multi-turn session memory with Redis chat message history
- [ ] Server-Sent Events (SSE) streaming endpoint (`/api/ai-search/chat/stream`)
- [ ] LangGraph agent orchestration with tool-calling for catalog filtering
- [ ] RAGAS automated evaluation and LangSmith tracing

---

## 📄 License

This project is licensed under the MIT License.
