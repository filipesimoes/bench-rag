# Pipeline Configuration Guide

This document is the authoritative technical reference for configuring, customizing, and executing Retrieval-Augmented Generation (RAG) pipelines in the **Modular RAG Evaluation Framework for Criminal Investigation Reports** (`bench-rag`).

For mathematical definitions of evaluation metrics, forensic taxonomies ($T1, T2, T3$), and complexity levels ($C1, C2, C3$), see [docs/methodology.md](file:///home/filipe/projetos/bench-rag/docs/methodology.md).

---

## 1. Overview & Architecture

The framework is built upon Clean Architecture and Dependency Inversion. RAG pipelines are declared declaratively via YAML configuration files and instantiated through dynamic factories:

```
                  ┌───────────────────────────────┐
                  │    Pipeline YAML Config File  │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │    PipelineConfig (Pydantic)  │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │  build_pipeline_from_config() │
                  └──────────────┬────────────────┘
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
  DocumentParser              Chunker             EmbeddingModel
  (PyMuPDF / pdfplumber)  (Recursive / Token)     (OpenAI / Mock)
          │                      │                      │
          └──────────────────────┬──────────────────────┘
                                 │
                                 ▼
                            VectorStore
                          (Qdrant / Memory)
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
  StandardDenseRAGPipeline               HybridRAGPipeline
   (Dense Vector Top-K)              (Dense + BM25 Okapi + RRF)
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                            GeneratorLLM
                          (OpenAI / Mock)
                                 │
                                 ▼
                         StandardEvaluationEngine
                          (Judge LLM / Mock)
```

### Configuration Lifecycle
1. **YAML Parsing**: The YAML file is loaded into Python primitives.
2. **Pydantic Validation**: Values are validated against `PipelineConfig` rules, enforcing types, literal choices, and defaults.
3. **Component Assembly**: Factories (`get_parser`, `get_chunker`, `get_embedding_model`, `get_vector_store`, `get_generator`, `get_judge`) wire the concrete implementations.
4. **Pipeline Execution**: The pipeline is evaluated or indexed via the CLI (`bench-rag`).

---

## 2. Configuration Hierarchy & Precedence

`bench-rag` utilizes a 4-tier precedence model to resolve endpoints, models, and credentials:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Pipeline YAML Override (per-pipeline model/endpoint)          │
│    (e.g., generator_base_url, generator_model)                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │ fallback if not specified
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Specific Service Environment Variable (.env)                  │
│    (e.g., GENERATOR_BASE_URL, GENERATOR_API_KEY)                │
└────────────────────────────────┬────────────────────────────────┘
                                 │ fallback if not specified
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Global Service Fallback Environment Variable (.env)           │
│    (e.g., OPENAI_BASE_URL, OPENAI_API_KEY)                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │ fallback if not specified
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Internal System Defaults (Pydantic Settings)                 │
│    (e.g., "gpt-4o-mini", "text-embedding-3-small", 1536)        │
└─────────────────────────────────────────────────────────────────┘
```

### Precedence Mapping Table

| Component | YAML Field Override | Specific `.env` Variable | Global Fallback `.env` | System Default |
| :--- | :--- | :--- | :--- | :--- |
| **Generator Endpoint** | `generator_base_url` | `GENERATOR_BASE_URL` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| **Generator API Key** | *(Via `.env`)* | `GENERATOR_API_KEY` | `OPENAI_API_KEY` | `mock-key` |
| **Generator Model** | `generator_model` | `GENERATOR_MODEL` | — | `gpt-4o-mini` |
| **Generator Enable Thinking** | `generator_enable_thinking` | `GENERATOR_ENABLE_THINKING` | — | `null` |
| **Generator Dialect** | `generator_dialect` | `GENERATOR_DIALECT` | — | `auto` |
| **Judge Endpoint** | `judge_base_url` | `JUDGE_BASE_URL` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| **Judge API Key** | *(Via `.env`)* | `JUDGE_API_KEY` | `OPENAI_API_KEY` | `mock-key` |
| **Judge Model** | `judge_model` | `JUDGE_MODEL` | — | `gpt-4o-mini` |
| **Judge Enable Thinking** | `judge_enable_thinking` | `JUDGE_ENABLE_THINKING` | — | `null` |
| **Judge Dialect** | `judge_dialect` | `JUDGE_DIALECT` | — | `auto` |
| **Embedding Endpoint** | `embedding_base_url` | `EMBEDDING_BASE_URL` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| **Embedding API Key** | *(Via `.env`)* | `EMBEDDING_API_KEY` | `OPENAI_API_KEY` | `mock-key` |
| **Embedding Model** | `embedding_model` | `EMBEDDING_MODEL` | — | `text-embedding-3-small` |
| **Embedding Dimension**| — | `EMBEDDING_DIMENSION`| — | `1536` |
| **Reranker Endpoint** | `reranker_base_url` | `RERANKER_BASE_URL` | — | `http://localhost:8000/v1` |
| **Reranker API Key** | *(Via `.env`)* | `RERANKER_API_KEY` | `OPENAI_API_KEY` | `mock-key` |
| **Reranker Model** | `reranker_model` | `RERANKER_MODEL` | — | `BAAI/bge-reranker-v2-m3` |
| **Qdrant Host / Port** | — | `QDRANT_HOST`, `QDRANT_PORT` | — | `qdrant`, `6333` |

---

## 3. Exhaustive Parameter Reference

All options configured in pipeline YAML files map directly to the `PipelineConfig` schema.

| Parameter | Type / Allowed Values | Default | Description & Functional Impact | Constraints & Validation |
| :--- | :--- | :--- | :--- | :--- |
| `name` | `str` | `"standard_dense_rag"` | Human-readable identifier for the pipeline. Used in generated reports, CSV outputs, and comparative head-to-head metrics. | Must be a non-empty string. |
| `pipeline_type` | `"standard"` \| `"hybrid"` | `"standard"` | Core retrieval architecture: <br>• `"standard"`: Dense semantic vector Top-K retrieval.<br>• `"hybrid"`: Dual dense vector + BM25 Okapi lexical search merged via Reciprocal Rank Fusion (RRF). | Only `"standard"` or `"hybrid"` accepted. |
| `parser_type` | `"pymupdf"` \| `"pdfplumber"` | `"pymupdf"` | PDF parsing engine: <br>• `"pymupdf"`: High-speed native parsing (PyMuPDF) for plain-text extraction.<br>• `"pdfplumber"`: Layout-aware parser that formats tables into structured `[TABLE DATA]` markdown blocks. | Only `"pymupdf"` or `"pdfplumber"` accepted. |
| `chunker_type` | `"recursive"` \| `"token"` | `"recursive"` | Text splitting strategy: <br>• `"recursive"`: Hierarchical splitting by `["\n\n", "\n", ". ", " ", ""]` to preserve sentence and paragraph cohesion.<br>• `"token"`: Sliding token/word window splitting. | Only `"recursive"` or `"token"` accepted. |
| `chunk_size` | `int` | `1000` | Target chunk size. In characters for `recursive`, or word/token count for `token`. | Must be $> 0$ and strictly $> chunk\_overlap$. |
| `chunk_overlap` | `int` | `200` | Number of characters or tokens shared between consecutive chunks to preserve contextual continuity at boundaries. | Must be $\ge 0$ and strictly $< chunk\_size$. Raises `ValueError` if $\ge chunk\_size$. |
| `embedding_type` | `"openai"` \| `"mock"` | `"mock"` | Embedding provider: <br>• `"mock"`: Deterministic MD5-based pseudo-embeddings for offline testing and zero API costs.<br>• `"openai"`: Remote embedding endpoint (`/v1/embeddings`) compatible with OpenAI, Ollama, vLLM, and LiteLLM. | Only `"openai"` or `"mock"` accepted. |
| `vector_store_type`| `"qdrant"` \| `"memory"` | `"memory"` | Vector storage engine: <br>• `"memory"`: In-memory NumPy vector store. Non-persistent, ideal for fast unit/integration tests.<br>• `"qdrant"`: Distributed, persistent Qdrant vector database service running in Docker. | Only `"qdrant"` or `"memory"` accepted. |
| `generator_type` | `"openai"` \| `"mock"` | `"mock"` | Generation LLM provider: <br>• `"mock"`: Deterministic offline response generator.<br>• `"openai"`: Remote chat completion endpoint (`/v1/chat/completions`). | Only `"openai"` or `"mock"` accepted. |
| `judge_type` | `"openai"` \| `"mock"` | `"mock"` | LLM-as-a-Judge evaluation engine: <br>• `"mock"`: Heuristic lexical/token overlap scoring.<br>• `"openai"`: Multi-criteria forensic evaluation (Faithfulness, Answer Relevance, Context Recall) returning structured JSON. | Only `"openai"` or `"mock"` accepted. |
| `top_k` | `int` | `5` | Number of context chunks retrieved from the vector store and injected into the LLM synthesis prompt. | Must be $\ge 1$. |
| `collection_name` | `str` | `"police_reports"` | Target collection/index name in the vector store. | Valid alphanumeric / underscore identifier. |
| `distance_metric` | `"cosine"` \| `"euclidean"` \| `"dot"` | `"cosine"` | Distance metric for vector similarity calculation in Qdrant and Memory stores. | Only `"cosine"`, `"euclidean"`, or `"dot"` accepted. |
| `rrf_k` | `int` | `60` | Rank smoothing constant ($k_0$) used in Reciprocal Rank Fusion formula: $\text{RRF}(c) = \sum \frac{w}{k_0 + r(c)}$. High values reduce the penalty for lower ranks. *(Hybrid only)*. | Must be $> 0$. Default is standard $60$. |
| `bm25_weight` | `float` | `0.5` | Weight multiplier applied to lexical BM25 reciprocal ranks in RRF fusion. *(Hybrid only)*. | Must be $\ge 0.0$. |
| `dense_weight` | `float` | `0.5` | Weight multiplier applied to dense vector reciprocal ranks in RRF fusion. *(Hybrid only)*. | Must be $\ge 0.0$. |
| `generator_base_url` | `str \| None` | `null` | Custom base URL override for the Generator LLM (e.g. `"http://host.docker.internal:11434/v1"`). | Valid HTTP/HTTPS URL or `null`. |
| `generator_model` | `str \| None` | `null` | Model identifier override for the Generator LLM (e.g. `"gpt-4o"` or `"llama3.1:8b"`). | String identifier or `null`. |
| `generator_enable_thinking` | `bool \| None` | `null` | Controls reasoning/thinking in OpenAI/vLLM generators. When `false`, suppresses `<think>` blocks on reasoning models. If `null`, falls back to `.env` or system default. | `true`, `false`, or `null`. |
| `generator_dialect` | `"auto"` \| `"vllm"` \| `"gemini"` \| `"openai"` | `"auto"` | API dialect for Generator LLM. Controls whether `reasoning_effort` (Gemini, OpenAI) or `chat_template_kwargs` (vLLM) is used. When `"auto"`, infers from `generator_base_url`. | `"auto"`, `"vllm"`, `"gemini"`, `"openai"`. |
| `judge_base_url` | `str \| None` | `null` | Custom base URL override for the LLM Judge (enables using an independent, high-capacity auditing model). | Valid HTTP/HTTPS URL or `null`. |
| `judge_model` | `str \| None` | `null` | Model identifier override for the LLM Judge (e.g. `"gpt-4o"`). | String identifier or `null`. |
| `judge_enable_thinking` | `bool \| None` | `null` | Controls reasoning/thinking for the Judge LLM. When `null` (default), preserves compatibility with endpoints like Google Gemini that reject extra fields. | `true`, `false`, or `null`. |
| `judge_dialect` | `"auto"` \| `"vllm"` \| `"gemini"` \| `"openai"` | `"auto"` | API dialect for Judge LLM. When `"auto"`, infers from `judge_base_url` (e.g. Google Gemini endpoint uses `reasoning_effort`). | `"auto"`, `"vllm"`, `"gemini"`, `"openai"`. |
| `embedding_base_url`| `str \| None` | `null` | Custom base URL override for the Embedding endpoint (e.g. Text Embeddings Inference or Ollama). | Valid HTTP/HTTPS URL or `null`. |
| `embedding_model` | `str \| None` | `null` | Model identifier override for Embeddings (e.g. `"text-embedding-3-large"` or `"nomic-embed-text"`). | String identifier or `null`. |
| `reranker_type` | `"none"` \| `"mock"` \| `"http"` \| `"vllm"` | `"none"` | Second-stage cross-encoder re-ranking engine: <br>• `"none"`: Disabled (direct passthrough).<br>• `"mock"`: Deterministic term-overlap mock for fast tests.<br>• `"http"` / `"vllm"`: Universal `/v1/rerank` client (vLLM, TEI, Jina, Cohere). | Only `"none"`, `"mock"`, `"http"`, `"vllm"`, `"tei"`, `"cohere"`, `"jina"` accepted. |
| `reranker_model` | `str \| None` | `null` | Model identifier for the reranker (e.g. `"BAAI/bge-reranker-v2-m3"`). Falls back to `.env` or system default. | String identifier or `null`. |
| `reranker_base_url` | `str \| None` | `null` | Custom base URL for the rerank endpoint (e.g. `"http://host.docker.internal:8000/v1"`). | Valid HTTP/HTTPS URL or `null`. |
| `rerank_fetch_k` | `int` | `20` | Number of candidate chunks retrieved in the first stage (dense/hybrid) to be scored by the reranker. | Must be $\ge 1$ (internally enforced as $\ge top\_k$). |

---

## 4. Component Deep Dive & Variations

### 4.1 Retrieval Paradigm: Standard vs. Hybrid

- **Standard Dense RAG (`pipeline_type: standard`)**:
  - Generates query embedding $v_q = \text{Embed}(q)$ and searches for top-$k$ nearest neighbors via vector similarity.
  - **Best suited for**: Conceptual questions, thematic categorization, and implicit relationships where lexical terms vary (e.g. matching "laranja" to "interposta pessoa").
  - **Weakness**: May struggle with exact alphanumeric strings, tax IDs, or rare surnames if not explicitly emphasized in the dense embedding space.

- **Hybrid RAG (`pipeline_type: hybrid`)**:
  - Executes dual retrieval: fetches $2 \times \text{top\_k}$ candidates from dense vector search and $2 \times \text{top\_k}$ candidates from BM25 Okapi lexical index.
  - Combines ranks via Reciprocal Rank Fusion:
    $$\text{Score}_{\text{RRF}}(c) = \frac{w_{\text{dense}}}{k_0 + r_{\text{dense}}(c)} + \frac{w_{\text{bm25}}}{k_0 + r_{\text{bm25}}(c)}$$
  - **Best suited for**: Police and criminal reports requiring precision on exact identifiers (CNPJ, CPF, vehicle license plates, bank wire numbers, specific dates) combined with semantic reasoning.

### 4.2 Two-Stage Retrieval: Dynamic Between `top_k` and `rerank_fetch_k`

The relationship between `top_k` and `rerank_fetch_k` establishes the **Two-Stage Retrieval funnel**, separating broad recall (coarse retrieval) from deep semantic precision (fine re-ranking):

- **`top_k`**: The final cutoff size. Specifies the exact number of top-ranking chunks delivered to the Generator LLM context window and scored by evaluation metrics (Recall@K, Precision@K, MRR).
- **`rerank_fetch_k`**: The initial candidate pool size retrieved during Stage 1. Provides a broader candidate selection for the cross-encoder reranker to inspect and re-score before finalizing `top_k`.

#### Architectural Matrix Across Pipeline Variations

| Pipeline Architecture | Reranker Active? | Role of `rerank_fetch_k` | Role of `top_k` | Chunk Flow Progression |
| :--- | :---: | :--- | :--- | :--- |
| **Standard Dense RAG** | **No** (`reranker_type: "none"`) | **Ignored** | Directly dictates VectorStore retrieval and final LLM context size | $k \to k$ (Single-stage vector search) |
| **Standard Dense RAG** | **Yes** (`reranker_type: "vllm"` / `"mock"`) | Sets the number of nearest neighbors retrieved by vector search | Sets the final cutoff after cross-attention re-ranking | $K_{\text{fetch}} \to k$ (Vector search $\to$ Reranker) |
| **Hybrid RAG** | **No** (`reranker_type: "none"`) | **Ignored** | Governs the initial dual fetch ($2 \times k$) and cuts the RRF output directly | $(2k \text{ dense} + 2k \text{ BM25}) \xrightarrow{\text{RRF}} k$ |
| **Hybrid RAG** | **Yes** (`reranker_type: "vllm"` / `"mock"`) | Sets the candidate pool size selected from RRF passed to the reranker | Sets the final cutoff after cross-attention re-ranking | $(K_{\text{fetch}} \text{ dense} + K_{\text{fetch}} \text{ BM25}) \xrightarrow{\text{RRF}} K_{\text{fetch}} \xrightarrow{\text{Reranker}} k$ |

#### Execution Flow Breakdown

1. **Standard Dense RAG without Rerank**:
   - `fetch_k = top_k` (e.g., 5).
   - `VectorStore.search(top_k=5)` $\to$ directly passed to Generator LLM prompt.
   - `rerank_fetch_k` is bypassed entirely.

2. **Standard Dense RAG with Rerank**:
   - `fetch_k = max(rerank_fetch_k, top_k)` (e.g., 20).
   - Stage 1: `VectorStore.search(top_k=20)` retrieves the 20 closest vector embeddings.
   - Stage 2: `Reranker.rerank(results=20, top_k=5)` performs cross-attention over all 20 query-chunk pairs, reordering them and returning only the 5 most relevant.
   - Initial retrieval rank and score are preserved in chunk metadata (`initial_retrieval_rank`, `initial_retrieval_score`).

3. **Hybrid RAG without Rerank**:
   - `candidate_pool_size = top_k` (e.g., 5).
   - Dual retrieval: fetches $\max(2 \times \text{top\_k}, 10)$ candidates from dense vector search and BM25 search.
   - Reciprocal Rank Fusion (RRF) scores the union of candidates.
   - The top `top_k` (5) chunks from RRF are sliced directly: `final_results = rrf_scores[:top_k]`.
   - `rerank_fetch_k` is bypassed entirely.

4. **Hybrid RAG with Rerank**:
   - `candidate_pool_size = max(rerank_fetch_k, top_k)` (e.g., 20).
   - Dual retrieval: fetches $\max(\text{candidate\_pool\_size}, 2 \times \text{top\_k}, 10)$ (e.g., 20) candidates from dense vector search and 20 from BM25.
   - Reciprocal Rank Fusion (RRF) scores all candidates and selects the top `candidate_pool_size` (20) chunks.
   - Stage 2: `Reranker.rerank(results=20, top_k=5)` evaluates the 20 RRF winners with the cross-encoder model (e.g. `BAAI/bge-reranker-v2-m3` on vLLM), delivering the top 5 chunks to the prompt.

> [!NOTE]
> **Safety Boundary Guarantee**: In all pipelines, the framework internally enforces `candidate_pool_size = max(rerank_fetch_k, top_k)`. This guarantees that even if a configuration inadvertently specifies `rerank_fetch_k: 3` and `top_k: 5`, the pipeline will never attempt to produce 5 results from only 3 candidates.

### 4.3 Document Parsers: PyMuPDF vs. pdfplumber

- **`parser_type: pymupdf`**:
  - Extremely fast C++-backed PDF extraction (`PyMuPDF`).
  - Extracts text streams per page while generating standard metadata (`page_number`, `char_count`).
  - **Use when**: High document volume, fast turnaround, or documents consisting primarily of narrative text.

- **`parser_type: pdfplumber`**:
  - Parses visual layout and table structures, converting tabular rows into markdown-formatted `[TABLE DATA]` tags.
  - Records `table_count` in page metadata.
  - **Use when**: Reports contain financial asset sheets, wire transfer logs, equity distributions, or seizure records formatted as tables.

### 4.4 Chunkers: Recursive Character vs. Token

- **`chunker_type: recursive`**:
  - Attempts to split along natural structural boundaries: double line breaks (`\n\n`), single line breaks (`\n`), sentence endings (`. `), words (` `), and characters (`""`).
  - Preserves paragraph integrity whenever possible up to `chunk_size`.
  - **Recommended for**: General investigative text to prevent splitting across sentences.

- **`chunker_type: token`**:
  - Splits text strictly based on word/token count sliding windows.
  - **Recommended for**: Rigid context budget management when working with models with strict input token limitations.

### 4.5 Vector Stores: In-Memory vs. Qdrant

- **`vector_store_type: memory`**:
  - Uses NumPy arrays in RAM.
  - Computes exact vector distances (Cosine, Euclidean, or Dot product).
  - No database setup required; resets when process terminates.
  - **Use when**: Quick unit tests, mock runs, or lightweight local verification.

- **`vector_store_type: qdrant`**:
  - Connects to containerized Qdrant instance via HTTP REST (`http://qdrant:6333`).
  - Persistent HNSW index with payload metadata storage.
  - **Use when**: Production benchmarks, scaling beyond memory limits, and multi-session persistence.

### 4.6 Providers & Endpoints: Mock vs. Cloud vs. Local Self-Hosted

- **Mock Mode (`mock`)**:
  - Deterministic execution across embeddings, generator, and judge.
  - Requires no network connection and incurs no API cost.
- **Cloud OpenAI-Compatible (`openai`)**:
  - Standard endpoint (`https://api.openai.com/v1`) using models like `gpt-4o-mini`, `gpt-4o`, `text-embedding-3-small`.
- **Local Self-Hosted (Ollama / vLLM / LocalAI)**:
  - Compatible with any server exposing OpenAI-compatible routes (`/v1/chat/completions` and `/v1/embeddings`).
  - Crucial for processing confidential or legally sensitive police inquiry documents *on-premises*.

---

## 5. Configuration Recipes & Examples

The following annotated YAML recipes illustrate standard architectural configurations.

### Recipe 1: Fast Offline Mock Pipeline (Zero Cost, CI/CD)
```yaml
# configs/recipe_mock.yaml
name: offline_fast_mock
pipeline_type: standard
parser_type: pymupdf
chunker_type: recursive
chunk_size: 1000
chunk_overlap: 200
embedding_type: mock
vector_store_type: memory
generator_type: mock
judge_type: mock
top_k: 5
collection_name: police_reports
distance_metric: cosine
```

### Recipe 2: Standard Dense Production (OpenAI + Qdrant)
```yaml
# configs/recipe_standard_openai.yaml
name: standard_dense_openai_qdrant
pipeline_type: standard
parser_type: pymupdf
chunker_type: recursive
chunk_size: 1000
chunk_overlap: 150
embedding_type: openai
vector_store_type: qdrant
generator_type: openai
judge_type: openai
top_k: 5
collection_name: police_reports
distance_metric: cosine

# Specific overrides (falls back to .env if omitted)
generator_model: gpt-4o-mini
judge_model: gpt-4o-mini
embedding_model: text-embedding-3-small
```

### Recipe 3: Forensic Hybrid Pipeline (Dense + BM25 Lexical + RRF)
```yaml
# configs/recipe_hybrid_forensic.yaml
name: hybrid_bm25_dense_forensic
pipeline_type: hybrid
parser_type: pymupdf
chunker_type: recursive
chunk_size: 800
chunk_overlap: 150
embedding_type: openai
vector_store_type: qdrant
generator_type: openai
judge_type: openai
top_k: 5
collection_name: police_reports
distance_metric: cosine

# Hybrid Reciprocal Rank Fusion settings
rrf_k: 60
bm25_weight: 0.6     # Slightly higher weight for exact forensic keyword matching (CPFs, plates)
dense_weight: 0.4    # Semantic similarity weight
```

### Recipe 4: Table-Aware Forensic Pipeline (`pdfplumber` + Tables)
```yaml
# configs/recipe_table_aware.yaml
name: table_aware_financial_pipeline
pipeline_type: standard
parser_type: pdfplumber  # Extracts structured [TABLE DATA] blocks from financial reports
chunker_type: recursive
chunk_size: 1200         # Larger chunk size to avoid splitting tabular records
chunk_overlap: 250
embedding_type: openai
vector_store_type: qdrant
generator_type: openai
judge_type: openai
top_k: 6
collection_name: police_reports
distance_metric: cosine
```

### Recipe 5: Local On-Premises Pipeline (Ollama / LocalAI / vLLM)
```yaml
# configs/recipe_local_onprem.yaml
name: local_onprem_confidential
pipeline_type: hybrid
parser_type: pymupdf
chunker_type: recursive
chunk_size: 1000
chunk_overlap: 200
embedding_type: openai
vector_store_type: qdrant
generator_type: openai
judge_type: openai
top_k: 5
collection_name: police_reports
distance_metric: cosine

# Endpoints pointing to local OpenAI-compatible inference servers
generator_base_url: "http://host.docker.internal:11434/v1"
generator_model: "llama3.1:8b"

judge_base_url: "http://host.docker.internal:11434/v1"
judge_model: "llama3.1:8b"

embedding_base_url: "http://host.docker.internal:11434/v1"
embedding_model: "nomic-embed-text"
```

### Recipe 6: Decoupled Multi-Provider Architecture
```yaml
# configs/recipe_decoupled_endpoints.yaml
name: decoupled_multi_provider
pipeline_type: standard
parser_type: pymupdf
chunker_type: recursive
chunk_size: 1000
chunk_overlap: 200
embedding_type: openai
vector_store_type: qdrant
generator_type: openai
judge_type: openai
top_k: 5
collection_name: police_reports
distance_metric: cosine

# Fast/low-cost inference for query generation
generator_base_url: "https://api.groq.com/openai/v1"
generator_model: "llama-3.3-70b-versatile"

# High-capability auditor model for LLM-as-a-Judge
judge_base_url: "https://api.openai.com/v1"
judge_model: "gpt-4o"

# High-dimensional embeddings
embedding_base_url: "https://api.openai.com/v1"
embedding_model: "text-embedding-3-large"
```

---

## 6. Execution Guide (via Docker)

All operations should be run via Docker without local host dependencies:

### 1. Ingest Documents
Parse PDF reports and verify character and page extraction:
```bash
docker compose run --rm app python -m bench_rag.cli ingest \
    --input-dir data/raw_reports \
    --parser-type pymupdf
```

### 2. Index Documents into Vector Store
Parse, chunk, embed, and index documents using a target pipeline configuration:
```bash
docker compose run --rm app python -m bench_rag.cli index \
    --input-dir data/raw_reports \
    --pipeline-config configs/pipeline_default.yaml
```

### 3. Evaluate a Single Pipeline
Run benchmark queries from `ground_truth.csv` and export reports:
```bash
docker compose run --rm app python -m bench_rag.cli evaluate \
    --benchmark-csv data/ground_truth.csv \
    --pipeline-config configs/pipeline_default.yaml \
    --output-dir reports/run_default \
    --top-k 5
```

### 4. Compare Two Pipeline Configurations Head-to-Head
Evaluate both pipelines under identical ground-truth conditions and generate comparative delta reports:
```bash
docker compose run --rm app python -m bench_rag.cli compare \
    --benchmark-csv data/ground_truth.csv \
    --config-a configs/pipeline_default.yaml \
    --config-b configs/pipeline_hybrid.yaml \
    --output-dir reports/comparison_run \
    --top-k 5
```

---

## 7. Related Documentation

- [Methodology & Metric Formulations](file:///home/filipe/projetos/bench-rag/docs/methodology.md): Detailed mathematical definitions of Precision@K, Recall@K, Hit Rate, MRR, ROUGE-L, Token F1, and LLM-as-a-Judge criteria.
- [Project Readme](file:///home/filipe/projetos/bench-rag/README.md): Quickstart guide, Docker setup, and project structure.
