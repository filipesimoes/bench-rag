# Modular RAG Evaluation Framework for Criminal Investigation Reports

A production-grade, modular Python framework containerized with Docker and `docker-compose` to ingest, index, and evaluate multiple Retrieval-Augmented Generation (RAG) architectures against a ground-truth dataset of police investigation reports (PDF format).

> [!IMPORTANT]
> **Containerized Environment**: All development, execution, and testing must be performed via Docker. No packages, dependencies, or runtime tools are required on the host machine.

---

## Architecture Overview

The system adheres to Clean Architecture and Dependency Inversion principles. Core components are defined via abstract base classes and `typing.Protocol` interfaces, ensuring zero coupling:

- `DocumentParser`: Extracts clean text and metadata from forensic PDF reports. Includes high-speed PyMuPDF and layout/table-aware pdfplumber.
- `Chunker`: Generic recursive character chunker with configurable separators, chunk sizes, and overlaps, as well as token-based chunking.
- `EmbeddingModel`: OpenAI-compatible remote embedding endpoints (`/v1/embeddings`) and deterministic offline mock embeddings for testing.
- `VectorStore`: Vector database abstraction supporting containerized Qdrant and fast in-memory stores with Cosine, Euclidean, and Dot distance metrics.
- `GeneratorLLM`: Interface targeting OpenAI-compatible HTTP endpoints (`/v1/chat/completions`) with temperature and max token controls.
- `RAGPipeline`: End-to-end coordinator supporting **Standard Dense RAG** and **Hybrid RAG** (Dense Vector + BM25 Lexical with Reciprocal Rank Fusion).
- `EvaluationEngine`: Computes multi-dimensional metrics aggregated across investigative typologies ($T1, T2, T3$) and complexity levels ($C1, C2, C3$).

```text
                 Forensic Police Inquiry Reports (PDF)
                                │
                                ▼
                       DocumentParser (PyMuPDF / pdfplumber)
                                │
                                ▼
                       Chunker (Recursive Character)
                                │
                                ▼
              EmbeddingModel (OpenAI-compatible / Mock)
                                │
                                ▼
                     VectorStore (Qdrant / Memory)
                                │
     ┌──────────────────────────┴──────────────────────────┐
     ▼                                                     ▼
Standard Dense RAG Pipeline                           Hybrid RAG Pipeline
(Dense Vector Top-K)                                  (Dense + BM25 Okapi + RRF)
     │                                                     │
     └──────────────────────────┬──────────────────────────┘
                                │
                                ▼
                        GeneratorLLM
                                │
                                ▼
                     Evaluation Engine & Taxonomies
   (Precision@K, Recall@K, HitRate, MRR, ROUGE-L, Token F1, LLM-as-a-Judge)
                                │
                                ▼
                    Markdown / JSON / CSV Reports
```

---

## Evaluation Taxonomies

### Investigative Typologies ($T$)
- **`T1` (Identification & Associations)**: Authorship, nominee strawmen (*laranjas*), corporate ownership trees, beneficial ownership chains, command hierarchies.
- **`T2` (Criminal Dynamics)**: Modus operandi, money laundering mechanics (smurfing, over-invoicing, crypto mixers), criminal jargon, asset concealment tactics.
- **`T3` (Tracking & Chronology)**: Asset flow trails, banking wire sequences, chronological timeline of suspicious activities.

### Complexity Levels ($C$)
- **`C1` (Low / Factual)**: Single-document, literal fact extraction (e.g., account owner, license plate, company registration).
- **`C2` (Medium / Relational)**: Multi-document synthesis across disparate inquiry reports testing retrieval completeness.
- **`C3` (High / Analytical)**: Inferential analysis regarding criminal intent (*mens rea* / *dolo*), conspiracy awareness, or complex organizational structure.

---

## Quickstart Guide (via Docker)

### 1. Clone & Prepare Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(If you do not specify live API keys, the framework runs smoothly using deterministic offline mock models).*

You can configure distinct endpoints and credentials for each component in `.env` (or via pipeline YAML configs):
- **Generator LLM**: `GENERATOR_BASE_URL`, `GENERATOR_API_KEY`, `GENERATOR_MODEL`
- **Judge LLM**: `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`
- **Embedding Model**: `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`
- **Global Fallback**: `OPENAI_BASE_URL`, `OPENAI_API_KEY`

> [!TIP]
> For an exhaustive reference of all pipeline options, parameter descriptions, architectural variants, and ready-to-use YAML recipes, consult the [Pipeline Configuration Guide](docs/pipeline_configuration.md).

### 2. Build Containers
Build the multi-stage Docker development container:
```bash
docker compose build
```

### 3. Generate Sample Forensic Dataset
Generate 3 realistic synthetic forensic police inquiry reports (PDF) and the benchmark ground-truth CSV:
```bash
docker compose run --rm app python -m bench_rag.cli seed-sample-data
```
This generates:
- `data/raw_reports/IP_001_2023_FRAUD.pdf` (T1 focus: Shell corporations, nominees, beneficial owners)
- `data/raw_reports/IP_002_2023_LAUNDERING.pdf` (T2 focus: Smurfing, trade over-invoicing, crypto mixers)
- `data/raw_reports/IP_003_2023_CONSPIRACY.pdf` (T3 focus: Wire timelines, telecomm intercepts, intent)
- `data/ground_truth.csv` (Benchmark questions covering each pair of $(T1..T3) \times (C1..C3)$)

### 4. Ingest & Inspect Documents
Inspect metadata extracted from the forensic PDFs:
```bash
docker compose run --rm app python -m bench_rag.cli ingest --input-dir data/raw_reports
```

### 5. Start Qdrant & Index Documents
Start the containerized Qdrant vector database:
```bash
docker compose up -d qdrant
```
Index all documents:
```bash
docker compose run --rm app python -m bench_rag.cli index \
    --input-dir data/raw_reports \
    --pipeline-config configs/pipeline_default.yaml
```

### 6. Run Benchmark Evaluation
Evaluate a single pipeline (e.g. Standard Dense RAG):
```bash
docker compose run --rm app python -m bench_rag.cli evaluate \
    --benchmark-csv data/ground_truth.csv \
    --pipeline-config configs/pipeline_default.yaml \
    --input-dir data/raw_reports \
    --output-dir reports/run_default
```
Outputs generated in `reports/run_default/`:
- `evaluation_report.md`: Markdown report with global summary, typology breakdown, complexity breakdown, taxonomy matrix, and granular query results.
- `run_summary.json`: Complete machine-readable summary.
- `samples_evaluation.csv`: Tabular sample-level metrics.

### 7. Compare Standard vs. Hybrid RAG Architectures
Directly benchmark Standard Dense RAG vs. Hybrid RAG (Dense + BM25 + Reciprocal Rank Fusion):
```bash
docker compose run --rm app python -m bench_rag.cli compare \
    --benchmark-csv data/ground_truth.csv \
    --config-a configs/pipeline_default.yaml \
    --config-b configs/pipeline_hybrid.yaml \
    --input-dir data/raw_reports \
    --output-dir reports/comparison_run
```
Outputs:
- `reports/comparison_run/comparison_report.md`: Side-by-side head-to-head comparison table with absolute deltas across all retrieval, lexical, and LLM-judge metrics.

---

## Running the Automated Test Suite

Execute unit and integration tests inside Docker:
```bash
docker compose run --rm app pytest -v --cov=src/bench_rag
```

---

## Privacy, Confidentiality & Synthetic Testing Policy

> [!CAUTION]
> **Zero Real Data in Code and Tests**:
> It is strictly forbidden to hardcode, commit, or register in `src/`, `tests/`, `configs/`, or documentation any real personal names, real company names, real court document numbers (PJe IDs), CPFs, RGs, or specific case facts extracted from actual police inquiry reports.
> - All unit test mocks and fixtures must use synthetic placeholders (e.g., `SERVIDOR TESTE`, dummy IDs `0000...`).
> - Integration tests validating real files must test structural and quantitative invariants without hardcoded real-world entity assertions.
> - Automated enforcement is built into `tests/unit/test_privacy_compliance.py`.
> - For full governance rules, see [AGENTS.md](AGENTS.md) and [docs/privacy_guidelines.md](docs/privacy_guidelines.md).

---

## Project Structure

```text
bench-rag/
├── Dockerfile                  # Multi-stage Dockerfile (development, production)
├── docker-compose.yml          # App & Qdrant vector database orchestration
├── pyproject.toml              # Build config, dependencies, tools
├── .env.example                # Environment variables template
├── README.md                   # System documentation and user guide
├── docs/
│   ├── methodology.md          # Scientific methodology, metric formulas, taxonomies
│   └── pipeline_configuration.md # Exhaustive pipeline configuration guide and recipes
├── configs/
│   ├── pipeline_default.yaml   # Standard Dense RAG configuration
│   └── pipeline_hybrid.yaml    # Hybrid (Dense + BM25 RRF) configuration
├── data/
│   ├── raw_reports/            # Ingested forensic PDF investigation reports
│   ├── ground_truth.csv        # Benchmark queries, answers, and inquiry references
│   └── sample/
│       └── seed.py             # Synthetic forensic report & benchmark generator
├── src/bench_rag/
│   ├── core/                   # Clean Architecture Protocols & Dataclasses
│   │   ├── parser.py           # Document & DocumentParser Protocol
│   │   ├── chunker.py          # Chunk & Chunker Protocol
│   │   ├── embedding.py        # EmbeddingModel Protocol
│   │   ├── vector_store.py     # SearchResult, VectorStore Protocol, DistanceMetric
│   │   ├── generator.py        # GenerationResult & GeneratorLLM Protocol
│   │   ├── pipeline.py         # RAGResponse & RAGPipeline Protocol
│   │   └── evaluator.py        # Metric models, SampleEvaluation, EvaluationSummary
│   ├── parsers/                # PyMuPDF & pdfplumber implementations
│   ├── chunkers/               # RecursiveCharacterChunker & TokenChunker
│   ├── embeddings/             # OpenAI-compatible & Mock embeddings
│   ├── vector_stores/          # Qdrant client & In-Memory vector stores
│   ├── generators/             # OpenAI-compatible & Mock LLM generators
│   ├── pipelines/              # StandardDenseRAG & HybridRAG implementations
│   ├── evaluation/             # Metrics (retrieval, lexical, judge) & taxonomy
│   ├── reporting/              # Markdown, JSON, and CSV report generators
│   └── cli.py                  # Typer & Rich command-line application
└── tests/
    ├── unit/                   # Unit tests for all components
    └── integration/            # End-to-end pipeline and evaluation flow tests
```
