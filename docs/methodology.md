# Evaluation Methodology & Experimental Protocol

This document outlines the theoretical foundation, mathematical metric formulations, investigative taxonomies, and experimental evaluation protocol implemented in the **Modular RAG Evaluation Framework for Criminal Investigation Reports**.

---

## 1. Investigative Taxonomies

Evaluating RAG architectures on forensic police investigation reports requires specialized categorization because criminal reports differ significantly from generic web text. They contain formal police structures (Portarias, Autos de Apreensão, Relatórios de Missão), legal qualifications, financial layering, and telecommunication intercepts.

### 1.1 Typology Categories (T)
The benchmark partitions queries into three distinct investigative typologies:

- **`T1` — Identification & Associations**:
  - Focuses on uncovering beneficial owners (*proprietários de fato*), nominee strawmen (*laranjas*), corporate registration data (tax IDs, equity distribution), shell corporations, command structures, and family/business association trees.
  - *Example*: Identifying that a nominee earning minimum wage legally owns US$ 14M in corporate equity via an undisclosed power of attorney held by the real mastermind.

- **`T2` — Criminal Dynamics & Modus Operandi**:
  - Focuses on mechanics of illicit schemes, money laundering typologies (smurfing cash deposits below reporting limits, trade-based over-invoicing, cryptocurrency mixers), concealment maneuvers, and criminal slang.
  - *Example*: Analyzing how cash is structured into deposits below US$ 10,000 across multiple bank branches and converted into over-invoiced industrial import machinery.

- **`T3` — Tracking & Chronology**:
  - Focuses on reconstructed timelines, sequential events, international SWIFT bank wire progressions, and telecommunication intercept sequences.
  - *Example*: Tracking capital displaced from a primary shell entity across offshore banks to the ultimate purchase of luxury vessels, and correlating that timeline with intercepted orders to destroy evidence.

---

### 1.2 Complexity Levels (C)

- **`C1` — Low / Factual**:
  - Literal, single-document information extraction. The target fact (e.g. suspect's tax ID, registered company address, wire date) exists verbatim in a single page/passage.
  - *Goal*: Tests basic retrieval precision and factual extraction fidelity.

- **`C2` — Medium / Relational**:
  - Cross-document relational aggregation. Answering the query requires synthesizing facts located across multiple separate inquiry volumes or distinct investigation reports.
  - *Goal*: Tests retrieval recall, exhaustiveness, and multi-hop reasoning.

- **`C3` — High / Analytical**:
  - Broad inferential synthesis concerning criminal intent (*mens rea* / *dolo*), conscious participation in unlawful schemes, or top-level organizational command. Facts must be correlated with behavioral evidence (e.g., ordering records burned upon learning of warrants).
  - *Goal*: Tests high-level reasoning, contextual completeness, and resistance to hallucination.

---

## 2. Metric Formulations

### 2.1 Retrieval Metrics
Let $C_k = [c_1, c_2, \dots, c_m]$ (where $m \le k$) be the ranked list of retrieved chunks at cutoff $k$, let $\text{doc}(c_i)$ be the normalized source inquiry document identifier of chunk $c_i$, and let $G = \{g_1, g_2, \dots, g_{|G|}\}$ be the ground-truth set of cited inquiry references (`inquiry_references`).

#### Precision@k (Chunk-Level Signal-to-Noise Ratio)
Measures the fraction of the top-$k$ cutoff window occupied by retrieved chunks that originate from a relevant ground-truth inquiry document (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**):
$$
\text{Precision}@k = \frac{|\{ i \in \{1, \dots, m\} \mid \text{doc}(c_i) \in G \}|}{\max(1, k)}
$$

#### Recall@k (Document-Level Ground-Truth Coverage)
Measures the fraction of unique ground-truth inquiry documents that are represented by at least one chunk in $C_k$ (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**):
$$
\text{Recall}@k = \frac{|\{ \text{doc}(c_i) \mid c_i \in C_k \} \cap G|}{|G|}
$$

#### Hit Rate@k
Binary indicator confirming baseline evidentiary retrieval success (**Range: $[0.0, 1.0]$ (binary $\{0, 1\}$ per query) — Closer to $1.0$ is better**):
$$
\text{HitRate}@k = \begin{cases} 1 & \text{if } \exists c_i \in C_k \text{ s.t. } \text{doc}(c_i) \in G \\ 0 & \text{otherwise} \end{cases}
$$

#### Mean Reciprocal Rank (MRR)
Measures the reciprocal of the 1-indexed rank of the first retrieved chunk belonging to a ground-truth document (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**):
$$
\text{MRR} = \begin{cases} \frac{1}{\min \{ i \in \{1, \dots, m\} \mid \text{doc}(c_i) \in G \}} & \text{if } \exists c_i \in C_k \text{ s.t. } \text{doc}(c_i) \in G \\ 0 & \text{otherwise} \end{cases}
$$

---

### 2.2 Citation Attribution Metrics
Let $D_{\text{cited}}$ be the set of unique normalized inquiry identifiers explicitly cited in the generator's response, $G$ be the ground-truth reference set, and $D_{\text{ret}} = \{ \text{doc}(c_i) \mid c_i \in C_k \}$ be the set of unique inquiry documents present in the retrieved context:

1. **Citation Precision**:
   Proportion of inquiry documents cited in the model's answer that are present in the ground-truth references $G$: $\frac{|D_{\text{cited}} \cap G|}{|D_{\text{cited}}|}$ (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**).
2. **Citation Recall**:
   Coverage of target ground-truth reference documents explicitly cited in the generated answer: $\frac{|D_{\text{cited}} \cap G|}{|G|}$ (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**).
3. **Citation Hallucination Rate**:
   Proportion of cited documents that **were not present in the retrieved context**: $\frac{|D_{\text{cited}} \setminus D_{\text{ret}}|}{|D_{\text{cited}}|}$, identifying critical forensic source hallucinations (**Range: $[0.0, 1.0]$ — Closer to $0.0$ is better**).

---

### 2.3 Lexical Generation Metrics
Let $C_{\text{cand}}$ be the generated answer tokens and $C_{\text{ref}}$ be the ground-truth reference answer tokens, extracted using a forensic-aware tokenizer that preserves compound numeric and document identifiers (`\d+(?:[.,/-]\d+)*|\w+`).

#### Token F1
Harmonic mean of token-level precision and recall (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**):
$$
\text{Precision}_{\text{tok}} = \frac{|C_{\text{cand}} \cap C_{\text{ref}}|}{|C_{\text{cand}}|}, \quad \text{Recall}_{\text{tok}} = \frac{|C_{\text{cand}} \cap C_{\text{ref}}|}{|C_{\text{ref}}|}
$$

$$
F_1 = \frac{2 \cdot \text{Precision}_{\text{tok}} \cdot \text{Recall}_{\text{tok}}}{\text{Precision}_{\text{tok}} + \text{Recall}_{\text{tok}}}
$$

#### ROUGE-L (Longest Common Subsequence — Symmetric Harmonic F1)
Symmetric harmonic F1 score based on the Longest Common Subsequence (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**):
$$
R_{\text{LCS}} = \frac{\text{LCS}(C_{\text{cand}}, C_{\text{ref}})}{|C_{\text{ref}}|}, \quad P_{\text{LCS}} = \frac{\text{LCS}(C_{\text{cand}}, C_{\text{ref}})}{|C_{\text{cand}}|}
$$
$$
\text{ROUGE-L} = \frac{2 \cdot R_{\text{LCS}} \cdot P_{\text{LCS}}}{R_{\text{LCS}} + P_{\text{LCS}}} \quad (\beta = 1.0)
$$

#### BLEU-4
Let $N = \min(4, |C_{\text{cand}}|, |C_{\text{ref}}|)$ and uniform weights $w_n = 1/N$ (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**):
$$
\text{BLEU-4} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \ln p_n \right)
$$
where $\text{BP} = \min\left(1, e^{1 - \frac{|C_{\text{ref}}|}{|C_{\text{cand}}|}}\right)$, $p_1$ is exact unigram precision, and higher-order $p_n$ ($n > 1$) uses exact $n$-gram precision when overlap $> 0$ and add-1 denominator smoothing $\frac{1}{|G_n(C_{\text{cand}})| + 1}$ when overlap $= 0$.

---

### 2.4 LLM-as-a-Judge Forensic Metrics
A dedicated Judge model evaluates the pipeline output on a continuous scale $[0.0, 1.0]$ across four dimensions:

1. **Faithfulness (Groundedness)**:
   Measures whether every factual claim in the *Generated Answer* is substantiated by the *Retrieved Context*. Penalizes forensic hallucinations (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**).
2. **Answer Relevance**:
   Measures whether the *Generated Answer* directly addresses the specific *Investigative Question* asked, without evasion or extraneous fluff (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**).
3. **Context Recall**:
   Measures whether the key factual points present in the *Reference Answer* are present in the *Retrieved Context*, evaluating retrieval sufficiency (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**).
4. **Answer Correctness**:
   Measures whether the *Generated Answer* accurately and completely captures the key factual points of the *Reference Answer*, evaluating end-to-end generation accuracy (**Range: $[0.0, 1.0]$ — Closer to $1.0$ is better**).

---

## 3. RAG Architectures & Reciprocal Rank Fusion (RRF)

The framework benchmarks two primary RAG paradigms:

### 3.1 Standard Dense RAG
1. Encodes query into dense vector $v_q = \text{Embed}(q)$.
2. Retrieves top-$k$ nearest chunks in embedding space via cosine similarity:

$$
\text{sim}(v_q, v_c) = \frac{v_q \cdot v_c}{\|v_q\| \|v_c\|}
$$

3. Synthesizes answer using context evidence.

### 3.2 Hybrid RAG with Reciprocal Rank Fusion (RRF)
Combines dense semantic vector retrieval with BM25 Okapi lexical matching. Dense retrieval excels at conceptual synonyms (e.g. *nominee* vs *strawman*), while BM25 excels at exact keyword matching (e.g., tax IDs, license plates, bank account numbers).

For candidate chunk $c$, given rank $r_{\text{dense}}(c)$ in vector search and rank $r_{\text{bm25}}(c)$ in BM25:

$$
\text{RRF}(c) = \frac{w_{\text{dense}}}{k_0 + r_{\text{dense}}(c)} + \frac{w_{\text{bm25}}}{k_0 + r_{\text{bm25}}(c)}
$$

where $k_0 = 60$ is the standard smoothing constant, and $w_{\text{dense}} = w_{\text{bm25}} = 0.5$ by default.

### 3.3 Two-Stage Retrieval with Cross-Encoder Reranking (Optional)
To overcome the bi-encoder semantic compression bottleneck and the purely positional heuristic of RRF, both Standard and Hybrid pipelines optionally execute a second-stage cross-encoder re-ranking pass:

1. **Stage 1 (Coarse Retrieval)**: Retrieves an expanded candidate pool of size $K_{\text{fetch}} = \max(k_{\text{fetch}}, k)$ (typically 20-30 chunks, configured via `rerank_fetch_k`) via dense vector similarity or hybrid RRF fusion.
2. **Stage 2 (Fine Re-ranking)**: The candidate chunks and query $q$ are jointly evaluated with full cross-attention by a dedicated reranker model (such as `BAAI/bge-reranker-v2-m3` running on vLLM):

$$
s(q, c) = \text{CrossEncoder}(q, c)
$$

   Chunks are sorted by descending $s(q, c)$, and the top-$k$ (e.g., $k=5$) highest-scoring chunks are delivered to the generator LLM prompt. The initial retrieval rank and score are preserved in the chunk metadata for displacement auditing.

#### Formulation Matrix: Candidate Pool ($K_{\text{fetch}}$) vs. Final Cutoff ($k$)

| Architectural Mode | Reranker State | Stage 1 Candidate Pool ($K_{\text{fetch}}$) | Stage 2 / Final Context Size ($k$) |
| :--- | :---: | :--- | :--- |
| **Standard Dense** | **Disabled** | $K_{\text{fetch}} = k$ (e.g. 5) | $k$ (direct vector similarity slice) |
| **Standard Dense** | **Enabled** | $K_{\text{fetch}} = \max(k_{\text{fetch}}, k)$ (e.g. 20 via `rerank_fetch_k`) | $k$ (top-$k$ cross-encoder scores) |
| **Hybrid RRF** | **Disabled** | $\max(2k, 10)$ dense + $\max(2k, 10)$ BM25 | $k$ (direct RRF reciprocal rank slice) |
| **Hybrid RRF** | **Enabled** | ($K_{\text{fetch}}$ dense + $K_{\text{fetch}}$ BM25) $\to$ RRF $\to K_{\text{fetch}}$ | $k$ (top-$k$ cross-encoder scores) |

---

## 4. Experimental Protocol
1. **Document Ingestion**: Cleanly parse official forensic PDFs, extracting text, page numbering, and metadata.
2. **Indexing**: Partition documents into overlapping chunks, embed them, and index them into Qdrant.
3. **Inference Execution**: Query the candidate pipeline across all rows in `ground_truth.csv`.
4. **Metric Aggregation**: Aggregate scores globally, across typologies ($T1, T2, T3$), across complexities ($C1, C2, C3$), and across the complete $3 \times 3$ taxonomy matrix.
5. **Report Generation**: Export Markdown, JSON, and CSV run artifacts.
