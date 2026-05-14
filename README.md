# LLM-Powered Biomedical RAG System — Evidence Synthesis over Pathology Literature

An advanced, production-grade Retrieval-Augmented Generation system for evidence synthesis over biomedical and pathology literature, built on **LangGraph**, **OpenAI**, and **PubMed**. Designed to support researchers and clinicians with grounded, citation-backed answers derived from peer-reviewed abstracts.

---

## Overview

This system implements an agentic reasoning pipeline that retrieves, ranks, and synthesizes evidence from PubMed pathology literature in response to biomedical queries. It combines hybrid dense-sparse retrieval with a LangGraph-orchestrated reasoning loop to deliver faithful, traceable responses with source-level attribution.

---

## Architecture

The system is powered by a **stateful LangGraph StateGraph** that orchestrates multi-step retrieval, relevance grading, query rewriting, and answer generation.

### Agentic Reasoning Loop
- **Query Router**: Classifies incoming queries and routes them to the retrieval pipeline or directly to the LLM for general knowledge questions
- **Corrective RAG (C-RAG)**: Grades retrieved documents for relevance — ambiguous or low-relevance results trigger automatic query rewriting and fallback web retrieval via TavilySearch
- **Answer Grader**: Evaluates generated answers for hallucination and groundedness before returning to the user
- **Tool Calling**: Agent can invoke the PubMed API directly for targeted literature lookup and a calculator tool for quantitative reasoning

### Hybrid Retrieval Pipeline
- **Dense Retrieval**: FAISS vector store with OpenAI embeddings for semantic similarity search over PubMed pathology abstracts
- **Sparse Retrieval**: BM25 keyword index over the same corpus for exact term matching
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse retrieval rankings into a single unified ranked list
- **HyDE Query Expansion**: Generates a hypothetical answer before embedding the query, improving retrieval recall for complex biomedical questions
- **Cross-Encoder Reranking**: Re-scores retrieved chunks against the original query for final precision ranking

### Evaluation
- **RAGAS Framework**: Pipeline evaluated on faithfulness and context recall against a held-out question set derived from pathology literature
- **Baseline Comparison**: Hybrid retrieval + HyDE evaluated against naive cosine similarity baseline

---

## Core Components

### 1. Document Ingestion (`ingest.py`)
- Fetches PubMed abstracts using the E-Utilities API via `pymed`
- Targets computational pathology and digital pathology literature
- Chunks abstracts with sentence-aware boundaries and indexes into FAISS and BM25

### 2. Hybrid Retriever (`retriever.py`)
- Runs dense and sparse retrieval in parallel
- Merges results using RRF scoring
- Applies cross-encoder reranking on the fused candidate set

### 3. LangGraph Agent (`graph.py`)
- Defines the StateGraph with nodes for routing, retrieval, grading, rewriting, and generation
- Manages conditional edges based on relevance grades and hallucination checks
- Streams intermediate reasoning steps via SSE output

### 4. Evaluation (`evaluate.py`)
- Generates a question set from indexed abstracts
- Runs the full pipeline and baseline through RAGAS
- Reports faithfulness, context recall, and answer relevancy scores

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- API Keys: OpenAI, Tavily

### Installation

```bash
git clone https://github.com/ShaMiT19/LLM-Powered-Biomedical-RAG
cd LLM-Powered-Biomedical-RAG
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
# Add your API keys to .env
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### Ingest PubMed Data

```bash
python ingest.py --query "computational pathology" --max_results 1000
```

### Run the Application

```bash
streamlit run app.py
```

### Run Evaluation

```bash
python evaluate.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | OpenAI GPT-4o-mini |
| Dense Retrieval | FAISS + OpenAI Embeddings |
| Sparse Retrieval | BM25 (rank-bm25) |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| Literature Source | PubMed E-Utilities (pymed) |
| Web Fallback | TavilySearch |
| Evaluation | RAGAS |
| Frontend | Streamlit |

---

## Key Design Decisions

**Why hybrid retrieval?** Dense embeddings capture semantic similarity but miss exact terminology critical in biomedical contexts (gene names, procedure codes, drug names). BM25 handles exact term matching. RRF fusion consistently outperforms either approach alone.

**Why HyDE?** Biomedical queries are often short and underspecified. Generating a hypothetical answer before embedding the query produces a richer vector that retrieves more relevant abstracts than embedding the raw query directly.

**Why Corrective RAG?** PubMed retrieval can return topically adjacent but clinically irrelevant results. The relevance grading node prevents low-quality context from reaching the LLM, reducing hallucination risk on technical biomedical questions.

---

## Project Structure

```
├── ingest.py           # PubMed data fetching and indexing
├── retriever.py        # Hybrid BM25 + FAISS retriever with RRF
├── graph.py            # LangGraph StateGraph definition
├── evaluate.py         # RAGAS evaluation pipeline
├── app.py              # Streamlit interface
├── requirements.txt
└── .env.example
```

---

*Built as a research tool for evidence-based biomedical question answering. Not intended for clinical diagnosis or treatment decisions.*