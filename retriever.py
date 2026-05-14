"""
retriever.py
Hybrid retriever combining:
  - Dense: FAISS + OpenAI embeddings
  - Sparse: BM25 (rank-bm25)
  - Fusion: Reciprocal Rank Fusion (RRF)
  - Query expansion: HyDE (Hypothetical Document Embeddings)
  - Reranking: cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
"""

import os
from typing import Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Cross-encoder for reranking
CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

RRF_K = 60          # RRF constant
TOP_K_DENSE = 20    # candidates from dense retrieval
TOP_K_SPARSE = 20   # candidates from BM25
TOP_K_FINAL = 5     # after reranking


# ── HyDE ─────────────────────────────────────────────────────────────────────

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """You are a biomedical research expert. 
Write a concise, factual paragraph that would appear in a PubMed abstract 
answering the following question. Do not add disclaimers.

Question: {question}

Hypothetical abstract paragraph:"""
)


def hyde_expand(question: str) -> str:
    """Generate a hypothetical answer to expand the query vector."""
    chain = HYDE_PROMPT | LLM
    return chain.invoke({"question": question}).content


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def dense_retrieve(query: str, faiss_index: FAISS, k: int = TOP_K_DENSE) -> list[Document]:
    return faiss_index.similarity_search(query, k=k)


def sparse_retrieve(query: str, bm25: BM25Okapi, chunks: list[Document], k: int = TOP_K_SPARSE) -> list[Document]:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices]


# ── RRF ───────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = RRF_K
) -> list[Document]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Score = sum of 1 / (k + rank) across both lists.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs, start=1):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


# ── Cross-encoder reranking ───────────────────────────────────────────────────

def rerank(query: str, docs: list[Document], top_n: int = TOP_K_FINAL) -> list[Document]:
    """Rerank fused candidates with a cross-encoder."""
    if not docs:
        return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = CROSS_ENCODER.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


# ── Public interface ──────────────────────────────────────────────────────────

def hybrid_retrieve(
    question: str,
    faiss_index: FAISS,
    bm25: BM25Okapi,
    chunks: list[Document],
    use_hyde: bool = True,
) -> list[Document]:
    """
    Full hybrid retrieval pipeline:
      1. Optionally expand query with HyDE
      2. Dense retrieval (FAISS)
      3. Sparse retrieval (BM25)
      4. RRF fusion
      5. Cross-encoder reranking
    Returns top-k reranked documents.
    """
    # Step 1: HyDE query expansion
    retrieval_query = hyde_expand(question) if use_hyde else question

    # Step 2 & 3: Parallel retrieval
    dense_docs = dense_retrieve(retrieval_query, faiss_index)
    sparse_docs = sparse_retrieve(retrieval_query, bm25, chunks)

    # Step 4: RRF fusion
    fused = reciprocal_rank_fusion(dense_docs, sparse_docs)

    # Step 5: Cross-encoder reranking
    final_docs = rerank(question, fused)

    return final_docs
