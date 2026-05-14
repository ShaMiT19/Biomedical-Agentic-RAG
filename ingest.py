"""
ingest.py
Fetches PubMed abstracts on computational/digital pathology,
builds a FAISS dense index and a BM25 sparse index.
Run once before starting the app:
    python ingest.py --query "computational pathology" --max_results 500
"""

import os
import json
import pickle
import argparse
import requests
from xml.etree import ElementTree as ET

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
INDEX_DIR = "indexes"


# ── PubMed helpers ────────────────────────────────────────────────────────────

def pubmed_search(query: str, retmax: int = 500) -> list[str]:
    """Return a list of PubMed IDs for a query."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def pubmed_fetch(pmids: list[str]) -> str:
    """Fetch XML for a list of PubMed IDs."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_pubmed_xml(xml_text: str) -> list[Document]:
    """Parse PubMed XML into LangChain Documents."""
    root = ET.fromstring(xml_text)
    docs = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = article.findtext(".//ArticleTitle", default="")
        abstract = " ".join(
            t.text or "" for t in article.findall(".//AbstractText")
        )
        mesh = [
            m.text for m in article.findall(".//MeshHeading/DescriptorName")
        ]
        content = f"{title}\n\n{abstract}".strip()
        if content:
            docs.append(Document(
                page_content=content,
                metadata={"pmid": pmid, "title": title, "mesh": mesh, "source": "PubMed"}
            ))
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs)


# ── Index builders ────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[Document]) -> FAISS:
    print(f"Building FAISS index over {len(chunks)} chunks...")
    return FAISS.from_documents(chunks, EMBEDDINGS)


def build_bm25_index(chunks: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    print(f"Building BM25 index over {len(chunks)} chunks...")
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks


# ── Persistence ───────────────────────────────────────────────────────────────

def save_indexes(faiss_index: FAISS, bm25: BM25Okapi, chunks: list[Document]):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_index.save_local(os.path.join(INDEX_DIR, "faiss_index"))
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"Indexes saved to ./{INDEX_DIR}/")


def load_indexes() -> tuple[FAISS, BM25Okapi, list[Document]]:
    faiss_index = FAISS.load_local(
        os.path.join(INDEX_DIR, "faiss_index"),
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
        data = pickle.load(f)
    return faiss_index, data["bm25"], data["chunks"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="computational pathology digital pathology", type=str)
    parser.add_argument("--max_results", default=500, type=int)
    args = parser.parse_args()

    print(f"Searching PubMed: '{args.query}' (max {args.max_results})...")
    pmids = pubmed_search(args.query, retmax=args.max_results)
    print(f"Found {len(pmids)} articles. Fetching...")

    # Fetch in batches of 100 to avoid API limits
    all_docs = []
    batch_size = 100
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        xml = pubmed_fetch(batch)
        all_docs.extend(parse_pubmed_xml(xml))
        print(f"  Parsed {len(all_docs)} documents so far...")

    print(f"Total documents parsed: {len(all_docs)}")
    chunks = chunk_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")

    faiss_index = build_faiss_index(chunks)
    bm25, chunks = build_bm25_index(chunks)
    save_indexes(faiss_index, bm25, chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
