"""
evaluate.py
Evaluates the RAG pipeline using RAGAS metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Recall (requires reference answers)

Usage:
    python evaluate.py --n_questions 50
"""

import os
import json
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from graph import workflow, RAGState
from ingest import load_indexes

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))


# ── Question generation ───────────────────────────────────────────────────────

QGEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a biomedical research expert. 
Read the following abstract and generate {n} specific, answerable questions 
that can be answered using only this abstract. 
Return questions as a JSON list of strings.

Abstract:
{abstract}

Questions (JSON list):"""
)


def generate_questions_from_chunks(
    chunks: list[Document],
    n_questions: int = 50,
    questions_per_chunk: int = 2
) -> list[dict]:
    """Generate question-context pairs from indexed chunks."""
    qa_pairs = []
    import random
    sample = random.sample(chunks, min(n_questions // questions_per_chunk, len(chunks)))

    for chunk in sample:
        try:
            chain = QGEN_PROMPT | LLM
            result = chain.invoke({
                "abstract": chunk.page_content[:800],
                "n": questions_per_chunk
            })
            raw = result.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            questions = json.loads(raw)
            for q in questions:
                qa_pairs.append({
                    "question": q,
                    "reference_context": chunk.page_content
                })
        except Exception as e:
            print(f"  Skipping chunk (question gen failed): {e}")
            continue

    return qa_pairs[:n_questions]


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline_on_questions(qa_pairs: list[dict]) -> list[dict]:
    """Run the full RAG pipeline on each question and collect results."""
    results = []
    for i, pair in enumerate(qa_pairs):
        print(f"  Running question {i+1}/{len(qa_pairs)}: {pair['question'][:60]}...")
        try:
            state = RAGState(question=pair["question"])
            result = workflow.invoke(state)
            results.append({
                "question": pair["question"],
                "answer": result.get("answer", ""),
                "contexts": [doc.page_content for doc in result.get("documents", [])],
                "ground_truth": pair.get("reference_context", "")
            })
        except Exception as e:
            print(f"  Pipeline failed for question: {e}")
            continue
    return results


# ── Baseline runner (naive cosine only) ───────────────────────────────────────

def run_baseline_on_questions(
    qa_pairs: list[dict],
    faiss_index,
    n_retrieve: int = 5
) -> list[dict]:
    """Run naive cosine similarity retrieval as baseline for comparison."""
    results = []
    for pair in qa_pairs:
        try:
            docs = faiss_index.similarity_search(pair["question"], k=n_retrieve)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = ChatPromptTemplate.from_template(
                "Use the context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
            chain = prompt | LLM
            answer = chain.invoke({"context": context, "question": pair["question"]}).content
            results.append({
                "question": pair["question"],
                "answer": answer,
                "contexts": [doc.page_content for doc in docs],
                "ground_truth": pair.get("reference_context", "")
            })
        except Exception as e:
            print(f"  Baseline failed: {e}")
            continue
    return results


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas(results: list[dict], label: str) -> dict:
    """Run RAGAS evaluation and print results."""
    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    print(f"\n{'='*50}")
    print(f"RAGAS Results — {label}")
    print(f"{'='*50}")
    for k, v in scores.items():
        print(f"  {k:<25}: {v:.4f}")
    return dict(scores)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    print("Loading indexes...")
    faiss_index, bm25, chunks = load_indexes()

    print(f"Generating {args.n_questions} questions from indexed chunks...")
    qa_pairs = generate_questions_from_chunks(chunks, n_questions=args.n_questions)
    print(f"Generated {len(qa_pairs)} question-context pairs.")

    print("\nRunning full hybrid RAG pipeline...")
    hybrid_results = run_pipeline_on_questions(qa_pairs)

    print("\nRunning naive cosine baseline...")
    baseline_results = run_baseline_on_questions(qa_pairs, faiss_index)

    print("\nEvaluating with RAGAS...")
    hybrid_scores = run_ragas(hybrid_results, "Hybrid RAG (HyDE + BM25 + RRF + Reranker)")
    baseline_scores = run_ragas(baseline_results, "Baseline (Naive Cosine Similarity)")

    # Save results
    output = {
        "n_questions": len(qa_pairs),
        "hybrid_scores": hybrid_scores,
        "baseline_scores": baseline_scores,
        "improvement": {
            k: round(hybrid_scores.get(k, 0) - baseline_scores.get(k, 0), 4)
            for k in hybrid_scores
        }
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print("\nAnswer Relevancy Improvement:")
    print(f"  Baseline : {baseline_scores.get('answer_relevancy', 0):.4f}")
    print(f"  Hybrid   : {hybrid_scores.get('answer_relevancy', 0):.4f}")
    print(f"  Delta    : +{output['improvement'].get('answer_relevancy', 0):.4f}")


if __name__ == "__main__":
    main()
