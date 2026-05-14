"""
graph.py
LangGraph agentic RAG pipeline for biomedical evidence synthesis.

Flow:
  query → route_query
            ├─ "retrieve"  → retrieve → grade_documents
            │                    ├─ "generate"   → generate → grade_answer → END
            │                    └─ "rewrite"    → rewrite_query → retrieve (loop)
            └─ "direct_llm"  → generate → END
"""

import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from retriever import hybrid_retrieve
from ingest import load_indexes

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
TAVILY = TavilySearchResults(max_results=3) if os.getenv("TAVILY_API_KEY") else None

# Load indexes at startup
FAISS_INDEX, BM25, CHUNKS = load_indexes()


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(BaseModel):
    question: str = Field(..., description="User's biomedical question")
    rewritten_question: str = Field(default="", description="HyDE-expanded or rewritten query")
    documents: list[Document] = Field(default_factory=list, description="Retrieved documents")
    answer: str = Field(default="", description="Generated answer")
    citations: list[str] = Field(default_factory=list, description="Source PMIDs or URLs")
    generation_count: int = Field(default=0, description="Number of generation attempts")
    retrieval_verdict: str = Field(default="", description="CORRECT / AMBIGUOUS / INCORRECT")
    hallucination_flag: bool = Field(default=False, description="Whether answer is grounded")


# ── Structured outputs ────────────────────────────────────────────────────────

class RouteDecision(BaseModel):
    route: Literal["retrieve", "direct_llm"] = Field(
        description="'retrieve' for domain-specific questions, 'direct_llm' for general ones"
    )

class GradeDocuments(BaseModel):
    verdict: Literal["CORRECT", "AMBIGUOUS", "INCORRECT"] = Field(
        description="CORRECT if docs are relevant, INCORRECT if not, AMBIGUOUS if mixed"
    )

class GradeAnswer(BaseModel):
    grounded: bool = Field(description="True if answer is supported by the documents")


# ── Node functions ────────────────────────────────────────────────────────────

def route_query(state: RAGState) -> dict:
    """Decide whether to retrieve or answer directly from LLM."""
    prompt = ChatPromptTemplate.from_template(
        """You are a routing engine for a biomedical RAG system.
Decide whether the question requires retrieval from PubMed literature 
or can be answered directly from general LLM knowledge.

Route to 'retrieve' for: specific pathology findings, clinical studies, 
experimental results, recent research, domain-specific terminology.
Route to 'direct_llm' for: definitions, general concepts, simple factual questions.

Question: {question}"""
    )
    chain = prompt | LLM.with_structured_output(RouteDecision)
    result = chain.invoke({"question": state.question})
    return {"rewritten_question": state.question, "retrieval_verdict": result.route}


def retrieve(state: RAGState) -> dict:
    """Hybrid retrieve: FAISS + BM25 + RRF + cross-encoder reranking."""
    query = state.rewritten_question or state.question
    docs = hybrid_retrieve(query, FAISS_INDEX, BM25, CHUNKS, use_hyde=True)
    citations = [
        doc.metadata.get("pmid", doc.metadata.get("url", "Unknown"))
        for doc in docs
    ]
    return {"documents": docs, "citations": citations}


def grade_documents(state: RAGState) -> dict:
    """Grade retrieved documents for relevance."""
    prompt = ChatPromptTemplate.from_template(
        """You are a retrieval quality evaluator for a biomedical RAG system.
Assess whether the retrieved documents are relevant to the question.

Question: {question}

Documents:
{documents}

Return:
- CORRECT if documents are clearly relevant
- INCORRECT if documents are clearly irrelevant
- AMBIGUOUS if some are relevant and some are not"""
    )
    chain = prompt | LLM.with_structured_output(GradeDocuments)
    doc_text = "\n\n---\n\n".join(doc.page_content[:300] for doc in state.documents)
    result = chain.invoke({"question": state.question, "documents": doc_text})
    return {"retrieval_verdict": result.verdict}


def rewrite_query(state: RAGState) -> dict:
    """Rewrite the query when retrieved documents are irrelevant."""
    prompt = ChatPromptTemplate.from_template(
        """You are a biomedical query optimizer.
The current query did not retrieve relevant documents.
Rewrite it to be more specific and use biomedical terminology that 
would appear in PubMed abstracts.

Original query: {question}
Rewritten query:"""
    )
    chain = prompt | LLM
    result = chain.invoke({"question": state.question})
    return {"rewritten_question": result.content.strip()}


def generate(state: RAGState) -> dict:
    """Generate an answer grounded in retrieved documents."""
    if state.documents:
        context = "\n\n---\n\n".join(doc.page_content for doc in state.documents)
        prompt = ChatPromptTemplate.from_template(
            """You are a biomedical research assistant providing evidence-based answers.
Use ONLY the provided context to answer. If the context does not contain 
enough information, say so clearly. Do not fabricate information.

For each key claim, indicate which source supports it by referencing the PMID.

Context:
{context}

Question: {question}

Answer (with citations):"""
        )
        chain = prompt | LLM
        answer = chain.invoke({"context": context, "question": state.question}).content
    else:
        # Direct LLM path — no retrieval
        prompt = ChatPromptTemplate.from_template(
            """You are a knowledgeable biomedical assistant.
Answer the following question clearly and accurately.

Question: {question}

Answer:"""
        )
        chain = prompt | LLM
        answer = chain.invoke({"question": state.question}).content

    # Fallback: web search if answer is thin
    if len(answer.strip()) < 100 and TAVILY:
        web_results = TAVILY.run(state.question)
        if web_results:
            web_context = "\n".join(
                r.get("content", "") for r in web_results if isinstance(r, dict)
            )
            answer += f"\n\n[Supplementary web evidence]:\n{web_context}"

    return {
        "answer": answer,
        "generation_count": state.generation_count + 1
    }


def grade_answer(state: RAGState) -> dict:
    """Check whether the answer is grounded in the retrieved documents."""
    if not state.documents:
        return {"hallucination_flag": False}

    prompt = ChatPromptTemplate.from_template(
        """You are a hallucination detector for a biomedical RAG system.
Assess whether the answer is supported by the provided documents.
Return True if the answer is grounded, False if it contains unsupported claims.

Documents:
{documents}

Answer: {answer}"""
    )
    chain = prompt | LLM.with_structured_output(GradeAnswer)
    doc_text = "\n\n".join(doc.page_content[:300] for doc in state.documents)
    result = chain.invoke({"documents": doc_text, "answer": state.answer})
    return {"hallucination_flag": not result.grounded}


# ── Routing conditions ────────────────────────────────────────────────────────

def route_after_routing(state: RAGState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state.retrieval_verdict == "retrieve" else "generate"


def route_after_grading(state: RAGState) -> Literal["generate", "rewrite_query"]:
    if state.retrieval_verdict == "INCORRECT" and state.generation_count < 2:
        return "rewrite_query"
    return "generate"


def route_after_answer(state: RAGState) -> Literal["END", "rewrite_query"]:
    if state.hallucination_flag and state.generation_count < 2:
        return "rewrite_query"
    return "END"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("route_query", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)

    graph.add_edge(START, "route_query")
    graph.add_conditional_edges("route_query", route_after_routing, {
        "retrieve": "retrieve",
        "generate": "generate"
    })
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grading, {
        "generate": "generate",
        "rewrite_query": "rewrite_query"
    })
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "grade_answer")
    graph.add_conditional_edges("grade_answer", route_after_answer, {
        "END": END,
        "rewrite_query": "rewrite_query"
    })

    return graph.compile()


workflow = build_graph()
