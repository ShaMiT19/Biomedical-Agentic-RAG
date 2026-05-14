"""
app.py
Streamlit interface for the LLM-Powered Biomedical RAG System.
"""

import streamlit as st
from graph import workflow, RAGState

st.set_page_config(
    page_title="Biomedical Evidence Synthesis",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 LLM-Powered Biomedical RAG System")
st.markdown("Evidence synthesis over PubMed pathology literature using hybrid retrieval and agentic reasoning.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system retrieves and synthesizes evidence from PubMed pathology abstracts using:
    - **Hybrid Retrieval**: FAISS dense + BM25 sparse search fused with RRF
    - **HyDE**: Hypothetical Document Embedding for query expansion
    - **Cross-Encoder Reranking**: ms-marco-MiniLM for precision ranking
    - **Agentic Loop**: LangGraph with query rewriting and hallucination checking
    - **Corrective RAG**: Automatic fallback when retrieved docs are irrelevant
    """)

    st.header("Example Questions")
    examples = [
        "What deep learning methods are used for glomeruli segmentation in kidney pathology?",
        "How does spatial transcriptomics improve digital pathology analysis?",
        "What are the latest approaches for tumor microenvironment characterization?",
        "How are vision transformers applied to whole-slide image analysis?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["question_input"] = ex

# ── Main input ────────────────────────────────────────────────────────────────
question = st.text_area(
    "Ask a biomedical or pathology research question:",
    value=st.session_state.get("question_input", ""),
    height=100,
    placeholder="e.g. What methods are used for nuclei segmentation in H&E images?"
)

run = st.button("🔍 Search & Synthesize", type="primary")

if run and question.strip():
    with st.spinner("Running agentic retrieval pipeline..."):
        state = RAGState(question=question.strip())
        result = workflow.invoke(state)

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("📄 Synthesized Answer")
    st.markdown(result.get("answer", "No answer generated."))

    # ── Citations ─────────────────────────────────────────────────────────────
    citations = result.get("citations", [])
    if citations:
        st.subheader("📚 Sources")
        for cite in citations:
            if cite and cite != "Unknown":
                if cite.isdigit():
                    st.markdown(f"- [PubMed {cite}](https://pubmed.ncbi.nlm.nih.gov/{cite}/)")
                else:
                    st.markdown(f"- {cite}")

    # ── Retrieved documents ───────────────────────────────────────────────────
    docs = result.get("documents", [])
    if docs:
        with st.expander(f"📑 Retrieved Chunks ({len(docs)} documents)"):
            for i, doc in enumerate(docs, 1):
                pmid = doc.metadata.get("pmid", "")
                title = doc.metadata.get("title", "")
                st.markdown(f"**[{i}] {title}** {'(PMID: ' + pmid + ')' if pmid else ''}")
                st.markdown(doc.page_content[:400] + "...")
                st.divider()

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    with st.expander("⚙️ Pipeline Details"):
        st.markdown(f"**Retrieval Verdict:** {result.get('retrieval_verdict', 'N/A')}")
        st.markdown(f"**Generation Attempts:** {result.get('generation_count', 1)}")
        st.markdown(f"**Hallucination Check Passed:** {not result.get('hallucination_flag', False)}")
        rewritten = result.get("rewritten_question", "")
        if rewritten and rewritten != question:
            st.markdown(f"**Rewritten Query:** {rewritten}")

elif run and not question.strip():
    st.warning("Please enter a question.")
