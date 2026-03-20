import streamlit as st
from sentence_transformers import SentenceTransformer, util

# 1. Singleton Loader with Caching (Prevents memory bloat)
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

def compute_relevance_embedding_score(query: str, paper: dict, query_embedding=None) -> float:

    model = load_embedding_model()
    if not model:
        return 0.0
    text = (paper.get("title", "") + " " + paper.get("abstract", ""))
    if not text.strip():
        return 0.0
    # Use pre-computed query embedding if provided
    if query_embedding is None:
        query_embedding = model.encode(query, convert_to_tensor=True)
    text_emb = model.encode(text, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(query_embedding, text_emb)[0][0])

