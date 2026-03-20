"""
clustering.py
=============
Paper clustering with UMAP + Ward + Silhouette.
Notebook-proven parameters: n_components=15, n_neighbors=20, min_dist=0.0

Cluster naming priority chain (config.py LABELLING.priority):
  A — Groq/Gemini LLM    (best quality, uses API)
  B — KeyBERT            (free, semantic keyphrases, needs keybert package)
  C — c-TF-IDF           (BERTopic-style, free, no extra packages)
  D — Keyword frequency  (always works, current fallback)
"""

import os
import re
import sys
import time
import math
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from embedding_utils import load_embedding_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import LABELLING, CLUSTERING
except ImportError:
    LABELLING   = {"priority": ["groq", "keybert", "ctfidf", "keyword"]}
    CLUSTERING  = {"umap_components": 15, "umap_neighbors": 20, "umap_min_dist": 0.0}

# ── Optional imports ────────────────────────────────────────────────
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# LABELLING CHAIN — A → B → C → D
# ─────────────────────────────────────────────────────────────────────

def _name_cluster(titles: List[str], abstracts: List[str]) -> str:
    """
    Try each labelling method in priority order.
    Returns the first one that succeeds.
    """
    priority = LABELLING.get("priority", ["groq", "keybert", "ctfidf", "keyword"])
    text     = " ".join(titles + [a[:200] for a in abstracts])

    for method in priority:
        try:
            if method == "groq":
                result = _name_via_groq(titles)
                if result:
                    return result

            elif method == "keybert":
                result = _name_via_keybert(text)
                if result:
                    return result

            elif method == "ctfidf":
                result = _name_via_ctfidf(titles, abstracts)
                if result:
                    return result

            elif method == "keyword":
                return _name_via_keywords(titles)

        except Exception as e:
            print(f"[Clustering] {method} labelling failed: {e}")
            continue

    return _name_via_keywords(titles)


# ── Option A — Groq LLM ──────────────────────────────────────────────

def _get_groq_client():
    """
    Cluster naming uses Groq specifically.
    Falls back to next method if unavailable.
    """
    if not GROQ_AVAILABLE:
        return None
    api_key = None
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _name_via_groq(titles: List[str]) -> Optional[str]:
    client = _get_groq_client()
    if not client:
        return None
    if not LABELLING.get("groq", {}).get("enabled", True):
        return None

    model       = LABELLING.get("groq", {}).get("model", "llama-3.1-8b-instant")
    sample      = "\n".join(f"- {t}" for t in titles[:5])
    prompt      = (
        "You are a research taxonomy expert. Given these paper titles, "
        "reply with ONLY a short 3-7 word research theme. No punctuation at end.\n\n"
        f"Titles:\n{sample}\n\nTheme:"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply only with the theme name."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3,
            max_tokens=20,
        )
        name = resp.choices[0].message.content.strip()
        name = re.sub(r'^[\"\'"]|[\"\'"]$', '', name).strip(" .")
        return name if name else None
    except Exception as e:
        print(f"[Clustering] Groq failed: {e}")
        if LABELLING.get("groq", {}).get("fallback_on_ratelimit", True):
            time.sleep(1)
        return None


# ── Option B — KeyBERT ───────────────────────────────────────────────

_keybert_model = None

def _name_via_keybert(text: str) -> Optional[str]:
    """
    Uses KeyBERT to extract the most representative keyphrase.
    Requires: pip install keybert
    """
    if not KEYBERT_AVAILABLE:
        return None
    if not LABELLING.get("keybert", {}).get("enabled", True):
        return None

    global _keybert_model
    if _keybert_model is None:
        try:
            _keybert_model = KeyBERT(model='all-MiniLM-L6-v2')
        except Exception:
            return None

    cfg      = LABELLING.get("keybert", {})
    ngram    = tuple(cfg.get("ngram_range", (2, 4)))
    top_n    = cfg.get("top_n", 1)
    diversity= cfg.get("diversity", 0.5)

    try:
        keywords = _keybert_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram,
            stop_words='english',
            top_n=top_n,
            diversity=diversity,
            use_mmr=True,
        )
        if keywords:
            return keywords[0][0].title()
    except Exception:
        pass
    return None


# ── Option C — c-TF-IDF (BERTopic style) ────────────────────────────

def _name_via_ctfidf(titles: List[str], abstracts: List[str]) -> Optional[str]:
    """
    Class-based TF-IDF: treats all text in this cluster as one 'document',
    compares against a background corpus of common academic words.
    Identifies terms that are distinctive for THIS cluster.
    No extra dependencies needed — uses sklearn's TfidfVectorizer.
    """
    # Combine titles (weighted 3x) + abstracts for this cluster
    cluster_text = " ".join(titles * 3 + [a[:300] for a in abstracts])

    # Background vocabulary: common academic English words to penalise
    background_words = {
        'study', 'paper', 'propose', 'present', 'method', 'approach',
        'result', 'show', 'model', 'system', 'framework', 'algorithm',
        'data', 'experiment', 'evaluation', 'performance', 'network',
        'learning', 'training', 'test', 'baseline', 'dataset', 'novel',
        'existing', 'improve', 'achieve', 'demonstrate', 'state', 'art',
        'problem', 'solution', 'task', 'work', 'research', 'using',
        'based', 'new', 'large', 'high', 'low', 'different', 'various'
    }

    try:
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
        )
        vectorizer.fit([cluster_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores  = vectorizer.transform([cluster_text]).toarray()[0]

        # Filter out background words and very short terms
        scored = []
        for term, score in zip(feature_names, tfidf_scores):
            words = term.split()
            if any(w in background_words for w in words):
                continue
            if len(term) < 5:
                continue
            scored.append((term, score))

        if not scored:
            return None

        # Take top phrase
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0].title()

    except Exception:
        return None


# ── Option D — Keyword frequency (always works) ─────────────────────

def _name_via_keywords(titles: List[str]) -> str:
    """Simple frequency-based fallback. Always works."""
    if not titles:
        return "Research Cluster"

    stop = {
        'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they',
        'have', 'been', 'were', 'using', 'based', 'approach', 'method', 'study',
        'analysis', 'research', 'paper', 'work', 'results', 'novel', 'deep', 'via',
        'towards', 'survey', 'review', 'efficient', 'effective', 'learning'
    }
    words = []
    for t in titles:
        for w in re.findall(r'\b[a-zA-Z]{4,}\b', t.lower()):
            if w not in stop:
                words.append(w)

    if not words:
        return "Research Cluster"

    top = [w.title() for w, _ in Counter(words).most_common(3)]
    return " & ".join(top)


# ─────────────────────────────────────────────────────────────────────
# MAIN CLUSTERER
# ─────────────────────────────────────────────────────────────────────

class PaperClusterer:
    """
    UMAP + Agglomerative Ward clustering with silhouette-based k selection.
    Notebook-proven parameters used throughout.
    """

    def __init__(self):
        self.model             = None
        self.embedding_available = False
        self.vectorizer        = TfidfVectorizer(
            max_features=1000, stop_words='english', ngram_range=(1, 2)
        )

        try:
            self.model = load_embedding_model()
            if self.model:
                self.embedding_available = True
        except Exception as e:
            print(f"[Clustering] Could not load embedding model: {e}")

    def cluster_papers(
        self,
        papers: List[Dict],
        n_clusters: Optional[int] = None
    ) -> Dict[int, Dict]:
        """
        Main entry point. Returns dict of {cluster_id: cluster_info}.
        """
        papers = [p for p in papers if p.get("accessibility") != "inaccessible"]

        if len(papers) < 2:
            return {0: {
                'name':            'All Papers',
                'description':     'Single cluster — only one paper found.',
                'papers':          papers,
                'paper_count':     len(papers),
                'avg_year':        0,
                'avg_citations':   0.0,
                'top_venues':      [],
                'silhouette_score': None,
                'start_here_idx':  0,
            }}

        texts          = self._build_texts(papers)
        raw_embeddings = self._get_embeddings(texts)
        embeddings     = self._apply_umap(raw_embeddings)

        if n_clusters is None:
            n_clusters, best_score = self._find_optimal_k(embeddings, papers)
        else:
            best_score = None

        labels = self._run_clustering(embeddings, n_clusters)

        for paper, label in zip(papers, labels):
            paper['cluster'] = int(label)

        clusters = {}
        for cid in range(n_clusters):
            cluster_papers = [p for p, lbl in zip(papers, labels) if lbl == cid]
            if cluster_papers:
                clusters[cid] = self._build_cluster_info(cid, cluster_papers, best_score)

        return clusters

    # ── Internal methods ─────────────────────────────────────────────

    def _build_texts(self, papers: List[Dict]) -> List[str]:
        texts = []
        for p in papers:
            parts = []
            if p.get('title'):
                parts.append(p['title'])
            if p.get('abstract'):
                parts.append(p['abstract'][:500])
            if p.get('venue'):
                parts.append(p['venue'])
            texts.append(' '.join(parts))
        return texts

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.embedding_available:
            try:
                return self.model.encode(texts, show_progress_bar=False)
            except Exception as e:
                print(f"[Clustering] Embedding failed: {e}. Using TF-IDF.")
        try:
            return self.vectorizer.fit_transform(texts).toarray()
        except Exception:
            return np.random.rand(len(texts), 100)

    def _apply_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Notebook-proven: n_components=15, n_neighbors=20, min_dist=0.0
        """
        if not UMAP_AVAILABLE:
            return embeddings

        n       = embeddings.shape[0]
        nc      = min(CLUSTERING.get("umap_components", 15), n - 1)
        nn      = min(CLUSTERING.get("umap_neighbors", 20), max(2, n - 1))
        md      = CLUSTERING.get("umap_min_dist", 0.0)
        metric  = CLUSTERING.get("umap_metric", "cosine")

        if nc < 2:
            return embeddings

        try:
            reducer = umap.UMAP(
                n_components=nc,
                n_neighbors=nn,
                min_dist=md,
                metric=metric,
                random_state=42
            )
            reduced = reducer.fit_transform(embeddings)
            print(f"[Clustering] UMAP: {embeddings.shape[1]}-dim → {nc}-dim")
            return reduced
        except Exception as e:
            print(f"[Clustering] UMAP failed: {e}")
            return embeddings

    def _find_optimal_k(self, embeddings: np.ndarray, papers: List[Dict]):
        """
        Adaptive k range from notebook:
          max_k = min(sqrt(N/2), abs_max_k, N//3)
        Ensures minimum 3 papers per cluster.
        """
        n      = len(papers)
        max_k  = min(
            int(math.sqrt(n / 2)),
            CLUSTERING.get("abs_max_k", 15),
            n // 3
        )
        max_k  = max(max_k, 2)
        min_k  = CLUSTERING.get("min_k", 2)

        best_k, best_score = min_k, -1.0

        for k in range(min_k, max_k + 1):
            try:
                labels = AgglomerativeClustering(
                    n_clusters=k, linkage='ward'
                ).fit_predict(embeddings)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(embeddings, labels)
                print(f"[Clustering] k={k}, silhouette={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k     = k
            except Exception as e:
                print(f"[Clustering] k={k} error: {e}")

        print(f"[Clustering] Best k={best_k}, score={best_score:.4f}")
        return best_k, best_score

    def _run_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        if embeddings.shape[0] <= n_clusters:
            return np.arange(len(embeddings))
        try:
            return AgglomerativeClustering(
                n_clusters=n_clusters, linkage='ward'
            ).fit_predict(embeddings)
        except Exception as e:
            print(f"[Clustering] Clustering failed: {e}")
            return np.zeros(len(embeddings), dtype=int)

    def _build_cluster_info(
        self,
        cluster_id: int,
        papers: List[Dict],
        silhouette: Optional[float]
    ) -> Dict:
        titles    = [p['title'] for p in papers if p.get('title')]
        abstracts = [p.get('abstract', '') for p in papers]
        venues    = [p.get('venue', '') for p in papers if p.get('venue')]
        years     = [p.get('year') for p in papers if p.get('year')]

        cluster_name = _name_cluster(titles, abstracts)

        if years:
            yr = f"{min(years)}–{max(years)}" if min(years) != max(years) else str(min(years))
        else:
            yr = "recent years"

        description = f"{len(papers)} papers spanning {yr}, theme: {cluster_name}."
        if silhouette is not None:
            description += f" (Silhouette: {silhouette:.2f})"

        citations = [p.get('citations', 0) or 0 for p in papers]
        avg_cites = round(float(np.mean(citations)), 1) if citations else 0.0
        avg_year  = int(np.mean(years)) if years else 0

        # Identify "Start Here" paper — highest citations in cluster
        sorted_papers = sorted(
            papers,
            key=lambda p: (p.get('citations') or 0),
            reverse=True
        )
        start_here_idx = 0
        if sorted_papers:
            start_title = sorted_papers[0].get('title', '')
            for i, p in enumerate(papers):
                if p.get('title') == start_title:
                    start_here_idx = i
                    break

        return {
            'name':            cluster_name,
            'description':     description,
            'paper_count':     len(papers),
            'avg_year':        avg_year,
            'avg_citations':   avg_cites,
            'top_venues':      [v for v, _ in Counter(venues).most_common(3)],
            'silhouette_score': round(silhouette, 4) if silhouette else None,
            'papers':          papers,
            'start_here_idx':  start_here_idx,
        }