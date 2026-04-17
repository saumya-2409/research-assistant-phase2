"""
pipeline.py
===========
Research pipeline — fetch → access check → relevance → rank → summarise → cluster.
Called from main.py via run_pipeline().
Returns a result dict so main.py only handles UI.
"""

import time
import streamlit as st
import concurrent.futures
from collections import Counter
from typing import Dict, Any

from embedding_utils   import load_embedding_model, compute_relevance_embedding_score
from summarizer        import FullPaperSummarizer
from fetchers          import IntelligentMultiSourceFetcher
from clustering        import PaperClusterer
from utils.utility     import rank_papers
from config            import RETRIEVAL
from database          import get_cached_search, save_to_cache, save_search


def run_pipeline(query: str, papers_per_source: int,
                 summarizer: FullPaperSummarizer,
                 user_id: int = None,
                 force_refresh: bool = False) -> Dict[str, Any]:
    """
    Full research pipeline. Writes progress via st.write() inside a 
    st.status() context — call this from inside `with st.status(...) as status:`.

    Returns:
        {
          'papers_data':       list of accessible paper dicts with ai_summary,
          'full_text_papers':  subset with full-text summaries,
          'suggested_papers':  restricted/inaccessible papers,
          'clusters':          cluster dict from PaperClusterer,
          'source_stats':      per-source relevance stats dict,
          'error':             error string if pipeline failed, else None,
        }
    """
    empty = {
        'papers_data': [], 'full_text_papers': [], 'suggested_papers': [],
        'clusters': {}, 'source_stats': {}, 'error': None
    }

    # ── Cache check ────────────────────────────────────────────────────
    if not force_refresh:
        cached = get_cached_search(query)
        if cached:
            cached_papers   = cached.get('papers', [])
            cached_clusters = cached.get('clusters', {})
            # If user wants more papers than cached, run fresh
            if len(cached_papers) >= papers_per_source:
                st.write(f"⚡ **Loaded from cache** — {len(cached_papers)} papers "
                         f"(run with Force Refresh to re-fetch)")
                accessible  = [p for p in cached_papers
                                if p.get('accessibility') != 'inaccessible']
                restricted  = [p for p in cached_papers
                                if p.get('accessibility') == 'inaccessible']
                full_text   = [p for p in accessible
                                if p.get('abstract_summary_status') == 'generated_from_fulltext']
                return {**empty,
                        'papers_data': accessible[:papers_per_source],
                        'full_text_papers': full_text,
                        'suggested_papers': restricted,
                        'clusters': cached_clusters}

    # ── STEP 1: Fetch ──────────────────────────────────────────────────
    st.write("1️⃣ **Fetching papers from all sources...**")
    t0 = time.time()
    fetcher = IntelligentMultiSourceFetcher()
    raw_papers, total_fetched = fetcher.fetch_papers(
        query, sources=None, papers_per_source=50, user_requested=None
    )
    st.write(f"   ✅ {total_fetched} unique papers fetched in {time.time()-t0:.1f}s")

    if not raw_papers:
        return {**empty, 'error': 'No papers found. Try different keywords.'}

    # ── STEP 2: Access check ───────────────────────────────────────────
    st.write("2️⃣ **Checking accessibility...**")
    accessible, restricted = [], []
    for p in raw_papers:
        has_content = (
            p.get('pdf_available') or 
            p.get('extracted_content') or
            p.get('is_open_access') or 
            len(p.get('abstract', '')) > 50
        )
        (accessible if has_content else restricted).append(p)
    st.write(f"   ✅ {len(accessible)} accessible · {len(restricted)} restricted")

    if not accessible:
        return {**empty, 'suggested_papers': restricted,
                'error': 'All papers are behind paywalls. Try ArXiv-focused queries.'}

    # ── STEP 3: Relevance scoring ──────────────────────────────────────
    st.write("3️⃣ **Scoring semantic relevance...**")
    _model   = load_embedding_model()
    qemb     = _model.encode(query, convert_to_tensor=True) if _model else None
    scored, source_stats = [], {}
    for p in accessible:
        score = compute_relevance_embedding_score(query, p, query_embedding=qemb)
        p['relevance_score'] = round(score, 3)
        src = p.get('source', 'unknown')
        source_stats.setdefault(src, {'total': 0, 'passed': 0})
        source_stats[src]['total'] += 1
        if score >= RETRIEVAL["relevance_threshold"]:
            scored.append(p)
            source_stats[src]['passed'] += 1

    for src, stats in sorted(source_stats.items()):
        pct = stats['passed'] / stats['total'] * 100 if stats['total'] else 0
        st.write(f"   {src}: {stats['passed']}/{stats['total']} relevant ({pct:.0f}%)")

    if not scored:
        return {**empty, 'suggested_papers': restricted, 'source_stats': source_stats,
                'error': 'No relevant papers found. Try broader keywords.'}

    # ── STEP 4: Rank + slice ───────────────────────────────────────────
    st.write("4️⃣ **Ranking by quality & relevance...**")
    ranked = rank_papers(scored)
    final  = ranked[:papers_per_source]
    comp   = Counter(p.get('source', 'unknown') for p in final)
    st.write("   **Source mix:** " + " | ".join(f"{s}: {n}" for s, n in comp.most_common()))

    # ── STEP 5: Summarise ──────────────────────────────────────────────
    st.write(f"5️⃣ **Generating AI summaries for {len(final)} papers...**")
    progress   = st.progress(0)
    status_txt = st.empty()
    papers_data, full_text_papers = [], []
    done = 0

    def _process(p):
        try:
            s = summarizer.summarize_paper(p, use_full_text=True, query=query)
            if not isinstance(s, dict):
                s = {}
        except Exception:
            s = {}
        p['ai_summary'] = s
        explicit = s.get('accessibility')
        p['accessibility'] = (
            'inaccessible'
            if explicit == 'inaccessible' and not p.get('abstract')
            else 'accessible'
        )
        p['abstract_summary_status'] = s.get('abstract_summary_status', 'extractive_fallback')
        return p

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_process, p): p for p in final}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception:
                result = futures[future]
                result.update({'accessibility': 'accessible',
                               'abstract_summary_status': 'extractive_fallback',
                               'ai_summary': {}})
            done += 1
            pct   = done / len(final)
            badge = ('📄' if result.get('abstract_summary_status') == 'generated_from_fulltext'
                     else '⚡' if result.get('abstract_summary_status') == 'generated_from_abstract'
                     else '📝')
            progress.progress(pct)
            status_txt.markdown(f"**{badge} {done}/{len(final)}** — "
                                f"{result.get('title','')[:55]}...")
            if result.get('accessibility') == 'accessible':
                papers_data.append(result)
                if (result.get('extracted_content') or
                        result.get('abstract_summary_status') == 'generated_from_fulltext'):
                    full_text_papers.append(result)
            else:
                restricted.append(result)

    progress.empty()
    status_txt.empty()

    # ── STEP 6: Cluster ────────────────────────────────────────────────
    clusters = {}
    if papers_data:
        st.write("6️⃣ **Clustering by research theme...**")
        try:
            clusterer = PaperClusterer()
            clusters  = clusterer.cluster_papers(papers_data)
            st.write(f"   ✅ {len(clusters)} clusters identified")
        except Exception as ce:
            st.warning(f"Clustering skipped: {ce}")

    # ── STEP 7: Persist to cache + history ────────────────────────────
    try:
        # Mark restricted papers so cache knows
        for p in restricted:
            p.setdefault('accessibility', 'inaccessible')
        save_to_cache(query, papers_data + restricted, clusters)
    except Exception:
        pass

    if user_id:
        try:
            save_search(user_id, query, len(papers_data), len(clusters))
        except Exception:
            pass

    return {
        'papers_data':      papers_data,
        'full_text_papers': full_text_papers,
        'suggested_papers': restricted,
        'clusters':         clusters,
        'source_stats':     source_stats,
        'error':            None,
    }
