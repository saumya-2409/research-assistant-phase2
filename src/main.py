"""
AI Research Assistant — main Streamlit entry point.
Handles UI layout, search orchestration, and results display.
"""

import streamlit as st
import warnings
import os
import logging
import time
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
from typing import List, Dict
from dotenv import load_dotenv
from collections import Counter
from embedding_utils import load_embedding_model, compute_relevance_embedding_score
from summarizer import FullPaperSummarizer
from fetchers import IntelligentMultiSourceFetcher
from utils.utility import rank_papers
from utils.display import (render_welcome_screen, render_suggested_paper,
                     render_paper_ui, render_paper_inline, render_header,
                     render_metrics, render_saved_paper_card)
from clustering import PaperClusterer
from config import RETRIEVAL
from database import init_database
from utils.export import export_to_excel, generate_bibtex

# ── Auth ──────────────────────────────────────────────────────────────
try:
    from auth import render_auth_gate, render_user_menu
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False

load_dotenv()

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger().setLevel(logging.ERROR)

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DB + Auth gate ────────────────────────────────────────────────────
try:
    init_database()
except Exception:
    pass

if AUTH_AVAILABLE:
    render_auth_gate()

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background: #F5F4FF !important; }

#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E8E6FF !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #9B97C4 !important;
    margin-bottom: 8px !important;
    margin-top: 20px !important;
}

.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1200px !important;
}

.app-header {
    background: linear-gradient(135deg, #5B4EE8 0%, #7B6FF0 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header h1 {
    color: white !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.app-header p {
    color: rgba(255,255,255,0.80) !important;
    font-size: 14px !important;
    margin: 4px 0 0 0 !important;
}
.header-badge {
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 6px 14px;
    color: white;
    font-size: 12px;
    font-weight: 500;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 9px 20px !important;
    transition: all 0.15s ease !important;
    border: 1px solid #E0DEFF !important;
    background: white !important;
    color: #3D35A8 !important;
}
.stButton > button:hover {
    background: #F0EEFF !important;
    border-color: #5B4EE8 !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: #5B4EE8 !important;
    color: white !important;
    border-color: #5B4EE8 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #4A3DD4 !important;
}

/* Text input */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1px solid #E0DEFF !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    background: white !important;
    color: #1A1744 !important;
}
.stTextInput > div > div > input:focus {
    border-color: #5B4EE8 !important;
    box-shadow: 0 0 0 3px rgba(91,78,232,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #B0ACDF !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1px solid #E8E6FF !important;
    gap: 3px !important;
    margin-bottom: 20px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #8B87C0 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #5B4EE8 !important;
    color: white !important;
}

/* Tags */
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    margin-right: 5px;
    margin-bottom: 4px;
}
.tag-purple  { background: #EEEDFE; color: #3C3489; }
.tag-teal    { background: #E1F5EE; color: #0F6E56; }
.tag-amber   { background: #FAEEDA; color: #633806; }
.tag-coral   { background: #FAECE7; color: #712B13; }
.tag-blue    { background: #E6F1FB; color: #0C447C; }
.tag-green   { background: #EAF3DE; color: #27500A; }
.tag-gray    { background: #F1EFE8; color: #444441; }

/* Paper cards */
.paper-card {
    background: white;
    border: 1px solid #E8E6FF;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 12px;
    transition: box-shadow 0.15s, transform 0.15s;
}
.paper-card:hover {
    box-shadow: 0 4px 20px rgba(91,78,232,0.08);
    transform: translateY(-1px);
}

/* Cluster cards */
.cluster-card {
    background: white;
    border: 1px solid #E8E6FF;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 10px;
}

/* Step dots */
.step-dot {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 600; flex-shrink: 0;
}
.step-done   { background: #E1F5EE; color: #0F6E56; }
.step-active { background: #5B4EE8; color: white; }
.step-wait   { background: #F1EFE8; color: #B4B2A9; }

/* Info boxes */
.warning-box {
    background: #FAEEDA; border: 1px solid #F5C875;
    border-radius: 12px; padding: 14px 16px;
    font-size: 13px; color: #633806; margin-bottom: 12px;
}
.success-box {
    background: #E1F5EE; border: 1px solid #9FE1CB;
    border-radius: 12px; padding: 14px 16px;
    font-size: 13px; color: #0F6E56; margin-bottom: 12px;
}

/* Expanders */
[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #E8E6FF !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin-bottom: 10px !important;
}
[data-testid="stExpander"] summary {
    padding: 14px 18px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: #1A1744 !important;
}
[data-testid="stExpander"] summary:hover { background: #F8F7FF !important; }

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background: #5B4EE8 !important; border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div {
    background: #EAE8FF !important; border-radius: 4px !important;
}

hr { border-color: #E8E6FF !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F5F4FF; }
::-webkit-scrollbar-thumb { background: #C8C4F0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #5B4EE8; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
_defaults = {
    'papers_data':          [],
    'full_text_papers':     [],
    'suggested_papers':     [],
    'clusters':             {},
    'processing':           False,
    'cached_query':         '',
    'current_page':         1,
    'saved_papers_session': [],
    'last_query':           '',
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if 'summarizer' not in st.session_state:
    st.session_state.summarizer = FullPaperSummarizer()

# ── Header ────────────────────────────────────────────────────────────
render_header(
    len(st.session_state.get("papers_data", [])),
    len(set(p.get("source", "") for p in st.session_state.get("papers_data", [])))
)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Research Parameters")
    query = st.text_input(
        "Research Topic",
        placeholder="e.g., Generative AI in Healthcare",
        help="Enter the specific topic you want to analyse"
    )

    st.markdown("### ⚙️ Configuration")
    papers_per_source = st.slider("Papers to Analyse", 10, 100, 30, 10)

    col_start, col_clear = st.columns(2)
    with col_start:
        start_btn = st.button(
            "🚀 Start", type="primary",
            disabled=st.session_state.processing or not query,
            use_container_width=True,
        )
    with col_clear:
        clear_btn = st.button(
            "🗑️ Clear", type="secondary", use_container_width=True,
        )

    if clear_btn:
        for key in ['papers_data', 'full_text_papers', 'suggested_papers',
                    'clusters', 'current_page', 'last_query']:
            st.session_state[key] = [] if key not in ('clusters', 'current_page', 'last_query') else ({} if key == 'clusters' else (1 if key == 'current_page' else ''))
        st.rerun()

    st.markdown("---")

    # User menu (shown if auth is active and user is logged in)
    if AUTH_AVAILABLE:
        try:
            render_user_menu()
        except Exception:
            pass

    st.caption("© 2026 AI Research Assistant")

    # ── PIPELINE (inside sidebar status) ─────────────────────────────
    if start_btn and query.strip():
        st.session_state.processing = True
        st.session_state.last_query = query.strip()

        with st.status("🚀 Initiating Research Sequence...", expanded=True) as status:
            try:
                # STEP 1: Fetch
                st.write("1️⃣ **Fetching papers from all sources...**")
                t0 = time.time()
                fetcher = IntelligentMultiSourceFetcher()
                raw_papers, total_fetched = fetcher.fetch_papers(
                    query, sources=None, papers_per_source=50, user_requested=None
                )
                st.write(f"   Found {total_fetched} unique papers in {time.time()-t0:.1f}s")

                if not raw_papers:
                    status.update(label="❌ No papers found. Try different keywords.", state="error")
                    st.session_state.processing = False
                    st.stop()

                # STEP 2: Access check
                st.write("2️⃣ **Checking paper accessibility...**")
                accessible, restricted = [], []
                for p in raw_papers:
                    has_content = (
                        p.get('pdf_available') or p.get('extracted_content') or
                        p.get('is_open_access') or len(p.get('abstract', '')) > 50
                    )
                    (accessible if has_content else restricted).append(p)
                st.write(f"   {len(accessible)} accessible · {len(restricted)} restricted")

                if not accessible:
                    status.update(label="⚠️ All papers are behind paywalls.", state="error")
                    st.session_state.suggested_papers = restricted
                    st.session_state.papers_data = []
                    st.session_state.processing = False
                    st.stop()

                # STEP 3: Relevance filter
                st.write("3️⃣ **Scoring relevance...**")
                _model  = load_embedding_model()
                qemb    = _model.encode(query, convert_to_tensor=True) if _model else None
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
                    status.update(label="⚠️ No relevant papers. Try broader keywords.", state="error")
                    st.session_state.processing = False
                    st.stop()

                # STEP 4: Rank + slice
                st.write("4️⃣ **Ranking papers...**")
                ranked      = rank_papers(scored)
                final       = ranked[:papers_per_source]
                composition = Counter(p.get('source', 'unknown') for p in final)
                st.write("   **Source breakdown:** " +
                         " | ".join(f"{s}: {n}" for s, n in composition.most_common()))

                # STEP 5: Summarise
                st.write(f"5️⃣ **Generating AI summaries for {len(final)} papers...**")
                summarizer_instance = st.session_state.summarizer
                progress = st.progress(0)
                status_txt = st.empty()
                papers_data, full_text_papers, suggested_papers = [], [], list(restricted)
                done = 0

                def _process(p):
                    try:
                        s = summarizer_instance.summarize_paper(p, use_full_text=True, query=query)
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
                            result['accessibility'] = 'accessible'
                            result['abstract_summary_status'] = 'extractive_fallback'
                            result['ai_summary'] = {}
                        done += 1
                        pct   = done / len(final)
                        badge = ('📄' if result.get('abstract_summary_status') == 'generated_from_fulltext'
                                 else '⚡' if result.get('abstract_summary_status') == 'generated_from_abstract'
                                 else '📝')
                        progress.progress(pct)
                        status_txt.markdown(f"**{badge} {done}/{len(final)}** — {result.get('title','')[:55]}...")
                        if result.get('accessibility') == 'accessible':
                            papers_data.append(result)
                            if (result.get('extracted_content') or
                                    result.get('abstract_summary_status') == 'generated_from_fulltext'):
                                full_text_papers.append(result)
                        else:
                            suggested_papers.append(result)

                progress.empty()
                status_txt.empty()

                # STEP 6: Save state
                st.session_state.papers_data      = papers_data
                st.session_state.full_text_papers  = full_text_papers
                st.session_state.suggested_papers  = suggested_papers
                st.session_state.processing        = False
                st.session_state.current_page      = 1

                status.update(label=f"✅ Done — {len(papers_data)} papers analysed",
                              state="complete", expanded=False)

                # STEP 7: Cluster
                if papers_data:
                    st.write("6️⃣ **Clustering by research theme...**")
                    try:
                        clusterer = PaperClusterer()
                        st.session_state.clusters = clusterer.cluster_papers(papers_data)
                    except Exception as ce:
                        st.warning(f"Clustering skipped: {ce}")
                        st.session_state.clusters = {}

                st.balloons()

            except Exception as e:
                status.update(label="❌ An error occurred.", state="error")
                st.error(f"Error: {e}")
                st.session_state.processing = False


# ── MAIN CONTENT ─────────────────────────────────────────────────────
if st.session_state.papers_data:

    papers_data   = st.session_state.papers_data
    full_text     = st.session_state.full_text_papers
    clusters      = st.session_state.clusters
    restricted    = st.session_state.suggested_papers

    render_metrics(papers_data, full_text, clusters)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "📄 Papers & Summaries",
        "📚 Saved Papers",
        "📤 Export & Restricted",
    ])

    # ── TAB 1: DASHBOARD ─────────────────────────────────────────────
    with tab1:
        st.markdown("### Research Dashboard")

        if clusters:
            cluster_names  = []
            cluster_counts = []
            cluster_cites  = []
            cluster_years  = []

            for cid, info in clusters.items():
                name = info.get('name', f'Cluster {int(cid)+1}')
                # Truncate long names for charts
                short = name[:30] + '…' if len(name) > 30 else name
                cluster_names.append(short)
                cluster_counts.append(info.get('paper_count', len(info.get('papers', []))))
                cluster_cites.append(info.get('avg_citations', 0))
                cluster_years.append(info.get('avg_year', 0))

            COLORS = ['#5B4EE8', '#1D9E75', '#D85A30', '#BA7517',
                      '#0C447C', '#712B13', '#27500A', '#3C3489']

            # Chart row
            col1, col2 = st.columns(2, gap="large")

            with col1:
                # Paper count bar — horizontal, easy to read
                fig = go.Figure(go.Bar(
                    x=cluster_counts,
                    y=cluster_names,
                    orientation='h',
                    marker_color=COLORS[:len(cluster_names)],
                    text=cluster_counts,
                    textposition='outside',
                ))
                fig.update_layout(
                    title="Papers per Research Theme",
                    xaxis_title="Number of Papers",
                    yaxis_title="",
                    font=dict(family="Inter", size=12),
                    height=max(250, len(cluster_names) * 52),
                    margin=dict(l=10, r=40, t=40, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#E8E6FF'),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Avg citations scatter — year on x, cites on y, size = paper count
                fig2 = go.Figure(go.Scatter(
                    x=cluster_years if any(cluster_years) else cluster_names,
                    y=cluster_cites,
                    mode='markers+text',
                    marker=dict(
                        size=[max(18, c * 3) for c in cluster_counts],
                        color=COLORS[:len(cluster_names)],
                        opacity=0.85,
                    ),
                    text=cluster_names,
                    textposition='top center',
                    textfont=dict(size=10),
                ))
                fig2.update_layout(
                    title="Avg Citations by Theme (bubble = paper count)",
                    xaxis_title="Avg Publication Year",
                    yaxis_title="Avg Citations",
                    font=dict(family="Inter", size=12),
                    height=max(250, len(cluster_names) * 52),
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#E8E6FF'),
                    yaxis=dict(gridcolor='#E8E6FF'),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Cluster cards
            st.markdown("### Research Themes")
            for i, (cid, info) in enumerate(clusters.items()):
                cp          = info.get('papers', [])
                n_extracted = len([p for p in cp if p.get('extracted_content')])
                years       = [int(p['year']) for p in cp if str(p.get('year','')).isdigit()]
                yr_range    = f"{min(years)}–{max(years)}" if years else "–"
                color       = COLORS[i % len(COLORS)]

                st.markdown(f"""
                <div class="cluster-card" style="border-left:4px solid {color};">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                        <div style="width:10px;height:10px;border-radius:50%;
                                    background:{color};flex-shrink:0;"></div>
                        <div style="font-size:15px;font-weight:600;color:#1A1744;">
                            {info.get('name', f'Cluster {int(cid)+1}')}
                        </div>
                    </div>
                    <p style="color:#64748b;font-size:13px;margin-bottom:12px;">
                        {info.get('description', '')[:120]}
                    </p>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;">
                        <span style="background:#F1EFFE;color:#5B4EE8;padding:4px 10px;
                            border-radius:6px;font-size:12px;font-weight:500;">
                            {len(cp)} papers
                        </span>
                        <span style="background:#F1EFFE;color:#5B4EE8;padding:4px 10px;
                            border-radius:6px;font-size:12px;font-weight:500;">
                            {n_extracted} full-text
                        </span>
                        <span style="background:#F1EFFE;color:#5B4EE8;padding:4px 10px;
                            border-radius:6px;font-size:12px;font-weight:500;">
                            ⌀ {info.get('avg_citations', 0)} citations
                        </span>
                        <span style="background:#F1EFFE;color:#5B4EE8;padding:4px 10px;
                            border-radius:6px;font-size:12px;font-weight:500;">
                            {yr_range}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run a search to see the research landscape dashboard.")

    # ── TAB 2: PAPERS & SUMMARIES ─────────────────────────────────────
    with tab2:
        st.markdown("### Papers & Summaries")

        # ── Filter controls ───────────────────────────────────────────
        all_labels   = sorted(set(p.get('paper_label', '') for p in papers_data if p.get('paper_label')))
        all_clusters = {}
        for cid, info in clusters.items():
            for p in info.get('papers', []):
                all_clusters[p.get('title', '')] = info.get('name', f'Cluster {int(cid)+1}')

        cluster_options = ['All'] + sorted(set(all_clusters.values()))
        label_options   = ['All'] + all_labels

        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc1:
            sel_label = st.selectbox(
                "Filter by type",
                label_options,
                key="filter_label",
                help="Foundational (highly cited, older), Current (recent, cited), Emerging (newest)"
            )
        with fc2:
            sel_cluster = st.selectbox(
                "Filter by research theme",
                cluster_options,
                key="filter_cluster"
            )
        with fc3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("↺ Reset Filters", use_container_width=True):
                st.session_state.filter_label   = 'All'
                st.session_state.filter_cluster = 'All'
                st.rerun()

        # Apply filters
        filtered = papers_data
        if sel_label != 'All':
            filtered = [p for p in filtered if p.get('paper_label') == sel_label]
        if sel_cluster != 'All':
            filtered = [p for p in filtered
                        if all_clusters.get(p.get('title', '')) == sel_cluster]

        if sel_label != 'All' or sel_cluster != 'All':
            st.caption(f"Showing {len(filtered)} of {len(papers_data)} papers after filters")

        if not filtered:
            st.info("No papers match the selected filters. Try adjusting or resetting them.")
        else:
            # Pagination
            ITEMS_PER_PAGE = 10
            total_pages = max(1, (len(filtered) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            if st.session_state.current_page > total_pages:
                st.session_state.current_page = 1

            st.markdown(
                f"<div style='color:#64748b;margin-bottom:15px;font-size:0.95rem;'>"
                f"Page <strong>{st.session_state.current_page}</strong> of "
                f"<strong>{total_pages}</strong> "
                f"<span style='color:#94a3b8;'>({len(filtered)} papers)</span></div>",
                unsafe_allow_html=True
            )

            start = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
            for i, paper in enumerate(filtered[start:start + ITEMS_PER_PAGE]):
                render_paper_ui(paper, idx=start + i)

            st.markdown("---")
            p1, p2, p3 = st.columns([1, 3, 1])
            with p1:
                if st.button("← Prev", disabled=(st.session_state.current_page == 1),
                             use_container_width=True, key="prev_btn"):
                    st.session_state.current_page -= 1
                    st.rerun()
            with p2:
                st.markdown(
                    f"<div style='text-align:center;padding-top:8px;color:#64748b;font-weight:500;'>"
                    f"Page {st.session_state.current_page} / {total_pages}</div>",
                    unsafe_allow_html=True
                )
            with p3:
                if st.button("Next →", disabled=(st.session_state.current_page == total_pages),
                             use_container_width=True, key="next_btn"):
                    st.session_state.current_page += 1
                    st.rerun()

    # ── TAB 3: SAVED PAPERS ───────────────────────────────────────────
    with tab3:
        st.markdown("### 📚 Saved Papers")
        saved = st.session_state.get('saved_papers_session', [])

        if not saved:
            st.markdown("""
            <div style="text-align:center;padding:40px 20px;color:#9B97C4;">
                <div style="font-size:32px;margin-bottom:12px;">🔖</div>
                <div style="font-size:16px;font-weight:500;margin-bottom:6px;">No saved papers yet</div>
                <div style="font-size:13px;">
                    Click the <strong>🔖 Save</strong> button inside any paper card to add it here.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(saved)} saved paper{'s' if len(saved) != 1 else ''}**")

            # Bulk export saved papers
            if len(saved) > 0:
                saved_bib = generate_bibtex(saved)
                st.download_button(
                    "📥 Download Saved Papers as BibTeX",
                    data=saved_bib,
                    file_name="saved_papers.bib",
                    mime="text/plain",
                    use_container_width=False,
                )

            st.markdown("---")
            # Render each saved paper
            for i, paper in enumerate(saved):
                render_saved_paper_card(paper, idx=i)

    # ── TAB 4: EXPORT & RESTRICTED ────────────────────────────────────
    with tab4:
        st.markdown("### 📤 Export")

        ecol1, ecol2 = st.columns(2, gap="large")

        with ecol1:
            st.markdown("""
            <div style="background:white;border:1px solid #E8E6FF;border-radius:14px;
                        padding:20px;margin-bottom:12px;">
                <div style="font-size:16px;font-weight:600;color:#1A1744;margin-bottom:6px;">
                    📊 Excel Spreadsheet
                </div>
                <div style="font-size:13px;color:#64748b;margin-bottom:14px;">
                    All papers organised by cluster, with metadata, abstracts,
                    relevance scores, and a cluster summary sheet.
                    Colour-coded by theme. Ready for review or sharing.
                </div>
            </div>
            """, unsafe_allow_html=True)
            try:
                excel_bytes = export_to_excel(
                    papers_data, clusters,
                    st.session_state.get('last_query', 'research')
                )
                st.download_button(
                    "📥 Download Excel (.xlsx)",
                    data=excel_bytes,
                    file_name="research_papers.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e:
                st.error(f"Could not generate Excel: {e}")

        with ecol2:
            st.markdown("""
            <div style="background:white;border:1px solid #E8E6FF;border-radius:14px;
                        padding:20px;margin-bottom:12px;">
                <div style="font-size:16px;font-weight:600;color:#1A1744;margin-bottom:6px;">
                    📝 BibTeX Reference File
                </div>
                <div style="font-size:13px;color:#64748b;margin-bottom:14px;">
                    All papers as a <code>.bib</code> file — import directly into
                    Overleaf, Zotero, Mendeley, or any LaTeX project.
                    One citation key per paper, auto-deduplicated.
                </div>
            </div>
            """, unsafe_allow_html=True)
            try:
                bibtex_str = generate_bibtex(papers_data)
                st.download_button(
                    "📥 Download BibTeX (.bib)",
                    data=bibtex_str,
                    file_name="research_papers.bib",
                    mime="text/plain",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e:
                st.error(f"Could not generate BibTeX: {e}")

        st.markdown("---")
        st.markdown("### 🔒 Restricted Papers")
        st.markdown("""
        <div class="warning-box">
            These papers were found but require a subscription or institutional access.
            Citations are still available — you can look them up via your library.
        </div>
        """, unsafe_allow_html=True)

        if not restricted:
            st.markdown("""
            <div class="success-box">
                ✅ All found papers were accessible — nothing is restricted.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(restricted)} restricted papers**")
            for paper in restricted:
                render_suggested_paper(paper)

else:
    render_welcome_screen()

# ── Footer ─────────────────────────────────────────────────────────────
if not st.session_state.processing:
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:0.85rem;padding-bottom:1rem;'>"
        "🔬 AI Research Assistant · 6 sources · AI summaries · Citation export"
        "</div>",
        unsafe_allow_html=True
    )
