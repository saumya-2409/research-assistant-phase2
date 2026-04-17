"""
AI Research Assistant — main Streamlit entry point.
Handles UI layout, search orchestration, and results display.
"""

#Imports
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
from pipeline import run_pipeline
from database import get_cached_search, save_search, get_saved_papers, save_paper


# ── Auth ──────────────────────────────────────────────────────────────
try:
    from auth import render_auth_gate, render_user_menu
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False


# For cleaning up the Python environment and controlling runtime behavior
load_dotenv()   # Loads environment variables from a .env file into your system environment; Keeps sensitive data (API keys, secrets) out of your code

warnings.filterwarnings("ignore") # Suppresses all warning messages
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Controls parallelism in Hugging Face tokenizers
logging.getLogger().setLevel(logging.ERROR) # Sets global logging level to only show errors

# ── Streamlit configuration ───────────────────────────────────────────────────────
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

# ── Loading CSS ───────────────────────────────────────────────────────────────
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("src/style.css")

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
    'filter_reset_count':   0
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
    
    st.session_state.force_refresh = st.checkbox(
        "🔄 Force re-fetch (ignore cache)", value=False,
        help="By default, results for the same query are cached for 7 days. Check this to re-fetch fresh papers."
    )

    if clear_btn:
        for key in ['papers_data', 'full_text_papers', 'suggested_papers',
                    'clusters', 'current_page', 'last_query']:
            st.session_state[key] = [] if key not in ('clusters', 'current_page', 'last_query') else ({} if key == 'clusters' else (1 if key == 'current_page' else ''))
        st.rerun()
  
    # User menu (shown if auth is active and user is logged in)
    if AUTH_AVAILABLE:
      try:
        render_user_menu()
      except Exception:
        pass
    
    st.markdown("---")
    st.caption("© 2026 AI Research Assistant")

    # ── PIPELINE (inside sidebar status) ─────────────────────────────
    if start_btn and query.strip():
        st.session_state.processing = True
        st.session_state.last_query = query.strip()
        user_id = (st.session_state.get('user') or {}).get('id')
        force   = st.session_state.get('force_refresh', False)

        with st.status("🚀 Running Research Pipeline...", expanded=True) as status:
            result = run_pipeline(
                query          = query.strip(),
                papers_per_source = papers_per_source,
                summarizer     = st.session_state.summarizer,
                user_id        = user_id,
                force_refresh  = force,
            )
            
            if result['error']:
                st.error(result['error'])
            else:
                st.session_state.papers_data      = result['papers_data']
                st.session_state.full_text_papers  = result['full_text_papers']
                st.session_state.suggested_papers  = result['suggested_papers']
                st.session_state.clusters          = result['clusters']
                st.session_state.current_page      = 1
                status.update(label=f"✅ Done — {len(result['papers_data'])} papers analysed",
                            state="complete", expanded=False)
                st.balloons()
            
            st.session_state.processing   = False
            st.session_state.force_refresh = False

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
        # Uses reset_count in widget keys so resetting forces fresh widgets
        reset_n = st.session_state.get('filter_reset_count', 0)

        all_labels   = sorted(set(p.get('paper_label', '') for p in papers_data if p.get('paper_label')))
        all_clusters = {}
        for cid, info in clusters.items():
            for p in info.get('papers', []):
                all_clusters[p.get('title', '')] = info.get('name', f'Cluster {int(cid)+1}')

        label_options   = ['All'] + all_labels        
        cluster_options = ['All'] + sorted(set(all_clusters.values()))
        

        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc1:
            sel_label = st.selectbox(
                "Filter by type", label_options,
                key=f"filter_label_{reset_n}",
                help="Foundational = highly cited older work · Current = recent & cited · Emerging = newest"
            )
        with fc2:
            sel_cluster = st.selectbox(
                "Filter by research theme", cluster_options,
                key=f"filter_cluster_{reset_n}",
            )
        with fc3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("↺ Reset", use_container_width=True):
              st.session_state['filter_reset_count'] = reset_n + 1
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
