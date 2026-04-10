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
import concurrent.futures
from typing import List, Dict
from dotenv import load_dotenv
from collections import Counter
from embedding_utils import load_embedding_model, compute_relevance_embedding_score
from summarizer import FullPaperSummarizer
from fetchers import IntelligentMultiSourceFetcher
from utils.utility import rank_papers
from utils.display import render_welcome_screen, render_suggested_paper, render_paper_ui, render_paper_inline, render_header, render_metrics
from clustering import PaperClusterer
from config import RETRIEVAL


load_dotenv()

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger().setLevel(logging.ERROR)

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon=" § ",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'summarizer' not in st.session_state:
    st.session_state.summarizer = FullPaperSummarizer()
    print("[App Debug] Summarizer singleton created")

# BEAUTIFUL DESIGN CSS (PRESERVED)

# ── DESIGN SYSTEM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background: #F5F4FF !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
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

/* ── Sidebar slider ── */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: #5B4EE8 !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #5B4EE8 !important;
    font-weight: 600 !important;
}

/* ── Main content padding ── */
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1100px !important;
}

/* ── App header ── */
.app-header {
    background: #5B4EE8;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
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
    color: rgba(255,255,255,0.75) !important;
    font-size: 14px !important;
    margin: 4px 0 0 0 !important;
}
.header-badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 6px 14px;
    color: white;
    font-size: 12px;
    font-weight: 500;
}

/* ── Buttons ── */
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
    border-color: #4A3DD4 !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1px solid #E0DEFF !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    background: white !important;
    color: #1A1744 !important;
    transition: border-color 0.15s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #5B4EE8 !important;
    box-shadow: 0 0 0 3px rgba(91,78,232,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #B0ACDF !important; }

/* ── Tabs ── */
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

/* ── Metric cards ── */
.metric-card {
    background: white;
    border: 1px solid #E8E6FF;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.metric-number {
    font-size: 32px;
    font-weight: 600;
    color: #5B4EE8;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #9B97C4;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Paper cards ── */
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
.paper-card-accent {
    border-left: 3px solid #5B4EE8 !important;
    border-radius: 0 14px 14px 0 !important;
}
.paper-title {
    font-size: 15px;
    font-weight: 600;
    color: #1A1744;
    margin-bottom: 6px;
    line-height: 1.4;
}
.paper-meta {
    font-size: 12px;
    color: #9B97C4;
    margin-bottom: 10px;
}

/* ── Source / status tags ── */
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

/* ── Cluster cards ── */
.cluster-card {
    background: white;
    border: 1px solid #E8E6FF;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    transition: all 0.15s;
}
.cluster-card:hover {
    border-color: #5B4EE8;
    box-shadow: 0 4px 16px rgba(91,78,232,0.08);
}
.cluster-name {
    font-size: 15px;
    font-weight: 600;
    color: #1A1744;
    margin-bottom: 4px;
}
.cluster-meta {
    font-size: 12px;
    color: #9B97C4;
}

/* ── Step progress ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.step-dot {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 600;
    flex-shrink: 0;
}
.step-done   { background: #E1F5EE; color: #0F6E56; }
.step-active { background: #5B4EE8; color: white; }
.step-wait   { background: #F1EFE8; color: #B4B2A9; }

/* ── Info / warning boxes ── */
.info-box {
    background: #F0EEFF;
    border: 1px solid #D4CFFF;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 13px;
    color: #3D35A8;
    margin-bottom: 12px;
}
.warning-box {
    background: #FAEEDA;
    border: 1px solid #F5C875;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 13px;
    color: #633806;
    margin-bottom: 12px;
}
.success-box {
    background: #E1F5EE;
    border: 1px solid #9FE1CB;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 13px;
    color: #0F6E56;
    margin-bottom: 12px;
}

/* ── Gap analysis cards ── */
.gap-card {
    background: white;
    border: 1px solid #E8E6FF;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
}
.gap-card-method { border-left: 3px solid #D85A30; border-radius: 0 14px 14px 0; }
.gap-card-eval   { border-left: 3px solid #1D9E75; border-radius: 0 14px 14px 0; }
.gap-card-app    { border-left: 3px solid #5B4EE8; border-radius: 0 14px 14px 0; }
.gap-card-theory { border-left: 3px solid #BA7517; border-radius: 0 14px 14px 0; }
.gap-title {
    font-size: 14px;
    font-weight: 600;
    color: #1A1744;
    margin-bottom: 8px;
}
.gap-text {
    font-size: 13px;
    color: #5C5888;
    line-height: 1.6;
}

/* ── Suggested / restricted papers ── */
.restricted-card {
    background: white;
    border: 1px solid #F5C4B3;
    border-left: 3px solid #D85A30;
    border-radius: 0 14px 14px 0;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.restricted-title {
    font-size: 14px;
    font-weight: 600;
    color: #1A1744;
    margin-bottom: 6px;
}
.restricted-meta { font-size: 12px; color: #9B97C4; margin-bottom: 8px; }

/* ── Expanders ── */
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
}
[data-testid="stExpander"] summary:hover { background: #F8F7FF !important; }

/* ── st.info / st.success / st.warning overrides ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: #5B4EE8 !important;
    border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div {
    background: #EAE8FF !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border-color: #E8E6FF !important; }

/* ── Link buttons ── */
.stLinkButton > a {
    border-radius: 8px !important;
    border: 1px solid #E0DEFF !important;
    color: #5B4EE8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    background: white !important;
}
.stLinkButton > a:hover {
    background: #F0EEFF !important;
    border-color: #5B4EE8 !important;
}

/* ── st.status ── */
[data-testid="stStatusWidget"] {
    border-radius: 14px !important;
    border: 1px solid #E8E6FF !important;
    background: white !important;
}

/* ── Caption ── */
.stCaption { color: #9B97C4 !important; font-size: 12px !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border: 1px solid #E0DEFF !important;
    background: white !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F5F4FF; }
::-webkit-scrollbar-thumb { background: #C8C4F0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #5B4EE8; }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'papers_data' not in st.session_state:
    st.session_state.papers_data = []
if 'full_text_papers' not in st.session_state:
    st.session_state.full_text_papers = []
if 'suggested_papers' not in st.session_state:
    st.session_state.suggested_papers = []
if 'clusters' not in st.session_state:
    st.session_state.clusters = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'cached_query' not in st.session_state:
    st.session_state.cached_query = ''
if 'papers_to_save' not in st.session_state:
    st.session_state.papers_to_save = []

# ==================== GAP ANALYSIS ====================

class ResearchGapAnalyzer:
    def analyze_gaps(self, papers: List[Dict]) -> Dict[str, List[str]]:
        gaps = {
            'methodological_gaps': [],
            'evaluation_gaps': [], 
            'application_gaps': [],
            'theoretical_gaps': []
        }
        
        # Use extracted content for better gap analysis
        all_content = []
        for paper in papers:
            content = paper.get('extracted_content', '') or paper.get('abstract', '')
            all_content.append(content)
        
        combined_content = ' '.join(all_content).lower()
        
        if 'dataset' in combined_content or 'limited' in combined_content:
            gaps['methodological_gaps'].extend([
                'Limited dataset diversity across different domains and applications',
                'Lack of standardized evaluation protocols for cross-method comparison',
                'Insufficient attention to computational efficiency and scalability issues'
            ])
        
        if 'experiment' in combined_content or 'evaluation' in combined_content:
            gaps['evaluation_gaps'].extend([
                'Need for more comprehensive real-world testing scenarios',
                'Lack of longitudinal studies assessing long-term performance',
                'Limited evaluation on edge cases and adversarial conditions'
            ])
        
        if 'application' in combined_content or 'real-world' in combined_content:
            gaps['application_gaps'].extend([
                'Gap between laboratory results and industrial deployment',
                'Limited integration with existing systems and workflows',
                'Insufficient consideration of regulatory and ethical constraints'
            ])
        
        if 'theoretical' in combined_content or 'analysis' in combined_content:
            gaps['theoretical_gaps'].extend([
                'Lack of theoretical foundations for empirical observations',
                'Limited understanding of failure modes and boundary conditions',
                'Insufficient mathematical analysis of convergence properties'
            ])
        
        return gaps

# ==================== MAIN APPLICATION ====================

render_header(len(st.session_state.get("papers_data",[])), len(set(p.get("source","") for p in st.session_state.get("papers_data",[]))))

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### 🔍 Research Parameters")    
    query = st.text_input("Research Topic", placeholder="e.g., Generative AI in Healthcare", help="Enter the specific topic you want to analyze")

    st.markdown("### ⚙️ Configuration")
    papers_per_source = st.slider("Papers to Analyse",10,100,30,10)


    # --- Buttons Layout (Side-by-Side) ---
    col_start, col_clear = st.columns(2)
    with col_start:
        # Start Analysis Button
        start_btn = st.button(
            "🚀 Start", type="primary", 
            disabled=st.session_state.processing or not query,
            use_container_width=True,
        )

    with col_clear:
        # Clear Results Button
        clear_btn = st.button(
            "🗑️ Clear", 
            type="secondary",
            use_container_width=True,
        )
    
    # Handle Clear Logic
    if clear_btn:
        for key in ['papers_data','full_text_papers','suggested_papers','clusters','current_page']:
            st.session_state[key] = [] if key != 'clusters' else {}
            if key == 'current_page': st.session_state[key] = 1 # Reset pagination
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("© 2026 Intelligent Research Assistant")
    
    # ── MAIN AREA — processing and results ──────────────────────────────
    # Handle Start Logic
    if start_btn and query.strip():
        st.session_state.processing = True
        
        # Create the status container
        with st.status("🚀 Initiating Research Sequence...", expanded=True) as status:
            try:
                # ── STEP 1: FETCH ALL AVAILABLE PAPERS ──────────────────────
                # Always fetch the maximum the API will return.
                # We rank and slice AFTER filtering, not before.
                st.write("1️⃣ **Scouting Sources:** Fetching latest research...")
                t0 = time.time()
                fetcher = IntelligentMultiSourceFetcher()

                # Pass papers_per_source=100 to fetch max from API.
                # user_requested is used later for slicing AFTER filtering.
                raw_papers, total_fetched = fetcher.fetch_papers(
                    query,
                    sources=None,
                    papers_per_source=50,
                    user_requested=None       # do NOT slice yet
                )
                st.write(f"   All sources: {total_fetched} unique papers in {time.time()-t0:.1f}s")

                if not raw_papers:
                    status.update(
                        label="❌ No papers found. Try different keywords.",
                        state="error"
                    )
                    st.session_state.processing = False
                    st.stop()

                # ── STEP 2: REMOVE INACCESSIBLE PAPERS ──────────────────────
                # Access check already ran inside fetch_papers (IntelligentPaperAccessor).
                # Here we hard-remove papers marked inaccessible so they never
                # reach the relevance filter, ranker, or summariser.
                # They go directly to suggested_papers for the Restricted tab.
                st.write("2️⃣ **Access Check:** Separating accessible from restricted papers...")

                pre_filter_accessible = []
                pre_filter_restricted = []

                for p in raw_papers:
                    # fetcher sets pdf_available=True if it found a PDF or extracted content
                    # papers with no URL AND no abstract are genuinely useless
                    has_content = (
                        p.get('pdf_available') or
                        p.get('extracted_content') or
                        p.get('is_open_access') or
                        len(p.get('abstract', '')) > 50
                    )
                    if has_content:
                        pre_filter_accessible.append(p)
                    else:
                        pre_filter_restricted.append(p)

                st.write(
                    f"   {len(pre_filter_accessible)} accessible  |  "
                    f"{len(pre_filter_restricted)} restricted (moved to Restricted tab)"
                )

                if not pre_filter_accessible:
                    status.update(
                        label="⚠️ All papers are behind paywalls. Try ArXiv-focused queries.",
                        state="error"
                    )
                    st.session_state.suggested_papers = pre_filter_restricted
                    st.session_state.papers_data      = []
                    st.session_state.full_text_papers  = []
                    st.session_state.clusters          = {}
                    st.session_state.processing        = False
                    st.stop()

                # ── STEP 3: RELEVANCE FILTER ─────────────────────────────────
                # Run semantic relevance scoring on ALL accessible papers.
                # This runs BEFORE slicing so the user's N papers are the
                # most relevant N, not just the first N that happened to arrive.
                st.write("3️⃣ **Relevance Filter:** Scoring all papers against query...")

                # Encode query ONCE before the loop
                _model = load_embedding_model()
                query_emb = _model.encode(query, convert_to_tensor=True) if _model else None

                scored_papers = []
                source_stats  = {}   # track per-source survival rates

                for p in pre_filter_accessible:
                    score = compute_relevance_embedding_score(query, p, query_embedding=query_emb)
                    p['relevance_score'] = round(score, 3)
                    src = p.get('source', 'unknown')
                    source_stats.setdefault(src, {'total': 0, 'passed': 0})
                    source_stats[src]['total'] += 1
                    if score >= RETRIEVAL["relevance_threshold"]:
                        scored_papers.append(p)
                        source_stats[src]['passed'] += 1

                # Show the per-source breakdown so user can see which sources contributed
                st.write("**Relevance filter results by source:**")
                for src, stats in sorted(source_stats.items()):
                    pct = (stats['passed'] / stats['total'] * 100) if stats['total'] else 0
                    st.write(
                        f"  {src}: {stats['passed']}/{stats['total']} passed "
                        f"({pct:.0f}% relevant)"
                    )

                if not scored_papers:
                    status.update(
                        label="⚠️ No relevant papers found. Try broader keywords.",
                        state="error"
                    )
                    st.warning("All papers scored below the relevance threshold (0.25).")
                    st.session_state.processing = False
                    st.stop()

                # ── STEP 4: RANK AND SLICE TO USER'S REQUESTED NUMBER ────────
                # Now rank all relevant papers by citation count + recency.
                # Then slice to exactly what the user asked for.
                # This is the correct place to apply the user's N — after
                # filtering, not before, so the user gets the best N papers.

                ranked_papers   = rank_papers(scored_papers)
                total_relevant  = len(ranked_papers)
                final_papers    = ranked_papers[:papers_per_source]

                # Show source composition of final selection
                source_composition = Counter(p.get('source', 'unknown') for p in final_papers)
                st.write(
                    f"**Final {len(final_papers)} papers by source:** " +
                    " | ".join(f"{src}: {n}" for src, n in source_composition.most_common())
                )

                # ── STEP 5: GENERATE SUMMARIES ───────────────────────────────
                # Run AI summarisation on the final set only.
                # This is the expensive step — we deliberately kept it last
                # so we only summarise papers the user will actually see.
                st.write(f"5️⃣ **Deep Reading:** Generating AI summaries for {len(final_papers)} papers...")

                summarizer_instance = st.session_state.summarizer
                total_to_summarise  = len(final_papers)

                summary_progress = st.progress(0)
                summary_status   = st.empty()
                summary_caption  = st.empty()

                def process_single_paper(p):
                    try:
                        summary = summarizer_instance.summarize_paper(
                            p, use_full_text=True, query=query
                        )
                        if not isinstance(summary, dict):
                            summary = {}
                    except Exception:
                        summary = {}

                    p['ai_summary']              = summary
                    # Default to accessible — paper already passed access check.
                    # Only mark inaccessible if summariser explicitly says so AND
                    # there is no abstract to fall back on.
                    explicit_status = summary.get('accessibility')
                    if explicit_status == 'inaccessible' and not p.get('abstract'):
                        p['accessibility'] = 'inaccessible'
                    else:
                        p['accessibility'] = 'accessible'

                    p['abstract_summary_status'] = summary.get(
                        'abstract_summary_status', 'extractive_fallback'
                    )
                    return p

                papers_data      = []
                full_text_papers = []
                suggested_papers = list(pre_filter_restricted)
                done_count       = 0

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_paper = {
                        executor.submit(process_single_paper, p): p
                        for p in final_papers
                    }
                    for future in concurrent.futures.as_completed(future_to_paper):
                        try:
                            result = future.result()
                        except Exception:
                            # Individual paper failed — use the original paper with no summary
                            result = future_to_paper[future]
                            result['accessibility']          = 'accessible'
                            result['abstract_summary_status'] = 'extractive_fallback'
                            result['ai_summary']             = {}

                        done_count += 1
                        pct    = done_count / total_to_summarise
                        title  = result.get('title', '')[:55]
                        badge  = (
                            '📄' if result.get('abstract_summary_status') == 'generated_from_fulltext'
                            else '⚡' if result.get('abstract_summary_status') == 'generated_from_abstract'
                            else '📝'
                        )

                        summary_progress.progress(pct)
                        summary_status.markdown(
                            f"**{badge} {done_count}/{total_to_summarise}** — {title}..."
                        )
                        summary_caption.caption(
                            f"{int(pct * 100)}% complete"
                        )

                        if result.get('accessibility') == 'accessible':
                            papers_data.append(result)
                            if (result.get('extracted_content') or
                                    result.get('abstract_summary_status') == 'generated_from_fulltext'):
                                full_text_papers.append(result)
                        else:
                            suggested_papers.append(result)

                summary_progress.empty()
                summary_status.empty()
                summary_caption.empty()

                # ── STEP 6: SAVE STATE ───────────────────────────────────────
                st.session_state.papers_data      = papers_data
                st.session_state.full_text_papers  = full_text_papers
                st.session_state.suggested_papers  = suggested_papers
                st.session_state.processing        = False

                status.update(
                    label=f"✅ Done — {len(papers_data)} papers analysed",
                    state="complete",
                    expanded=False
                )

                # ── STEP 7: CLUSTERING ───────────────────────────────────────
                if papers_data:
                    st.write("6️⃣ **Smart Clustering:** Grouping papers by research theme...")
                    try:
                        clusterer = PaperClusterer()
                        st.session_state.clusters = clusterer.cluster_papers(papers_data)
                    except Exception as ce:
                        st.warning(f"Clustering skipped: {ce}")
                        st.session_state.clusters = {}
                else:
                    st.session_state.clusters = {}

                st.balloons()

            except Exception as e:
                status.update(
                    label="❌ An error occurred during analysis.",
                    state="error"
                )
                st.error(f"Error: {str(e)}")
                st.session_state.processing = False

    

# ==================== MAIN CONTENT ====================
if st.session_state.papers_data:
    # Enhanced metrics
    render_metrics(st.session_state.papers_data, st.session_state.full_text_papers, st.session_state.clusters)
    
    # Clean tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        " Dashboard", 
        " Papers & Summaries", 
        " Research Gaps",
        " Restricted Reading"
    ])
    
    with tab1:
        st.markdown("### Research Dashboard")
        st.markdown("*Intelligent analysis with enhanced content extraction*")
        
        if st.session_state.clusters:
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_names = []
                cluster_counts = []
                for cluster_id, cluster_info in st.session_state.clusters.items():
                    cluster_names.append(cluster_info['name'])
                    cluster_counts.append(cluster_info.get('paper_count', len(cluster_info.get('papers', []))))

                
                colors = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#667eea', '#764ba2']
                
                fig = px.pie(
                    values=cluster_counts,
                    names=cluster_names,
                    title="Research Areas Distribution",
                    color_discrete_sequence=colors
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    font_family="Inter",
                    title_font_size=16,
                    font_size=12,
                    showlegend=True
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                citation_data = []
                area_names = []
                for cluster_info in st.session_state.clusters.values():
                    citation_data.append(cluster_info.get('avg_citations', 0))
                    area_names.append(cluster_info['name'])
                
                fig = px.bar(
                    x=citation_data,
                    y=area_names,
                    orientation='h',
                    title="Average Citations by Research Area",
                    color=citation_data,
                    color_continuous_scale=["#f8fafc", "#667eea", "#764ba2"]
                )
                fig.update_layout(
                    font_family="Inter",
                    title_font_size=16,
                    font_size=12,
                    xaxis_title="Average Citations",
                    yaxis_title="Research Area"
                )
                st.plotly_chart(fig, width='stretch')
            
            # Enhanced cluster cards
            st.markdown("### Research Themes")
            
            for cluster_id, cluster_info in st.session_state.clusters.items():
                paper_count = len(cluster_info['papers'])# Safe count from filtered papers list
                extracted_in_cluster = len([p for p in cluster_info['papers'] if p.get('extracted_content')])
                
                st.markdown(f"""
                <div class="cluster-card">
                    <div class="cluster-name">{cluster_info['name']}</div>
                    <p style="color: #64748b; margin-bottom: 1rem;">{cluster_info['description']}</p>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                            {paper_count} papers
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                            {extracted_in_cluster} content extracted
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                            {cluster_info.get('avg_citations', 0)} avg citations
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                            {cluster_info.get('avg_year', 2024)}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Complete analysis to see research themes and dashboard metrics.")
   
    with tab2:
        st.markdown("### Papers & Summaries")
        # Process any pending saves from paper cards
        if st.session_state.get('papers_to_save'):
            for paper_to_save in st.session_state.papers_to_save:
                st.success(f"✅ Saved: {paper_to_save.get('title','')[:50]}...")
            st.session_state.papers_to_save = []
        papers_data = st.session_state.get("papers_data", [])
        
        if not papers_data:
            st.info("No accessible papers available. Try another query or enable more sources.")
        else:
            # --- PAGINATION CONFIG ---
            ITEMS_PER_PAGE = 10
            
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
                
            total_count = len(papers_data)
            total_pages = max(1, (total_count + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            
            if st.session_state.current_page > total_pages:
                st.session_state.current_page = 1

            # --- INFO TEXT (TOP) ---
            # Just text, no buttons here
            st.markdown(f"""
            <div style="color: #64748b; margin-bottom: 15px; font-size: 0.95rem;">
                Showing Page <strong>{st.session_state.current_page}</strong> of <strong>{total_pages}</strong> 
                <span style="color: #94a3b8;">({total_count} total papers)</span>
            </div>
            """, unsafe_allow_html=True)

            # --- DISPLAY CURRENT PAGE ---
            start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            
            current_batch = papers_data[start_idx:end_idx]
            
            for p_idx, paper in enumerate(current_batch):
                global_idx = start_idx + p_idx
                render_paper_ui(paper, idx=global_idx)
            
            st.markdown("---")

            # --- PAGINATION CONTROLS (BOTTOM ONLY) ---
            c1, c2, c3 = st.columns([1, 3, 1])
            
            with c1:
                if st.button("← Previous", key="btn_prev", disabled=(st.session_state.current_page == 1), use_container_width=True):
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with c2:
                # Centered page indicator for easy reading at bottom
                st.markdown(f"""
                <div style="text-align: center; padding-top: 8px; color: #64748b; font-weight: 500;">
                    Page {st.session_state.current_page}
                </div>
                """, unsafe_allow_html=True)

            with c3:
                if st.button("Next →", key="btn_next", disabled=(st.session_state.current_page == total_pages), use_container_width=True):
                    st.session_state.current_page += 1
                    st.rerun()

    with tab3:
        st.markdown("### Research Gaps Analysis")
        st.markdown("*Enhanced gap analysis using extracted content*")
        
        if st.session_state.papers_data:
            gap_analyzer = ResearchGapAnalyzer()
            gaps = gap_analyzer.analyze_gaps(st.session_state.papers_data)
            
            for gap_type, gap_list in gaps.items():
                if gap_list:
                    gap_title = gap_type.replace('_', ' ').title()
                    
                    st.markdown(f"""
                    <div class="gap-card">
                        <h4 style="margin-bottom: 0.8rem; color: #7c3aed;">{gap_title}</h4>
                        <ul style="color: #374151; margin: 0; padding-left: 1.5rem;">
                    """, unsafe_allow_html=True)
                    
                    for gap in gap_list:
                        st.markdown(f"<li style='margin-bottom: 0.5rem;'>{gap}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
        else:
            st.info("Complete paper analysis to identify research gaps and opportunities.")
    
    with tab4:
        st.markdown("### Restricted Reading")

        
        st.markdown("""
        <div class="warning-box">
            <strong>Restricted Access:</strong> Access to these papers requires a subscription or paid access.
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.suggested_papers:
            st.success("Excellent! All papers are accessible. Check 'Papers & Summaries' for complete analysis with extracted content.")
        else:
            st.markdown(f"**{len(st.session_state.suggested_papers)} papers requiring paid/institutional access**")
            
            for i, paper in enumerate(st.session_state.suggested_papers, 1):
                render_suggested_paper(paper)

else:
    render_welcome_screen()


# Clean footer
if not st.session_state.processing:
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'> Intelligent research assistant with content extraction</div>", unsafe_allow_html=True)
