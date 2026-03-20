import streamlit as st
from typing import List, Dict

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _source_tag_class(source: str) -> str:
    s = source.lower()
    if "arxiv"    in s: return "tag-purple"
    if "semantic" in s: return "tag-blue"
    if "openalex" in s: return "tag-teal"
    if "crossref" in s: return "tag-amber"
    if "core"     in s: return "tag-green"
    if "pubmed"   in s: return "tag-coral"
    return "tag-gray"


def _label_badge(label: str) -> str:
    badges = {
        'Foundational': 'background:#EEEDFE;color:#3C3489',
        'Current':      'background:#E1F5EE;color:#0F6E56',
        'Emerging':     'background:#FAEEDA;color:#633806',
    }
    icons = {'Foundational': '⭐', 'Current': '✅', 'Emerging': '🚀'}
    style = badges.get(label, '')
    icon  = icons.get(label, '')
    if not style:
        return ''
    return (
        f'<span style="{style};padding:2px 8px;border-radius:20px;'
        f'font-size:10px;font-weight:600;margin-right:4px;">'
        f'{icon} {label}</span>'
    )


def _format_authors(authors_raw) -> str:
    if isinstance(authors_raw, list):
        au = ", ".join(str(a) for a in authors_raw[:3])
        return au + " et al." if len(authors_raw) > 3 else au
    return str(authors_raw) if authors_raw else ""


# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────

def render_header(n_papers: int = 0, n_sources: int = 0):
    badge = (
        f'<span class="header-badge">{n_papers} papers · {n_sources} sources</span>'
        if n_papers else
        '<span class="header-badge">6 sources · AI-powered</span>'
    )
    st.markdown(f"""
    <div class="app-header">
        <div>
            <h1>AI Research Assistant</h1>
            <p>Real papers. Organised. Ready to use.</p>
        </div>
        {badge}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────

def render_metrics(papers_data, full_text_papers, clusters):
    sources_count = len(set(p.get('source', '') for p in papers_data))
    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, len(papers_data),      "Papers Analysed"),
        (c2, len(full_text_papers), "Accessible"),
        (c3, sources_count,         "Sources Used"),
        (c4, len(clusters),         "Clusters Found"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{num}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAPER BODY — shared between expander and inline modes
# ─────────────────────────────────────────────────────────────────────

def _render_paper_body(paper: dict, idx: int = 0, show_start_here: bool = False):
    """
    Renders the inside content of a paper card.
    Called from both render_paper_ui (expander) and render_paper_inline (flat).
    idx must be unique per paper on the page to avoid duplicate widget keys.
    """
    def s(x): return "" if x is None else str(x)

    year    = s(paper.get("year", ""))
    cites   = s(paper.get("citations") or "N/A")
    source  = s(paper.get("source", ""))
    tag_cls = _source_tag_class(source)
    label   = paper.get("paper_label", "")
    authors = _format_authors(paper.get("authors") or [])

    is_fulltext = paper.get("abstract_summary_status") == "generated_from_fulltext"
    status_tag  = (
        '<span class="tag tag-teal">full text</span>'
        if is_fulltext else
        '<span class="tag tag-amber">abstract only</span>'
    )
    label_html  = _label_badge(label)
    start_html  = (
        '<span style="background:#5B4EE8;color:white;padding:2px 8px;'
        'border-radius:20px;font-size:10px;font-weight:600;margin-right:4px;">'
        '⭐ Read This First</span>'
    ) if show_start_here else ''

    # Metadata row
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:12px;">
        {start_html}{status_tag}
        <span class="tag {tag_cls}">{source}</span>
        {label_html}
        <span style="font-size:12px;color:#9B97C4;">📅 {year}</span>
        <span style="font-size:12px;color:#9B97C4;">🎓 {cites} citations</span>
        <span style="font-size:12px;color:#9B97C4;">✍️ {authors}</span>
    </div>
    """, unsafe_allow_html=True)

    # Extract summary fields
    summary        = paper.get("ai_summary") or {}
    problem        = summary.get("Research_Problem", "").strip()
    objective      = summary.get("Research_Objective", "").strip()
    implications   = summary.get("Aim_of_Study", "").strip()
    limitations    = summary.get("limitations_and_future_work", "")
    if isinstance(limitations, list):
        limitations = " ".join(limitations)
    limitations    = limitations.strip() if limitations else ""
    lit_para       = summary.get("Literature_Review_Paragraph", "").strip()
    key_metrics    = [m for m in (summary.get("Key_Metrics") or []) if m]
    mma            = summary.get("Methodology_Approach") or {}
    method         = (mma.get("Method", "") if isinstance(mma, dict) else "").strip()
    process        = (mma.get("Process", "") if isinstance(mma, dict) else "").strip()
    data_h         = (mma.get("Data_Handling", "") if isinstance(mma, dict) else "").strip()
    results_f      = (mma.get("Results_Format", "") if isinstance(mma, dict) else "").strip()
    findings       = [f for f in (summary.get("Key_Findings") or []) if f]
    if isinstance(findings, str):
        findings   = [findings] if findings else []
    keywords       = [k for k in (summary.get("Keywords") or []) if k]

    working_url    = paper.get("working_url") or paper.get("url", "")
    pdf_url        = paper.get("pdf_url", "")

    # ── Summary tabs ─────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["📋 Summary", "📖 Literature Review ¶", "📊 Metrics"])

    with t1:
        # Only show rows that have content — issue #9
        has_row1 = problem or objective
        has_method = any([method, process, data_h, results_f])
        has_findings = bool(findings)
        has_row3 = implications or limitations

        if has_row1:
            r1c1, r1c2 = st.columns(2, gap="medium")
            if problem:
                with r1c1:
                    st.info(f"**🧐 Problem Statement**\n\n{problem}")
            if objective:
                with r1c2:
                    st.success(f"**🎯 Research Objective**\n\n{objective}")

        if has_method or has_findings:
            r2c1, r2c2 = st.columns(2, gap="medium")
            if has_method:
                with r2c1:
                    with st.container(border=True):
                        st.markdown("**⚙️ Methodology**")
                        if method:    st.markdown(f"- **Method:** {method}")
                        if process:   st.markdown(f"- **Process:** {process}")
                        if data_h:    st.markdown(f"- **Data:** {data_h}")
                        if results_f: st.markdown(f"- **Results:** {results_f}")
            if has_findings:
                with r2c2:
                    with st.container(border=True):
                        st.markdown("**💡 Key Findings**")
                        for f in findings:
                            st.markdown(f"- {f}")

        if has_row3:
            r3c1, r3c2 = st.columns(2, gap="medium")
            if implications:
                with r3c1:
                    with st.container(border=True):
                        st.markdown("**🚀 Implications**")
                        st.write(implications)
            if limitations:
                with r3c2:
                    with st.container(border=True):
                        st.markdown("**⚠️ Limitations & Future Work**")
                        st.write(limitations)

        if not any([has_row1, has_method, has_findings, has_row3]):
            abstract = paper.get("abstract", "")
            if abstract:
                st.markdown(f"**Abstract:** {abstract[:500]}{'...' if len(abstract)>500 else ''}")
            else:
                st.caption("No summary available for this paper.")

        if keywords:
            st.caption(f"**Keywords:** {', '.join(str(k) for k in keywords[:12])}")

    with t2:
        # Literature review — issue #7: use text_area (scrollable, bigger, selectable)
        st.markdown("""
        <div style="background:#F0EEFF;border:1px solid #D4CFFF;border-radius:8px;
                    padding:10px 14px;margin-bottom:10px;font-size:12px;color:#3D35A8;">
            Select all text below (Ctrl+A) and copy into your literature review.
            Always verify facts before submitting.
        </div>
        """, unsafe_allow_html=True)

        if lit_para:
            # text_area is scrollable, selectable, copyable — issue #2 & #7
            st.text_area(
                "Literature review paragraph",
                value=lit_para,
                height=180,
                key=f"lit_{idx}_{hash(paper.get('title',''))%9999}",
                label_visibility="collapsed"
            )
        else:
            # Build a basic one from available data
            authors_list = paper.get("authors", [])
            au   = authors_list[0].split()[-1] if authors_list else "Authors"
            et   = " et al." if len(authors_list) > 1 else ""
            year = paper.get("year", "")
            body = problem or (paper.get("abstract", "")[:200] + "...")
            basic = f"{au}{et} ({year}) {body} {'Key findings include: ' + findings[0] if findings else ''}"
            st.text_area(
                "Literature review paragraph",
                value=basic,
                height=140,
                key=f"lit_basic_{idx}_{hash(paper.get('title',''))%9999}",
                label_visibility="collapsed"
            )
            st.caption("Basic auto-generated paragraph. Run with LLM for richer version.")

    with t3:
        if key_metrics:
            st.markdown("**📊 Key Metrics & Numbers from this paper:**")
            for m in key_metrics:
                st.markdown(f"- {m}")
        else:
            abstract = paper.get("abstract", "")
            if abstract:
                st.caption("No specific metrics extracted. Read the full paper for quantitative results.")
            else:
                st.caption("No metrics available.")

    # ── Action buttons ────────────────────────────────────────────────
    st.divider()
    btn_cols = st.columns(4)
    access_type = paper.get("access_type", "")
    btn_label   = "Access Paper (PDF)" if access_type == "direct_pdf" else "🔗 Open Paper"
    if working_url:
        with btn_cols[0]:
            st.link_button(btn_label, working_url, use_container_width=True)
    if pdf_url and pdf_url != working_url:
        with btn_cols[1]:
            st.link_button("📄 Direct PDF", pdf_url, use_container_width=True)
    # Save button — issue #8: uses unique key, stores in session state for main.py to handle
    with btn_cols[2]:
        save_key = f"save_btn_{idx}_{hash(paper.get('title',''))%9999}"
        if st.button("🔖 Save", key=save_key, use_container_width=True):
            if 'papers_to_save' not in st.session_state:
                st.session_state.papers_to_save = []
            st.session_state.papers_to_save.append(paper)
            st.toast("Paper saved to your library!")


# ─────────────────────────────────────────────────────────────────────
# PAPER CARD — with expander (for tab2 / Papers list)
# ─────────────────────────────────────────────────────────────────────

def render_paper_ui(paper: dict, idx: int = 0, show_start_here: bool = False):
    """Standard paper card with collapsible expander. Use in Papers tab."""
    def s(x): return "" if x is None else str(x)

    title       = s(paper.get("title") or
                    (paper.get("ai_summary") or {}).get("Title") or
                    "Research Paper")
    is_fulltext = paper.get("abstract_summary_status") == "generated_from_fulltext"
    start_badge = "⭐ " if show_start_here else ""
    label       = f"{'📄' if is_fulltext else '⚡'} {start_badge}{title}"

    with st.expander(label, expanded=False):
        _render_paper_body(paper, idx=idx, show_start_here=show_start_here)


# ─────────────────────────────────────────────────────────────────────
# PAPER CARD — flat/inline (for tab1 / Cluster view, no nesting)
# ─────────────────────────────────────────────────────────────────────

def render_paper_inline(paper: dict, idx: int = 0, show_start_here: bool = False):
    """
    Flat paper card without expander wrapper.
    Use inside cluster toggles in tab1 to avoid nested expanders.
    """
    def s(x): return "" if x is None else str(x)

    title = s(paper.get("title") or
               (paper.get("ai_summary") or {}).get("Title") or
               "Research Paper")
    is_fulltext = paper.get("abstract_summary_status") == "generated_from_fulltext"
    start_badge = "⭐ " if show_start_here else ""
    icon        = "📄" if is_fulltext else "⚡"

    st.markdown(f"""
    <div style="background:white;border:1px solid #E8E6FF;border-radius:14px;
                padding:16px 20px;margin-bottom:10px;">
        <div style="font-size:15px;font-weight:600;color:#1A1744;margin-bottom:8px;">
            {icon} {start_badge}{title}
        </div>
    """, unsafe_allow_html=True)

    _render_paper_body(paper, idx=idx, show_start_here=show_start_here)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# RESTRICTED PAPER
# ─────────────────────────────────────────────────────────────────────

def render_suggested_paper(paper: Dict):
    authors = _format_authors(paper.get("authors", []))
    url     = paper.get("url", "#")
    title   = paper.get("title", "Unknown Title")
    src     = paper.get("source", "Unknown")
    year    = paper.get("year", "")
    cites   = paper.get("citations", "N/A")

    st.markdown(f"""
    <div class="restricted-card">
        <div class="restricted-title">{title}</div>
        <div class="restricted-meta">
            {authors} &nbsp;·&nbsp; {src} &nbsp;·&nbsp; {year}
            &nbsp;·&nbsp; {cites} citations
        </div>
        <a href="{url}" target="_blank"
           style="font-size:13px;color:#D85A30;font-weight:500;text-decoration:none;">
            Access via library / Sci-Hub →
        </a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────────────────────────────

def render_welcome_screen():
    st.markdown("""
    <div style="text-align:center;padding:40px 20px 28px;">
        <div style="font-size:40px;margin-bottom:12px;">🔬</div>
        <div style="font-size:22px;font-weight:600;color:#1A1744;margin-bottom:8px;">
            What are you researching?
        </div>
        <div style="font-size:14px;color:#9B97C4;">
            Enter a topic in the sidebar and click Search.
            The system queries 6 academic databases simultaneously.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-bottom:24px;">
        <div style="font-size:11px;font-weight:600;color:#9B97C4;
                    text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">
            Try these topics
        </div>
        <span class="tag tag-purple">transformer attention</span>
        <span class="tag tag-teal">federated learning</span>
        <span class="tag tag-amber">CRISPR gene therapy</span>
        <span class="tag tag-coral">climate neural networks</span>
        <span class="tag tag-blue">RAG knowledge graphs</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    steps = [
        ("1", "step-done",   "Enter your research topic",
         "Any keyword, author name, or research area"),
        ("2", "step-done",   "Set paper count (10–100)",
         "System fetches max from each API, shows you the best N ranked by citations & recency"),
        ("3", "step-active", "Click Search",
         "6 sources queried in parallel — ArXiv, OpenAlex, CrossRef, Semantic Scholar, CORE, PubMed"),
        ("4", "step-wait",   "Get your research brief",
         "Papers clustered, AI summaries, literature review paragraphs ready to paste"),
    ]

    for num, cls, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:16px;">
            <div class="step-dot {cls}">{num}</div>
            <div>
                <div style="font-size:14px;font-weight:600;
                            color:#1A1744;margin-bottom:3px;">{title}</div>
                <div style="font-size:13px;color:#9B97C4;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    sources = [
        ("ArXiv",            ARXIV_AVAILABLE, "No key needed"),
        ("Semantic Scholar", True,            "Free key recommended"),
        ("OpenAlex",         True,            "No key · 250M works"),
        ("CrossRef",         True,            "No key · 130M papers"),
        ("CORE",             True,            "Free key recommended"),
        ("PubMed",           True,            "No key · biomedical"),
    ]

    col1, col2 = st.columns(2)
    for i, (name, available, note) in enumerate(sources):
        col  = col1 if i % 2 == 0 else col2
        dot  = "#1D9E75" if available else "#D85A30"
        lbl  = "ready" if available else "install required"
        tcls = "tag-teal" if available else "tag-coral"
        with col:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;
                        background:white;border:1px solid #E8E6FF;
                        border-radius:10px;padding:10px 14px;margin-bottom:8px;">
                <div style="width:8px;height:8px;border-radius:50%;
                            background:{dot};flex-shrink:0;"></div>
                <div>
                    <div style="font-size:13px;font-weight:500;color:#1A1744;">{name}</div>
                    <div style="font-size:11px;color:#9B97C4;">{note}</div>
                </div>
                <div style="margin-left:auto;">
                    <span class="tag {tcls}">{lbl}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)