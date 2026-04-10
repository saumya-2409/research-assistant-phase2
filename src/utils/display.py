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

import streamlit.components.v1 as components

def _copy_button(text: str, key: str):
    """Renders a small JS-powered copy-to-clipboard button."""
    escaped = text.replace('`', '\\`').replace('\\', '\\\\').replace('\n', '\\n')
    components.html(f"""
    <button onclick="
        navigator.clipboard.writeText(`{escaped}`)
            .then(() => {{
                this.innerText = '✅ Copied!';
                setTimeout(() => this.innerText = '📋 Copy', 1800);
            }})
            .catch(() => this.innerText = '❌ Failed');
    " style="
        background:#5B4EE8;color:white;border:none;
        padding:7px 16px;border-radius:8px;cursor:pointer;
        font-size:13px;font-weight:500;font-family:Inter,sans-serif;
        transition:background 0.15s;
    ">📋 Copy</button>
    """, height=42)
    
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
# CITATION FORMATTERS
# ─────────────────────────────────────────────────────────────────────

def _format_apa(paper: dict) -> str:
    """Generate APA 7th edition citation string."""
    authors = paper.get('authors', []) or []
    year    = paper.get('year', 'n.d.')
    title   = paper.get('title', 'Untitled')
    source  = paper.get('source', '')
    url     = paper.get('url', '') or paper.get('pdf_url', '') or ''
    doi     = paper.get('doi', '') or ''

    if isinstance(authors, list) and authors:
        clean = [str(a).strip() for a in authors if a]
        if not clean:
            au_str = 'Unknown Author'
        elif len(clean) == 1:
            au_str = clean[0]
        elif len(clean) <= 20:
            au_str = ', '.join(clean[:-1]) + ', & ' + clean[-1]
        else:
            au_str = ', '.join(clean[:19]) + ', ... ' + clean[-1]
    else:
        au_str = str(authors).strip() if authors else 'Unknown Author'

    cite = f"{au_str} ({year}). {title}."
    if source and source.lower() not in ['unknown', '']:
        cite += f" *{source}*."
    if doi:
        cite += f" https://doi.org/{doi}"
    elif url:
        cite += f" {url}"
    return cite


def _format_mla(paper: dict) -> str:
    """Generate MLA 9th edition citation string."""
    authors = paper.get('authors', []) or []
    year    = paper.get('year', 'n.d.')
    title   = paper.get('title', 'Untitled')
    source  = paper.get('source', '')
    url     = paper.get('url', '') or ''

    if isinstance(authors, list) and authors:
        clean = [str(a).strip() for a in authors if a]
        if not clean:
            au_str = 'Unknown Author'
        elif len(clean) == 1:
            parts = clean[0].rsplit(' ', 1)
            au_str = f"{parts[-1]}, {parts[0]}" if len(parts) > 1 else clean[0]
        else:
            parts = clean[0].rsplit(' ', 1)
            first = f"{parts[-1]}, {parts[0]}" if len(parts) > 1 else clean[0]
            au_str = first + ', et al.' if len(clean) > 1 else first
    else:
        au_str = str(authors).strip() if authors else 'Unknown Author'

    cite = f'{au_str}. "{title}."'
    if source and source.lower() not in ['unknown', '']:
        cite += f" *{source}*,"
    cite += f" {year}."
    if url:
        cite += f" {url}."
    return cite


def _format_bibtex_entry(paper: dict) -> str:
    """Generate a single BibTeX @article entry."""
    authors = paper.get('authors', []) or []
    year    = str(paper.get('year', '0000'))
    title   = paper.get('title', 'Untitled')
    url     = paper.get('url', '') or ''
    doi     = paper.get('doi', '') or ''

    if isinstance(authors, list) and authors:
        surname    = str(authors[0]).strip().split()[-1].lower() if authors[0] else 'unknown'
        author_str = ' and '.join(str(a) for a in authors)
    else:
        surname    = 'unknown'
        author_str = str(authors) if authors else 'Unknown'

    # Sanitise key
    key = f"{surname}{year}"
    key = ''.join(c for c in key if c.isalnum())

    entry  = f"@article{{{key},\n"
    entry += f"  title   = {{{title}}},\n"
    entry += f"  author  = {{{author_str}}},\n"
    entry += f"  year    = {{{year}}},\n"
    if doi:
        entry += f"  doi     = {{{doi}}},\n"
    if url:
        entry += f"  url     = {{{url}}}\n"
    entry += "}"
    return entry


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
            <h1>🔬 AI Research Assistant</h1>
            <p>Real papers. Organised. Ready to use.</p>
        </div>
        {badge}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# METRIC CARDS — use st.metric (always renders, no CSS dependency)
# ─────────────────────────────────────────────────────────────────────

def render_metrics(papers_data, full_text_papers, clusters):
    sources_count = len(set(p.get('source', '') for p in papers_data))
    avg_cites     = (
        int(sum(int(p.get('citations') or 0) for p in papers_data) / len(papers_data))
        if papers_data else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("📄 Papers", len(papers_data))
    with c2:
        st.metric("🔓 Accessible", len(full_text_papers))
    with c3:
        st.metric("🗄️ Sources", sources_count)
    with c4:
        st.metric("🗂️ Clusters", len(clusters))
    with c5:
        st.metric("🎓 Avg Citations", avg_cites)


# ─────────────────────────────────────────────────────────────────────
# PAPER BODY — shared between expander and inline modes
# ─────────────────────────────────────────────────────────────────────

def _render_paper_body(paper: dict, idx: int = 0, show_start_here: bool = False):
    def s(x): return "" if x is None else str(x)

    year    = s(paper.get("year", ""))
    cites   = s(paper.get("citations") or "N/A")
    source  = s(paper.get("source", ""))
    tag_cls = _source_tag_class(source)
    label   = paper.get("paper_label", "")
    authors = _format_authors(paper.get("authors") or [])
    rel_score = paper.get("relevance_score", None)

    is_fulltext = paper.get("abstract_summary_status") == "generated_from_fulltext"
    status_tag  = (
        '<span class="tag tag-teal">📄 full text</span>'
        if is_fulltext else
        '<span class="tag tag-amber">⚡ abstract only</span>'
    )
    label_html  = _label_badge(label)
    start_html  = (
        '<span style="background:#5B4EE8;color:white;padding:2px 8px;'
        'border-radius:20px;font-size:10px;font-weight:600;margin-right:4px;">'
        '⭐ Start Here</span>'
    ) if show_start_here else ''

    rel_html = (
        f'<span style="font-size:12px;color:#9B97C4;">🎯 {rel_score:.2f} relevance</span>'
        if rel_score is not None else ''
    )

    # Metadata row
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:12px;">
        {start_html}{status_tag}
        <span class="tag {tag_cls}">{source}</span>
        {label_html}
        <span style="font-size:12px;color:#9B97C4;">📅 {year}</span>
        <span style="font-size:12px;color:#9B97C4;">🎓 {cites} citations</span>
        {rel_html}
        <span style="font-size:12px;color:#9B97C4;">✍️ {authors}</span>
    </div>
    """, unsafe_allow_html=True)

    # Extract summary fields
    summary      = paper.get("ai_summary") or {}
    problem      = summary.get("Research_Problem", "").strip()
    objective    = summary.get("Research_Objective", "").strip()
    implications = summary.get("Aim_of_Study", "").strip()
    limitations  = summary.get("limitations_and_future_work", "")
    if isinstance(limitations, list):
        limitations = " ".join(limitations)
    limitations  = limitations.strip() if limitations else ""
    lit_para     = summary.get("Literature_Review_Paragraph", "").strip()
    key_metrics  = [m for m in (summary.get("Key_Metrics") or []) if m]
    mma          = summary.get("Methodology_Approach") or {}
    method       = (mma.get("Method",       "") if isinstance(mma, dict) else "").strip()
    process      = (mma.get("Process",      "") if isinstance(mma, dict) else "").strip()
    data_h       = (mma.get("Data_Handling","") if isinstance(mma, dict) else "").strip()
    results_f    = (mma.get("Results_Format","")if isinstance(mma, dict) else "").strip()
    findings     = [f for f in (summary.get("Key_Findings") or []) if f]
    if isinstance(findings, str):
        findings = [findings] if findings else []
    keywords     = [k for k in (summary.get("Keywords") or []) if k]
    abstract     = paper.get("abstract", "") or ""

    working_url = paper.get("working_url") or paper.get("url", "")
    pdf_url     = paper.get("pdf_url", "")

    # ── 4 summary tabs ────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["📋 Summary", "📖 Lit Review ¶", "📎 Cite"])

    # ── Tab 1: Summary ────────────────────────────────────────────────
    with t1:
        has_llm = any([problem, objective, method, findings, implications, limitations])

        if has_llm:
            # Row 1: Problem + Objective side by side
            if problem or objective:
                col_a, col_b = st.columns(2, gap="medium")
                with col_a:
                    if problem:
                        st.markdown("""<div style="background:#F0F4FF;border-left:3px solid #5B4EE8;
                            border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:8px;">
                            <div style="font-size:11px;font-weight:700;color:#5B4EE8;
                                text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">
                            🧐 Problem Statement</div>""", unsafe_allow_html=True)
                        st.write(problem)
                        st.markdown("</div>", unsafe_allow_html=True)
                with col_b:
                    if objective:
                        st.markdown("""<div style="background:#F0FFF8;border-left:3px solid #1D9E75;
                            border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:8px;">
                            <div style="font-size:11px;font-weight:700;color:#1D9E75;
                                text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">
                            🎯 Objective</div>""", unsafe_allow_html=True)
                        st.write(objective)
                        st.markdown("</div>", unsafe_allow_html=True)

            # Row 2: Methodology + Key Findings side by side
            has_method   = any([method, process, data_h, results_f])
            has_findings = bool(findings)
            if has_method or has_findings:
                col_c, col_d = st.columns(2, gap="medium")
                with col_c:
                    if has_method:
                        with st.container(border=True):
                            st.markdown("**⚙️ Methodology**")
                            if method:    st.markdown(f"**Method:** {method}")
                            if process:   st.markdown(f"**Process:** {process}")
                            if data_h:    st.markdown(f"**Data:** {data_h}")
                            if results_f: st.markdown(f"**Results format:** {results_f}")
                with col_d:
                    if has_findings:
                        with st.container(border=True):
                            st.markdown("**💡 Key Findings**")
                            for f in findings[:4]:
                                st.markdown(f"- {f}")

            # Row 3: Implications + Limitations side by side
            if implications or limitations:
                col_e, col_f = st.columns(2, gap="medium")
                with col_e:
                    if implications:
                        with st.container(border=True):
                            st.markdown("**🚀 Implications**")
                            st.write(implications)
                with col_f:
                    if limitations:
                        with st.container(border=True):
                            st.markdown("**⚠️ Limitations & Future Work**")
                            st.write(limitations)
        else:
            # No LLM summary — show abstract as fallback
            if abstract:
                st.markdown(f"""
                <div style="background:#FAFAFA;border:1px solid #E8E6FF;border-radius:8px;
                    padding:14px 16px;line-height:1.6;color:#374151;font-size:13px;">
                    <div style="font-size:11px;font-weight:700;color:#9B97C4;
                        text-transform:uppercase;margin-bottom:8px;">Abstract</div>
                    {abstract[:600]}{'...' if len(abstract) > 600 else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("No summary available. Run with a configured LLM API key for AI summaries.")

        if keywords:
            st.caption(f"**Keywords:** {', '.join(str(k) for k in keywords[:12])}")

    # ── Tab 2: Literature Review ──────────────────────────────────────
    with t2:
        st.markdown("""
        <div style="background:#F0EEFF;border:1px solid #D4CFFF;border-radius:8px;
                    padding:10px 14px;margin-bottom:10px;font-size:12px;color:#3D35A8;">
            ℹ️ Select all (Ctrl+A) and copy into your literature review.
            Always verify facts before submitting.
        </div>
        """, unsafe_allow_html=True)

        if lit_para:
            st.text_area(
                "Literature review paragraph",
                value=lit_para,
                height=180,
                key=f"lit_{idx}_{hash(paper.get('title',''))%99991}",
                label_visibility="collapsed"
            )
        else:
            authors_list = paper.get("authors", []) or []
            au   = str(authors_list[0]).split()[-1] if authors_list else "Authors"
            et   = " et al." if len(authors_list) > 1 else ""
            yr   = paper.get("year", "n.d.")
            body = problem or (abstract[:250] + "..." if abstract else "conducted a study in this area.")
            find = f" Key findings include: {findings[0]}" if findings else ""
            basic = f"{au}{et} ({yr}) {body}{find}"
            st.text_area(
                "Literature review paragraph",
                value=basic,
                height=140,
                key=f"lit_basic_{idx}_{hash(paper.get('title',''))%99991}",
                label_visibility="collapsed"
            )
            st.caption("Auto-generated from abstract. Configure an LLM for richer paragraphs.")

    # ── Tab 3: Cite ───────────────────────────────────────────────────
    with t3:
   
        cite_key = f"cite_{idx}_{hash(paper.get('title',''))%99991}"

        apa_str    = _format_apa(paper)
        mla_str    = _format_mla(paper)
        bibtex_str = _format_bibtex_entry(paper)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**APA 7th Edition**")
            st.text_area("APA", value=apa_str, height=90,
                     key=f"apa_{cite_key}", label_visibility="collapsed")
        with col2:
            st.markdown("**MLA 9th Edition**")
            st.text_area("MLA", value=mla_str, height=90,
                     key=f"mla_{cite_key}", label_visibility="collapsed")

        st.markdown("**BibTeX**")
        st.text_area("BibTeX", value=bibtex_str, height=130,
                     key=f"bib_{cite_key}", label_visibility="collapsed")

        st.markdown("""
        <div style="background:#FFF8E7;border:1px solid #F5C875;border-radius:8px;
                    padding:10px 14px;font-size:12px;color:#633806;margin-top:8px;">
            ⚠️ Auto-generated citations. Always verify against the original paper
            for accuracy before submitting to journals or assignments.
        </div>
        """, unsafe_allow_html=True)

    # ── Action buttons ─────────────────────────────────────────────────
    st.divider()
    btn_cols = st.columns(4)
    access_type = paper.get("access_type", "")
    btn_label   = "📥 Download PDF" if access_type == "direct_pdf" else "🔗 Open Paper"
    if working_url:
        with btn_cols[0]:
            st.link_button(btn_label, working_url, use_container_width=True)
    if pdf_url and pdf_url != working_url:
        with btn_cols[1]:
            st.link_button("📄 Direct PDF", pdf_url, use_container_width=True)
    with btn_cols[2]:
        save_key = f"save_btn_{idx}_{hash(paper.get('title',''))%99991}"
        if st.button("🔖 Save", key=save_key, use_container_width=True):
            if 'saved_papers_session' not in st.session_state:
                st.session_state.saved_papers_session = []
            # Avoid duplicates
            existing_titles = {p.get('title') for p in st.session_state.saved_papers_session}
            if paper.get('title') not in existing_titles:
                st.session_state.saved_papers_session.append(paper)
                st.toast("✅ Paper saved to your library!")
            else:
                st.toast("Already in your library.")


# ─────────────────────────────────────────────────────────────────────
# PAPER CARD — with expander (Papers tab)
# ─────────────────────────────────────────────────────────────────────

def render_paper_ui(paper: dict, idx: int = 0, show_start_here: bool = False):
    def s(x): return "" if x is None else str(x)

    title       = s(paper.get("title") or
                    (paper.get("ai_summary") or {}).get("Title") or
                    "Research Paper")
    is_fulltext = paper.get("abstract_summary_status") == "generated_from_fulltext"
    start_badge = "⭐ " if show_start_here else ""
    label_tag   = paper.get("paper_label", "")
    label_emoji = {"Foundational": "⭐", "Current": "✅", "Emerging": "🚀"}.get(label_tag, "")
    year        = s(paper.get("year", ""))
    cites       = s(paper.get("citations") or "")
    cites_str   = f" · {cites} cites" if cites and cites != "0" else ""
    icon        = "📄" if is_fulltext else "⚡"
    expander_label = f"{icon} {start_badge}{label_emoji} {title}  [{year}{cites_str}]"

    with st.expander(expander_label, expanded=False):
        _render_paper_body(paper, idx=idx, show_start_here=show_start_here)


# ─────────────────────────────────────────────────────────────────────
# PAPER CARD — flat/inline (Cluster view in tab1)
# ─────────────────────────────────────────────────────────────────────

def render_paper_inline(paper: dict, idx: int = 0, show_start_here: bool = False):
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
    apa     = _format_apa(paper)

    with st.expander(f"🔒 {title}", expanded=False):
        st.markdown(f"""
        <div style="font-size:12px;color:#9B97C4;margin-bottom:10px;">
            {authors} &nbsp;·&nbsp; {src} &nbsp;·&nbsp; {year}
            &nbsp;·&nbsp; {cites} citations
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**APA Citation:**")
        st.text_area("APA", value=apa, height=80,
                     key=f"restricted_apa_{hash(title)%99991}",
                     label_visibility="collapsed")
        if url and url != "#":
            st.link_button("🔗 Try to Access", url)


# ─────────────────────────────────────────────────────────────────────
# SAVED PAPER (compact card for saved papers tab)
# ─────────────────────────────────────────────────────────────────────

def render_saved_paper_card(paper: dict, idx: int = 0):
    title   = paper.get("title", "Unknown Title")
    authors = _format_authors(paper.get("authors", []))
    year    = paper.get("year", "")
    cites   = paper.get("citations", "N/A")
    src     = paper.get("source", "")
    url     = paper.get("url", "")
    apa     = _format_apa(paper)

    with st.expander(f"🔖 {title}", expanded=False):
        st.markdown(f"""
        <div style="font-size:12px;color:#9B97C4;margin-bottom:10px;">
            {authors} &nbsp;·&nbsp; {src} &nbsp;·&nbsp; {year}
            &nbsp;·&nbsp; {cites} citations
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Citation (APA):**")
            st.text_area("APA", value=apa, height=80,
                         key=f"saved_apa_{idx}_{hash(title)%99991}",
                         label_visibility="collapsed")
        with col2:
            if url:
                st.link_button("🔗 Open", url, use_container_width=True)
            remove_key = f"remove_saved_{idx}_{hash(title)%99991}"
            if st.button("🗑️ Remove", key=remove_key, use_container_width=True):
                st.session_state.saved_papers_session.pop(idx)
                st.rerun()


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
            Enter a topic in the sidebar and click Start.
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
         "System fetches max from APIs, shows you the best N ranked by citations & recency"),
        ("3", "step-active", "Click Start",
         "6 sources queried in parallel — ArXiv, OpenAlex, CrossRef, Semantic Scholar, CORE, PubMed"),
        ("4", "step-wait",   "Get your research brief",
         "Papers clustered, AI summaries, citation formats, and literature review paragraphs ready"),
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
