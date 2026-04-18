"""
Utility functions for paper processing.
"""

import hashlib
import requests
import re
import math
from typing import List, Dict
from collections import Counter
from datetime import datetime

CURRENT_YEAR = datetime.now().year


def is_paywalled_response(response: requests.Response) -> bool:
    if response.status_code in (401, 403):
        return True
    ctype = response.headers.get("Content-Type", "").lower()
    if "html" in ctype:
        return True
    if "pdf" not in ctype and not response.content.startswith(b"%PDF"):
        return True
    return False


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    if not text:
        return []
    text = text.lower()
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    words           = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    meaningful      = [w for w in words if w not in stop_words]
    return [w for w, _ in Counter(meaningful).most_common(max_keywords)]


def get_paper_label(paper: Dict) -> str:
    """
    Classify paper as Foundational / Current / Emerging.
    Based on age and citation count.
    """
    year      = int(paper.get('year') or 0)
    citations = int(paper.get('citations') or 0)
    age       = CURRENT_YEAR - year

    if age <= 2:
        return 'Emerging'
    elif age <= 5 and citations >= 20:
        return 'Current'
    elif citations >= 200:
        return 'Foundational'
    elif age <= 5:
        return 'Current'
    else:
        return 'Foundational'


def rank_papers(papers: List[Dict]) -> List[Dict]:
    """
    Rank papers using a composite score:

      score = relevance (heaviest) + age_adjusted_citations + recency +
              full_text_bonus + abstract_quality

    Rules:
      - Papers with 0 citations AND older than 2 years AND not ArXiv → penalised
      - Papers older than 10 years kept only if citations >= 300
        (landmark papers that must be known)
      - Recent papers (≤2 years) get full recency bonus regardless of citations
      - Full text available → significant bonus
    """
    qualified    = []
    noise_papers = []

    for p in papers:
        citations    = int(p.get('citations') or 0)
        year         = int(p.get('year') or 0)
        age          = CURRENT_YEAR - year if year else 99
        source       = (p.get('source') or '').lower()
        is_arxiv     = 'arxiv' in source
        abstract_len = len(p.get('abstract') or '')
        has_full     = bool(p.get('pdf_available') or p.get('extracted_content'))

        # Hard filters — noise
        # Very old papers with no citations are almost certainly noise
        if age > 10 and citations < 300:
            noise_papers.append(p)
            p['_rank_score']  = -99.0
            p['paper_label']  = 'Old'
            continue

        score = 0.0

        # ── Relevance (already computed by embedding step) ───────────
        score += float(p.get('relevance_score', 0.0)) * 5.0

        # ── Age-adjusted citation score ──────────────────────────────
        # Normalise by age so a 2022 paper with 150 citations is
        # comparable to a 2016 paper with 600 citations.
        adjusted_cites = citations / max(1, age)
        score += min(adjusted_cites * 0.5, 8.0)

        # ── Recency bonus ─────────────────────────────────────────────
        # Linear: 2025→5pts, 2024→4pts, 2023→3pts, ... 2020→0pts
        if year >= 2020:
            score += min(year - 2020, 5) * 0.8

        # ── Zero-citation penalty ─────────────────────────────────────
        # Recent papers and ArXiv papers legitimately have 0 citations.
        # Old non-ArXiv papers with 0 citations are usually noise.
        if citations == 0 and age > 2 and not is_arxiv:
            score -= 8.0

        # ── Full text available ───────────────────────────────────────
        score += 3.0 if has_full else 0.0

        # ── Abstract quality ──────────────────────────────────────────
        score += 2.0 if abstract_len > 200 else (1.0 if abstract_len > 80 else 0.0)

        p['_rank_score'] = round(score, 3)
        p['paper_label'] = get_paper_label(p)
        qualified.append(p)

    # Sort qualified papers best first, then append noise at the bottom
    qualified.sort(key=lambda x: x.get('_rank_score', 0), reverse=True)

    # Mark the top paper per cluster as "Start Here" if cluster info present
    # (This is set later by clustering — left as placeholder)

    return qualified + noise_papers


def categorize_papers(papers: List[Dict]) -> Dict[str, List[Dict]]:
    full_text, abstract_only = [], []
    for p in papers:
        if (p.get('pdf_available') or p.get('pdf_url') or
                'arxiv' in (p.get('source') or '').lower()):
            full_text.append(p)
        else:
            abstract_only.append(p)
    return {'full_text': full_text, 'abstract_only': abstract_only}


def validate_paper_data(paper: Dict) -> Dict:
    cleaned = {}
    cleaned['id']     = str(paper.get('id', ''))
    cleaned['title']  = clean_text(paper.get('title', 'Untitled'))
    cleaned['source'] = paper.get('source', 'unknown')

    authors = paper.get('authors', [])
    if isinstance(authors, list):
        cleaned['authors'] = [clean_text(str(a)) for a in authors if a]
    else:
        cleaned['authors'] = []

    year = paper.get('year')
    if year:
        try:
            y = int(year)
            if 1900 <= y <= 2030:
                cleaned['year'] = y
        except (ValueError, TypeError):
            pass

    abstract = paper.get('abstract', '')
    if abstract:
        cleaned['abstract'] = clean_text(str(abstract))

    for url_field in ['url', 'pdf_url']:
        url = paper.get(url_field)
        if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
            cleaned[url_field] = url

    venue = paper.get('venue')
    if venue:
        cleaned['venue'] = clean_text(str(venue))

    citations = paper.get('citations', 0)
    try:
        cleaned['citations'] = max(0, int(citations))
    except (ValueError, TypeError):
        cleaned['citations'] = 0

    for bf in ['pdf_available', 'full_text_available']:
        if bf in paper:
            cleaned[bf] = bool(paper[bf])

    return cleaned


def format_authors(authors: List[str], max_authors: int = 3) -> str:
    if not authors:
        return "Unknown Authors"
    clean = [a.strip() for a in authors if a and a.strip()]
    if not clean:
        return "Unknown Authors"
    if len(clean) <= max_authors:
        return ", ".join(clean)
    return f"{', '.join(clean[:max_authors])} et al."


def generate_paper_id(paper: Dict) -> str:
    authors   = paper.get('authors', [])
    id_string = f"{paper.get('title', '')}_{authors[0] if authors else ''}_{paper.get('year', '')}"
    return hashlib.md5(id_string.encode()).hexdigest()[:16]


def merge_paper_data(paper1: Dict, paper2: Dict) -> Dict:
    merged = paper1.copy()
    for key, value in paper2.items():
        if key not in merged or not merged[key]:
            merged[key] = value
        elif key == 'authors':
            existing = set(merged.get('authors', []))
            all_authors = list(merged.get('authors', []))
            for a in paper2.get('authors', []):
                if a not in existing:
                    all_authors.append(a)
            merged['authors'] = all_authors
        elif key == 'citations':
            merged[key] = max(merged.get(key, 0), value or 0)
        elif key in ['abstract', 'full_text']:
            if len(str(value or '')) > len(str(merged.get(key, ''))):
                merged[key] = value
    return merged


def deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    """
    Remove duplicates. Priority order for keeping:
      1. DOI match (exact duplicate across sources)
      2. Normalised title match (fuzzy)
    Keeps the version with highest information score.
    """
    if not papers:
        return []

    # First pass: deduplicate by DOI
    seen_dois   = {}
    no_doi      = []
    for p in papers:
        doi = (p.get('doi') or '').strip().lower()
        if doi:
            if doi not in seen_dois:
                seen_dois[doi] = p
            else:
                if _calculate_paper_score(p) > _calculate_paper_score(seen_dois[doi]):
                    seen_dois[doi] = p
        else:
            no_doi.append(p)

    doi_deduped = list(seen_dois.values())

    # Second pass: deduplicate by normalised title
    unique      = []
    seen_titles = set()
    for p in doi_deduped + no_doi:
        title = p.get('title', '').lower().strip()
        key   = re.sub(r'[^\w\s]', '', title)
        key   = re.sub(r'\s+', ' ', key).strip()
        if not key:
            continue
        if key not in seen_titles:
            unique.append(p)
            seen_titles.add(key)
        else:
            _handle_duplicate(unique, p, key)

    return unique


def _handle_duplicate(unique_papers, new_paper, title_key):
    for i, p in enumerate(unique_papers):
        t = re.sub(r'[^\w\s]', '', p.get('title', '').lower())
        t = re.sub(r'\s+', ' ', t).strip()
        if t == title_key:
            if _calculate_paper_score(new_paper) > _calculate_paper_score(p):
                unique_papers[i] = new_paper
            break


def _calculate_paper_score(paper: Dict) -> int:
    score = 0
    if paper.get('extracted_content'):  score += 50
    if paper.get('pdf_url'):            score += 20
    abstract = paper.get('abstract', '')
    if abstract and len(abstract) > 100: score += 10
    authors = paper.get('authors', [])
    if authors:                         score += min(len(authors), 5)
    if paper.get('year'):               score += 5
    citations = int(paper.get('citations') or 0)
    score += min(citations // 100, 10)
    return score
