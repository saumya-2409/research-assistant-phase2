"""
fetchers.py
===========
Fetches papers from 6 free sources silently in parallel.
User never selects sources — all sources run automatically.
Results are merged, deduplicated, and ranked before returning.

Sources:
  1. ArXiv          — cs, math, physics preprints (no key needed)
  2. Semantic Scholar — 200M papers with citations (free key optional)
  3. OpenAlex       — 250M works, best free API (no key needed)
  4. CrossRef       — 130M papers, all disciplines (no key needed)
  5. CORE           — 200M open-access papers (no key needed)
  6. PubMed         — biomedical, 35M papers (no key needed)
"""

import os
import re
import io
import time
import random
import logging
import requests
import concurrent.futures
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import urljoin

import streamlit as st

from utils.utility import deduplicate_papers

# Robust Streamlit thread context import
try:
    from streamlit.runtime.scriptrunner import add_script_run_context
except ImportError:
    try:
        from streamlit.runtime.script_run_context import add_script_run_context
    except ImportError:
        try:
            from streamlit.scriptrunner import add_script_run_context
        except ImportError:
            add_script_run_context = None

logger = logging.getLogger(__name__)

# Optional libraries
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# INTELLIGENT PAPER ACCESS DETECTOR (unchanged from original)
# ─────────────────────────────────────────────────────────────────────

class IntelligentPaperAccessor:
    """Intelligently detects and accesses papers from various sources."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        })

    def check_and_extract_paper_content(self, paper: Dict) -> Dict:
        paper = paper.copy()
        access_methods = []

        if paper.get('pdf_url'):
            access_methods.append(('direct_pdf', paper['pdf_url']))
        if paper.get('url'):
            access_methods.append(('paper_landing', paper['url']))
        if paper.get('semantic_scholar_id'):
            sid = paper['semantic_scholar_id']
            access_methods.append((
                'semantic_alternative',
                f"https://www.semanticscholar.org/paper/{sid}"
            ))
        if paper.get('doi'):
            access_methods.append(('doi_pdf', f"https://doi.org/{paper['doi']}"))

        extracted_content = None
        working_url       = None
        access_type       = None

        for method_name, url in access_methods:
            try:
                content = self.try_extract_content(url, method_name)
                if content and len(content) > 200:
                    extracted_content = content[:3000]
                    working_url       = url
                    access_type       = method_name
                    break
            except Exception:
                continue

        if extracted_content:
            paper['extracted_content'] = extracted_content
            paper['working_url']       = working_url
            paper['access_type']       = access_type
            paper['pdf_available']     = True

        return paper

    def try_extract_content(self, url: str, method_name: str) -> Optional[str]:
        try:
            response     = self.session.get(url, timeout=10, allow_redirects=True)
            if response.status_code != 200:
                return None
            content_type = response.headers.get('Content-Type', '').lower()

            if 'application/pdf' in content_type:
                content_len = len(response.content)
                if content_len < 500 * 1024 and PYPDF_AVAILABLE:
                    try:
                        reader = PdfReader(io.BytesIO(response.content))
                        text   = ''
                        for page in reader.pages[:5]:
                            text += (page.extract_text() or '') + '\n'
                        text = text.strip()[:4000]
                        if len(text) > 200:
                            return text
                    except Exception:
                        pass
                return f"PDF available ({content_len // 1024} KB)"

            elif 'text/html' in content_type:
                if not BEAUTIFULSOUP_AVAILABLE:
                    return None
                soup = BeautifulSoup(response.content, 'html.parser')
                for tag in soup(['script', 'style']):
                    tag.decompose()

                cues = ['pdf', 'download', 'full text', 'access pdf', 'view pdf']
                pdf_links = []
                for a in soup.find_all('a', href=True, limit=20):
                    href = a['href'].lower()
                    text = a.get_text(strip=True).lower()
                    if href.endswith('.pdf') or any(c in href or c in text for c in cues):
                        pdf_links.append(urljoin(url, a['href']))
                        if len(pdf_links) >= 3:
                            break

                for candidate in pdf_links:
                    try:
                        r2 = self.session.get(candidate, timeout=8, allow_redirects=True)
                        if (r2.status_code == 200 and
                                'application/pdf' in r2.headers.get('Content-Type', '')):
                            if PYPDF_AVAILABLE and len(r2.content) < 500 * 1024:
                                try:
                                    reader = PdfReader(io.BytesIO(r2.content))
                                    text   = ''
                                    for page in reader.pages[:5]:
                                        text += (page.extract_text() or '') + '\n'
                                    text = text.strip()[:4000]
                                    if len(text) > 200:
                                        return text
                                except Exception:
                                    pass
                    except Exception:
                        continue

                main = soup.find('main') or soup.find('article') or soup.body
                if main:
                    text = main.get_text(separator=' ', strip=True)[:3000]
                    if len(text) > 200:
                        return text
            return None
        except Exception:
            return None

# SOURCE 1 — ARXIV
class ArxivFetcher:
    """Fetches from ArXiv using arxiv-py. Free, no key needed."""

    def __init__(self):
        self.rate_limit_delay  = 0.5
        self.last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def search_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        if not ARXIV_AVAILABLE:
            return []
        try:
            search = arxiv.Search(
                query=f"all:{query}",
                max_results=min(max_results, 100),
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            papers = []
            for i, result in enumerate(arxiv.Client().results(search)):
                if i % 10 == 0:
                    self._rate_limit()
                papers.append({
                    'id':               result.entry_id,
                    'arxiv_id':         result.entry_id.split('/')[-1],
                    'title':            result.title,
                    'abstract':         (result.summary or '')[:1000],
                    'authors':          [a.name for a in result.authors],
                    'year':             int(result.published.year)
                                        if result.published else datetime.now().year,
                    'month':            result.published.month
                                        if result.published else 1,
                    'categories':       result.categories,
                    'url':              result.entry_id,
                    'pdf_url':          result.pdf_url,
                    'doi':              result.doi,
                    'source':           'ArXiv',
                    'citations':        None,
                    'pdf_available':    True,
                    'is_open_access':   True,
                })
                if len(papers) >= max_results:
                    break
            return papers
        except Exception as e:
            logger.warning(f"ArXiv error: {e}")
            return []

# SOURCE 2 — SEMANTIC SCHOLAR
class SemanticScholarFetcher:
    """Fetches from Semantic Scholar. Free key optional for higher limits."""

    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key  = None

        try:
            if "SEMANTIC_SCHOLAR_API_KEY" in st.secrets:
                self.api_key = st.secrets["SEMANTIC_SCHOLAR_API_KEY"]
        except Exception:
            pass
        if not self.api_key:
            self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

        self.rate_limit_delay  = 2.0 if not self.api_key else 1.0
        self.last_request_time = 0.0
        self.max_retries       = 3
        if not self.api_key:
            logger.warning(
                "Semantic Scholar running without API key — "
                "shared rate limit, may return no results. "
                "Get free key at semanticscholar.org/product/api"
            )

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed + random.uniform(0.1, 0.3))
        self.last_request_time = time.time()

    def search_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        if not query:
            return []

        last_error = None

        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                params  = {
                    'query':  query,
                    'limit':  min(max_results, 100),
                    'fields': ('paperId,title,abstract,authors,year,citationCount,'
                               'url,venue,openAccessPdf,externalIds,isOpenAccess')
                }
                headers = {
                    'User-Agent': 'Research Assistant (B.Tech Project)',
                    'Accept':     'application/json'
                }
                if self.api_key:
                    headers['x-api-key'] = self.api_key

                response = requests.get(
                    f"{self.base_url}/paper/search",
                    params=params, headers=headers, timeout=30
                )

                # Rate limited — wait the amount the server tells us, then retry
                if response.status_code == 429:
                    # Server sometimes returns Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    wait = float(retry_after) if retry_after else (3 + attempt * 3)
                    logger.warning(f"Semantic Scholar 429 — waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue

                # Auth error — try once without key, but don't permanently destroy it
                if response.status_code in (401, 403):
                    if self.api_key and attempt == 0:
                        logger.warning("Semantic Scholar key rejected, retrying without")
                        headers.pop('x-api-key', None)
                        response = requests.get(
                            f"{self.base_url}/paper/search",
                            params=params, headers=headers, timeout=30
                        )
                        # If this also fails, fall through to generic error
                    if response.status_code not in (200,):
                        last_error = f"Auth failed ({response.status_code})"
                        break   # no point retrying auth errors
                
                # Server error — retry with backoff
                if response.status_code in (500, 502, 503, 504):
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"Semantic Scholar {response.status_code} — waiting {wait}s")
                    time.sleep(wait)
                    continue

                # Any other non-200 — give up
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    break

                # Success
                papers = []
                for p in response.json().get('data', []):
                    if not p.get('title'):
                        continue
                    authors    = [a.get('name', '') for a in p.get('authors', [])[:5]]
                    ext        = p.get('externalIds', {})
                    arxiv_id   = ext.get('ArXiv')
                    doi        = ext.get('DOI')
                    oa_pdf     = p.get('openAccessPdf')
                    pdf_url    = oa_pdf['url'] if oa_pdf and oa_pdf.get('url') else None
                    if not pdf_url and arxiv_id:
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    paper_id   = p.get('paperId', '')
                    papers.append({
                        'id':                   paper_id,
                        'semantic_scholar_id':  paper_id,
                        'title':                p.get('title', ''),
                        'abstract':             (p.get('abstract') or '')[:1000],
                        'authors':              authors,
                        'year':                 int(p.get('year') or datetime.now().year),
                        'citations':            int(p.get('citationCount') or 0),
                        'url':  f"https://www.semanticscholar.org/paper/{paper_id}",
                        'pdf_url':              pdf_url,
                        'venue':                p.get('venue', ''),
                        'source':               'Semantic Scholar',
                        'pdf_available':        pdf_url is not None,
                        'is_open_access':       p.get('isOpenAccess', False),
                        'arxiv_id':             arxiv_id,
                        'doi':                  doi,
                    })
                return papers

            except requests.exceptions.Timeout:
                last_error = f"Timeout (attempt {attempt + 1})"
                time.sleep(2 * (attempt + 1))
                continue
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Semantic Scholar unexpected error: {e}")
                break

        logger.warning(f"Semantic Scholar failed after {self.max_retries} attempts: {last_error}")
        return []

# SOURCE 3 — OPENALEX  (best free API — 250M works, no key needed)
class OpenAlexFetcher:
    """
    OpenAlex is the best free academic API.
    250M works, 100K requests/day free, no key needed.
    Covers journals, conferences, and books across all disciplines.
    """

    BASE_URL = "https://api.openalex.org/works"

    def search_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        try:
            params  = {
                'search':   query,
                'per-page': min(max_results, 100),
                'select': (
                    'id,title,abstract_inverted_index,authorships,'
                    'publication_year,cited_by_count,primary_location,'
                    'open_access,doi,type'
                )
            }
            headers = {
                # Adding email gives access to the polite pool (faster)
                'User-Agent': 'mailto:research-assistant@example.com'
            }
            response = requests.get(
                self.BASE_URL, params=params, headers=headers, timeout=15
            )
            if response.status_code != 200:
                return []

            papers = []
            for item in response.json().get('results', []):
                title = item.get('title') or ''
                if not title or len(title) < 5:
                    continue

                # Reconstruct abstract from inverted index
                inv = item.get('abstract_inverted_index') or {}
                if inv:
                    positions = {pos: word
                                 for word, pos_list in inv.items()
                                 for pos in pos_list}
                    abstract = ' '.join(
                        positions[i] for i in sorted(positions)
                    )[:1000]
                else:
                    abstract = ''

                authors = [
                    a.get('author', {}).get('display_name', '')
                    for a in item.get('authorships', [])[:5]
                ]

                loc      = item.get('primary_location') or {}
                url      = loc.get('landing_page_url', '') or ''
                pdf_url  = loc.get('pdf_url', '') or ''
                oa       = item.get('open_access', {})
                doi      = item.get('doi', '')
                if doi and not doi.startswith('http'):
                    doi = f"https://doi.org/{doi}"

                papers.append({
                    'id':           item.get('id', ''),
                    'title':        title,
                    'abstract':     abstract,
                    'authors':      [a for a in authors if a],
                    'year':         item.get('publication_year') or datetime.now().year,
                    'citations':    item.get('cited_by_count', 0) or 0,
                    'url':          url or doi,
                    'pdf_url':      pdf_url or oa.get('oa_url', ''),
                    'doi':          doi,
                    'source':       'OpenAlex',
                    'is_open_access': oa.get('is_oa', False),
                    'pdf_available':  bool(pdf_url or oa.get('oa_url')),
                })
            return papers

        except Exception as e:
            logger.warning(f"OpenAlex error: {e}")
            return []

# SOURCE 4 — CROSSREF  (130M papers, all disciplines, no key needed)
class CrossRefFetcher:
    """
    CrossRef covers 130M+ papers including IEEE, ACM, Springer, Elsevier.
    Completely free, no key needed.
    Add your email to User-Agent for access to polite pool.
    """

    BASE_URL = "https://api.crossref.org/works"

    def search_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        try:
            params  = {
                'query': query,
                'rows':  min(max_results, 100),
                'select': (
                    'title,abstract,author,published,'
                    'URL,is-referenced-by-count,container-title,DOI'
                )
            }
            headers = {
                # Polite pool: add your email for better rate limits
                'User-Agent': 'Research-Assistant/1.0 (research@example.com)'
            }
            response = requests.get(
                self.BASE_URL, params=params, headers=headers, timeout=15
            )
            if response.status_code != 200:
                return []

            papers = []
            for item in response.json().get('message', {}).get('items', []):
                title_list = item.get('title', [])
                title      = title_list[0] if title_list else ''
                if not title or len(title) < 5:
                    continue

                # Strip any XML/HTML tags from CrossRef abstracts
                abstract_raw = item.get('abstract', '')
                abstract     = re.sub(r'<[^>]+>', '', abstract_raw)[:1000]

                authors = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in item.get('author', [])[:5]
                ]

                pub_date   = item.get('published', {})
                date_parts = pub_date.get('date-parts', [['']])[0]
                year       = int(date_parts[0]) if date_parts and date_parts[0] else datetime.now().year

                doi        = item.get('DOI', '')
                url        = item.get('URL', '') or (f"https://doi.org/{doi}" if doi else '')
                venue_list = item.get('container-title', [])
                venue      = venue_list[0] if venue_list else ''

                papers.append({
                    'id':           doi,
                    'title':        title,
                    'abstract':     abstract,
                    'authors':      [a for a in authors if a.strip()],
                    'year':         year,
                    'citations':    item.get('is-referenced-by-count', 0) or 0,
                    'url':          url,
                    'pdf_url':      '',   # CrossRef does not provide direct PDF links
                    'doi':          doi,
                    'venue':        venue,
                    'source':       'CrossRef',
                    'is_open_access': False,
                    'pdf_available':  False,
                })
            return papers

        except Exception as e:
            logger.warning(f"CrossRef error: {e}")
            return []

# SOURCE 5 — CORE  (200M open-access papers, no key needed for basic)
class COREFetcher:
    """
    CORE aggregates 200M+ open-access papers from repositories worldwide.
    Free basic access (10 req/min). Register at core.ac.uk for API key
    which gives 10K req/day.
    """

    BASE_URL = "https://api.core.ac.uk/v3/search/works"

    def __init__(self):
        self.api_key = os.getenv("CORE_API_KEY", "")

    def search_papers(self, query: str, max_results: int = 30) -> List[Dict]:
        try:
            params  = {
                'q':      query,
                'limit':  min(max_results, 100),
            }
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"

            response = requests.get(
                self.BASE_URL, params=params, headers=headers, timeout=15
            )
            if response.status_code not in (200, 201):
                return []

            papers = []
            for item in response.json().get('results', []):
                title = item.get('title', '') or ''
                if not title or len(title) < 5:
                    continue

                authors = [
                    a.get('name', '') for a in item.get('authors', [])[:5]
                ]
                year    = item.get('yearPublished')
                try:
                    year = int(year) if year else datetime.now().year
                except (ValueError, TypeError):
                    year = datetime.now().year

                download_url = item.get('downloadUrl', '') or ''
                doi          = item.get('doi', '') or ''

                papers.append({
                    'id':           item.get('id', ''),
                    'title':        title,
                    'abstract':     (item.get('abstract') or '')[:1000],
                    'authors':      [a for a in authors if a],
                    'year':         year,
                    'citations':    0,
                    'url':          download_url or (f"https://doi.org/{doi}" if doi else ''),
                    'pdf_url':      download_url,
                    'doi':          doi,
                    'source':       'CORE',
                    'is_open_access': True,   # CORE only has open-access papers
                    'pdf_available':  bool(download_url),
                })
            return papers

        except Exception as e:
            logger.warning(f"CORE error: {e}")
            return []

# SOURCE 6 — PUBMED  (35M biomedical papers, no key needed)
class PubMedFetcher:
    """
    PubMed covers 35M+ biomedical and life-science papers.
    Completely free, no key needed for basic access.
    Best source for medical, health, biology queries.
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search_papers(self, query: str, max_results: int = 30) -> List[Dict]:
        try:
            # Step 1: get PubMed IDs
            search_resp = requests.get(
                self.ESEARCH_URL,
                params={
                    'db':      'pubmed',
                    'term':    query,
                    'retmax':  min(max_results, 100),
                    'retmode': 'json'
                },
                timeout=15
            )
            if search_resp.status_code != 200:
                return []

            id_list = search_resp.json().get(
                'esearchresult', {}
            ).get('idlist', [])
            if not id_list:
                return []

            # Step 2: fetch details
            fetch_resp = requests.get(
                self.EFETCH_URL,
                params={
                    'db':      'pubmed',
                    'id':      ','.join(id_list),
                    'retmode': 'xml',
                    'rettype': 'abstract'
                },
                timeout=20
            )
            if fetch_resp.status_code != 200:
                return []

            root   = ET.fromstring(fetch_resp.text)
            papers = []

            for article in root.findall('.//PubmedArticle'):
                title    = article.findtext('.//ArticleTitle', '')
                abstract = article.findtext('.//AbstractText', '')
                year_str = article.findtext('.//PubDate/Year', '')
                pmid     = article.findtext('.//PMID', '')

                try:
                    year = int(year_str) if year_str else datetime.now().year
                except ValueError:
                    year = datetime.now().year

                authors = []
                for author in article.findall('.//Author')[:5]:
                    last = author.findtext('LastName', '')
                    fore = author.findtext('ForeName', '')
                    name = f"{fore} {last}".strip()
                    if name:
                        authors.append(name)

                doi_list = article.findall('.//ArticleId[@IdType="doi"]')
                doi      = doi_list[0].text if doi_list else ''

                url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                       if pmid else '')

                papers.append({
                    'id':           pmid,
                    'title':        title,
                    'abstract':     abstract[:1000],
                    'authors':      authors,
                    'year':         year,
                    'citations':    0,
                    'url':          url,
                    'pdf_url':      '',
                    'doi':          doi,
                    'source':       'PubMed',
                    'is_open_access': False,
                    'pdf_available':  False,
                })
            return papers

        except Exception as e:
            logger.warning(f"PubMed error: {e}")
            return []

# ─────────────────────────────────────────────────────────────────────
# INTELLIGENT MULTI-SOURCE FETCHER
# Runs all sources in parallel, merges, deduplicates, returns.
# User never sees which sources are queried.
# ─────────────────────────────────────────────────────────────────────

class IntelligentMultiSourceFetcher:
    """
    Fetches from all 6 free sources simultaneously.
    User only sees the final merged, deduplicated result.
    Source is stored as metadata on each paper for display.

    Source selection per query topic:
      - All queries    → ArXiv, Semantic Scholar, OpenAlex, CrossRef
      - Medical terms  → Also PubMed
      - Any query      → Also CORE (open-access supplement)
    """

    # Keywords that trigger PubMed inclusion
    BIOMEDICAL_TERMS = {
        'health', 'medical', 'clinical', 'disease', 'drug', 'patient',
        'hospital', 'treatment', 'diagnosis', 'biology', 'cancer',
        'genome', 'protein', 'neural', 'brain', 'pharma', 'therapy',
        'covid', 'virus', 'vaccine', 'biomedical', 'medicine', 'surgery'
    }

    def __init__(self):
        self.fetchers = {
            'arxiv':            ArxivFetcher(),
            'semantic_scholar': SemanticScholarFetcher(),
            'openalex':         OpenAlexFetcher(),
            'crossref':         CrossRefFetcher(),
            'core':             COREFetcher(),
            'pubmed':           PubMedFetcher(),
        }
        self.accessor = IntelligentPaperAccessor()

    def _select_sources(self, query: str) -> List[str]:
        """
        Automatically select which sources to query based on topic.
        Always uses the four main sources.
        Adds PubMed for biomedical queries.
        """
        # Always run these four
        sources = ['arxiv', 'semantic_scholar', 'openalex', 'crossref']

        # Add CORE — supplements with open-access papers
        sources.append('core')

        # Add PubMed if query contains biomedical terms
        query_lower = query.lower()
        if any(term in query_lower for term in self.BIOMEDICAL_TERMS):
            sources.append('pubmed')

        return sources


    def fetch_papers(
        self,
        query: str,
        sources=None,
        papers_per_source: int = 50,
        user_requested=None
    ):
        """
        Fetches from all relevant sources in parallel.
        Returns (papers, total_unique_found).

        Flow: Fetch all → Deduplicate → Access check → Return
        Deduplication happens BEFORE access check to avoid checking
        the same paper twice when it appears in multiple sources.

        papers_per_source: max results to request from each API
        user_requested: ignored here — slicing happens in main.py after
                        relevance filtering so user gets the best N papers
        """
        # Per-source API maximums
        SOURCE_MAXIMUMS = {
            'arxiv':            100,
            'semantic_scholar': 100,
            'openalex':         100,
            'crossref':         100,
            'core':              50,
            'pubmed':            50,
        }

        active_sources = self._select_sources(query)
        all_papers     = []

        # ── Phase 1: Parallel fetch from all sources ──────────────────
        def run_fetcher(source_name: str):
            fetcher = self.fetchers[source_name]
            limit   = SOURCE_MAXIMUMS.get(source_name, papers_per_source)
            t0      = time.time()
            results = fetcher.search_papers(query, limit)
            elapsed = time.time() - t0
            for p in results:
                p.setdefault('fetch_source', source_name)
            return source_name, results, elapsed

        source_counts       = {}
        source_results_data = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(active_sources)
        ) as executor:
            futures = {
                executor.submit(run_fetcher, src): src
                for src in active_sources
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    src_name, results, elapsed = future.result()
                    source_results_data[src_name] = (results, elapsed)
                    if results:
                        all_papers.extend(results)
                        source_counts[src_name] = len(results)
                    else:
                        source_counts[src_name] = 0
                except Exception as e:
                    logger.warning(f"Fetcher failed: {e}")

        # Render source results in main thread (safe — no threads active now)
        for src_name, (results, elapsed) in source_results_data.items():
            if results:
                st.success(
                    f"**{src_name.replace('_',' ').title()}**: "
                    f"{len(results)} papers  ({elapsed:.1f}s)"
                )
            else:
                st.caption(
                    f"{src_name.replace('_',' ').title()}: no results"
                )

        if not all_papers:
            st.warning("No papers found from any source.")
            return [], 0

        # ── Phase 2: Deduplicate BEFORE access check ──────────────────
        # Removes papers that appear in multiple sources so we don't
        # waste time running HTTP access checks on the same paper twice.
        deduplicated = deduplicate_papers(all_papers)
        st.write(
            f"   Deduplication: {len(all_papers)} raw → "
            f"{len(deduplicated)} unique papers"
        )

        # ── Phase 3: Access check on unique papers only ────────────────
        processed    = []
        progress_bar = st.progress(0)
        status_text  = st.empty()
        total        = len(deduplicated)

        def check_paper(paper):
            # Pure computation — no st.* calls inside threads
            return self.accessor.check_and_extract_paper_content(paper)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_map = {
                executor.submit(check_paper, p): p
                for p in deduplicated
            }
            done = 0
            for future in concurrent.futures.as_completed(future_map):
                processed.append(future.result())
                done += 1
                # Progress updates run in main thread — safe
                progress_bar.progress(done / total)
                status_text.text(f"Checking access: {done}/{total}")

        progress_bar.empty()
        status_text.empty()

        unique_papers = processed
        total_unique  = len(unique_papers)

        # ── Summary ────────────────────────────────────────────────────
        accessible = sum(
            1 for p in unique_papers
            if p.get('pdf_available') or p.get('working_url')
        )
        extracted = sum(
            1 for p in unique_papers
            if p.get('extracted_content')
        )
        active_source_count = len(
            [v for v in source_counts.values() if v > 0]
        )

        st.info(
            f"**Fetch complete** — "
            f"{total_unique} unique papers from "
            f"{active_source_count} sources | "
            f"{accessible} accessible | "
            f"{extracted} full-text extracted"
        )

        return unique_papers, total_unique