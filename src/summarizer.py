"""
summarizer.py
=============
LLM-based summarisation with:
  - Batched processing (5 papers per API call instead of 1)
  - Structured summary schema
  - Literature review paragraph (directly pasteable into a paper)
  - Reading order generation across all clusters
  - Research gap identification from actual content
"""

import os
import sys
import json
import time
import tempfile
import requests
import re
import logging
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import SUMMARISATION
except ImportError:
    SUMMARISATION = {
        "provider": "gemini",
        "gemini":   {"model": "gemma-3-12b-it", "max_tokens": 4000, "temperature": 0.1},
        "fallback": "extractive"
    }

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from groq import RateLimitError as GroqRateLimitError
except ImportError:
    GroqRateLimitError = Exception

MAX_CONTEXT_CHARS = 3000
BATCH_SIZE        = 5      # papers per LLM call
FULL_LLM_LIMIT    = 20     # top N papers get LLM summaries; rest get extractive


# ─────────────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────────────

def _try_fix_json(s: str) -> str:
    if not s:
        return s
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = re.sub(r'^[^\{]*\{', '{', s, count=1)
    s = re.sub(r',(\s*[\}\]])', r'\1', s)
    return s


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON object
    start, end = raw.find('{'), raw.rfind('}')
    if start != -1 and end != -1:
        candidate = raw[start:end+1]
        for attempt in [candidate, _try_fix_json(candidate)]:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
    return None


def _parse_json_array(raw: str) -> Optional[List[Dict]]:
    """Parse a JSON array from raw LLM output."""
    if not raw:
        return None
    # Strip markdown fences
    clean = re.sub(r'^```json|^```|```$', '', raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    # Try finding array
    start, end = clean.find('['), clean.rfind(']')
    if start != -1 and end != -1:
        try:
            parsed = json.loads(clean[start:end+1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _fill_schema(parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in ["Title", "Research_Problem", "Research_Objective",
              "Aim_of_Study", "limitations_and_future_work"]:
        val    = parsed.get(k) if isinstance(parsed, dict) else None
        out[k] = val if isinstance(val, str) and val.strip() else ""
    for k in ["Keywords", "Key_Findings"]:
        val    = parsed.get(k) if isinstance(parsed, dict) else None
        out[k] = [str(x) for x in val] if isinstance(val, list) else []
    mma = parsed.get("Methodology_Approach") if isinstance(parsed, dict) else None
    out["Methodology_Approach"] = {}
    for sk in ["Method", "Process", "Data_Handling", "Results_Format"]:
        sv = mma.get(sk) if isinstance(mma, dict) else None
        out["Methodology_Approach"][sk] = sv if isinstance(sv, str) and sv.strip() else ""
    # Literature review paragraph — new field
    lit = parsed.get("Literature_Review_Paragraph") if isinstance(parsed, dict) else None
    out["Literature_Review_Paragraph"] = lit if isinstance(lit, str) and lit.strip() else ""
    # Key metrics extracted from paper
    metrics = parsed.get("Key_Metrics") if isinstance(parsed, dict) else None
    out["Key_Metrics"] = [str(x) for x in metrics] if isinstance(metrics, list) else []
    return out


# ─────────────────────────────────────────────────────────────────────
# EXTRACTIVE FALLBACK
# ─────────────────────────────────────────────────────────────────────

def _extractive_summary(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Returns first 3 sentences of abstract as minimal summary."""
    abstract  = paper.get("abstract", "").strip()
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    snippet   = " ".join(sentences[:3]) if sentences else abstract[:300]

    # Build a basic lit review paragraph from available metadata
    authors  = paper.get("authors", [])
    year     = paper.get("year", "")
    first_au = authors[0].split()[-1] if authors else "Authors"
    et_al    = " et al." if len(authors) > 1 else ""
    lit_para = (
        f"{first_au}{et_al} ({year}) {snippet}"
        if year else snippet
    )

    return _fill_schema({
        "Title":                      paper.get("title", ""),
        "Research_Problem":           snippet,
        "Key_Findings":               [s for s in sentences[1:3] if s],
        "Literature_Review_Paragraph": lit_para,
        "Key_Metrics":                [],
    })


# ─────────────────────────────────────────────────────────────────────
# MAIN SUMMARISER CLASS
# ─────────────────────────────────────────────────────────────────────

class FullPaperSummarizer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.provider   = SUMMARISATION.get("provider", "gemini")
        cfg             = SUMMARISATION.get(self.provider, {})
        self.model      = cfg.get("model", "gemma-3-12b-it")
        self.max_tok    = cfg.get("max_tokens", 4000)
        self.temp       = cfg.get("temperature", 0.1)
        self.client     = None
        self.ollama_url = SUMMARISATION.get("ollama", {}).get(
            "base_url", "http://localhost:11434"
        )

        logger.info(f"[Summarizer] Provider: {self.provider}, Model: {self.model}")
        self._init_client()

    def _init_client(self):
        key_env_map = {
            "groq":      "GROQ_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "cohere":    "COHERE_API_KEY",
            "mistral":   "MISTRAL_API_KEY",
            "together":  "TOGETHER_API_KEY",
            "gemini":    "GEMINI_API_KEY",
        }
        api_key = None
        env_var = key_env_map.get(self.provider)
        if env_var:
            try:
                if env_var in st.secrets:
                    api_key = st.secrets[env_var]
            except Exception:
                pass
            api_key = api_key or os.getenv(env_var)

        try:
            if self.provider == "groq":
                from groq import Groq
                self.client = Groq(api_key=api_key) if api_key else None

            elif self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key) if api_key else None

            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

            elif self.provider == "cohere":
                import cohere
                self.client = cohere.Client(api_key) if api_key else None

            elif self.provider == "mistral":
                from mistralai import Mistral
                self.client = Mistral(api_key=api_key) if api_key else None

            elif self.provider == "together":
                from together import Together
                self.client = Together(api_key=api_key) if api_key else None

            elif self.provider == "gemini":
                import google.generativeai as genai
                key = api_key or os.getenv("GEMINI_API_KEY", "")
                genai.configure(api_key=key)
                self.client = genai.GenerativeModel(self.model)

            elif self.provider == "ollama":
                self.client = "ollama"

        except Exception as e:
            logger.error(f"[Summarizer] Failed to init {self.provider}: {e}")
            self.client = None

    # ── Raw LLM call ─────────────────────────────────────────────────

    def _llm_call(self, prompt: str) -> str:
        if not self.client:
            return ""

        try:
            if self.provider == "ollama":
                try:
                    ping = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                    models = [m["name"] for m in ping.json().get("models", [])]
                    if not models:
                        logger.error("[ollama] No models pulled.")
                        return ""
                    if not any(self.model.split(":")[0] in m for m in models):
                        self.model = models[0]
                except Exception as e:
                    logger.error(f"[ollama] Unreachable: {e}")
                    return ""

                resp = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model":   self.model,
                        "stream":  False,
                        "options": {"temperature": self.temp, "num_predict": self.max_tok},
                        "messages": [
                            {"role": "system", "content": "Output ONLY valid JSON. No markdown."},
                            {"role": "user",   "content": prompt}
                        ]
                    },
                    timeout=300
                )
                if resp.status_code != 200:
                    return ""
                data = resp.json()
                if "message" in data:
                    msg = data["message"]
                    return msg.get("content", "") if isinstance(msg, dict) else str(msg)
                return data.get("response", "")

            elif self.provider == "groq":
                time.sleep(0.5)
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temp, max_tokens=self.max_tok,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Output strictly valid JSON only."},
                        {"role": "user",   "content": prompt}
                    ]
                )
                return resp.choices[0].message.content.strip()

            elif self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temp, max_tokens=self.max_tok,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Output strictly valid JSON only."},
                        {"role": "user",   "content": prompt}
                    ]
                )
                return resp.choices[0].message.content.strip()

            elif self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tok,
                    system="Output strictly valid JSON only. No markdown.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content[0].text.strip()

            elif self.provider == "cohere":
                resp = self.client.chat(
                    model=self.model, temperature=self.temp, max_tokens=self.max_tok,
                    preamble="Output strictly valid JSON only.",
                    message=prompt
                )
                return resp.text.strip()

            elif self.provider == "mistral":
                resp = self.client.chat.complete(
                    model=self.model, temperature=self.temp, max_tokens=self.max_tok,
                    messages=[
                        {"role": "system", "content": "Output strictly valid JSON only."},
                        {"role": "user",   "content": prompt}
                    ]
                )
                return resp.choices[0].message.content.strip()

            elif self.provider == "together":
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temp, max_tokens=self.max_tok,
                    messages=[
                        {"role": "system", "content": "Output strictly valid JSON only."},
                        {"role": "user",   "content": prompt}
                    ]
                )
                return resp.choices[0].message.content.strip()

            elif self.provider == "gemini":
                time.sleep(2)   # stay under 30 RPM
                resp = self.client.generate_content(
                    f"Output strictly valid JSON only. No markdown.\n\n{prompt}"
                )
                return resp.text.strip()

        except GroqRateLimitError as e:
            wait = 5.0
            m    = re.search(r"try again in (\d+\.?\d*)s", str(e))
            if m:
                wait = float(m.group(1)) + 1.0
            logger.warning(f"[groq] Rate limit — waiting {wait:.1f}s")
            time.sleep(wait)
            return self._llm_call(prompt)

        except Exception as e:
            logger.error(f"[{self.provider}] LLM call failed: {e}")
            return ""

        return ""

    # ── Single paper summary ─────────────────────────────────────────

    def _build_prompt(self, text: str, meta: Dict, query: str, label: str) -> str:
        return f"""Analyse this academic paper and return a JSON summary.

Title: {meta.get('title', 'Unknown')}
Authors: {', '.join((meta.get('authors') or [])[:3])}
Year: {meta.get('year', '')}
Query context: {query}

Content ({label}):
{text[:MAX_CONTEXT_CHARS]}

Return ONLY this JSON (no other text):
{{
    "Title": "paper title",
    "Keywords": ["keyword1", "keyword2"],
    "Research_Problem": "what problem this paper solves (2-3 sentences)",
    "Research_Objective": "main goal",
    "Methodology_Approach": {{
        "Method": "technique or algorithm used",
        "Process": "how it was applied",
        "Data_Handling": "what data was used and how",
        "Results_Format": "how results are presented"
    }},
    "Aim_of_Study": "practical implications",
    "Key_Findings": ["finding 1 with any numbers/metrics", "finding 2"],
    "Key_Metrics": ["accuracy: X%", "dataset: Y", "baseline: Z"],
    "limitations_and_future_work": "limitations and next steps",
    "Literature_Review_Paragraph": "A 3-5 sentence paragraph in academic style that can be directly pasted into a literature review. Must include author surnames and year in parentheses citation format. Example: Smith et al. (2020) proposed... Their approach achieves... However, a limitation is..."
}}"""

    def summarize_paper(
        self,
        paper: Dict[str, Any],
        use_full_text: bool = True,
        query: str = ""
    ) -> Dict[str, Any]:

        if not self.client:
            result = _extractive_summary(paper)
            result.update({'accessibility': 'accessible',
                           'abstract_summary_status': 'extractive_fallback'})
            return result

        meta = {
            "title":   paper.get("title", "").strip(),
            "abstract":paper.get("abstract", "").strip(),
            "authors": paper.get("authors", []),
            "year":    paper.get("year", ""),
        }

        # Try full text
        if use_full_text and PYPDF2_AVAILABLE and paper.get('pdf_url'):
            extracted, _ = self._download_and_extract_pdf(paper['pdf_url'])
            if extracted:
                s = self._call_and_parse(extracted, meta, query, "Full Text")
                if s:
                    s.update({'accessibility': 'accessible',
                               'abstract_summary_status': 'generated_from_fulltext'})
                    return s

        # Abstract
        abstract = meta.get('abstract', '')
        if abstract:
            s = self._call_and_parse(abstract, meta, query, "Abstract")
            if s:
                s.update({'accessibility': 'accessible',
                           'abstract_summary_status': 'generated_from_abstract'})
                return s

        # Extractive fallback
        result = _extractive_summary(paper)
        result.update({'accessibility': 'accessible',
                       'abstract_summary_status': 'extractive_fallback'})
        return result

    def _call_and_parse(self, text, meta, query, label):
        raw    = self._llm_call(self._build_prompt(text, meta, query, label))
        if not raw:
            return None
        parsed = _parse_json(raw)
        if not parsed:
            return None
        if not (parsed.get("Research_Problem") or parsed.get("Key_Findings")):
            return None
        return _fill_schema(parsed)

    # ── BATCH summarisation (5 papers per call) ──────────────────────

    def summarize_batch(self, papers: List[Dict], query: str) -> List[Dict]:
        """
        Summarise up to 5 papers in a single LLM call.
        Returns list of summary dicts in input order.
        Falls back to extractive if batch parse fails.
        """
        if not self.client or not papers:
            return [_extractive_summary(p) for p in papers]

        papers_text = ""
        for i, p in enumerate(papers, 1):
            abstract = (p.get('abstract') or '')[:400]
            authors  = (p.get('authors') or [])
            au_str   = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
            papers_text += (
                f"\n--- PAPER {i} ---\n"
                f"Title: {p.get('title', 'Unknown')}\n"
                f"Authors: {au_str}\n"
                f"Year: {p.get('year', '')}\n"
                f"Abstract: {abstract}\n"
            )

        prompt = f"""Analyse {len(papers)} academic papers for the query: "{query}"

{papers_text}

Return a JSON ARRAY with exactly {len(papers)} objects in order.
Each object:
{{
  "Title": "paper title",
  "Keywords": ["k1", "k2"],
  "Research_Problem": "what problem this paper solves",
  "Research_Objective": "main goal",
  "Methodology_Approach": {{
    "Method": "technique used",
    "Process": "how applied",
    "Data_Handling": "data used",
    "Results_Format": "results format"
  }},
  "Aim_of_Study": "practical implications",
  "Key_Findings": ["finding 1", "finding 2"],
  "Key_Metrics": ["metric 1", "metric 2"],
  "limitations_and_future_work": "limitations",
  "Literature_Review_Paragraph": "3-5 sentence academic paragraph directly pasteable into literature review. Use Author(s) (year) citation format."
}}

Return ONLY the JSON array. No other text."""

        raw = self._llm_call(prompt)
        if raw:
            parsed = _parse_json_array(raw)
            if parsed and len(parsed) == len(papers):
                return [_fill_schema(item) for item in parsed]

        # Fallback: extractive for all in batch
        logger.warning(f"[Summarizer] Batch parse failed, using extractive for {len(papers)} papers")
        return [_extractive_summary(p) for p in papers]

    # ── Reading order ─────────────────────────────────────────────────

    def generate_reading_order(self, clusters: Dict, query: str) -> str:
        """
        Generate a suggested reading order for a newcomer.
        Returns a formatted string with 5-7 papers and why to read each.
        """
        if not self.client:
            return ""

        # Collect top paper per cluster (highest citations)
        candidates = []
        for cid, info in clusters.items():
            papers = sorted(
                info.get('papers', []),
                key=lambda p: p.get('citations') or 0,
                reverse=True
            )
            if papers:
                p = papers[0]
                candidates.append({
                    'title':     p.get('title', ''),
                    'year':      p.get('year', ''),
                    'citations': p.get('citations', 0),
                    'cluster':   info.get('name', ''),
                    'abstract':  (p.get('abstract') or '')[:200],
                })

        if not candidates:
            return ""

        papers_text = "\n".join(
            f"{i+1}. {c['title']} ({c['year']}, {c['citations']} citations) — {c['cluster']}"
            for i, c in enumerate(candidates)
        )

        prompt = f"""A student new to "{query}" needs a reading order.

Available papers (one per research cluster):
{papers_text}

Suggest 5 papers to read in order, from foundational to cutting-edge.
For each, give one sentence on why to read it at that point.
Format as a numbered list. Be specific and practical."""

        # Use a text (non-JSON) call for this
        raw = self._llm_call_text(prompt)
        return raw or ""

    def _llm_call_text(self, prompt: str) -> str:
        """Like _llm_call but for plain text output, not JSON."""
        if not self.client:
            return ""
        try:
            if self.provider == "gemini":
                time.sleep(2)
                resp = self.client.generate_content(prompt)
                return resp.text.strip()
            elif self.provider in ("groq", "openai", "mistral", "together"):
                time.sleep(0.5)
                create = (
                    self.client.chat.completions.create
                    if self.provider != "mistral"
                    else self.client.chat.complete
                )
                resp = create(
                    model=self.model, temperature=0.3, max_tokens=800,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content.strip()
            elif self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self.model, max_tokens=800,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content[0].text.strip()
        except Exception as e:
            logger.error(f"[{self.provider}] Text call failed: {e}")
        return ""

    # ── Research gaps from actual content ────────────────────────────

    def generate_research_gaps(self, clusters: Dict, query: str) -> str:
        """
        Generate actual research gaps by comparing cluster themes.
        Returns a formatted string with real gaps derived from the papers.
        """
        if not self.client or len(clusters) < 2:
            return ""

        cluster_summaries = []
        for cid, info in clusters.items():
            cluster_summaries.append(
                f"Cluster '{info.get('name', '')}': "
                f"{info.get('description', '')} "
                f"({info.get('paper_count', 0)} papers)"
            )

        summary_text = "\n".join(cluster_summaries)

        prompt = f"""Research query: "{query}"

These are the research clusters found in the literature:
{summary_text}

Identify 3-4 specific, concrete research gaps by:
1. Comparing what different clusters address vs what they leave open
2. Noting where clusters don't overlap but probably should
3. Identifying what the field has not yet studied

Be specific — name the actual topics, not generic phrases like "limited datasets".
Format as a numbered list with one sentence each."""

        return self._llm_call_text(prompt) or ""

    # ── PDF extraction ────────────────────────────────────────────────

    def _download_and_extract_pdf(self, pdf_url: str) -> Tuple[Optional[str], bool]:
        if not PYPDF2_AVAILABLE:
            return None, False
        try:
            resp = requests.get(pdf_url, timeout=20)
            if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
                return None, True
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                tmp = f.name
            text = ""
            with open(tmp, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages[:10]:
                    text += (page.extract_text() or "") + "\n\n"
            try:
                os.unlink(tmp)
            except Exception:
                pass
            return text.strip() or None, False
        except Exception as e:
            logger.warning(f"[PDF Extract] {e}")
            return None, True

    def _is_summary_useful(self, summary: Dict[str, Any]) -> bool:
        if not summary:
            return False
        return bool(
            summary.get("Research_Problem", "").strip() or
            summary.get("Key_Findings")
        )