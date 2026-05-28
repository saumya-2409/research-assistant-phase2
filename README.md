# 🔬 AI Research Assistant

> Search any topic. Get back a structured map of the academic literature — papers fetched live from 6 databases, automatically grouped into themes, with AI-generated summaries — in under 30 seconds.

**🚀 [Live App](https://research-assistant-phase2.streamlit.app/)** · 📹 [Demo Video](YOUR_VIDEO_LINK_HERE)

---

## What it does

Academic search engines return flat, unsorted lists. You get 200 papers with no idea how they relate or what the sub-topics are. This tool organises them automatically.

Type a topic → the app fetches real, verified papers from 6 sources in parallel → groups them into labelled research themes using NLP + ML → and generates a structured AI summary for each paper. No hallucinations, no fake citations — everything links back to a real DOI.

---

## Demo

[![Watch demo](assets/thumbnail.png)](YOUR_VIDEO_LINK_HERE)

---

## Screenshots

| Dashboard | Paper Cards |
|---|---|
| ![dashboard](assets/dashboard.png) | ![papers](assets/papers.png) |

---

## How it works

The backend is an 11-module Python pipeline:

```
Query
  → Parallel fetch across 6 APIs (ThreadPoolExecutor)
  → Deduplication + composite ranking (50% semantic similarity · 30% citations · 20% recency)
  → SBERT embedding (all-MiniLM-L6-v2, 384-dim)
  → UMAP compression (384-dim → 30-dim)
  → Silhouette-guided Ward clustering (auto cluster count)
  → LLM cluster labelling (LLaMA 3.1-8B via Groq)
  → Per-paper AI summarisation
  → Streamlit dashboard
```

Three engineering decisions worth calling out:

**1. UMAP before clustering, not optional.** Raw 384-dimensional SBERT vectors are geometrically too flat — all pairwise distances converge, making clustering nearly random. Compressing to 30 dimensions (chosen by sensitivity analysis across 5–50) fixes this and accounts for most of the accuracy improvement over a naive baseline.

**2. Automatic cluster count via Silhouette sweep.** Instead of asking the user to guess how many groups there are, the app sweeps `k` from 2 to `min(√(N/2), 15, N/3)` and picks the value that maximises the mean Silhouette score. No tuning required.

**3. Four-tier label fallback chain.** LLM labels are better, but APIs go down. The pipeline cascades: Groq (LLaMA) → KeyBERT → c-TF-IDF → keyword frequency. The app always produces a label.

---

## Tech stack

| | |
|---|---|
| **Language** | Python 3.10+ |
| **UI** | Streamlit |
| **Embeddings** | `sentence-transformers` — all-MiniLM-L6-v2 |
| **Dim. reduction** | `umap-learn` |
| **Clustering** | `scikit-learn` — AgglomerativeClustering (Ward) |
| **LLM** | LLaMA 3.1-8B via Groq API (configurable — see below) |
| **Data sources** | arXiv, Semantic Scholar, OpenAlex, CrossRef, CORE, PubMed |
| **Database** | SQLite |
| **Visualisation** | Plotly |
| **Export** | openpyxl (Excel), BibTeX |

---

## Features

- Fetches up to 300 papers per search across 6 sources simultaneously
- Deduplicates by DOI and normalised title; keeps the record with richest metadata
- Ranks by a weighted composite of semantic relevance, citation count, and recency
- Clusters papers into themes automatically — no user input on group count
- Generates 3–7 word, specific cluster labels (e.g. *"Federated Learning Privacy Guarantees"*, not *"Machine & Learning & Network"*)
- AI-structured summary per paper: research problem, methodology, key findings, limitations
- 7-day search cache — repeat queries return in under 1 second
- User authentication (login / signup) with session management
- Accessibility badges — Open Access vs. Paywalled, with PDF links where available
- Interactive Plotly scatter plot — hover any dot to see the paper title
- Export results as a formatted Excel spreadsheet or BibTeX file

---

## Repository structure

```
research-assistant-phase2/
├── src/
│   ├── main.py              # Streamlit app — UI, session state, tab routing
│   ├── fetchers.py          # 6 API clients, deduplication, paywall detection
│   ├── embedding_utils.py   # Sentence transformer (cached), relevance scoring
│   ├── clustering.py        # UMAP → Silhouette sweep → Ward clustering → label chain
│   ├── summarizer.py        # LLM client, JSON parsing + repair, extractive fallback
│   ├── config.py            # All system parameters in one place
│   ├── database.py          # SQLite — auth tables, search history, 7-day cache
│   ├── auth.py              # Login/signup UI, session gate
│   ├── utility.py           # Ranking, deduplication, text cleaning
│   ├── display.py           # Paper cards, metric widgets, badges
│   └── export.py            # Excel + BibTeX generation
├── notebooks/               # Benchmark evaluation, sensitivity analysis
├── assets/
├── requirements.txt
└── setup_project.py
```

---

## Run locally

```bash
git clone https://github.com/saumya-2409/research-assistant-phase2.git
cd research-assistant-phase2
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key_here"   # free tier, required
CORE_API_KEY = "your_key_here"   # optional — better open-access PDF coverage
```

arXiv, Semantic Scholar, OpenAlex, CrossRef, and PubMed need no keys.

```bash
streamlit run src/main.py
# → http://localhost:8501
```

---

## Configurable LLM backend

Swap in any provider by setting the right key — no code changes needed.

| Provider | Models | Cost |
|---|---|---|
| Groq | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile` | Free tier |
| Gemini | `gemini-1.5-flash`, `gemini-2.0-flash` | Free tier |
| Ollama | `llama3.2`, `mistral`, `gemma2` | Free (local) |
| OpenAI | `gpt-4o-mini`, `gpt-4o` | Paid |
| Anthropic | `claude-haiku-4-5`, `claude-sonnet-4-6` | Paid |

---

## Performance

Benchmarked on a 1,200-document, 12-category subset of 20 Newsgroups — chosen specifically because the categories heavily overlap in vocabulary (e.g. atheism vs. religion), which is a realistic stand-in for adjacent academic sub-fields sharing the same terminology.

| Model | Cluster Acc. | F1 | NMI |
|---|---|---|---|
| TF-IDF + K-Means | 40.8% | 40.6% | 0.426 |
| SBERT + Agglomerative *(no UMAP)* | 66.9% | 64.6% | 0.662 |
| GMM | 74.4% | 73.5% | 0.722 |
| Fuzzy C-Means | 75.5% | 75.4% | 0.719 |
| **This system** | **75.6%** | **74.4%** | **0.732** |

Adding UMAP alone jumps accuracy by **+8.7 points** and Silhouette score by **+0.451** vs. the no-UMAP baseline.

---

## What I'd add next

- [ ] Swap embedding model to SPECTER (citation-aware, trained on scientific papers)
- [ ] Migrate SQLite to PostgreSQL (Supabase) for persistence across cloud restarts
- [ ] Citation network graph — visualise which papers bridge different clusters
- [ ] "Chat with Paper" — RAG interface over retrieved full-text PDFs
- [ ] Multi-query mode — merge searches on different facets into one map
- [ ] Browser extension — cluster results directly on arXiv / Google Scholar

---

## License

[MIT](LICENSE) — use it, fork it, build on it.

---

Made by [Saumya Garg](https://linkedin.com/in/saumya-garg-1ab39224b) · [GitHub](https://github.com/saumya-2409)
