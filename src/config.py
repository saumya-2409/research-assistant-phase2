# config.py
# ─────────────────────────────────────────────────────────────────────
# Central configuration for the Research Assistant.
# ALL system behaviour is controlled from here.
# To upgrade any component, change the value in this file only.
# No other files need to be touched.
# ─────────────────────────────────────────────────────────────────────

# ── PAPER SOURCES ─────────────────────────────────────────────────────
# NOTE: Not yet wired — fetchers.py uses SOURCE_MAXIMUMS internally
SOURCES = {
    "arxiv": {
        "enabled": True,
        "max_results": 50,
        "api_key": None,        # Always free, no key needed
    },
    "semantic_scholar": {
        "enabled": True,
        "max_results": 50,
        "api_key": None,        # Free: 100 req/5min without key
                                # Paid key: much higher limits
                                # Set via env: SEMANTIC_SCHOLAR_API_KEY
    },
    "openalex": {
        "enabled": False,       # Set True to enable
        "max_results": 30,
        "email": None,          # Optional: polite pool (faster)
    },
    "crossref": {
        "enabled": False,       # Set True to enable
        "max_results": 30,
    },
    "pubmed": {
        "enabled": False,       # Good for biomedical queries
        "max_results": 30,
        "api_key": None,
    },
}

# ── EMBEDDING MODEL ───────────────────────────────────────────────────
# Free options ranked by quality (change model name to upgrade):
#   "all-MiniLM-L6-v2"     fast,  384-dim, ~72% accuracy,  80MB
#   "all-MiniLM-L12-v2"    better, 384-dim, ~75% accuracy, 120MB
#   "all-mpnet-base-v2"    good,   768-dim, ~79% accuracy, 420MB
#   "allenai-specter"      best for papers, ~82%, 420MB
#
# Paid options (requires API key in env):
#   "text-embedding-3-small"  OpenAI  → set OPENAI_API_KEY
#   "voyage-lite-02-instruct" Voyage  → set VOYAGE_API_KEY

# NOTE: Not yet wired — embedding_utils.py hardcodes the model name
EMBEDDING = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
}

# ── CLUSTERING PARAMETERS ─────────────────────────────────────────────
CLUSTERING = {
    # UMAP settings — proven optimal via sensitivity analysis
    "umap_components":  15,     # Output dimensions (15 > 10 per analysis)
    "umap_neighbors":   20,     # n_neighbors parameter
    "umap_min_dist":    0.0,    # Tighter clusters
    "umap_metric":     "cosine",

    # Silhouette sweep range
    # max_k is computed adaptively as min(sqrt(N/2), 15, N//3)
    # These are absolute bounds
    "min_k":           3,
    "abs_max_k":       15,

    # LLM labelling
    "titles_for_label": 5,      # Top N papers sent to LLM per cluster
}

# ── LLM FOR CLUSTER LABELS ────────────────────────────────────────────
# Priority chain: system tries each in order until one works
LABELLING = {
    "priority": ["groq", "keybert", "ctfidf", "keyword"],

    "groq": {
        "enabled":             True,
        "model":               "llama-3.1-8b-instant",
        "fallback_on_ratelimit": True,
        # Key loaded from env: GROQ_API_KEY
        # or from .streamlit/secrets.toml: [GROQ_API_KEY]
    },
    "openai": {
        "enabled": False,       # Paid
        "model":   "gpt-4o-mini",
        # Key from env: OPENAI_API_KEY
    },
    "keybert": {
        "enabled":      True,   # Always free
        "ngram_range": (2, 4),
        "top_n":        1,
        "diversity":    0.5,
    },
}

# ── LLM FOR SUMMARISATION ─────────────────────────────────────────────
# Set "provider" to whichever API key you have.
# System tries the chosen provider first, falls back through the chain.
# Free options: groq (best free), ollama (local)
# Paid options: openai, anthropic, cohere, mistral, together

SUMMARISATION = {

    # ── Active provider — change this one value to switch ────────────
    "provider": "gemini",   # options: groq | openai | anthropic |
                           #          cohere | mistral | together | ollama

    # ── GROQ (FREE — best free option) ───────────────────────────────
    # Models: llama-3.1-8b-instant, llama-3.3-70b-versatile,
    #         mixtral-8x7b-32768, gemma2-9b-it
    # Limits: 6,000 tokens/min on free tier
    # Key:    console.groq.com
    "groq": {
        "api_key":    None,          # set GROQ_API_KEY in .env
        "model":      "llama-3.1-8b-instant",
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── OPENAI (PAID) ─────────────────────────────────────────────────
    # Models: gpt-4o-mini (cheapest), gpt-4o, gpt-4-turbo
    # Cost:   gpt-4o-mini ~$0.15/1M input tokens
    # Key:    platform.openai.com
    "openai": {
        "api_key":    None,          # set OPENAI_API_KEY in .env
        "model":      "gpt-4o-mini",
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── ANTHROPIC CLAUDE (PAID) ───────────────────────────────────────
    # Models: claude-haiku-4-5-20251001 (cheapest), claude-sonnet-4-6 (best)
    # Cost:   Haiku ~$0.25/1M input tokens
    # Key:    console.anthropic.com
    "anthropic": {
        "api_key":    None,          # set ANTHROPIC_API_KEY in .env
        "model":      "claude-haiku-4-5-20251001",
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── COHERE (FREE TRIAL + PAID) ────────────────────────────────────
    # Models: command-r (free trial), command-r-plus (paid)
    # Free:   1,000 API calls/month on trial key
    # Key:    dashboard.cohere.com
    "cohere": {
        "api_key":    None,          # set COHERE_API_KEY in .env
        "model":      "command-r",
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── MISTRAL (FREE TRIAL + PAID) ───────────────────────────────────
    # Models: mistral-small-latest (cheapest), mistral-large-latest
    # Free:   free tier available at la plateforme
    # Key:    console.mistral.ai
    "mistral": {
        "api_key":    None,          # set MISTRAL_API_KEY in .env
        "model":      "mistral-small-latest",
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── TOGETHER AI (FREE CREDITS + PAID) ────────────────────────────
    # Models: meta-llama/Llama-3.3-70B-Instruct-Turbo (best open model)
    #         mistralai/Mixtral-8x7B-Instruct-v0.1
    # Free:   $5 free credits on signup
    # Key:    api.together.xyz
    "together": {
        "api_key":    None,          # set TOGETHER_API_KEY in .env
        "model":      "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── OLLAMA (FREE — runs locally, no internet needed) ─────────────
    # Models: llama3.2, mistral, gemma2, phi3 — whatever you pulled
    # Install: ollama.com → pull a model → runs on localhost:11434
    # No key needed — completely offline
    "ollama": {
        "api_key":    None,          # no key needed
        "base_url":   "http://localhost:11434",
        "model":      "llama3.2:1b",    # must be pulled: ollama pull llama3.2
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── GOOGLE GEMINI (FREE TIER + PAID) ─────────────────────────────
    # Free tier: 15 req/min, 1M tokens/day on Flash — best free quota
    # Models: gemini-1.5-flash (fastest/free), gemini-1.5-pro (better)
    #         gemini-2.0-flash (latest, still free tier)
    # Key:    aistudio.google.com → Get API key (free, no card needed)
    "gemini": {
        "api_key":    None,          # set GEMINI_API_KEY in .env
        "model":      "gemma-3-12b-it",
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── META LLAMA API (FREE CREDITS + PAID) ─────────────────────────
    # Direct from Meta — official Llama API
    # Models: Llama-3.3-70B-Instruct, Llama-3.1-8B-Instruct
    # Free:   $5 free credits on signup
    # Key:    llama.developer.meta.com
    # Note:   If you want Llama locally for free, use "ollama" above
    "llama": {
        "api_key":    None,          # set LLAMA_API_KEY in .env
        "model":      "Llama-3.3-70B-Instruct",
        "max_tokens": 4000,
        "temperature": 0.1,
    },

    # ── Fallback if all LLM calls fail ───────────────────────────────
    # "extractive" = returns first 3 sentences of abstract
    # "none"       = returns empty summary rather than crashing
    "fallback": "extractive",
}

# ── RETRIEVAL ─────────────────────────────────────────────────────────
RETRIEVAL = {
    "default_papers":       30,   # Shown to user by default
    "max_papers":          100,   # Maximum user can request
    "relevance_threshold": 0.25,  # Min cosine similarity to keep paper
    "min_abstract_len":     50,   # Filter papers with very short abstracts
}

# ── DISPLAY ───────────────────────────────────────────────────────────
DISPLAY = {
    "papers_per_page": 10,        # Pagination in Papers & Summaries tab
    "max_abstract_chars": 300,    # Truncation in cluster cards
}
