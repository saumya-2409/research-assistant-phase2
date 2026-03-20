# src/utils/__init__.py
# Makes utils/ a proper Python package.
# Re-exports the most commonly used functions so both:
#   from utils import deduplicate_papers          ← works
#   from utils.utility import deduplicate_papers  ← also works

from utils.utility import (
    deduplicate_papers,
    rank_papers, 
    clean_text,
    extract_keywords,
    validate_paper_data,
    format_authors,
    generate_paper_id,
    merge_paper_data,
    categorize_papers,
    is_paywalled_response,
)