"""
Utility modules for Fake News Detection Streamlit App.
"""

from .preprocessing import (
    MAX_LEN,
    VOCAB_SIZE,
    clean_text,
    extract_first_paragraph,
    combine_title_and_body,
    preprocess_for_prediction,
    get_prediction_label,
    validate_input
)

from .scraper import (
    scrape_article,
    validate_url
)

__all__ = [
    'MAX_LEN',
    'VOCAB_SIZE',
    'clean_text',
    'extract_first_paragraph',
    'combine_title_and_body',
    'preprocess_for_prediction',
    'get_prediction_label',
    'validate_input',
    'scrape_article',
    'validate_url'
]
