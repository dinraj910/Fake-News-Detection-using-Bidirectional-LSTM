"""
Preprocessing utilities for Fake News Detection.
Replicates notebook preprocessing behavior exactly.
"""

import re
from typing import Tuple, Optional, List
import numpy as np


# Constants matching training configuration
MAX_LEN = 500
VOCAB_SIZE = 20000


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    Removes extra whitespace while preserving content.
    """
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_first_paragraph(body: str, min_words: int = 20) -> str:
    """
    Extract the first meaningful paragraph from article body.
    
    Args:
        body: Full article body text
        min_words: Minimum words for a paragraph to be considered valid
    
    Returns:
        First valid paragraph or empty string if none found
    """
    if not body:
        return ""
    
    # Split into paragraphs
    paragraphs = body.split("\n")
    
    # Filter out short/noisy paragraphs
    valid_paragraphs = [
        p.strip() for p in paragraphs 
        if len(p.split()) >= min_words
    ]
    
    if not valid_paragraphs:
        # If no valid paragraph found, try to use the whole text
        # but only if it has some content
        cleaned_body = clean_text(body)
        if len(cleaned_body.split()) >= min_words:
            return cleaned_body
        return ""
    
    return valid_paragraphs[0]


def combine_title_and_body(title: str, body: str) -> str:
    """
    Combine title and first paragraph of body.
    Matches the training format: title + " " + text
    
    Args:
        title: News article title
        body: News article body (will extract first paragraph only)
    
    Returns:
        Combined text ready for tokenization
    """
    clean_title = clean_text(title)
    first_para = extract_first_paragraph(body)
    
    if clean_title and first_para:
        return f"{clean_title} {first_para}"
    elif clean_title:
        return clean_title
    elif first_para:
        return first_para
    return ""


def preprocess_for_prediction(
    tokenizer,
    text: str
) -> Tuple[np.ndarray, int]:
    """
    Preprocess text for model prediction.
    Replicates notebook preprocessing exactly.
    
    Args:
        tokenizer: Loaded Keras Tokenizer
        text: Combined title + first paragraph text
    
    Returns:
        Tuple of (padded_sequence, word_count)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    word_count = len(text.split())
    
    # Tokenize
    sequences = tokenizer.texts_to_sequences([text])
    
    # Pad sequences (post padding, post truncating - matches training)
    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )
    
    return padded, word_count


def get_prediction_label(probability: float) -> Tuple[str, str]:
    """
    Convert probability to label.
    Model output: P(Real) - probability of being real news.
    
    Args:
        probability: Model output (sigmoid probability)
    
    Returns:
        Tuple of (label, confidence_description)
    """
    if probability > 0.5:
        label = "Real News"
        confidence = probability
    else:
        label = "Fake News"
        confidence = 1 - probability
    
    if confidence > 0.9:
        desc = "Very High Confidence"
    elif confidence > 0.75:
        desc = "High Confidence"
    elif confidence > 0.6:
        desc = "Moderate Confidence"
    else:
        desc = "Low Confidence"
    
    return label, desc


def validate_input(text: str, min_words: int = 10) -> Tuple[bool, str]:
    """
    Validate input text before prediction.
    
    Args:
        text: Input text to validate
        min_words: Minimum required words
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Input text is empty."
    
    word_count = len(text.split())
    
    if word_count < min_words:
        return False, f"Input too short ({word_count} words). Minimum {min_words} words recommended for reliable prediction."
    
    return True, ""
