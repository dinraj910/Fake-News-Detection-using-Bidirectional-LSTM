"""
Fake News Detection - Streamlit Web Application

A professional web application for detecting fake news using a 
Bidirectional LSTM model trained on the WELFake dataset.

Features:
- Manual input mode (Title + Body)
- URL scraping mode
- Real-time prediction with confidence scores
"""

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from pathlib import Path
import sys

# Add the app directory to path for imports
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

from utils import (
    MAX_LEN,
    combine_title_and_body,
    preprocess_for_prediction,
    get_prediction_label,
    validate_input,
    scrape_article,
    validate_url,
    clean_text,
    extract_first_paragraph
)


# =============================================================================
# Configuration
# =============================================================================

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths to model files (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "fake_news_bilstm.h5"
TOKENIZER_PATH = PROJECT_ROOT / "model" / "tokenizer.pkl"


# =============================================================================
# Model Loading (Cached)
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained Keras model. Cached to prevent reloading."""
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"


@st.cache_resource
def load_tokenizer():
    """Load the trained tokenizer. Cached to prevent reloading."""
    try:
        with open(str(TOKENIZER_PATH), "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer, None
    except Exception as e:
        return None, f"Failed to load tokenizer: {str(e)}"


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_news(model, tokenizer, text: str) -> dict:
    """
    Make prediction on preprocessed text.
    
    Args:
        model: Loaded Keras model
        tokenizer: Loaded tokenizer
        text: Preprocessed text (title + first paragraph)
    
    Returns:
        Dictionary with prediction results
    """
    # Validate input
    is_valid, error_msg = validate_input(text, min_words=10)
    
    # Preprocess
    padded, word_count = preprocess_for_prediction(tokenizer, text)
    
    # Check if tokenizer produced valid sequences
    if padded.sum() == 0:
        return {
            "success": False,
            "error": "Tokenizer returned empty sequence. Input may contain only unknown words."
        }
    
    # Predict
    probability = float(model.predict(padded, verbose=0)[0][0])
    
    # Get label and confidence
    label, confidence_desc = get_prediction_label(probability)
    
    return {
        "success": True,
        "label": label,
        "probability_real": probability,
        "probability_fake": 1 - probability,
        "confidence_description": confidence_desc,
        "word_count": word_count,
        "warning": None if is_valid else error_msg
    }


# =============================================================================
# UI Components
# =============================================================================

def display_header():
    """Display application header."""
    st.title("📰 Fake News Detector")
    st.markdown("""
    **Powered by Bidirectional LSTM Deep Learning Model**
    
    Analyze news articles to detect potential fake news based on linguistic patterns.
    """)
    
    # Disclaimer
    st.info("""
    ⚠️ **Disclaimer:** This model detects stylistic patterns based on the training dataset (WELFake). 
    It does **not** verify factual truth. Always cross-reference with trusted sources.
    """)


def display_prediction_results(result: dict, processed_text: str = None):
    """Display prediction results with visual components."""
    
    if not result.get("success"):
        st.error(f"❌ {result.get('error', 'Unknown error occurred.')}")
        return
    
    # Warning for short inputs
    if result.get("warning"):
        st.warning(f"⚠️ {result['warning']}")
    
    # Main prediction display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result["label"] == "Real News":
            st.success(f"### ✅ {result['label']}")
        else:
            st.error(f"### ❌ {result['label']}")
    
    with col2:
        st.metric(
            label="Probability Real",
            value=f"{result['probability_real']:.2%}"
        )
    
    with col3:
        st.metric(
            label="Probability Fake",
            value=f"{result['probability_fake']:.2%}"
        )
    
    # Confidence bar
    st.markdown("---")
    st.markdown(f"**Confidence Level:** {result['confidence_description']}")
    
    # Progress bar showing confidence
    confidence = max(result['probability_real'], result['probability_fake'])
    st.progress(confidence)
    
    # Word count info
    st.caption(f"📝 Processed input: {result['word_count']} words | Max tokens: {MAX_LEN}")
    
    # Show processed text in expander
    if processed_text:
        with st.expander("📄 View Processed Text"):
            st.text(processed_text[:1000] + "..." if len(processed_text) > 1000 else processed_text)


def display_extracted_content(title: str, paragraph: str):
    """Display extracted content from URL scraping."""
    st.markdown("---")
    st.markdown("### 📋 Extracted Content")
    
    st.markdown("**Title:**")
    st.info(title if title else "No title extracted")
    
    st.markdown("**First Paragraph:**")
    st.info(paragraph if paragraph else "No paragraph extracted")


def manual_input_mode(model, tokenizer):
    """Manual input mode UI and logic."""
    st.markdown("### ✍️ Manual Input Mode")
    st.markdown("Enter the news article title and body text below.")
    
    # Input fields
    title = st.text_input(
        "News Title",
        placeholder="Enter the news headline...",
        key="manual_title"
    )
    
    body = st.text_area(
        "News Body",
        placeholder="Enter the news article body text...",
        height=200,
        key="manual_body"
    )
    
    # Analyze button
    if st.button("🔍 Analyze", key="manual_analyze", type="primary"):
        if not title and not body:
            st.warning("Please enter at least a title or body text.")
            return
        
        with st.spinner("Analyzing..."):
            # Combine title and first paragraph (replicating training format)
            processed_text = combine_title_and_body(title, body)
            
            if not processed_text:
                st.error("Could not process the input. Please check your text.")
                return
            
            # Make prediction
            result = predict_news(model, tokenizer, processed_text)
            
            # Display results
            display_prediction_results(result, processed_text)


def url_scraping_mode(model, tokenizer):
    """URL scraping mode UI and logic."""
    st.markdown("### 🌐 URL Scraping Mode")
    st.markdown("Enter a news article URL to automatically extract and analyze the content.")
    
    # URL input
    url = st.text_input(
        "News Article URL",
        placeholder="https://example.com/news-article",
        key="url_input"
    )
    
    # Analyze button
    if st.button("🔍 Scrape & Analyze", key="url_analyze", type="primary"):
        # Validate URL
        is_valid, error_msg = validate_url(url)
        if not is_valid:
            st.warning(error_msg)
            return
        
        with st.spinner("Scraping article..."):
            # Scrape article
            scrape_result = scrape_article(url)
            
            if not scrape_result["success"]:
                st.error(f"❌ {scrape_result['error']}")
                
                # Show partial content if available
                if scrape_result.get("title"):
                    st.info(f"**Title extracted:** {scrape_result['title']}")
                return
            
            # Display extracted content
            display_extracted_content(
                scrape_result["title"],
                scrape_result["first_paragraph"]
            )
            
            # Combine for prediction (title + first paragraph only)
            title = scrape_result["title"] or ""
            first_para = scrape_result["first_paragraph"] or ""
            
            processed_text = f"{title} {first_para}".strip()
            
            # Validate combined text
            word_count = len(processed_text.split())
            if word_count < 50:
                st.warning(f"⚠️ Extracted content is short ({word_count} words). Prediction may be less reliable.")
            
            # Make prediction
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            result = predict_news(model, tokenizer, processed_text)
            display_prediction_results(result, processed_text)


def display_sidebar():
    """Display sidebar with information."""
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This application uses a **Bidirectional LSTM** neural network 
        trained on the **WELFake dataset** to detect potential fake news.
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Details")
        st.markdown(f"""
        - **Architecture:** Bidirectional LSTM
        - **Embedding Size:** 128
        - **LSTM Units:** 128
        - **Max Sequence Length:** {MAX_LEN}
        - **Vocabulary Size:** 20,000
        - **Output:** Binary (Real/Fake)
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Best Practices")
        st.markdown("""
        For most accurate results:
        - Provide both title and article body
        - Use the first paragraph of the article
        - Avoid very short inputs (< 50 words)
        - Avoid noisy text (ads, navigation links)
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Limitations")
        st.markdown("""
        - Model detects **stylistic patterns**, not factual accuracy
        - Performance may vary for:
          - Very short texts
          - Non-English content
          - Topics not in training data
        - Always verify with trusted sources
        """)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Load model and tokenizer
    model, model_error = load_model()
    tokenizer, tokenizer_error = load_tokenizer()
    
    # Check for loading errors
    if model_error:
        st.error(f"❌ Model Loading Error: {model_error}")
        st.markdown(f"Expected model path: `{MODEL_PATH}`")
        st.stop()
    
    if tokenizer_error:
        st.error(f"❌ Tokenizer Loading Error: {tokenizer_error}")
        st.markdown(f"Expected tokenizer path: `{TOKENIZER_PATH}`")
        st.stop()
    
    st.success("✅ Model and tokenizer loaded successfully!")
    
    # Mode selection
    st.markdown("---")
    mode = st.radio(
        "Select Input Mode:",
        ["✍️ Manual Input", "🌐 URL Scraping"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Display selected mode
    if mode == "✍️ Manual Input":
        manual_input_mode(model, tokenizer)
    else:
        url_scraping_mode(model, tokenizer)
    
    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit | Model: Bidirectional LSTM | Dataset: WELFake")


if __name__ == "__main__":
    main()
