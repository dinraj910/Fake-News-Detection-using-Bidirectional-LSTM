"""
URL scraping utilities for Fake News Detection.
Uses newspaper3k for article extraction.
"""

from typing import Tuple, Optional, Dict, Any
from newspaper import Article


def scrape_article(url: str) -> Dict[str, Any]:
    """
    Scrape article from URL using newspaper3k.
    
    Extracts:
    - Title
    - Full text
    - First meaningful paragraph (>= 20 words)
    
    Args:
        url: URL of the news article
    
    Returns:
        Dictionary with keys:
        - success: bool
        - title: str or None
        - full_text: str or None
        - first_paragraph: str or None
        - error: str or None
    """
    result = {
        "success": False,
        "title": None,
        "full_text": None,
        "first_paragraph": None,
        "error": None
    }
    
    try:
        # Initialize and download article
        article = Article(url)
        article.download()
        article.parse()
        
        title = article.title
        full_text = article.text
        
        if not title and not full_text:
            result["error"] = "Could not extract any content from the URL. The site may be blocking scrapers."
            return result
        
        if not full_text:
            result["error"] = "Could not extract article text. Only title found."
            result["title"] = title
            return result
        
        # Split into paragraphs and filter
        paragraphs = full_text.split("\n")
        
        # Remove very short paragraphs (noise like navigation, footers, etc.)
        valid_paragraphs = [
            p.strip() for p in paragraphs 
            if len(p.split()) >= 20
        ]
        
        if not valid_paragraphs:
            # If no valid paragraph, try to use full text if it has content
            if len(full_text.split()) >= 20:
                first_paragraph = full_text.strip()
            else:
                result["error"] = "No valid paragraph found (all paragraphs are too short)."
                result["title"] = title
                result["full_text"] = full_text
                return result
        else:
            first_paragraph = valid_paragraphs[0]
        
        result["success"] = True
        result["title"] = title
        result["full_text"] = full_text
        result["first_paragraph"] = first_paragraph
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide more user-friendly error messages
        if "403" in error_msg or "forbidden" in error_msg.lower():
            result["error"] = "Access denied. The website is blocking automated requests."
        elif "404" in error_msg or "not found" in error_msg.lower():
            result["error"] = "Page not found. Please check the URL."
        elif "timeout" in error_msg.lower():
            result["error"] = "Request timed out. The server took too long to respond."
        elif "connection" in error_msg.lower():
            result["error"] = "Connection error. Please check your internet connection."
        else:
            result["error"] = f"Failed to scrape article: {error_msg}"
        
        return result


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Basic URL validation.
    
    Args:
        url: URL string to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not url.strip():
        return False, "Please enter a URL."
    
    url = url.strip()
    
    if not url.startswith(('http://', 'https://')):
        return False, "URL must start with http:// or https://"
    
    if '.' not in url:
        return False, "Invalid URL format."
    
    return True, ""
