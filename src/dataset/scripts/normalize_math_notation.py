import re

from bs4 import BeautifulSoup


def normalize_math_notation(text: str) -> str:
    """Clean and normalize a math text string from FARMA for RAG indexing.

    Applies three layers of normalization in order:
    1. Strip HTML tags and decode HTML entities (e.g. ``&lt;`` → ``<``).
    2. Normalize LaTeX variants to a canonical form so equivalent expressions
       produce the same embedding (e.g. ``\\dfrac`` → ``\\frac``).
    3. Normalize decimal separators and collapse whitespace.
    """
    text = _strip_html(text)
    text = _normalize_latex(text)
    text = _normalize_formatting(text)
    return text


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities using BeautifulSoup."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def _normalize_latex(text: str) -> str:
    """Collapse LaTeX display/style variants to their canonical forms."""
    replacements = [
        (r"\\dfrac", r"\\frac"),
        (r"\\tfrac", r"\\frac"),
        (r"\\left\s*\(", r"("),
        (r"\\right\s*\)", r")"),
        (r"\\left\s*\[", r"["),
        (r"\\right\s*\]", r"]"),
        (r"\\cdot", r"*"),
        (r"\\times", r"*"),
        (r"²", r"^2"),
        (r"³", r"^3"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def _normalize_formatting(text: str) -> str:
    """Normalize decimal separators and collapse whitespace."""
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
