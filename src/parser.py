"""
Parsing module: extract raw text from PDF (or .txt) resumes and job descriptions.
Handles messy formatting gracefully - falls back across pages, strips excess whitespace.
"""
import os
import re
import pdfplumber


def extract_text_from_pdf(filepath: str) -> str:
    """Extract all text from a PDF file, page by page."""
    text_chunks = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text(filepath: str) -> str:
    """Extract text from a file, dispatching by extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        raw = extract_text_from_pdf(filepath)
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return clean_text(raw)


def clean_text(text: str) -> str:
    """Normalize whitespace, fix common PDF extraction artifacts."""
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix hyphenated line-breaks e.g. "develop-\nment" -> "development"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    return text.strip()


def extract_candidate_name(text: str, fallback: str) -> str:
    """
    Best-effort guess at candidate name: usually the first non-empty line
    of a resume, if it looks like a name (short, alphabetic, no digits/@ etc).
    Falls back to filename if no good candidate found.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        if len(line) > 60:
            continue
        if any(ch.isdigit() for ch in line):
            continue
        if "@" in line or "http" in line.lower():
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
            return line.title()
    return fallback
