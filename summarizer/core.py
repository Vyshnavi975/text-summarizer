"""
core.py
=======

The single entry point used by both the CLI and the web UI. Decides
whether to use the LLM backend (if an API key is configured and the
call succeeds) or fall back to the from-scratch extractive summarizer.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from . import extractive
from .llm import LLMUnavailableError, llm_backend_name, summarize_with_llm

VALID_LENGTHS = ("short", "medium", "long")


def get_backend_name() -> str:
    """Return the backend that would be used right now: 'openai',
    'anthropic', or 'extractive' (demo mode)."""
    return llm_backend_name() or "extractive"


def summarize(
    text: str,
    length: str = "medium",
    force_extractive: bool = False,
) -> Tuple[str, str]:
    """Summarize ``text`` and return ``(summary, backend_used)``.

    Args:
        text: Source document text.
        length: 'short', 'medium', or 'long'.
        force_extractive: If True, always use the offline extractive
            summarizer even if an API key is configured (useful for
            tests and the CLI's ``--demo`` flag).

    The function never raises for a missing/failed LLM call — it
    transparently falls back to the extractive summarizer so the tool
    always produces a result.
    """
    text = (text or "").strip()
    if not text:
        return "", "extractive"

    length = (length or "medium").lower()
    if length not in VALID_LENGTHS and not length.isdigit():
        length = "medium"

    if not force_extractive and llm_backend_name():
        try:
            summary = summarize_with_llm(text, length=length)
            return summary.strip(), llm_backend_name()
        except LLMUnavailableError:
            pass  # fall through to extractive backend below

    summary = extractive.summarize_text(text, length=length)
    return summary, "extractive"


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """Read a .txt file, tolerating minor encoding issues."""
    with open(path, "r", encoding=encoding, errors="replace") as fh:
        return fh.read()
