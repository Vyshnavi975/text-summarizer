"""
summarizer
==========

A small, dependency-light text summarization toolkit.

Two summarization backends are available:

* ``summarizer.extractive`` — a from-scratch extractive summarizer
  (word-frequency scored sentences, TextRank-style graph ranking as a
  fallback) that needs no external services or API keys. This is the
  "demo mode" backend and always works.

* ``summarizer.llm`` — an LLM-based abstractive summarizer that calls
  the OpenAI API when ``OPENAI_API_KEY`` is set (Anthropic is
  supported too, via ``ANTHROPIC_API_KEY``, as an alternative
  provider).

``summarizer.core.summarize()`` picks whichever backend is appropriate
and is the single entry point used by both the CLI (``summarize.py``)
and the web UI (``app.py``).
"""

from .core import summarize, get_backend_name

__all__ = ["summarize", "get_backend_name"]
__version__ = "1.0.0"
