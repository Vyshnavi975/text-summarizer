"""
llm.py
======

LLM-based abstractive summarization. Used automatically whenever an
API key is available in the environment:

* ``OPENAI_API_KEY``    -> OpenAI (``openai`` package)
* ``ANTHROPIC_API_KEY`` -> Anthropic Claude (``anthropic`` package), as an
  alternative provider

If neither key is set, or the relevant client library isn't
installed, or the API call fails for any reason, callers should catch
``LLMUnavailableError`` and fall back to the extractive summarizer in
``summarizer.extractive`` — see ``summarizer.core.summarize`` for the
orchestration logic. This module never crashes the whole program: any
problem is normalized into ``LLMUnavailableError``.
"""

from __future__ import annotations

import os
from typing import Optional

_LENGTH_INSTRUCTIONS = {
    "short": "in about 2-3 sentences",
    "medium": "in about 1 short paragraph (4-6 sentences)",
    "long": "in about 2-3 paragraphs, covering the main points and important supporting details",
}


class LLMUnavailableError(RuntimeError):
    """Raised when no LLM backend can be used (no key, missing SDK,
    or the API call failed). Callers should fall back to the
    extractive summarizer when they see this."""


def _length_instruction(length: str) -> str:
    return _LENGTH_INSTRUCTIONS.get((length or "medium").lower(), _LENGTH_INSTRUCTIONS["medium"])


def _build_prompt(text: str, length: str) -> str:
    instruction = _length_instruction(length)
    return (
        "Summarize the following article for a general reader. "
        f"Write the summary {instruction}. "
        "Be faithful to the source, do not add outside information, and "
        "do not include a preamble like 'Here is a summary' — output only "
        "the summary text itself.\n\n"
        f"ARTICLE:\n{text}"
    )


def summarize_with_openai(text: str, length: str, api_key: str, model: Optional[str] = None) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailableError(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from exc

    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_prompt(text, length)}],
            temperature=0.3,
            max_tokens=1024,
        )
        summary = (response.choices[0].message.content or "").strip()
        if not summary:
            raise LLMUnavailableError("OpenAI API returned an empty response.")
        return summary
    except LLMUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any SDK/network error
        raise LLMUnavailableError(f"OpenAI API call failed: {exc}") from exc


def summarize_with_anthropic(text: str, length: str, api_key: str, model: Optional[str] = None) -> str:
    """Alternative provider — used only when ANTHROPIC_API_KEY is set
    and OPENAI_API_KEY is not."""
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailableError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        ) from exc

    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": _build_prompt(text, length)}],
        )
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        summary = "".join(parts).strip()
        if not summary:
            raise LLMUnavailableError("Anthropic API returned an empty response.")
        return summary
    except LLMUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any SDK/network error
        raise LLMUnavailableError(f"Anthropic API call failed: {exc}") from exc


def summarize_with_llm(text: str, length: str = "medium") -> str:
    """Summarize using whichever LLM provider has a key configured.

    Prefers OpenAI (the default provider) if both keys happen to be
    set, falling back to Anthropic. Raises ``LLMUnavailableError`` if
    no provider is usable, so callers can fall back to the extractive
    summarizer.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key:
        return summarize_with_openai(text, length, openai_key)
    if anthropic_key:
        return summarize_with_anthropic(text, length, anthropic_key)

    raise LLMUnavailableError(
        "No LLM API key found. Set OPENAI_API_KEY to enable LLM-based "
        "summarization (or ANTHROPIC_API_KEY as an alternative)."
    )


def llm_backend_name() -> Optional[str]:
    """Return which LLM provider *would* be used, without calling it,
    or None if neither key is configured."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None
