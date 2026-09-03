"""
webapp.py
=========

Flask application factory for the AI Text Summarizer web UI. A single
page lets the user paste text (or upload a .txt file), pick a summary
length, and see the result along with which backend produced it.

Run via the root-level ``app.py``:

    python app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import os

from flask import Flask, render_template, request

from . import extractive
from .core import get_backend_name, summarize

MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB upload cap
ALLOWED_EXTENSIONS = {".txt"}

# The package lives in <project_root>/summarizer/, but templates/static
# live at <project_root>/templates and <project_root>/static, so point
# Flask at the project root explicitly rather than relying on its
# default (which looks next to this module).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(_PROJECT_ROOT, "templates"),
        static_folder=os.path.join(_PROJECT_ROOT, "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        error = None
        submitted_text = ""
        selected_length = "medium"

        if request.method == "POST":
            selected_length = request.form.get("length", "medium")
            submitted_text = request.form.get("text", "")

            uploaded_file = request.files.get("file")
            if uploaded_file and uploaded_file.filename:
                filename = uploaded_file.filename
                if not filename.lower().endswith(".txt"):
                    error = "Please upload a plain .txt file."
                else:
                    try:
                        raw = uploaded_file.read()
                        submitted_text = raw.decode("utf-8", errors="replace")
                    except Exception as exc:  # noqa: BLE001
                        error = f"Could not read the uploaded file: {exc}"

            if error is None:
                text = submitted_text.strip()
                if not text:
                    error = "Please paste some text or upload a .txt file to summarize."
                elif len(text.split()) < 20:
                    error = "Please provide a longer piece of text (at least ~20 words) for a meaningful summary."
                else:
                    summary, backend = summarize(text, length=selected_length)
                    stats = extractive.summary_stats(text, summary)
                    result = {
                        "summary": summary,
                        "backend": backend,
                        "backend_label": _backend_label(backend),
                        "stats": stats,
                    }

        return render_template(
            "index.html",
            result=result,
            error=error,
            submitted_text=submitted_text,
            selected_length=selected_length,
            active_backend=_backend_label(get_backend_name()),
        )

    @app.errorhandler(413)
    def too_large(_exc):
        return render_template(
            "index.html",
            result=None,
            error="That file is too large (2 MB max). Please upload a smaller .txt file.",
            submitted_text="",
            selected_length="medium",
            active_backend=_backend_label(get_backend_name()),
        ), 413

    return app


def _backend_label(backend: str) -> str:
    return {
        "openai": "OpenAI (LLM mode)",
        "anthropic": "Anthropic Claude (LLM mode)",
        "extractive": "Extractive algorithm (demo mode — no API key set)",
    }.get(backend, backend)
