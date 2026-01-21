#!/usr/bin/env python3
"""
app.py — Flask web UI entry point.

Usage:
    python app.py
    # then open http://127.0.0.1:5000

Set FLASK_DEBUG=1 for auto-reload during development, and PORT to
change the listening port (default 5000).
"""
import os

from summarizer.webapp import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
