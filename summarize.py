#!/usr/bin/env python3
"""
summarize.py — command-line entry point.

Usage:
    python summarize.py --file sample_articles/climate_report.txt --length short
    python summarize.py --file article.txt --length long --demo --stats
    cat article.txt | python summarize.py --length medium

Run `python summarize.py --help` for the full list of options.
"""
import sys

from summarizer.cli import main

if __name__ == "__main__":
    sys.exit(main())
