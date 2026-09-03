"""
cli.py
======

Implementation of the ``summarize.py`` command-line interface. Kept
separate from the thin root-level script so the logic is importable
and testable.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import extractive
from .core import get_backend_name, read_text_file, summarize


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize.py",
        description="Summarize a text file or piped/stdin text from the command line.",
        epilog=(
            "Examples:\n"
            "  python summarize.py --file sample_articles/climate_report.txt --length short\n"
            "  python summarize.py --file article.txt --length long --demo\n"
            "  cat article.txt | python summarize.py --length medium\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a .txt file to summarize. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--length", "-l",
        type=str,
        default="medium",
        choices=["short", "medium", "long"],
        help="Summary length preset (default: medium).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force the offline extractive summarizer even if an API key is set.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print compression statistics (word/sentence counts) after the summary.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Print only the summary text (no headers/backend info) — useful for piping.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.file:
        try:
            text = read_text_file(args.file)
        except FileNotFoundError:
            print(f"error: file not found: {args.file}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"error: could not read {args.file}: {exc}", file=sys.stderr)
            return 1
    else:
        if sys.stdin.isatty():
            print(
                "error: no --file given and no piped input detected.\n"
                "       Usage: python summarize.py --file path.txt --length short",
                file=sys.stderr,
            )
            return 1
        text = sys.stdin.read()

    if not text.strip():
        print("error: input text is empty.", file=sys.stderr)
        return 1

    summary, backend = summarize(text, length=args.length, force_extractive=args.demo)

    if not summary:
        print("error: could not extract a summary from the given text.", file=sys.stderr)
        return 1

    if args.quiet:
        print(summary)
        return 0

    label = {
        "openai": "OpenAI (LLM)",
        "extractive": "Extractive / demo mode (no API key configured)",
    }.get(backend, backend)

    print(f"\n=== Summary ({args.length}, backend: {label}) ===\n")
    print(summary)

    if args.stats:
        stats = extractive.summary_stats(text, summary)
        print("\n--- Stats ---")
        print(f"Original: {stats['original_words']} words, {stats['original_sentences']} sentences")
        print(f"Summary:  {stats['summary_words']} words, {stats['summary_sentences']} sentences")
        print(f"Reduction: {stats['reduction_pct']}%")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
