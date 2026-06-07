#!/usr/bin/env python3
"""Generate a Claude prompt with absolute document paths."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a MinerU private document Q&A prompt from local files."
    )
    parser.add_argument("files", nargs="+", help="PDF, DOCX, PPTX, image, or HTML files")
    parser.add_argument(
        "-q",
        "--question",
        default="Summarize the key points, risks, decisions, and next actions.",
        help="Question to ask after parsing the documents",
    )
    args = parser.parse_args()

    paths = [Path(item).expanduser().resolve() for item in args.files]

    print("Use MinerU to parse these private document files:")
    print()
    for path in paths:
        print(f"- {path}")
    print()
    print("Question:")
    print(args.question)
    print()
    print("Instructions:")
    print("- Answer only from the parsed document content.")
    print("- Cite the document filename and page or section when available.")
    print("- If the evidence is missing, say what is missing.")
    print("- Do not use web knowledge unless explicitly requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

