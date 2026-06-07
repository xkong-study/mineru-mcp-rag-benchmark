"""CLI for the LangChain document orchestration demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import DocumentOrchestrator, synthesize_with_llm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangChain document orchestration demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index markdown documents into JSON")
    ingest.add_argument("paths", nargs="+", help="Markdown files or directories")
    ingest.add_argument("-o", "--output", required=True, help="Index output path")

    ask = subparsers.add_parser("ask", help="Ask a question against an index")
    ask.add_argument("question", help="Question to answer")
    ask.add_argument("--index", required=True, help="Index JSON file")
    ask.add_argument("--use-llm", action="store_true", help="Use an OpenAI-backed LangChain synthesis layer")

    demo = subparsers.add_parser("demo", help="Run the sample project demo")
    demo.add_argument("--sample-dir", default="data/sample_docs", help="Directory of sample markdown docs")
    demo.add_argument("--question", default="What are the key decisions, risks, and next actions?", help="Demo question")

    return parser


def _expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in values:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Input path not found: {path}. "
                "Pass a real Markdown file or a directory that contains .md/.markdown/.txt files."
            )
        if path.is_dir():
            children = sorted(child for child in path.iterdir() if child.suffix.lower() in {".md", ".markdown", ".txt"})
            if not children:
                raise FileNotFoundError(
                    f"No Markdown-like files found in directory: {path}. "
                    "Add .md, .markdown, or .txt files, or point ingest at a specific file."
                )
            paths.extend(children)
        else:
            paths.append(path)
    return paths


def command_ingest(args: argparse.Namespace) -> int:
    paths = _expand_inputs(args.paths)
    orchestrator = DocumentOrchestrator.from_paths(paths)
    orchestrator.save_index(Path(args.output).expanduser().resolve())
    print(json.dumps({"indexed_files": [str(path) for path in paths], "chunks": len(orchestrator.chunks)}, indent=2))
    return 0


def command_ask(args: argparse.Namespace) -> int:
    orchestrator = DocumentOrchestrator.from_index(Path(args.index).expanduser().resolve())
    result = orchestrator.run(args.question)
    if args.use_llm:
        result.answer = synthesize_with_llm(args.question, result.evidence)
        result.metadata["synthesis"] = "langchain-openai"
    print(orchestrator.to_json(result))
    return 0


def command_demo(args: argparse.Namespace) -> int:
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    orchestrator = DocumentOrchestrator.from_paths(sorted(path for path in sample_dir.iterdir() if path.suffix.lower() in {".md", ".markdown", ".txt"}))
    result = orchestrator.run(args.question)
    print(orchestrator.to_json(result))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        return command_ingest(args)
    if args.command == "ask":
        return command_ask(args)
    if args.command == "demo":
        return command_demo(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
