#!/usr/bin/env python3
"""Local MCP-vs-RAG benchmark helper.

The RAG implementation is intentionally dependency-free. It indexes parsed
Markdown text and answers with extractive snippets so the benchmark can run
without API keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "why",
}


@dataclass
class Chunk:
    chunk_id: str
    source: str
    heading: str
    text: str
    tokens: list[str]


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in TOKEN_RE.finditer(text))
        if token not in STOPWORDS
    ]


def split_markdown_sections(text: str) -> list[str]:
    starts = [match.start() for match in HEADING_RE.finditer(text)]
    if not starts:
        return [text]

    sections: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def split_by_window(text: str, max_tokens: int = 180, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(words), step):
        part = words[start : start + max_tokens]
        if part:
            chunks.append(" ".join(part))
        if start + max_tokens >= len(words):
            break
    return chunks


def chunk_text(text: str, max_tokens: int = 180, overlap: int = 30) -> list[str]:
    chunks: list[str] = []
    for section in split_markdown_sections(text):
        if len(section.split()) <= max_tokens:
            chunks.append(section)
        else:
            chunks.extend(split_by_window(section, max_tokens=max_tokens, overlap=overlap))
    return chunks


def load_documents(paths: list[Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_text(text)):
            heading = chunk.splitlines()[0].lstrip("# ").strip() if chunk.splitlines() else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}-{index + 1}",
                    source=str(path),
                    heading=heading,
                    text=chunk,
                    tokens=tokenize(chunk) + tokenize(heading) * 3,
                )
            )
    return chunks


def build_idf(chunks: list[Chunk]) -> dict[str, float]:
    doc_count = len(chunks)
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(chunk.tokens))

    return {
        term: math.log((doc_count + 1) / (frequency + 1)) + 1
        for term, frequency in document_frequency.items()
    }


def vectorize(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    length = max(1, len(tokens))
    return {
        term: (count / length) * idf.get(term, 0.0)
        for term, count in counts.items()
    }


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def save_index(chunks: list[Chunk], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunks": [asdict(chunk) for chunk in chunks]}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(path: Path) -> list[Chunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in payload["chunks"]]


def ask_rag(chunks: list[Chunk], question: str, top_k: int = 3) -> dict[str, object]:
    start = time.perf_counter()
    idf = build_idf(chunks)
    question_vector = vectorize(tokenize(question), idf)

    scored = []
    question_tokens = set(tokenize(question))
    for chunk in chunks:
        score = cosine(question_vector, vectorize(chunk.tokens, idf))
        heading_overlap = question_tokens.intersection(tokenize(chunk.heading))
        score += 0.25 * len(heading_overlap)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [(score, chunk) for score, chunk in scored[:top_k] if score > 0]

    snippets = []
    for score, chunk in selected:
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        body = " ".join(lines[1:]) if len(lines) > 1 else chunk.text
        snippet = body[:500]
        snippets.append(
            {
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "heading": chunk.heading,
                "score": round(score, 4),
                "snippet": snippet,
            }
        )

    if snippets:
        answer = " ".join(item["snippet"] for item in snippets)
    else:
        answer = "No matching evidence was found in the indexed documents."

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "question": question,
        "mode": "rag",
        "latency_ms": latency_ms,
        "answer": answer,
        "evidence": snippets,
    }


def command_ingest(args: argparse.Namespace) -> int:
    paths = [Path(item).expanduser().resolve() for item in args.files]
    chunks = load_documents(paths)
    save_index(chunks, Path(args.output).expanduser().resolve())
    print(f"Indexed {len(paths)} document(s) into {len(chunks)} chunk(s).")
    print(f"Index: {Path(args.output).expanduser().resolve()}")
    return 0


def command_ask(args: argparse.Namespace) -> int:
    chunks = load_index(Path(args.index).expanduser().resolve())
    result = ask_rag(chunks, args.question, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    chunks = load_index(Path(args.index).expanduser().resolve())
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    results = [ask_rag(chunks, item["question"], top_k=args.top_k) for item in questions]

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".csv":
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["mode", "question", "latency_ms", "answer", "evidence_count"],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "mode": result["mode"],
                        "question": result["question"],
                        "latency_ms": result["latency_ms"],
                        "answer": result["answer"],
                        "evidence_count": len(result["evidence"]),
                    }
                )
    else:
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    avg_latency = sum(item["latency_ms"] for item in results) / max(1, len(results))
    print(f"Ran {len(results)} RAG benchmark question(s).")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Output: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local RAG over MinerU-parsed Markdown.")
    subparsers = parser.add_subparsers(required=True)

    ingest = subparsers.add_parser("ingest", help="Index parsed Markdown documents")
    ingest.add_argument("files", nargs="+", help="Markdown/text documents to index")
    ingest.add_argument("-o", "--output", default="data/rag_index/index.json")
    ingest.set_defaults(func=command_ingest)

    ask = subparsers.add_parser("ask", help="Ask a question using the local RAG index")
    ask.add_argument("question")
    ask.add_argument("-i", "--index", default="data/rag_index/index.json")
    ask.add_argument("-k", "--top-k", type=int, default=3)
    ask.set_defaults(func=command_ask)

    benchmark = subparsers.add_parser("benchmark", help="Run a batch RAG benchmark")
    benchmark.add_argument("-i", "--index", default="data/rag_index/index.json")
    benchmark.add_argument("-q", "--questions", default="benchmark/questions.json")
    benchmark.add_argument("-o", "--output", default="reports/rag-results.json")
    benchmark.add_argument("-k", "--top-k", type=int, default=3)
    benchmark.set_defaults(func=command_benchmark)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
