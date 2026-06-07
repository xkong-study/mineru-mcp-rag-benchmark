"""Document loading and lightweight retrieval helpers."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
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
    "how",
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


def save_index(chunks: list[Chunk], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunks": [asdict(chunk) for chunk in chunks]}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_index(path: Path) -> list[Chunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in payload["chunks"]]


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


def score_chunks(question: str, chunks: list[Chunk]) -> list[tuple[float, Chunk]]:
    idf = build_idf(chunks)
    question_tokens = tokenize(question)
    question_vector = vectorize(question_tokens, idf)
    question_term_set = set(question_tokens)

    scored: list[tuple[float, Chunk]] = []
    for chunk in chunks:
        score = cosine(question_vector, vectorize(chunk.tokens, idf))
        heading_overlap = question_term_set.intersection(tokenize(chunk.heading))
        score += 0.25 * len(heading_overlap)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def search(question: str, chunks: list[Chunk], top_k: int = 3) -> list[dict[str, object]]:
    selected = [(score, chunk) for score, chunk in score_chunks(question, chunks)[:top_k] if score > 0]
    results: list[dict[str, object]] = []
    for score, chunk in selected:
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        body = " ".join(lines[1:]) if len(lines) > 1 else chunk.text
        results.append(
            {
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "heading": chunk.heading,
                "score": round(score, 4),
                "snippet": body[:500],
            }
        )
    return results


def extract_sentences(text: str, max_sentences: int = 4) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return sentences[:max_sentences]

