"""LangChain orchestration over private document chunks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .corpus import Chunk, extract_sentences, load_documents, load_index, save_index, search, score_chunks


COMPARE_PATTERNS = (
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bdifference\b",
    r"\bbetween\b",
)
SUMMARY_PATTERNS = (
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\boverview\b",
    r"\bbrief\b",
)
ACTION_PATTERNS = (
    r"\bnext steps\b",
    r"\baction items\b",
    r"\brisks\b",
    r"\bdecisions\b",
    r"\bfollow up\b",
)


@dataclass
class RouteDecision:
    route: str
    rationale: str


@dataclass
class OrchestrationResult:
    route: str
    question: str
    answer: str
    evidence: list[dict[str, object]]
    metadata: dict[str, Any]


def classify_route(question: str) -> RouteDecision:
    haystack = question.casefold()
    if any(re.search(pattern, haystack) for pattern in COMPARE_PATTERNS):
        return RouteDecision(route="compare", rationale="Question asks for a comparison or difference.")
    if any(re.search(pattern, haystack) for pattern in SUMMARY_PATTERNS):
        return RouteDecision(route="summarize", rationale="Question asks for a summary or overview.")
    if any(re.search(pattern, haystack) for pattern in ACTION_PATTERNS):
        return RouteDecision(route="extract", rationale="Question asks for risks, decisions, or action items.")
    return RouteDecision(route="qa", rationale="Default document question answering route.")


def _format_evidence(evidence: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for item in evidence:
        heading = item.get("heading", "")
        source = item.get("source", "")
        score = item.get("score", 0)
        snippet = item.get("snippet", "")
        lines.append(f"- [{heading}]({source}) score={score}: {snippet}")
    return "\n".join(lines)


def _extractive_answer(question: str, chunks: list[Chunk], top_k: int = 3) -> tuple[str, list[dict[str, object]]]:
    evidence = search(question, chunks, top_k=top_k)
    if not evidence:
        return "No matching evidence was found in the indexed documents.", evidence

    snippets = []
    for item in evidence:
        snippet = str(item.get("snippet", ""))
        snippets.extend(extract_sentences(snippet, max_sentences=2))
    answer = " ".join(snippets) if snippets else "No matching evidence was found in the indexed documents."
    return answer, evidence


def _compare_answer(question: str, chunks: list[Chunk]) -> tuple[str, list[dict[str, object]]]:
    evidence = search(question, chunks, top_k=6)
    by_source: dict[str, list[str]] = {}
    for item in evidence:
        source = str(item.get("source", ""))
        by_source.setdefault(source, []).append(str(item.get("snippet", "")))

    if len(by_source) < 2:
        answer, evidence = _extractive_answer(question, chunks, top_k=4)
        return answer, evidence

    lines = ["Comparison summary:"]
    for source, snippets in by_source.items():
        merged = " ".join(snippets[:2])
        lines.append(f"- {Path(source).name}: {merged[:260]}")
    return "\n".join(lines), evidence


def _summarize_answer(question: str, chunks: list[Chunk]) -> tuple[str, list[dict[str, object]]]:
    evidence = search(question, chunks, top_k=5)
    if not evidence:
        evidence = search("summary risks decisions next actions", chunks, top_k=5)

    bullets = []
    for item in evidence[:4]:
        bullets.append(f"- {item.get('heading', '')}: {item.get('snippet', '')}")
    if not bullets:
        bullets.append("- No matching evidence was found in the indexed documents.")
    return "\n".join(bullets), evidence


def build_langchain_chain(orchestrator: "DocumentOrchestrator"):
    try:
        from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
    except Exception:
        return None

    return (
        RunnablePassthrough.assign(route=RunnableLambda(lambda payload: orchestrator.classify(payload["question"]).route))
        | RunnableBranch(
            (lambda payload: payload["route"] == "compare", RunnableLambda(lambda payload: orchestrator.compare(payload["question"]))),
            (lambda payload: payload["route"] == "summarize", RunnableLambda(lambda payload: orchestrator.summarize(payload["question"]))),
            (lambda payload: payload["route"] == "extract", RunnableLambda(lambda payload: orchestrator.extract(payload["question"]))),
            RunnableLambda(lambda payload: orchestrator.answer(payload["question"])),
        )
    )


class DocumentOrchestrator:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._chain = build_langchain_chain(self)

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "DocumentOrchestrator":
        return cls(load_documents(paths))

    @classmethod
    def from_index(cls, path: Path) -> "DocumentOrchestrator":
        return cls(load_index(path))

    def save_index(self, output: Path) -> None:
        save_index(self.chunks, output)

    def classify(self, question: str) -> RouteDecision:
        return classify_route(question)

    def answer(self, question: str) -> OrchestrationResult:
        answer, evidence = _extractive_answer(question, self.chunks)
        return OrchestrationResult(
            route="qa",
            question=question,
            answer=answer,
            evidence=evidence,
            metadata={"strategy": "extractive"},
        )

    def compare(self, question: str) -> OrchestrationResult:
        answer, evidence = _compare_answer(question, self.chunks)
        return OrchestrationResult(
            route="compare",
            question=question,
            answer=answer,
            evidence=evidence,
            metadata={"strategy": "multi-source comparison"},
        )

    def summarize(self, question: str) -> OrchestrationResult:
        answer, evidence = _summarize_answer(question, self.chunks)
        return OrchestrationResult(
            route="summarize",
            question=question,
            answer=answer,
            evidence=evidence,
            metadata={"strategy": "extractive summary"},
        )

    def extract(self, question: str) -> OrchestrationResult:
        answer, evidence = _extractive_answer(question, self.chunks, top_k=5)
        return OrchestrationResult(
            route="extract",
            question=question,
            answer=answer,
            evidence=evidence,
            metadata={"strategy": "risk/action extraction"},
        )

    def run(self, question: str) -> OrchestrationResult:
        if self._chain is not None:
            result = self._chain.invoke({"question": question})
            if isinstance(result, OrchestrationResult):
                return result
            return OrchestrationResult(
                route=str(result.get("route", "qa")),
                question=question,
                answer=str(result.get("answer", "")),
                evidence=list(result.get("evidence", [])),
                metadata=dict(result.get("metadata", {})),
            )

        route = self.classify(question).route
        if route == "compare":
            return self.compare(question)
        if route == "summarize":
            return self.summarize(question)
        if route == "extract":
            return self.extract(question)
        return self.answer(question)

    def to_json(self, result: OrchestrationResult) -> str:
        return json.dumps(asdict(result), indent=2, ensure_ascii=False)


def synthesize_with_llm(question: str, evidence: list[dict[str, object]], model_name: str = "gpt-4o-mini") -> str:
    """Optional LangChain synthesis layer.

    This is only used when the caller explicitly wants a model-generated
    answer and has installed the optional LangChain OpenAI dependencies.
    """

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise RuntimeError("Install langchain-openai to enable LLM synthesis.") from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer only from the provided evidence. If evidence is thin, say so.",
            ),
            (
                "user",
                "Question: {question}\n\nEvidence:\n{evidence}\n\nWrite a concise answer with no speculation.",
            ),
        ]
    )
    chain = prompt | ChatOpenAI(model=model_name, temperature=0) | StrOutputParser()
    return chain.invoke({"question": question, "evidence": _format_evidence(evidence)})

