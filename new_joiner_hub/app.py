"""Browser UI + LangGraph routing for the new joiner knowledge hub."""

from __future__ import annotations

import argparse
import html
import json
import os
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import parse_qs, urlparse

from langgraph.graph import END, StateGraph

from .retrieval import Chunk, extract_sentences, load_documents, search


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8088
KB_DIR = Path(__file__).resolve().parent / "knowledge_base"

CATEGORY_LABELS = {
    "all": "All",
    "welcome": "Welcome",
    "product": "Product",
    "engineering": "Engineering",
    "security": "Security",
    "support": "Support",
    "people": "People & Process",
}

SAMPLE_PROMPTS = {
    "all": [
        "What should I do in my first week?",
        "Where is the latest product overview?",
        "How do I request access to staging?",
    ],
    "welcome": [
        "What should I do in my first week?",
        "What are the first week checklist items?",
    ],
    "product": [
        "What does the product do?",
        "Explain the main product areas.",
    ],
    "engineering": [
        "How do I release a feature safely?",
        "What is the code review flow?",
    ],
    "security": [
        "What do I need before I can access staging?",
        "Which docs can be shared externally?",
    ],
    "support": [
        "How do I escalate a production issue?",
        "What information should I capture in an incident?",
    ],
    "people": [
        "Who should I contact for engineering workflow questions?",
        "Who is the first contact for onboarding help?",
    ],
}


@dataclass
class CatalogItem:
    slug: str
    title: str
    category: str
    path: str
    summary: str


class HubState(TypedDict, total=False):
    question: str
    route_mode: str
    category: str
    top_k: int
    route: str
    routing_reason: str
    evidence: list[dict[str, object]]
    answer: str
    synthesis: str
    use_llm: bool


def slug_to_category(slug: str) -> str:
    slug = slug.lower()
    if "welcome" in slug:
        return "welcome"
    if "product" in slug:
        return "product"
    if "engineering" in slug:
        return "engineering"
    if "access" in slug or "security" in slug:
        return "security"
    if "support" in slug:
        return "support"
    if "people" in slug or "process" in slug:
        return "people"
    return "all"


def titleize_slug(slug: str) -> str:
    base = slug.replace("_", " ").replace("-", " ")
    return base[:1].upper() + base[1:]


def summarize_markdown(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        if not line.startswith("#"):
            return line[:220]
    return ""


def load_catalog(kb_dir: Path) -> tuple[list[CatalogItem], list[Chunk]]:
    docs: list[CatalogItem] = []
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        slug = path.stem
        category = slug_to_category(slug)
        docs.append(
            CatalogItem(
                slug=slug,
                title=titleize_slug(slug),
                category=category,
                path=str(path),
                summary=summarize_markdown(path),
            )
        )
        chunks.extend(load_documents([path]))
    return docs, chunks


def category_filter(chunks: list[Chunk], category: str) -> list[Chunk]:
    if category == "all":
        return chunks
    filtered = [chunk for chunk in chunks if slug_to_category(Path(chunk.source).stem) == category]
    return filtered or chunks


def classify_route(question: str) -> tuple[str, str]:
    text = question.casefold()
    if any(token in text for token in ("compare", "difference", "versus", "vs")):
        return "compare", "Question asks for a comparison."
    if any(token in text for token in ("summarize", "summary", "overview", "give me an overview")):
        return "summarize", "Question asks for a summary."
    if any(token in text for token in ("next steps", "action item", "checklist", "what should i do", "what do i need")):
        return "extract", "Question asks for actions or next steps."
    return "qa", "Default question-answering route."


def route_node(state: HubState) -> dict[str, Any]:
    route_mode = state.get("route_mode", "auto")
    if route_mode and route_mode != "auto":
        return {"route": route_mode, "routing_reason": "Route selected manually in the UI."}
    route, reason = classify_route(state["question"])
    return {"route": route, "routing_reason": reason}


def route_selector(state: HubState) -> str:
    return state.get("route", "qa")


def retrieve_node(state: HubState) -> dict[str, Any]:
    question = state["question"]
    category = state.get("category", "all")
    top_k = int(state.get("top_k", 4))
    pool = category_filter(CHUNKS, category)
    evidence = search(question, pool, top_k=top_k)
    return {"evidence": evidence}


def _build_answer(route: str, evidence: list[dict[str, object]], question: str) -> str:
    if not evidence:
        return "No matching evidence was found in the selected knowledge base."

    if route == "compare":
        by_source: dict[str, list[str]] = {}
        for item in evidence:
            by_source.setdefault(str(item.get("source", "")), []).append(str(item.get("snippet", "")))
        lines = ["Comparison summary:"]
        if len(by_source) == 1:
            source, snippets = next(iter(by_source.items()))
            lines.append(f"- {Path(source).name}: {' '.join(snippets[:2])[:320]}")
        else:
            for source, snippets in by_source.items():
                lines.append(f"- {Path(source).name}: {' '.join(snippets[:2])[:250]}")
        return "\n".join(lines)

    if route == "summarize":
        bullets = []
        for item in evidence[:4]:
            bullets.append(f"- {item.get('heading', '')}: {item.get('snippet', '')}")
        return "\n".join(bullets)

    if route == "extract":
        if len(evidence) == 1:
            return str(evidence[0].get("snippet", ""))
        items = []
        for index, item in enumerate(evidence[:5], start=1):
            snippet = str(item.get("snippet", ""))
            items.append(f"{index}. {snippet}")
        return "\n".join(items)

    bullets = []
    for item in evidence[:3]:
        bullets.extend(extract_sentences(str(item.get("snippet", "")), max_sentences=2))
    return " ".join(bullets)


def answer_node(state: HubState) -> dict[str, Any]:
    route = state.get("route", "qa")
    answer = _build_answer(route, state.get("evidence", []), state["question"])
    return {"answer": answer}


def maybe_synthesize(question: str, route: str, evidence: list[dict[str, object]], base_answer: str) -> tuple[str, str]:
    if not os.getenv("OPENAI_API_KEY"):
        return base_answer, "disabled"

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except Exception:
        return base_answer, "unavailable"

    evidence_text = "\n".join(
        f"- {Path(str(item.get('source', ''))).name} [{item.get('heading', '')}]: {item.get('snippet', '')}"
        for item in evidence
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an internal onboarding assistant. Answer only from the provided evidence. If evidence is weak, say so."),
            (
                "user",
                "Route: {route}\nQuestion: {question}\n\nEvidence:\n{evidence}\n\nWrite a concise answer for a new joiner.",
            ),
        ]
    )
    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    answer = chain.invoke({"route": route, "question": question, "evidence": evidence_text})
    return answer, "enabled"


def build_graph():
    graph = StateGraph(HubState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer_qa", answer_node)
    graph.add_node("answer_summarize", answer_node)
    graph.add_node("answer_compare", answer_node)
    graph.add_node("answer_extract", answer_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_selector,
        {
            "qa": "answer_qa",
            "summarize": "answer_summarize",
            "compare": "answer_compare",
            "extract": "answer_extract",
        },
    )
    graph.add_edge("answer_qa", END)
    graph.add_edge("answer_summarize", END)
    graph.add_edge("answer_compare", END)
    graph.add_edge("answer_extract", END)
    return graph.compile()


CATALOG, CHUNKS = load_catalog(KB_DIR)
GRAPH = build_graph()


def run_query(question: str, category: str = "all", route_mode: str = "auto", top_k: int = 4, use_llm: bool = False) -> dict[str, Any]:
    result = GRAPH.invoke(
        {
            "question": question,
            "category": category or "all",
            "route_mode": route_mode or "auto",
            "top_k": top_k,
        }
    )
    answer = str(result.get("answer", ""))
    synthesis = "disabled"
    if use_llm:
        answer, synthesis = maybe_synthesize(question, str(result.get("route", "qa")), list(result.get("evidence", [])), answer)
    result["answer"] = answer
    result["synthesis"] = synthesis
    result["agent_steps"] = [
        "route",
        "retrieve",
        f"answer_{result.get('route', 'qa')}",
        "optional_llm_synthesis" if use_llm else "extractive_fallback",
    ]
    result["catalog"] = [asdict(item) for item in CATALOG]
    return result


def get_doc(slug: str) -> dict[str, Any] | None:
    for item in CATALOG:
        if item.slug == slug:
            path = Path(item.path)
            return {
                "slug": item.slug,
                "title": item.title,
                "category": item.category,
                "summary": item.summary,
                "path": item.path,
                "content": path.read_text(encoding="utf-8"),
            }
    return None


def render_index_html() -> str:
    categories_json = json.dumps(CATEGORY_LABELS)
    prompts_json = json.dumps(SAMPLE_PROMPTS)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>New Joiner Knowledge Hub</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --panel-2: #fbfcfe;
      --text: #101828;
      --muted: #475467;
      --border: #d0d5dd;
      --accent: #155eef;
      --accent-2: #eef4ff;
      --good: #067647;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      min-height: 100vh;
    }}
    header {{
      padding: 20px 24px 0;
    }}
    .title {{
      font-size: 28px;
      line-height: 1.15;
      margin: 0 0 6px;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 16px;
      padding: 16px 24px 24px;
      align-items: start;
    }}
    .sidebar, .main {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 18px 40px rgba(16, 24, 40, 0.08);
    }}
    .sidebar {{
      padding: 16px;
      position: sticky;
      top: 16px;
    }}
    .main {{
      padding: 16px;
      min-width: 0;
    }}
    .section-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
      margin: 0 0 12px;
      font-weight: 700;
    }}
    .category-list {{
      display: grid;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .category-btn {{
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      cursor: pointer;
      text-align: left;
      font: inherit;
    }}
    .category-btn.active {{
      border-color: var(--accent);
      background: var(--accent-2);
      color: #1d4ed8;
      font-weight: 600;
    }}
    .doc-list {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .doc-card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: #fff;
    }}
    .doc-card button {{
      appearance: none;
      border: 0;
      background: transparent;
      color: var(--accent);
      font: inherit;
      font-weight: 600;
      padding: 0;
      cursor: pointer;
      text-align: left;
    }}
    .doc-meta, .muted {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .query-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 180px;
      gap: 12px;
      margin-bottom: 12px;
    }}
    textarea, select, input[type="number"] {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }}
    textarea {{
      min-height: 110px;
      resize: vertical;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .control {{
      display: grid;
      gap: 6px;
    }}
    .control label {{
      font-size: 12px;
      color: var(--muted);
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}
    .primary, .secondary {{
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }}
    .primary {{
      background: var(--accent);
      color: white;
    }}
    .secondary {{
      background: #eef4ff;
      color: #1d4ed8;
    }}
    .panel {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 14px;
      background: #fff;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: inherit;
      line-height: 1.55;
    }}
    .evidence-list {{
      display: grid;
      gap: 10px;
    }}
    .evidence-item {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: #fff;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 12px;
    }}
    .chip {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      background: #fff;
      cursor: pointer;
      font: inherit;
    }}
    .pill {{
      display: inline-flex;
      border-radius: 999px;
      padding: 4px 10px;
      background: #eef4ff;
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 600;
      margin-right: 8px;
    }}
    .doc-preview {{
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: #fcfcfd;
      white-space: pre-wrap;
      line-height: 1.5;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
    }}
    .split {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    @media (max-width: 1100px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
      }}
      .controls, .query-grid, .split {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1 class="title">New Joiner Knowledge Hub</h1>
    <p class="subtitle">LangGraph routes questions across onboarding docs, then shows evidence so new joiners can learn without hunting through folders.</p>
  </header>
  <div class="shell">
    <aside class="sidebar">
      <div class="section-title">Knowledge Areas</div>
      <div class="category-list" id="categoryList"></div>
      <div class="section-title">Quick Prompts</div>
      <div class="chip-row" id="promptRow"></div>
      <div class="section-title">Documents</div>
      <div class="doc-list" id="docList"></div>
    </aside>
    <main class="main">
      <div class="section-title">Ask</div>
      <div class="query-grid">
        <textarea id="question" placeholder="Ask about onboarding, product, engineering, security, support, or people/process..."></textarea>
        <div style="display:grid; gap:10px;">
          <select id="routeMode">
            <option value="auto">Auto route</option>
            <option value="qa">QA</option>
            <option value="summarize">Summarize</option>
            <option value="compare">Compare</option>
            <option value="extract">Extract</option>
          </select>
          <select id="categorySelect"></select>
          <input id="topK" type="number" min="1" max="10" value="4" />
        </div>
      </div>
      <div class="controls">
        <div class="control"><label for="topK">Top K</label><input id="topK2" type="number" min="1" max="10" value="4" /></div>
        <div class="control"><label for="useLlm">LLM synthesis</label><select id="useLlm"><option value="false">Off</option><option value="true">On if available</option></select></div>
        <div class="control"><label for="categorySelect2">Knowledge area</label><select id="categorySelect2"></select></div>
        <div class="control"><label for="routeMode2">Route mode</label><select id="routeMode2"><option value="auto">Auto</option><option value="qa">QA</option><option value="summarize">Summarize</option><option value="compare">Compare</option><option value="extract">Extract</option></select></div>
      </div>
      <div class="actions">
        <button class="primary" id="askBtn">Ask hub</button>
        <button class="secondary" id="clearBtn">Clear</button>
      </div>
      <div class="split">
        <section class="panel">
          <div class="section-title">Answer</div>
          <div class="muted" id="metaLine">Pick a prompt or ask your own question.</div>
          <pre id="answer">The answer will appear here.</pre>
        </section>
        <section class="panel">
          <div class="section-title">Document Preview</div>
          <div class="muted" id="previewMeta">Click a document on the left to preview it.</div>
          <div class="doc-preview" id="docPreview">No document selected.</div>
        </section>
      </div>
      <section class="panel">
        <div class="section-title">Evidence</div>
        <div class="evidence-list" id="evidenceList"></div>
      </section>
    </main>
  </div>
  <script>
    const categories = {categories_json};
    const prompts = {prompts_json};
    const categoryKeys = Object.keys(categories);
    let currentCategory = "all";
    let docs = [];

    const categoryList = document.getElementById("categoryList");
    const docList = document.getElementById("docList");
    const promptRow = document.getElementById("promptRow");
    const categorySelect = document.getElementById("categorySelect");
    const categorySelect2 = document.getElementById("categorySelect2");
    const routeMode = document.getElementById("routeMode");
    const routeMode2 = document.getElementById("routeMode2");
    const topK = document.getElementById("topK");
    const topK2 = document.getElementById("topK2");
    const useLlm = document.getElementById("useLlm");
    const question = document.getElementById("question");
    const answer = document.getElementById("answer");
    const evidenceList = document.getElementById("evidenceList");
    const metaLine = document.getElementById("metaLine");
    const previewMeta = document.getElementById("previewMeta");
    const docPreview = document.getElementById("docPreview");

    function syncSelects() {{
      categorySelect.value = currentCategory;
      categorySelect2.value = currentCategory;
      routeMode2.value = routeMode.value;
      topK2.value = topK.value;
    }}

    function renderCategories() {{
      categoryList.innerHTML = "";
      categoryKeys.forEach((key) => {{
        const btn = document.createElement("button");
        btn.className = "category-btn" + (currentCategory === key ? " active" : "");
        btn.textContent = categories[key];
        btn.onclick = () => setCategory(key);
        categoryList.appendChild(btn);

        const opt1 = document.createElement("option");
        opt1.value = key;
        opt1.textContent = categories[key];
        categorySelect.appendChild(opt1);
        const opt2 = document.createElement("option");
        opt2.value = key;
        opt2.textContent = categories[key];
        categorySelect2.appendChild(opt2);
      }});
      syncSelects();
    }}

    function setCategory(category) {{
      currentCategory = category;
      renderCategories();
      renderDocs();
    }}

    function renderPrompts() {{
      promptRow.innerHTML = "";
      const items = prompts[currentCategory] || prompts.all;
      items.forEach((prompt) => {{
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = prompt;
        chip.onclick = () => {{
          question.value = prompt;
          question.focus();
        }};
        promptRow.appendChild(chip);
      }});
    }}

    function renderDocs() {{
      const items = docs.filter((doc) => currentCategory === "all" || doc.category === currentCategory);
      docList.innerHTML = "";
      if (!items.length) {{
        docList.innerHTML = '<div class="muted">No docs in this category.</div>';
        return;
      }}
      items.forEach((doc) => {{
        const card = document.createElement("div");
        card.className = "doc-card";
        card.innerHTML = `
          <button data-slug="${{doc.slug}}">${{doc.title}}</button>
          <div class="doc-meta">${{doc.category}}</div>
          <div class="doc-meta">${{doc.summary || ""}}</div>
        `;
        card.querySelector("button").onclick = () => loadDoc(doc.slug);
        docList.appendChild(card);
      }});
    }}

    async function loadCatalog() {{
      const res = await fetch("/api/catalog");
      const data = await res.json();
      docs = data.docs;
      renderCategories();
      renderPrompts();
      renderDocs();
    }}

    async function loadDoc(slug) {{
      const res = await fetch(`/api/doc?slug=${{encodeURIComponent(slug)}}`);
      const data = await res.json();
      previewMeta.textContent = `${{data.title}} · ${{categories[data.category] || data.category}}`;
      docPreview.textContent = data.content;
    }}

    async function askHub() {{
      const payload = {{
        question: question.value.trim(),
        category: categorySelect2.value,
        route_mode: routeMode.value,
        top_k: parseInt(topK2.value, 10) || 4,
        use_llm: useLlm.value === "true",
      }};
      if (!payload.question) {{
        answer.textContent = "Type a question first.";
        return;
      }}
      answer.textContent = "Thinking...";
      evidenceList.innerHTML = "";
      metaLine.textContent = "Running LangGraph route...";
      const res = await fetch("/api/query", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
      }});
      const data = await res.json();
      answer.textContent = data.answer || "";
      metaLine.innerHTML = `<span class="pill">${{data.route}}</span>${{data.routing_reason}} · synthesis: ${{data.synthesis}}`;
      evidenceList.innerHTML = "";
      (data.evidence || []).forEach((item) => {{
        const node = document.createElement("div");
        node.className = "evidence-item";
        node.innerHTML = `
          <div><strong>${{item.heading || "Untitled"}}</strong></div>
          <div class="doc-meta">${{item.source}}</div>
          <div class="doc-meta">score ${{item.score}}</div>
          <div style="margin-top:8px; line-height:1.5;">${{item.snippet || ""}}</div>
        `;
        evidenceList.appendChild(node);
      }});
    }}

    document.getElementById("askBtn").onclick = askHub;
    document.getElementById("clearBtn").onclick = () => {{
      question.value = "";
      answer.textContent = "The answer will appear here.";
      metaLine.textContent = "Pick a prompt or ask your own question.";
      evidenceList.innerHTML = "";
    }};
    categorySelect.onchange = (e) => setCategory(e.target.value);
    categorySelect2.onchange = (e) => setCategory(e.target.value);
    routeMode.onchange = () => routeMode2.value = routeMode.value;
    routeMode2.onchange = () => routeMode.value = routeMode2.value;
    topK.onchange = () => topK2.value = topK.value;
    topK2.onchange = () => topK.value = topK2.value;

    loadCatalog();
  </script>
</body>
</html>
"""


class HubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            body = render_index_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/catalog":
            payload = {
                "docs": [asdict(item) for item in CATALOG],
                "categories": CATEGORY_LABELS,
            }
            body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/doc":
            slug = parse_qs(parsed.query).get("slug", [""])[0]
            doc = get_doc(slug)
            if not doc:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return
            body = json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/health":
            body = json.dumps({"ok": True, "docs": len(CATALOG), "chunks": len(CHUNKS)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/query":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing question")
            return

        result = run_query(
            question=question,
            category=str(payload.get("category", "all")),
            route_mode=str(payload.get("route_mode", "auto")),
            top_k=int(payload.get("top_k", 4)),
            use_llm=bool(payload.get("use_llm", False)),
        )
        body = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), HubHandler)
    url = f"http://{host}:{port}"
    print(f"New Joiner Knowledge Hub listening on {url}")
    print(f"Knowledge base: {KB_DIR}")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="New joiner knowledge hub")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the local browser UI")
    serve_parser.add_argument("--host", default=DEFAULT_HOST)
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_parser.add_argument("--open", action="store_true", help="Open the UI in the default browser")

    ask_parser = subparsers.add_parser("ask", help="Run one agent query and print JSON")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--category", default="all", choices=sorted(CATEGORY_LABELS))
    ask_parser.add_argument("--route", default="auto", choices=["auto", "qa", "summarize", "compare", "extract"])
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument("--use-llm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "ask":
        result = run_query(
            question=args.question,
            category=args.category,
            route_mode=args.route,
            top_k=args.top_k,
            use_llm=args.use_llm,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if getattr(args, "open", False):
        webbrowser.open_new(f"http://{args.host}:{args.port}")
    serve(getattr(args, "host", DEFAULT_HOST), getattr(args, "port", DEFAULT_PORT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
