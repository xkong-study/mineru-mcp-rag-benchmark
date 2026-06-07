# MinerU-Powered MCP vs RAG Document Q&A Benchmark

Personal project and technical write-up for comparing two ways to ask questions over private PDFs, Word documents, and PowerPoint files after MinerU parsing.

- MCP mode: Claude Desktop calls MinerU Open MCP Server and answers from parsed documents on demand.
- RAG mode: MinerU-parsed Markdown is chunked, indexed locally, retrieved by question, and used as evidence.

The project shows when MCP is enough for private ad hoc Q&A and when a team should graduate to an indexed RAG workflow.

## What This Project Contains

- Claude Desktop MCP configuration templates for local `uvx` and remote MinerU MCP modes.
- A reusable prompt workflow for private document Q&A.
- A dependency-free local RAG benchmark CLI using TF-IDF and cosine similarity.
- Sample parsed Markdown documents and benchmark questions.
- Technical write-ups comparing MCP, API/SDK, and RAG approaches.
- Validation and benchmark report templates for team adoption.

## Architecture

```text
                    +-----------------------------+
                    | Private PDF / DOCX / PPTX   |
                    +--------------+--------------+
                                   |
                                   v
                         MinerU document parsing
                                   |
                         Parsed Markdown output
                                   |
           +-----------------------+-----------------------+
           |                                               |
           v                                               v
  MCP mode: on-demand Q&A                         RAG mode: indexed Q&A
  Claude Desktop -> MinerU MCP                    chunk -> index -> retrieve
  no pre-indexing                                 reusable local corpus
```

## Setup

### 1. Install `uv`

MinerU Open MCP can be launched with `uvx`. If `uvx` is not available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then check:

```bash
uvx --version
uvx mineru-open-mcp --help
```

### 2. Configure Claude Desktop MCP

Use one of the templates in `config/`:

- `config/claude_desktop_config.local.example.json`: local stdio server launched by Claude Desktop.
- `config/claude_desktop_config.remote.example.json`: remote streamable HTTP server at MinerU.

On macOS, Claude Desktop config is usually:

```text
~/Library/Application Support/Claude/claude_desktop_config.json
```

Merge the `mcpServers.mineru` entry into that file, then fully restart Claude Desktop.

### 3. Run MCP Mode

Use the prompt template in `prompts/document-qa.md`, or generate one:

```bash
python3 scripts/make_document_prompt.py ~/Documents/report.pdf -q "Summarize the risks and next actions."
```

Claude Desktop prompt:

```text
Use the local MCP server named mineru. Call its parse_documents tool on this local file path:

/Users/kongxiangrui/Documents/test.pdf

Then answer:
Summarize the main points, risks, decisions, and next actions.

Answer only from the parsed document content.
```

### 4. Run RAG Mode

The included RAG benchmark runs locally over Markdown. It does not require an API key.

Index the sample parsed document:

```bash
cd ~/mineru-mcp-private-doc-qa
python3 benchmark/rag_benchmark.py ingest data/sample_docs/mineru_mcp_test.md
```

Ask one question:

```bash
python3 benchmark/rag_benchmark.py ask "What are the main risks?"
```

Run the benchmark question set:

```bash
python3 benchmark/rag_benchmark.py benchmark -o reports/rag-results.json
```

## Benchmark Focus

This is a workflow benchmark, not a claim that local TF-IDF beats embedding-based RAG. The goal is to compare engineering tradeoffs:

- MCP setup simplicity vs. RAG indexing overhead
- First-question latency vs. repeated-question latency
- Ad hoc private files vs. reusable shared corpus
- Local path and sandbox failure modes
- Privacy implications of parsed Markdown and stored indexes

## Deliverables

- `benchmark/rag_benchmark.py`: local RAG ingest, ask, and benchmark CLI.
- `benchmark/questions.json`: benchmark question set.
- `data/sample_docs/`: sample MinerU-style parsed Markdown.
- `docs/benchmark-design.md`: MCP vs RAG benchmark design.
- `docs/technical-writeup.md`: portfolio-style technical write-up.
- `docs/mcp-vs-api-vs-rag.md`: adoption tradeoff matrix.
- `reports/benchmark-template.md`: report template for manual MCP/RAG comparison.
- `config/`: Claude Desktop MCP templates.
- `prompts/`: reusable document Q&A prompts.

## Resume Version

MinerU-Powered MCP vs RAG Document Q&A Benchmark  
Personal project and technical write-up | 2026

- Built a Claude Desktop workflow using MinerU Open MCP Server to parse private PDFs, Word documents, and PowerPoint files directly for on-demand document Q&A.
- Implemented a local RAG benchmark over MinerU-parsed Markdown to compare on-demand MCP parsing with indexed retrieval across setup cost, latency, repeated-query performance, sandboxed file paths, and privacy tradeoffs.

## LangChain Orchestration Add-On

The repository now also includes a LangChain-based document orchestration demo under `langchain_orchestrator/`.

It shows how to:

- route questions into `qa`, `summarize`, `compare`, and `extract` flows
- keep retrieval local while adding an optional LangChain synthesis layer
- separate orchestration logic from indexing logic
- present a cleaner resume-ready AI project than a plain vector search demo

## New Joiner Knowledge Hub

The repository now also includes a browser-based onboarding knowledge hub under `new_joiner_hub/`.

It shows how to:

- route new joiner questions across onboarding, product, engineering, security, support, and people/process docs
- present knowledge lookup in a simple local UI
- keep the routing logic in LangGraph while keeping retrieval local

## References

- MinerU ecosystem: https://mineru.net/ecosystem
- MinerU Open MCP registry summary: https://mcp.so/server/mineru-open-mcp/OpenDataLab
- MinerU Ecosystem GitHub: https://github.com/opendatalab/MinerU-Ecosystem
