# LangChain Document Orchestration Demo

This project turns the private-document workflow into a real LangChain orchestration example.

It does four things:

1. Ingests parsed Markdown documents into a local JSON index.
2. Routes each question into one of three paths: `qa`, `summarize`, or `compare`.
3. Uses a LangChain `RunnableBranch` for orchestration when the library is installed.
4. Optionally adds an LLM synthesis layer on top of the local evidence.

## Why this is useful

The project is not just a vector search demo. It shows how to:

- Route user intent before calling tools.
- Choose between extractive retrieval and summary/comparison flows.
- Keep sensitive files local while still using an LLM when needed.
- Separate orchestration logic from document indexing.

## Install

```bash
cd ~/mineru-mcp-private-doc-qa/langchain_orchestrator
python3 -m pip install -r requirements.txt
```

Optional if you want the synthesis layer:

```bash
export OPENAI_API_KEY=your_key_here
```

## Run

Index sample documents:

```bash
python3 -m langchain_orchestrator ingest ../data/sample_docs -o ../outputs/langchain_index.json
```

Ask a question:

```bash
python3 -m langchain_orchestrator ask "What are the key decisions and risks?" --index ../outputs/langchain_index.json
```

Run the demo:

```bash
python3 -m langchain_orchestrator demo
```

Use the LLM synthesis layer:

```bash
python3 -m langchain_orchestrator ask "What are the key decisions and risks?" --index ../outputs/langchain_index.json --use-llm
```

## Portfolio angle

Resume-ready wording:

- Built a LangChain-based document orchestration layer that routes private-document questions into retrieval, summary, and comparison paths.
- Added an optional LLM synthesis stage on top of locally indexed evidence while keeping document parsing and retrieval local.
- Separated orchestration, retrieval, and answer synthesis to support auditable private-document Q&A workflows.
- Published the project on GitHub: github.com/xkong-study/mineru-mcp-rag-benchmark.
