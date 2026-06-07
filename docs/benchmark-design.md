# MinerU-Powered MCP vs RAG Document Q&A Benchmark

## Goal

This project compares two ways to ask questions over private documents after MinerU parsing:

- MCP mode: Claude Desktop calls MinerU Open MCP Server and answers from the parsed document on demand.
- RAG mode: MinerU-parsed Markdown is chunked, indexed locally, retrieved by a query, and used as evidence.

## What Is Being Benchmarked

The benchmark focuses on workflow tradeoffs rather than model quality alone:

- Setup complexity
- First-question latency
- Repeated-question latency
- Ability to search across many documents
- Storage and privacy implications
- Failure modes around local paths and sandboxing

## MCP Mode

MCP mode is best for ad hoc Q&A over a small number of private files. It avoids pre-indexing and does not require a vector database. The main cost is that each document must be parsed when needed, and the MCP client must be configured correctly.

## RAG Mode

RAG mode is best when the same corpus is queried repeatedly or when many documents must be searched together. It requires an indexing step and persistent storage of parsed chunks.

The included RAG implementation is dependency-free and extractive. It uses TF-IDF and cosine similarity over Markdown chunks. This keeps the benchmark runnable without API keys, while still demonstrating the indexing and retrieval tradeoff.

## Benchmark Commands

Index parsed Markdown:

```bash
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

## Interpreting Results

MCP should win on setup simplicity and one-off private document Q&A. RAG should win on repeated questions over an already indexed document set. For large teams, RAG also supports shared search, access-control design, and reusable corpora, but adds operational complexity.

