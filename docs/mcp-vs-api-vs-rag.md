# MCP vs API/SDK vs RAG for Internal Document Access

## Decision Summary

Use MCP when the primary job is interactive Q&A over a small number of private files that users already have locally. Use an API/SDK when the product needs deterministic backend parsing, storage, batch jobs, or automated workflows. Use RAG when many documents need to be searchable repeatedly by many users.

## Tradeoff Matrix

| Approach | Best For | Strengths | Weaknesses |
| --- | --- | --- | --- |
| MCP | Claude Desktop/Cursor-style user workflows | Fast setup, no app backend, natural tool calling, good for ad hoc private documents | Depends on client config, path sandboxing can confuse users, not a shared knowledge base |
| API/SDK | Productized backend workflows | Deterministic integration, logging, auth, retry handling, batch processing | Requires app code, storage/security design, more engineering overhead |
| RAG | Repeated search over a document corpus | Fast repeated retrieval, cross-document search, scalable team knowledge access | Needs indexing, chunking, embeddings, access control, freshness management |

## Why MCP for This Project

This project optimizes for private, on-demand Q&A:

- No separate ingestion job.
- No persistent vector database.
- Lower setup burden for one user or a small team.
- Direct fit for Claude Desktop workflows.
- Users can decide exactly which local files are exposed in each prompt.

## When MCP Is Not Enough

Move beyond MCP if the workflow requires:

- A shared knowledge base across many users.
- Scheduled parsing or batch processing.
- Strict audit trails and access control.
- Retrieval across hundreds or thousands of documents.
- Backend-owned parsing results reused by multiple applications.

## Flash Mode vs Precision / Token Mode

Flash mode is useful for quick Markdown-only parsing of smaller files and can avoid token setup. Precision/token mode is better when files are larger, table/formula extraction matters, or richer output formats are needed.

For team adoption, document the default mode clearly:

- Flash mode: lightweight default for trials and small documents.
- Precision/token mode: production default for larger or high-value documents.

