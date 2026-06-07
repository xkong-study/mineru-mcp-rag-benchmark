# MinerU-Powered MCP vs RAG Document Q&A Benchmark

Personal project and technical write-up | 2026

## Problem

Teams often have private PDFs, Word documents, and slide decks that are useful for question answering. The engineering question is whether to parse documents on demand through MCP or build a reusable RAG index.

This project compares both paths using MinerU as the document parsing layer.

## Solution

The MCP workflow connects Claude Desktop to MinerU through MCP. The user provides absolute local file paths in the prompt. Claude invokes MinerU's document parsing tool, MinerU converts the documents into Markdown, and Claude answers from the parsed content.

The RAG workflow indexes MinerU-parsed Markdown locally. A benchmark CLI chunks the Markdown, retrieves relevant chunks with TF-IDF and cosine similarity, and returns extractive answers with evidence snippets.

Supported target inputs include PDFs, Word documents, PowerPoint files, images, and HTML pages, depending on the active MinerU mode and server version.

## Implementation

The project includes:

- Claude Desktop MCP configuration templates for local stdio and remote streamable HTTP modes.
- Prompt templates for document Q&A and meeting-pack summarization.
- A helper script that converts local file paths into a structured Claude prompt.
- A dependency-free local RAG benchmark CLI with ingest, ask, and batch benchmark commands.
- Sample parsed Markdown and benchmark questions.
- A technical comparison of MCP, API/SDK, and RAG options.
- A validation checklist covering install, file paths, privacy, and document format support.

## Key Design Choices

### On-demand parsing for MCP mode

MCP mode avoids a vector database because the target use case is ad hoc Q&A over a small number of files. This reduces setup complexity and avoids storing a second copy of document content in a retrieval system.

### Local indexing for RAG mode

RAG mode stores parsed Markdown chunks and supports repeated queries over the same corpus. The included implementation uses TF-IDF instead of embeddings so the benchmark can run without API keys.

### Absolute file paths

The prompt asks users to provide full local paths. Some MCP clients sandbox dragged files into temporary directories, which can make relative or UI-provided paths unreliable.

### Separate config templates

Local `uvx` mode and remote MinerU MCP mode are kept as separate templates. This makes the deployment choice explicit and avoids mixing local subprocess assumptions with remote MCP server assumptions.

### Explicit evidence rules

The prompt instructs Claude to answer only from parsed document content and to say when evidence is missing. This reduces unsupported synthesis when the document does not contain the requested answer.

## Tradeoffs

MCP gives the fastest path for interactive personal and small-team workflows. API/SDK integration is stronger for backend automation. RAG is stronger for shared, repeated search over a large corpus.

The main MCP risks are client configuration drift, token handling, local path sandboxing, and user confusion around file visibility. These are handled through config templates, `.gitignore`, prompt instructions, and validation checks.

## Outcome

The finished project demonstrates both approaches. MCP is simpler for one-off private document Q&A. RAG is better when the same parsed corpus is queried repeatedly or when many documents need cross-document search.

## Resume Version

Built a Claude Desktop workflow around MinerU Open MCP Server to read private PDFs, Word documents, and PowerPoint files directly for on-demand document Q&A.

Implemented a local RAG benchmark over MinerU-parsed Markdown to compare on-demand MCP parsing with indexed retrieval across setup cost, latency, repeated-query performance, sandboxed file paths, and privacy tradeoffs.
