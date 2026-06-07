# MinerU MCP Private Document Q&A Test File

Project: Internal Knowledge Access Prototype
Date: 2026-06-07

## Executive Summary

The project evaluates a lightweight document question-answering workflow using Claude Desktop and MinerU Open MCP Server. Instead of building a separate RAG knowledge base, the workflow parses private PDFs, Word documents, and PowerPoint files on demand.

## Key Decisions

1. Use MCP for the first prototype because it avoids a backend service, vector database, and pre-indexing pipeline.
2. Use absolute local file paths in prompts because dragged files can be moved into temporary sandbox directories.
3. Start with Flash mode for small documents and switch to token or Precision mode when larger documents or richer table extraction are needed.

## Risks

1. Claude may not call the local MCP tool unless the MinerU server is enabled in the current chat.
2. Local file paths will fail if the target file does not exist or if the client is running in a remote environment.
3. Flash mode has file size and page limits, so large files may require a MinerU API token.
4. Parsed Markdown and output logs should be treated as sensitive when source documents are private.

## Next Actions

1. Confirm that uvx and mineru-open-mcp are installed.
2. Confirm that Claude Desktop shows the mineru MCP server as connected.
3. Parse this PDF using the local MinerU parse_documents tool.
4. Ask Claude to summarize the document, list risks, and identify missing evidence.

## Success Criteria

The test succeeds if Claude uses the local MinerU MCP server, parses the PDF from the absolute path, and answers only from the document content.

