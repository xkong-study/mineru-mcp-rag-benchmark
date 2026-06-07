# Validation Checklist

## Environment

- `uvx --version` works if using local stdio mode.
- Claude Desktop has been restarted after config changes.
- MinerU server appears in Claude Desktop MCP tools.
- `MINERU_API_TOKEN` is set only when using precision/token mode.

## File Access

- Test with one PDF using an absolute path.
- Test with one DOCX using an absolute path.
- Test with one PPTX using an absolute path.
- Test a file path containing spaces.
- Avoid relying on dragged files unless the client shows the real path.

## Q&A Behavior

- Ask for a factual answer present in the document.
- Ask for an answer not present in the document and verify that Claude says evidence is missing.
- Ask for a table or slide-specific summary.
- Ask for decisions, risks, owners, and deadlines.

## Privacy

- Do not place private files inside the repo.
- Do not commit parsed outputs.
- Do not commit API tokens.
- Review MinerU mode and endpoint before using sensitive documents.

## Team Adoption

- Pick a default mode: Flash for trials, precision/token mode for production parsing.
- Document supported file formats and size/page limits for the chosen mode.
- Document where users should store files and how to copy absolute paths.
- Define when the team should switch to API/SDK or RAG.

