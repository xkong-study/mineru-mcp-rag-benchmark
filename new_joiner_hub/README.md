# New Joiner Knowledge Hub

This is a local browser UI for onboarding and internal knowledge lookup.

It combines:

- a small private knowledge base in Markdown
- a LangGraph routing layer for `qa`, `summarize`, `compare`, and `extract`
- a local HTTP UI so a new joiner can ask questions without leaving the browser

## What it demonstrates

- How to route onboarding questions by intent before retrieval
- How to search across internal docs with lightweight local evidence
- How to keep answer generation separate from retrieval
- How to present the whole workflow in a simple UI instead of a CLI

## Run

```bash
cd ~/mineru-mcp-private-doc-qa/new_joiner_hub
python3 -m new_joiner_hub
```

Then open the URL printed in the terminal.

Default port: `8088`

