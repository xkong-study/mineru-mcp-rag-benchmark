# LangGraph New Joiner Knowledge Agent

This is a local onboarding knowledge agent with both a browser UI and a CLI verification path.

It combines:

- a small private knowledge base in Markdown
- a LangGraph state machine for `route -> retrieve -> answer`
- explicit routes for `qa`, `summarize`, `compare`, and `extract`
- lightweight local retrieval over Markdown evidence
- optional LLM synthesis when `OPENAI_API_KEY` is available
- a local HTTP UI with category filtering, document preview, and evidence-backed answers

## What It Demonstrates

- Intent routing before retrieval, so the response format matches the task.
- Local evidence retrieval, so answers remain inspectable without a public vector store.
- Separation between retrieval, deterministic extractive answers, and optional LLM synthesis.
- A usable UI for category filtering, document preview, route selection, and evidence display.
- A CLI path that makes the agent behavior easy to test in interviews or demos.

## Agent Flow

```text
question
  -> route_node       qa | summarize | compare | extract
  -> retrieve_node    category-filtered local Markdown retrieval
  -> answer_node      route-specific extractive answer
  -> synthesis        optional LLM answer grounded in evidence
```

## Install

```bash
cd ~/mineru-mcp-private-doc-qa
python3 -m pip install -r new_joiner_hub/requirements.txt
```

Optional LLM synthesis:

```bash
export OPENAI_API_KEY=your_key_here
python3 -m pip install langchain-openai langchain-core
```

## Run the UI

```bash
cd ~/mineru-mcp-private-doc-qa
python3 -m new_joiner_hub serve --open
```

Default port: `8088`.

## Run a CLI Query

```bash
python3 -m new_joiner_hub ask "How do I release a feature safely?"
```

Force a route:

```bash
python3 -m new_joiner_hub ask "What should I do in my first week?" --route extract --category welcome
```

Use optional LLM synthesis:

```bash
python3 -m new_joiner_hub ask "Summarize the onboarding process" --use-llm
```

## Resume-Accurate Description

- Built a LangGraph onboarding knowledge agent that routes questions into QA, summarize, compare, or extract paths before local retrieval.
- Implemented category-filtered Markdown retrieval and evidence-backed responses, with optional LLM synthesis only when an API key is available.
- Added a local browser UI and CLI query path so the agent is both demoable and testable.
