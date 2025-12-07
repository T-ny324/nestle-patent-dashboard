# Azure OpenAI Quick Start Guide

## Setup

1. Install uv
Follow the instructions at https://docs.astral.sh/uv/.

2. Install dependencies:
```bash
uv sync
```

3. Run the examples using uv:
```bash
uv run openai-quickstart/example_embeddings.py
uv run openai-quickstart/example_llm.py
```

## What's Included

- `openai-quickstart/example_embeddings.py` - Generate embeddings for text
- `openai-quickstart/example_llm.py` - Use GPT-5 to generate labels and summaries
- `.env` - API credentials (Zinzan will share with you separately)
- `pyproject.toml` - uv dependencies

## Models Available

- **Embeddings**: `text-embedding-3-small`
- **Chat**: `gpt-5-mini` or `gpt-5.1`
