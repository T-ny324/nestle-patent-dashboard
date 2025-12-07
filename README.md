# AIM Reply Micro-Internship Winter 2025

Patent analysis project using AI embeddings and clustering.

## Quick Start

### 1. Install uv package manager
Follow instructions at https://docs.astral.sh/uv/

### 2. Install dependencies
```bash
uv sync
```

### 3. Set up environment variables
Create a `.env` file at the root with the credentials provided.

### 4. Test the setup
Run the example scripts to verify everything works:
```bash
uv run openai-quickstart/example_embeddings.py
uv run openai-quickstart/example_llm.py
```

## What's Included

- `data/` - Patent datasets (one for each intern)
- `openai-quickstart/` - Example scripts showing how to use Azure OpenAI
  - `example_embeddings.py` - Generate text embeddings
  - `example_llm.py` - Use GPT models for analysis
  - `README.md` - Quick reference for the API
- `.env.sample` - Template for environment variables
- `pyproject.toml` - Project dependencies

## Available Models

- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **Chat**: `gpt-5.1` or `gpt-5-mini`
