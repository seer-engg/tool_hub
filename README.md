# Tool Hub 🎯

A "Hub & Spoke" tool selection engine for AI agents. Combines semantic search with graph-based expansion to intelligently select tools—including dependencies you didn't know you needed.

## Features

- **🧠 Smart Ingestion**: Enriches tools with LLM-generated metadata (use cases, dependencies, neighbors)
- **🔍 Hybrid Retrieval**: Vector search (Hub) + graph expansion (Spoke) for complete toolkits
- **☁️ Pinecone Backend**: Cloud-native vector store with namespace isolation for integrations
- **⚡ Async-First**: Fully async API for high-performance tool queries

## Installation

```bash
pip install -e .
pip install -e ".[dev]"  # For development dependencies
```

## Quick Start

```python
import os
from tool_hub import ToolHub

# Initialize ToolHub
hub = ToolHub(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# Ingest tools (one-time, per integration)
await hub.ingest(
    tools=tools,  # List of OpenAI-format tools
    integration_name="github",  # Namespace for this integration
    max_workers=10
)

# Query tools
results = await hub.query(
    query="list GitHub repositories",
    integration_name=["github"],  # Optional: filter by integration(s)
    top_k=5
)
```

## API Reference

**Initialization:**
```python
ToolHub(
    openai_api_key: str,           # Required
    pinecone_index_name: str,      # Required
    pinecone_api_key: str,         # Required
    llm_model: str = "gpt-5-mini",
    embedding_model: str = "text-embedding-3-small",
    embedding_dimensions: Optional[int] = None
)
```

**Methods:**

- `async ingest(tools, integration_name, max_workers=10)` - Enrich and index tools to Pinecone
- `async query(query, integration_name=None, top_k=3)` - Query tools using semantic search

## Examples

See `examples/precompute_pinecone_index.py` for complete Pinecone indexing example.

## Environment Variables

```bash
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
```
