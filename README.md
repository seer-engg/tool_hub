# Tool Hub 🎯

A "Hub & Spoke" tool selection engine for AI agents. Combines semantic search with graph-based expansion to intelligently select tools—including dependencies you didn't know you needed.

## Features

- **🧠 Smart Ingestion**: Enriches tools with LLM-generated metadata (use cases, dependencies, neighbors)
- **🔍 Hybrid Retrieval**: Vector search (Hub) + graph expansion (Spoke) for complete toolkits
- **⚡ Fast**: Supports FAISS (local) and Pinecone (cloud) vector stores
- **☁️ Pinecone Support**: Query existing Pinecone indexes with thousands of pre-computed tools

## Installation

```bash
pip install -e .
pip install -e ".[dev]"  # For Pinecone support
```

## Quick Start

### FAISS Mode (Local)

```python
from tool_hub import ToolHub

hub = ToolHub(openai_api_key="...", pinecone_index_name="", pinecone_api_key="")
hub.ingest(tools)  # Your OpenAI-format tools
results = hub.query("I need to file a bug report")
```

### Pinecone Mode (Production)

```python
from tool_hub import ToolHub

hub = ToolHub(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# Query existing index (no ingestion needed)
results = await hub.query_pinecone(
    query="list GitHub repositories",
    integration_name="github",  # Optional filter
    top_k=5
)
```

## API Reference

**Initialization:**
```python
ToolHub(openai_api_key, pinecone_index_name, pinecone_api_key, 
        llm_model="gpt-5-mini", embedding_model="text-embedding-3-small")
```

**Methods:**
- `ingest(tools, max_workers=10)` - Build FAISS index (FAISS mode)
- `query(query, top_k=3)` - Query FAISS index (FAISS mode)
- `query_pinecone(query, integration_name=None, top_k=3)` - Query Pinecone (async)
- `ingest_to_pinecone(tools, max_workers=10)` - Index tools to Pinecone (async, one-time)
- `save(directory)` / `load(directory)` - Persist FAISS index

## Examples

See `examples/precompute_pinecone_index.py` for complete Pinecone indexing example.

## Environment Variables

```bash
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key      # For Pinecone mode
PINECONE_INDEX_NAME=your_index # For Pinecone mode
```
