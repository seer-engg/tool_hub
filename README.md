# Tool Hub Prototype

A "Hub & Spoke" tool selection engine for MCP toolkits.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   Export `OPENAI_API_KEY` and optionally `COMPOSIO_API_KEY`.
   ```bash
   export OPENAI_API_KEY=sk-...
   export COMPOSIO_API_KEY=...
   ```

## Usage

1. **Ingestion (One-time setup)**
   Fetches tools (mock or real), generates metadata using GPT-4, and builds the vector index.
   ```bash
   python ingestion.py
   ```
   *Outputs: `tools.index`, `tools_metadata.json`*

2. **Run Simulation**
   Runs a query through the retriever and simulates the LLM tool selection.
   ```bash
   python main.py
   ```

## Architecture

- **Ingestion**: Enriches raw tool definitions with "Use Cases", "Dependencies", and "Likely Neighbors" using an LLM.
- **Retrieval**:
  1. **Hub**: Vector search to find top semantic matches.
  2. **Spoke**: Graph traversal to pull in required dependencies (e.g., `delete_object` pulls in `list_objects`).

