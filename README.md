# Tool Hub 🎯

A "Hub & Spoke" tool selection engine for AI agents. It combines semantic search with graph-based expansion to intelligently select the right tools for the job—including dependencies you didn't know you needed.

## Features

- **🧠 Smart Ingestion**: Enriches tool definitions with "use cases", "dependencies", and "likely neighbors" using GPT-4.
- **🔍 Hybrid Retrieval**:
  - **Hub**: Vector search finds the most semantically relevant tools.
  - **Spoke**: Graph traversal pulls in required dependencies (e.g., `delete_file` automatically pulls `list_files`).
- **⚡ Fast**: Uses FAISS for high-performance vector similarity search.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import os
from tool_hub import ToolHub

# 1. Initialize
hub = ToolHub(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 2. Ingest your tools (OpenAI format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a new support ticket",
            "parameters": {...}
        }
    },
    # ... more tools
]
hub.ingest(tools)

# 3. Query
query = "I need to file a bug report"
selected_tools = hub.query(query)

print(f"Selected {len(selected_tools)} tools")
```

## How it Works

1. **Ingest**: The `ingest()` method processes your raw tool definitions. It uses an LLM to "dream up" metadata:
   - *Use Cases*: specific user intents this tool satisfies.
   - *Dependencies*: other tools required before this one can run.
   - *Neighbors*: tools often used in sequence.

2. **Retrieve**: The `query()` method first performs a vector search to find the best "anchor" tools. Then, it looks at the enriched metadata to pull in any "spoke" tools (dependencies and neighbors) to ensure the agent has a complete toolkit.
