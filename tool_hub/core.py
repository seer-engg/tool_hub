import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from .models import Tool, EnrichedTool, ToolFunction

class ToolHub:
    def __init__(
        self, 
        openai_api_key: str, 
        llm_model: str = "gpt-4.1", 
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the ToolHub.

        Args:
            openai_api_key: The OpenAI API key (required).
            llm_model: The model used for enrichment (default: gpt-4.1).
            embedding_model: The model used for vector embedding (default: text-embedding-3-small).
        """
        if not openai_api_key:
            raise ValueError("openai_api_key is required")
            
        self.client = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        self.index = None
        self.metadata: List[EnrichedTool] = []
        self.tool_map: Dict[str, EnrichedTool] = {}

    def ingest(self, tools: List[Union[Tool, Dict[str, Any]]], max_workers: int = 10):
        """
        Ingests a list of tools, enriches them with metadata, and builds a FAISS index.
        
        Args:
            tools: List of Tool objects or dictionaries matching OpenAI tool schema.
            max_workers: Number of concurrent threads for enrichment (default: 10).
        """
        print(f"Ingesting {len(tools)} tools...")
        
        # Normalize inputs
        normalized_tools = []
        for t in tools:
            if isinstance(t, dict):
                # Handle both full tool format {"type": "function", "function": {...}} and direct function/action definition
                if "function" in t:
                    normalized_tools.append(Tool.from_dict(t))
                elif "parameters" in t: # It's likely a direct function/action definition (e.g. from Composio's ActionModel)
                    # Adapt ActionModel structure to ToolFunction structure if needed
                    # Composio's ActionModel usually has 'name', 'description', 'parameters' which fits directly
                    normalized_tools.append(Tool(function=ToolFunction(**t)))
                else:
                     # Try to fit it blindly or skip
                     try:
                         normalized_tools.append(Tool.from_dict(t))
                     except:
                         print(f"Skipping invalid tool structure: {t.keys()}")
            elif isinstance(t, Tool):
                normalized_tools.append(t)
            else:
                raise ValueError(f"Unsupported tool type: {type(t)}")

        enriched_tools = []
        
        print(f"Enriching tools with concurrency (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tool = {
                executor.submit(self._enrich_tool_metadata, tool): tool 
                for tool in normalized_tools
            }
            
            # Use tqdm to show a progress bar for the completed futures
            for future in tqdm(as_completed(future_to_tool), total=len(normalized_tools), desc="Enriching Tools"):
                tool = future_to_tool[future]
                try:
                    enriched = future.result()
                    enriched_tools.append(enriched)
                except Exception as e:
                    print(f"Failed to enrich {tool.function.name}: {e}")
        
        self._build_index(enriched_tools)
        self.metadata = enriched_tools
        self.tool_map = {t.name: t for t in enriched_tools}
        print("Ingestion complete.")

    def save(self, directory: str):
        """Saves the index and metadata to disk."""
        if not self.index:
            raise RuntimeError("No index to save. Run ingest() first.")
            
        os.makedirs(directory, exist_ok=True)
        
        # 1. Save FAISS index
        index_path = os.path.join(directory, "tools.index")
        faiss.write_index(self.index, index_path)
        
        # 2. Save Metadata
        metadata_path = os.path.join(directory, "metadata.json")
        data = [t.model_dump() for t in self.metadata]
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Index and metadata saved to {directory}")

    def load(self, directory: str):
        """Loads the index and metadata from disk."""
        index_path = os.path.join(directory, "tools.index")
        metadata_path = os.path.join(directory, "metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Index or metadata not found in {directory}")
            
        # 1. Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # 2. Load Metadata
        with open(metadata_path, "r") as f:
            data = json.load(f)
            self.metadata = [EnrichedTool(**item) for item in data]
            
        self.tool_map = {t.name: t for t in self.metadata}
        print(f"Loaded {len(self.metadata)} tools from {directory}")

    def query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves tools using the 'Hub & Spoke' method.
        Returns a list of dictionaries compatible with OpenAI tool schema.
        """
        if not self.index:
            raise RuntimeError("Index not built. Call ingest() first.")

        # 1. Semantic Search (Hub)
        query_vector = self._get_embedding(query)
        distances, indices = self.index.search(query_vector, top_k)
        
        selected_tool_names = set()
        results: List[EnrichedTool] = []
        
        print(f"\n--- Anchor Tools (Vector Match) ---")
        for idx in indices[0]:
            if idx < len(self.metadata):
                tool = self.metadata[idx]
                if tool.name not in selected_tool_names:
                    print(f"Found: {tool.name}")
                    selected_tool_names.add(tool.name)
                    results.append(tool)

        # 2. Graph Expansion (Spoke)
        expanded_results: List[EnrichedTool] = []
        print(f"\n--- Expanded Tools (Graph Neighbors) ---")
        
        for tool in results:
            # Check dependencies
            for dep_name in tool.dependencies:
                match = self._find_tool_loose(dep_name)
                if match and match.name not in selected_tool_names:
                    print(f"Adding Dependency: {match.name} (needed by {tool.name})")
                    selected_tool_names.add(match.name)
                    expanded_results.append(match)
            
            # Check neighbors
            for neighbor_name in tool.likely_neighbors:
                match = self._find_tool_loose(neighbor_name)
                if match and match.name not in selected_tool_names:
                    print(f"Adding Neighbor: {match.name} (related to {tool.name})")
                    selected_tool_names.add(match.name)
                    expanded_results.append(match)

        final_selection = results + expanded_results
        
        # Convert back to OpenAI dict format
        return [t.original_tool.model_dump() for t in final_selection]

    def _enrich_tool_metadata(self, tool: Tool) -> EnrichedTool:
        """Uses LLM to generate rich metadata."""
        # print(f"Enriching {tool.function.name}...") # Reduced logging for concurrency
        prompt = f"""
        Analyze this tool definition:
        Name: {tool.function.name}
        Description: {tool.function.description}
        Parameters: {json.dumps(tool.function.parameters)}

        I need to build a smart retrieval index. Provide the following in JSON format:
        1. "use_cases": List of 3-5 specific user intent questions this tool solves (e.g. "How do I delete a file?").
        2. "dependencies": List of generic tool names that likely must run BEFORE this tool (e.g. "list_buckets" before "delete_bucket").
        3. "likely_neighbors": List of generic tool names likely used immediately BEFORE or AFTER this tool in a workflow.
        4. "required_entities": List of abstract entities required (e.g. "bucket_name", "user_id").
        5. "embedding_text": A consolidated paragraph combining name, description, and use cases for vector embedding.
        
        Return ONLY valid JSON matching this structure.
        """

        response = self.client.chat.completions.create(
            model=self.llm_model, 
            messages=[{"role": "system", "content": "You are a backend architect optimizing tool retrieval."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        content = json.loads(response.choices[0].message.content)
        
        return EnrichedTool(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=tool.function.parameters or {},
            use_cases=content.get('use_cases', []),
            dependencies=content.get('dependencies', []),
            likely_neighbors=content.get('likely_neighbors', []),
            required_entities=content.get('required_entities', []),
            embedding_text=content.get('embedding_text', ""),
            original_tool=tool
        )

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates embeddings using OpenAI."""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return np.array([response.data[0].embedding]).astype('float32')

    def _build_index(self, enriched_tools: List[EnrichedTool]):
        """Builds in-memory FAISS index."""
        if not enriched_tools:
            print("No tools to index.")
            return

        print("Generating embeddings for index...")
        texts = [t.embedding_text for t in enriched_tools]
        
        embeddings_list = []
        batch_size = 20
        # Use tqdm for embedding generation progress as well
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                input=batch_texts,
                model=self.embedding_model
            )
            embeddings_list.extend([item.embedding for item in response.data])

        embeddings = np.array(embeddings_list).astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Indexed {len(enriched_tools)} tools.")

    def _find_tool_loose(self, name_hint: str) -> Optional[EnrichedTool]:
        """Helper to find tool by loose name matching."""
        if name_hint in self.tool_map:
            return self.tool_map[name_hint]
        
        for real_name, tool_data in self.tool_map.items():
            if name_hint.lower() in real_name.lower() or real_name.lower() in name_hint.lower():
                return tool_data
        return None
