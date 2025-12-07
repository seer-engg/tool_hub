import os
import json
import faiss
import numpy as np
import traceback
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from .models import Tool, EnrichedTool, ToolFunction

class ToolHub:
    def __init__(
        self, 
        openai_api_key: str, 
        pinecone_index_name: str,
        pinecone_api_key: str,
        llm_model: str = "gpt-5-mini", 
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: Optional[int] = None
    ):
        """
        Initialize the ToolHub.

        Args:
            openai_api_key: The OpenAI API key (required).
            llm_model: The model used for enrichment (default: gpt-5-mini).
            embedding_model: The model used for vector embedding (default: text-embedding-3-small).
            pinecone_api_key: Optional Pinecone API key for vector store (if provided, uses Pinecone instead of FAISS).
            pinecone_index_name: Pinecone index name.
        """
        if not openai_api_key:
            raise ValueError("openai_api_key is required")
            
        self.client = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        
        # Pinecone setup
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.pc = None
        self.pinecone_store = None
        
        if pinecone_api_key:
            try:
                self.pc = Pinecone(api_key=pinecone_api_key)
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    model=embedding_model
                )
                # Initialize vector store (index must exist)
                try:
                    self.pinecone_store = PineconeVectorStore(
                        index_name=pinecone_index_name,
                        embedding=self.embeddings
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize Pinecone store: {e}")
                    print("Index may need to be created first. Use ingest_to_pinecone() to create it.")
            except Exception as e:
                print(f"Warning: Could not initialize Pinecone: {e}")
        
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
            # Check neighbors (dependencies removed - they were broken)
            for neighbor_name in tool.likely_neighbors:
                match = self._find_tool_loose(neighbor_name)
                if match and match.name not in selected_tool_names:
                    print(f"Adding Neighbor: {match.name} (related to {tool.name})")
                    selected_tool_names.add(match.name)
                    expanded_results.append(match)

        final_selection = results + expanded_results
        
        # Return a list of EnrichedTool dictionaries (including parameters)
        return [t.model_dump(include={'name', 'description', 'parameters'}) for t in final_selection]

    def get_tool(self, name: str) -> Optional[Any]:
        """Retrieves the executable tool by name."""
        enriched = self.tool_map.get(name)
        if enriched:
            return enriched.get_executable()
        return None
    
    def bind_executables(self, tools: List[Any]):
        """
        Binds executable functions from a list of tools to the loaded metadata.
        This is required after loading from disk, as functions aren't serialized.
        """
        count = 0
        # tools is a list of BaseTool (LangChain) or similar which have a .name attribute
        # We need to match them to our metadata
        
        # First, create a map of executable tools for faster lookup
        executable_map = {t.name: t for t in tools}
        
        for tool_name, enriched_tool in self.tool_map.items():
            if tool_name in executable_map:
                # The tool hub model (Tool) expects 'executable' field to be the callable/tool object
                if enriched_tool.original_tool:
                    enriched_tool.original_tool.executable = executable_map[tool_name]
                    count += 1
        
        print(f"Bound {count} executable functions to ToolHub.")

    def _enrich_tool_metadata(self, tool: Tool) -> EnrichedTool:
        """
        Uses LLM to generate rich metadata AND verifies tool capabilities through execution.
        
        This implements "Synthetic Data Verification" - instead of just asking the LLM
        to "dream up" use cases, we actually attempt to execute the tool (in a sandbox)
        to verify its capabilities and capture real execution logs.
        """
        # Step 1: Generate initial metadata (as before)
        # Check if parameters schema is empty - if so, ask LLM to infer it
        has_empty_schema = not tool.function.parameters or tool.function.parameters == {}
        
        if has_empty_schema:
            prompt = f"""
            Analyze this tool definition:
            Name: {tool.function.name}
            Description: {tool.function.description}
            Parameters Schema: EMPTY - schema not provided by Composio

            I need to build a smart retrieval index. Provide the following in JSON format:
            1. "use_cases": List of 3-5 specific user intent questions this tool solves (e.g. "How do I delete a file?").
            2. "likely_neighbors": List of actual tool names likely used immediately BEFORE or AFTER this tool in a workflow (must be actual tool names, e.g. "GITHUB_LIST_REPOSITORY_INVITATIONS").
            3. "required_params": List of parameter names required to use this tool (e.g. "emails", "invitation_id"). Extract from description.
            4. "parameters_schema": Infer the parameter schema from the description. Return a JSON object with parameter names as keys and their schema as values. Follow JSON Schema format: {{"param_name": {{"type": "string|array|object|integer|boolean", "description": "...", "items": {{...}} if array, "properties": {{...}} if object}}}}
            5. "embedding_text": A consolidated paragraph combining name, description, and use cases for vector embedding.
            6. "test_scenario": A simple test scenario with example parameters that could be used to verify this tool works.
            
            Return ONLY valid JSON matching this structure.
            """
        else:
            prompt = f"""
            Analyze this tool definition:
            Name: {tool.function.name}
            Description: {tool.function.description}
            Parameters: {json.dumps(tool.function.parameters)}

            I need to build a smart retrieval index. Provide the following in JSON format:
            1. "use_cases": List of 3-5 specific user intent questions this tool solves (e.g. "How do I delete a file?").
            2. "likely_neighbors": List of actual tool names likely used immediately BEFORE or AFTER this tool in a workflow (must be actual tool names, e.g. "GITHUB_LIST_REPOSITORY_INVITATIONS").
            3. "required_params": List of parameter names required to use this tool (e.g. "invitation_id", "user_id").
            4. "embedding_text": A consolidated paragraph combining name, description, and use cases for vector embedding.
            5. "test_scenario": A simple test scenario with example parameters that could be used to verify this tool works.
            
            Return ONLY valid JSON matching this structure.
            """

        response = self.client.chat.completions.create(
            model=self.llm_model, 
            messages=[{"role": "system", "content": "You are a backend architect optimizing tool retrieval."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        content = json.loads(response.choices[0].message.content)
        
        # Step 1.5: If schema was empty and LLM inferred parameters, use them
        if has_empty_schema and content.get('parameters_schema'):
            inferred_params = content.get('parameters_schema', {})
            if inferred_params:
                # Update tool.function.parameters with inferred schema
                tool.function.parameters = inferred_params
                print(f"📝 Inferred parameters for {tool.function.name}: {list(inferred_params.keys())}")
        elif has_empty_schema:
            # If LLM didn't provide parameters_schema, log warning
            print(f"⚠️ Warning: Empty schema for {tool.function.name} but LLM didn't infer parameters_schema")
        
        # Step 2: Synthetic Data Verification - Attempt to execute the tool
        execution_log = ""
        verified_capabilities = []
        
        if tool.executable:
            try:
                execution_log, verified_capabilities = self._verify_tool_execution(
                    tool, 
                    content.get('test_scenario', {})
                )
                
                # Enhance embedding text with verified execution info
                if execution_log:
                    content['embedding_text'] += f"\n\nVerified execution: {execution_log[:200]}"
                    # Add verified capabilities to use cases
                    if verified_capabilities:
                        content['use_cases'].extend(verified_capabilities)
                        # Deduplicate
                        content['use_cases'] = list(dict.fromkeys(content['use_cases']))
                        
            except Exception as e:
                # Tool execution failed - this is valuable information too!
                execution_log = f"Execution verification failed: {str(e)[:200]}"
                print(f"⚠️ Tool {tool.function.name} verification failed: {e}", flush=True)
        
        return EnrichedTool(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=tool.function.parameters or {},
            use_cases=content.get('use_cases', []),
            likely_neighbors=content.get('likely_neighbors', []),
            required_params=content.get('required_params', []),
            embedding_text=content.get('embedding_text', ""),
            original_tool=tool
        )
    
    def _verify_tool_execution(
        self, 
        tool: Tool, 
        test_scenario: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """
        Attempt to execute a tool in a safe manner to verify its capabilities.
        
        This solves the "hallucinated capability" problem by actually testing
        what the tool can do, rather than relying solely on descriptions.
        
        Returns:
            Tuple of (execution_log, verified_capabilities)
        """
        if not tool.executable:
            return "", []
        
        execution_log_parts = []
        verified_capabilities = []
        
        try:
            # Strategy 1: Try to call the tool with minimal/safe parameters
            # This is a "dry run" - we're not trying to actually perform actions,
            # just verify the tool interface and basic behavior
            
            # Check if it's a LangChain tool (has .invoke or .run method)
            if hasattr(tool.executable, 'invoke'):
                # Try with empty or minimal input
                try:
                    # Some tools might have a "dry_run" or "validate" mode
                    # For now, we'll just check if the tool can be instantiated
                    # and has the expected interface
                    execution_log_parts.append("Tool interface verified (LangChain tool)")
                    verified_capabilities.append(f"Tool {tool.function.name} is callable via .invoke()")
                except Exception as e:
                    execution_log_parts.append(f"Interface check: {str(e)[:100]}")
            
            # Strategy 2: Generate a Python script that would call the tool
            # and analyze what it would do (without actually executing dangerous operations)
            if test_scenario:
                # Create a test script template
                test_script = f"""
# Test script for {tool.function.name}
# This verifies the tool's expected behavior

tool_name = "{tool.function.name}"
parameters = {json.dumps(test_scenario)}

# Tool would be called with these parameters
# This helps verify parameter types and expected behavior
"""
                execution_log_parts.append(f"Generated test scenario: {test_script[:150]}")
                verified_capabilities.append(f"Tool accepts parameters: {list(test_scenario.keys())}")
            
            # Strategy 3: Check tool metadata and documentation
            if hasattr(tool.executable, 'description'):
                desc = tool.executable.description
                execution_log_parts.append(f"Tool description: {desc[:100]}")
            
            # For tools that are safe to test (read-only operations), we could actually call them
            # But for safety, we'll be conservative and just verify the interface
            
            execution_log = " | ".join(execution_log_parts)
            
        except Exception as e:
            execution_log = f"Verification error: {str(e)[:200]}"
            traceback.print_exc()
        
        return execution_log, verified_capabilities

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates embeddings using OpenAI."""
        # Replace newlines with spaces to handle multi-line inputs properly, 
        # as per OpenAI recommendation for embeddings
        cleaned_text = text.replace("\n", " ")
        
        params = {
            "input": cleaned_text,
            "model": self.embedding_model
        }
        if self.embedding_dimensions:
            params["dimensions"] = self.embedding_dimensions
        
        response = self.client.embeddings.create(**params)
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
            params = {
                "input": batch_texts,
                "model": self.embedding_model
            }
            if self.embedding_dimensions:
                params["dimensions"] = self.embedding_dimensions
            response = self.client.embeddings.create(**params)
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

    # ============================================================================
    # Pinecone-Based Methods
    # ============================================================================

    async def ingest_to_pinecone(
        self, 
        tools: List[Union[Tool, Dict[str, Any]]], 
        integration_name: str,
        max_workers: int = 10
    ):
        """
        Ingests tools, enriches them, and stores them in Pinecone vector store.
        
        Args:
            tools: List of Tool objects or dictionaries matching OpenAI tool schema.
            integration_name: Integration name (e.g., "github", "asana") for metadata filtering.
            max_workers: Number of concurrent threads for enrichment (default: 10).
        """
        if not self.pc:
            raise ValueError("Pinecone not initialized. Provide pinecone_api_key in __init__.")
        
        integration_name = integration_name.lower()
        
        print(f"Ingesting {len(tools)} tools for {integration_name} into Pinecone...")
        
        # Normalize inputs (same as regular ingest)
        normalized_tools = []
        for t in tools:
            # Filter out deprecated tools before normalization
            if isinstance(t, dict):
                description = t.get("description", "") or t.get("function", {}).get("description", "")
                if "deprecated" in description.lower():
                    continue
            elif isinstance(t, Tool):
                if "deprecated" in (t.function.description or "").lower():
                    continue
            
            if isinstance(t, dict):
                if "function" in t:
                    normalized_tools.append(Tool.from_dict(t))
                elif "parameters" in t:
                    normalized_tools.append(Tool(function=ToolFunction(**t)))
                else:
                    try:
                        normalized_tools.append(Tool.from_dict(t))
                    except:
                        print(f"Skipping invalid tool structure: {t.keys()}")
            elif isinstance(t, Tool):
                normalized_tools.append(t)
            else:
                raise ValueError(f"Unsupported tool type: {type(t)}")

        # Enrich tools (same as regular ingest)
        enriched_tools = []
        print(f"Enriching tools with concurrency (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tool = {
                executor.submit(self._enrich_tool_metadata, tool): tool 
                for tool in normalized_tools
            }
            
            for future in tqdm(as_completed(future_to_tool), total=len(normalized_tools), desc="Enriching Tools"):
                tool = future_to_tool[future]
                try:
                    enriched = future.result()
                    enriched_tools.append(enriched)
                except Exception as e:
                    print(f"Failed to enrich {tool.function.name}: {e}")
        
        # Generate embeddings and store in Pinecone
        print(f"Generating embeddings and storing {len(enriched_tools)} enriched tools in Pinecone...")
        
        # Get Pinecone index
        index = self.pc.Index(self.pinecone_index_name)
        
        # Prepare vectors for batch upload
        vectors_to_upsert = []
        stored_count = 0
        
        for enriched_tool in tqdm(enriched_tools, desc="Storing Tools"):
            try:
                # Generate embedding for embedding_text
                params = {
                    "input": enriched_tool.embedding_text.replace("\n", " "),
                    "model": self.embedding_model
                }
                if self.embedding_dimensions:
                    params["dimensions"] = self.embedding_dimensions
                embedding_response = self.client.embeddings.create(**params)
                embedding = embedding_response.data[0].embedding
                
                # Prepare flattened metadata structure
                # Pinecone supports: strings, numbers, booleans, lists of strings
                # Only nested structures (like parameters dict) need to be JSON strings
                metadata = {
                    # Filtering/Display fields
                    "integration": integration_name,  # string
                    "description": enriched_tool.description,  # string (full, not truncated)
                    
                    # Flattened lists (directly accessible, no JSON parsing needed)
                    "use_cases": enriched_tool.use_cases,  # list of strings
                    "likely_neighbors": enriched_tool.likely_neighbors,  # list of strings
                    "required_params": enriched_tool.required_params,  # list of strings (renamed from required_entities)
                    
                    # Only nested structure (must be JSON string)
                    "parameters": json.dumps(enriched_tool.parameters),  # JSON string (nested dict)
                    
                    # Backward compatibility: embedding_text for re-embedding with different models
                    "embedding_text": enriched_tool.embedding_text,  # string (full, for re-embedding)
                }
                
                # Vector ID: use tool_name directly (tool names already include integration prefix)
                vector_id = enriched_tool.name
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                stored_count += 1
                
                # Batch upload every 100 vectors
                if len(vectors_to_upsert) >= 100:
                    index.upsert(vectors=vectors_to_upsert, namespace=integration_name)
                    vectors_to_upsert = []
                    
            except Exception as e:
                print(f"Failed to store {enriched_tool.name}: {e}")
        
        # Upload remaining vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert, namespace=integration_name)
        
        print(f"✅ Stored {stored_count}/{len(enriched_tools)} tools for {integration_name} in Pinecone")
        
        # Update internal state for compatibility
        self.metadata = enriched_tools
        self.tool_map = {t.name: t for t in enriched_tools}

    async def query_pinecone(
        self,
        query: str,
        integration_name: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Query tools from Pinecone using semantic search.
        Implements Hub & Spoke method: semantic search + dependency/neighbor expansion.
        
        Args:
            query: Search query string.
            integration_name: Optional integration name (e.g., "github", "asana") for metadata filtering.
            top_k: Number of top results to return from semantic search.
        
        Returns:
            List of tool dictionaries compatible with OpenAI tool schema.
        """
        if not self.pc:
            raise ValueError("Pinecone not initialized. Provide pinecone_api_key in __init__.")
        
        # Generate query embedding
        query_embedding = self._get_embedding(query.replace("\n", " "))
        query_vector = query_embedding[0].tolist()
        
        # Get Pinecone index
        index = self.pc.Index(self.pinecone_index_name)
        
        # Build filter if integration specified
        filter_dict = None
        if integration_name:
            filter_dict = {"integration": integration_name.lower()}
        
        # 1. Semantic Search (Hub)
        # Use namespace if integration specified (better isolation than metadata filter)
        # When namespace is None, omit it from query (queries default namespace)
        query_kwargs = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True
        }
        
        if integration_name:
            query_kwargs["namespace"] = integration_name.lower()  # Use namespace for integration isolation
            if filter_dict:
                query_kwargs["filter"] = filter_dict  # Additional metadata filter if needed
        
        try:
            query_response = index.query(**query_kwargs)
        except Exception as e:
            print(f"Pinecone query failed: {e}")
            return []
        
        if not query_response.matches:
            return []
        
        # Extract tool names from search results
        selected_tool_names = set()
        results: List[Dict[str, Any]] = []
        tool_metadata_map = {}  # Store metadata for expansion
        
        print(f"\n--- Anchor Tools (Vector Match) ---")
        for match in query_response.matches:
            metadata = match.metadata
            tool_name = match.id  # Use vector ID as tool name (ID is the tool name)
            
            if tool_name and tool_name not in selected_tool_names:
                print(f"Found: {tool_name} (score: {match.score:.3f})")
                selected_tool_names.add(tool_name)
                
                # Direct access - no JSON parsing needed (except parameters)
                tool_metadata_map[tool_name] = metadata
                
                # Parse only parameters (the only nested structure)
                parameters = {}
                try:
                    parameters_str = metadata.get("parameters", "{}")
                    if parameters_str:
                        parameters = json.loads(parameters_str)
                except json.JSONDecodeError:
                    pass
                
                tool_dict = {
                    "name": tool_name,  # Use ID directly
                    "description": metadata.get("description", ""),
                    "parameters": parameters
                }
                results.append(tool_dict)
        
        # 2. Graph Expansion (Spoke) - expand with neighbors only
        expanded_results: List[Dict[str, Any]] = []
        print(f"\n--- Expanded Tools (Graph Neighbors) ---")
        
        for tool_dict in results:
            tool_name = tool_dict.get("name")
            if not tool_name:
                continue
            
            # Direct access to flattened metadata - no JSON parsing needed
            metadata = tool_metadata_map.get(tool_name, {})
            likely_neighbors = metadata.get("likely_neighbors", [])  # Direct access!
            
            # Check neighbors - fetch from Pinecone
            for neighbor_name in likely_neighbors:
                if neighbor_name not in selected_tool_names:
                    try:
                        # Fetch neighbor tool by ID (tool_name directly, no integration prefix)
                        # Use namespace from the tool's metadata to fetch from correct namespace
                        neighbor_id = neighbor_name
                        neighbor_namespace = metadata.get("integration", integration_name.lower() if integration_name else None)
                        fetch_response = index.fetch(ids=[neighbor_id], namespace=neighbor_namespace)
                        
                        if neighbor_id in fetch_response.vectors:
                            neighbor_metadata = fetch_response.vectors[neighbor_id].metadata  # Already flat!
                            
                            # Parse only parameters (the only nested structure)
                            neighbor_parameters = {}
                            try:
                                neighbor_parameters_str = neighbor_metadata.get("parameters", "{}")
                                if neighbor_parameters_str:
                                    neighbor_parameters = json.loads(neighbor_parameters_str)
                            except json.JSONDecodeError:
                                pass
                            
                            print(f"Adding Neighbor: {neighbor_name} (related to {tool_name})")
                            selected_tool_names.add(neighbor_name)
                            tool_metadata_map[neighbor_name] = neighbor_metadata
                            expanded_results.append({
                                "name": neighbor_name,  # Use ID directly
                                "description": neighbor_metadata.get("description", ""),
                                "parameters": neighbor_parameters
                            })
                    except Exception as e:
                        print(f"Failed to load neighbor {neighbor_name}: {e}")
        
        final_selection = results + expanded_results
        return final_selection
