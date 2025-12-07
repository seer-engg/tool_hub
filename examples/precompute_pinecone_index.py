"""
Pre-computation script for indexing all Composio tools into Pinecone vector store.

This script:
1. Fetches tools from Composio for each integration system
2. Uses ToolHub to enrich tools with LLM-generated metadata
3. Stores enriched tools in Pinecone with integration metadata filtering

Run once - index persists across all deployments.
Can be re-run safely (idempotent - upserts existing vectors).

Usage:
    python -m tool_hub.precompute_pinecone_index
"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from composio import Composio
from composio_langchain import LangchainProvider
from tool_hub import ToolHub
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# Static list of integration systems
INTEGRATIONS = [
    "GITHUB",
    "ASANA",
    "SLACK",
    "GMAIL",
    "GOOGLECALENDAR",
    "GOOGLEDOCS",
    "GOOGLESHEETS",
    "TELEGRAM",
    "TWITTER",
    # Add more integrations as needed
    # "NOTION",
    # "LINEAR",
    # "JIRA",
    # "CONFLUENCE",
    # etc.
]

# Test mode: limit number of tools per integration (set to None to process all)
TEST_MODE_LIMIT = None  # Set to None for full run (process all tools)


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int = 512):
    """
    Ensure Pinecone index exists, create if it doesn't.
    
    Args:
        pc: Pinecone client instance
        index_name: Name of the index
        dimension: Vector dimension (default: 512 for text-embedding-3-small)
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Change to your preferred region
            )
        )
        print(f"✅ Created index '{index_name}'")
    else:
        print(f"✅ Index '{index_name}' already exists")


async def precompute_integration(
    integration_name: str,
    composio_client: Composio,
    toolhub: ToolHub,
    user_id: str
) -> dict:
    """
    Pre-compute tools for a single integration.
    
    Returns:
        Dict with success status and tool count.
    """
    print(f"\n{'='*60}")
    print(f"Processing integration: {integration_name}")
    print(f"{'='*60}")
    
    try:
        # 1. Fetch tools from Composio
        print(f"Fetching tools from Composio for {integration_name}...")
        tools = composio_client.tools.get(
            user_id=user_id,
            toolkits=[integration_name],
            limit=2000
        )
        
        if not tools:
            print(f"⚠️ No tools found for {integration_name}")
            return {"success": False, "tool_count": 0, "error": "No tools found"}
        
        print(f"✅ Fetched {len(tools)} tools from Composio")
        
        # Filter out deprecated tools before normalization
        tools = [t for t in tools if "deprecated" not in (getattr(t, 'description', '') or '').lower()]
        print(f"📝 Filtered to {len(tools)} non-deprecated tools")
        
        # Test mode: limit number of tools if TEST_MODE_LIMIT is set
        if TEST_MODE_LIMIT and len(tools) > TEST_MODE_LIMIT:
            original_count = len(tools)
            tools = tools[:TEST_MODE_LIMIT]
            print(f"🧪 TEST MODE: Limiting to {TEST_MODE_LIMIT} tools (from {original_count})")
        
        # 2. Convert Composio tools to ToolHub format
        normalized_tools = []
        for tool in tools:
            try:
                # Extract tool definition
                tool_dict = {
                    "name": tool.name,
                    "description": getattr(tool, 'description', '') or '',
                    "parameters": {}
                }
                
                # Extract parameters from args_schema if available
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    try:
                        schema = tool.args_schema
                        if hasattr(schema, 'model_json_schema'):
                            schema_dict = schema.model_json_schema()
                        elif hasattr(schema, 'schema'):
                            schema_dict = schema.schema()
                        else:
                            schema_dict = {}
                        
                        tool_dict["parameters"] = schema_dict.get("properties", {})
                        
                        # Debug: Log if parameters are empty
                        if not tool_dict["parameters"]:
                            print(f"⚠️ Warning: Empty parameters for {tool.name} (schema keys: {list(schema_dict.keys())})")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not extract schema for {tool.name}: {e}")
                        tool_dict["parameters"] = {}  # Explicit empty dict
                
                normalized_tools.append(tool_dict)
            except Exception as e:
                print(f"Warning: Failed to normalize tool: {e}")
                continue
        
        if not normalized_tools:
            print(f"⚠️ No valid tools after normalization for {integration_name}")
            return {"success": False, "tool_count": 0, "error": "Normalization failed"}
        
        # 3. Enrich and store tools using ToolHub
        print(f"Enriching and storing {len(normalized_tools)} tools in Pinecone...")
        await toolhub.ingest_to_pinecone(
            tools=normalized_tools,
            integration_name=integration_name.lower(),
            max_workers=10
        )
        
        print(f"✅ Successfully indexed {len(normalized_tools)} tools for {integration_name}")
        return {"success": True, "tool_count": len(normalized_tools)}
        
    except Exception as e:
        print(f"❌ Error processing {integration_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "tool_count": 0, "error": str(e)}


async def main():
    """Main pre-computation function."""
    print("="*60)
    print("ToolHub + Pinecone Pre-Computation")
    print("="*60)
    
    # Check required environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    composio_user_id = os.getenv("COMPOSIO_USER_ID")
    
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    if not pinecone_key:
        print("❌ Error: PINECONE_API_KEY not set in environment")
        return
    
    composio_api_key = os.getenv("COMPOSIO_API_KEY")
    
    # Initialize clients
    print("\nInitializing clients...")
    toolhub = ToolHub(
        openai_api_key=openai_key,
        pinecone_index_name=pinecone_index_name,
        pinecone_api_key=pinecone_key,
        embedding_dimensions=512  # Match Pinecone index dimension
    )
    
    if composio_api_key:
        composio_client = Composio(api_key=composio_api_key, provider=LangchainProvider())
    else:
        composio_client = Composio(provider=LangchainProvider())
    
    # Initialize Pinecone and ensure index exists
    pc = Pinecone(api_key=pinecone_key)
    ensure_pinecone_index(pc, pinecone_index_name, dimension=512)
    
    print(f"✅ Clients initialized")
    print(f"   OpenAI API Key: {'✅ Set' if openai_key else '❌ Missing'}")
    print(f"   Pinecone API Key: {'✅ Set' if pinecone_key else '❌ Missing'}")
    print(f"   Pinecone Index: {pinecone_index_name}")
    print(f"   Composio User ID: {composio_user_id}")
    print(f"   Composio API Key: {'✅ Set' if composio_api_key else '⚠️ Not set (may be optional)'}")
    print(f"   Integrations to process: {len(INTEGRATIONS)}")
    if TEST_MODE_LIMIT:
        print(f"   🧪 TEST MODE: Limited to {TEST_MODE_LIMIT} tools per integration")
    else:
        print(f"   🚀 FULL MODE: Processing all tools")
    
    # Process each integration
    results = []
    for integration in INTEGRATIONS:
        result = await precompute_integration(
            integration_name=integration,
            composio_client=composio_client,
            toolhub=toolhub,
            user_id=composio_user_id
        )
        result["integration"] = integration
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("PRE-COMPUTATION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    total_tools = sum(r.get("tool_count", 0) for r in successful)
    
    print(f"\n✅ Successful: {len(successful)}/{len(INTEGRATIONS)} integrations")
    print(f"❌ Failed: {len(failed)}/{len(INTEGRATIONS)} integrations")
    print(f"📊 Total tools indexed: {total_tools}")
    
    if successful:
        print("\nSuccessful integrations:")
        for r in successful:
            print(f"  ✅ {r['integration']}: {r['tool_count']} tools")
    
    if failed:
        print("\nFailed integrations:")
        for r in failed:
            error = r.get("error", "Unknown error")
            print(f"  ❌ {r['integration']}: {error}")
    
    print("\n" + "="*60)
    print("Pre-computation complete!")
    print("="*60)
    print(f"\n✅ All tools are now indexed in Pinecone index: {pinecone_index_name}")
    print("   This index persists across all agent deployments.")
    print("   You can query it directly in Pinecone console for debugging.")


if __name__ == "__main__":
    asyncio.run(main())

