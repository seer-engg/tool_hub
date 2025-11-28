import os
from dotenv import load_dotenv
from tool_hub import ToolHub
from composio import Composio

# load .env file in the current directory
load_dotenv()

def main():
    openai_key = os.getenv("OPENAI_API_KEY")
    composio_key = os.getenv("COMPOSIO_API_KEY")
    user_id = os.getenv("COMPOSIO_USER_ID")
    
    if not openai_key:
        print("Error: OPENAI_API_KEY not set.")
        return

    if not composio_key:
        print("Skipping Composio example: COMPOSIO_API_KEY not set.")
        return

    # Initialize Hub
    hub = ToolHub(openai_api_key=openai_key)
    index_dir = "github_tool_index"

    if os.path.exists(index_dir):
        print("Loading cached index...")
        hub.load(index_dir)
    else:
        print("Fetching GitHub tools from Composio...")
        
        composio_client = Composio(api_key=composio_key)
        
        try:
            # Fetch tools for GitHub
            # The SDK returns a list of tool definitions directly usable by OpenAI
            tools = composio_client.tools.get(user_id=user_id, toolkits=["GITHUB", "ASANA"], limit=1500)
        except Exception as e:
            print(f"Error fetching tools: {e}")
            return
        
        # Ingest
        # We pass the raw tools list directly; ToolHub handles normalization
        hub.ingest(tools)
        hub.save(index_dir)
    
    # Query
    query = "assign a task for user onboarding to John Doe in Asana and create a new issue in github"
    print(f"\nQuery: '{query}'")
    
    selected_tools = hub.query(query, top_k=5)
    
    print(f"\nSelected {len(selected_tools)} tools:")
    for tool in selected_tools:
        print(f"- {tool['name']}")

if __name__ == "__main__":
    main()
