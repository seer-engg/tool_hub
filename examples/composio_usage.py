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

    print("Fetching GitHub tools from Composio...")
    
    composio_client = Composio(api_key=composio_key)
    
    try:
        # Fetch tools for GitHub
        # The SDK returns a list of tool definitions directly usable by OpenAI
        tools = composio_client.tools.get(user_id=user_id, toolkits=["GITHUB"], limit=1000)
    except Exception as e:
        print(f"Error fetching tools: {e}")
        return

    # Initialize Hub
    hub = ToolHub(openai_api_key=openai_key)
    
    # Ingest
    # We pass the raw tools list directly; ToolHub handles normalization
    hub.ingest(tools)
    
    # Query
    query = "Create a new issue about a bug in the login page"
    print(f"\nQuery: '{query}'")
    
    selected_tools = hub.query(query)
    
    print(f"\nSelected {len(selected_tools)} tools:")
    for tool in selected_tools:
        print(f"- {tool['function']['name']}")

if __name__ == "__main__":
    main()
