import asyncio
import os

from dotenv import load_dotenv  
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

async def run_memory_chat():
    """Run a chat using MCPAgent's built in conversation memory."""
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    config_file = "browser_mcp.json"

    print('Initializing Chat...')

    # Initialize the MCP client
    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model_name="qwen-wq-32b")

    # Initialize the agent
    agent = MCPAgent(
        client=client,
        llm=llm,
        max_steps=15,
        memory_enabled=True  # enable built-in conversation memory
    )
    print("\n====Interactive MCP Chat====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear the conversation history")
    print("==============================\n")

    try:
        # main Chat loop
        while True:
            # Get user input
            user_input = input("You: ")

            # Exit condition
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat...")
                break

            if user_input.lower() == 'clear':
                agent.clear_conversation_history()
                print("Conversation history cleared")
                continue

            print("Assistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nChat ended by user")
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())
        
