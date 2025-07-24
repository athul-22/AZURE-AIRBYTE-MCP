from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import json, os, logging, asyncio, time
from dotenv import load_dotenv
import openai

# Setup logging and load environment variables
logger = logging.getLogger(__name__)
load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Choose authentication method
USE_API_KEY = os.getenv("USE_API_KEY", "false").lower() == "true"

if not USE_API_KEY:
    # Initialize Azure credentials for AD authentication
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

def call_azure_openai_with_minimal_retry(client, **kwargs):
    """Call Azure OpenAI with minimal retry logic - optimized for higher tier"""
    max_retries = 2  # Reduced since you have higher limits
    base_delay = 10   # Much shorter delay
    
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"⏳ Brief rate limit. Waiting {base_delay} seconds...")
            time.sleep(base_delay)
        except Exception as e:
            raise e

async def run():
    # Initialize Azure OpenAI client with appropriate auth
    if USE_API_KEY:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-08-01-preview"
        )
    else:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT, 
            api_version="2024-08-01-preview",
            azure_ad_token_provider=token_provider
        )

    # Test Azure OpenAI connection first
    try:
        test_response = call_azure_openai_with_minimal_retry(
            client,
            model=AZURE_OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50
        )
        print(f"✅ Azure OpenAI connection successful!")
        print(f"Test response: {test_response.choices[0].message.content}")
        print(f"💰 Using credits - High performance mode enabled!")
    except Exception as e:
        print(f"❌ Azure OpenAI test failed: {e}")
        return

    # MCP client configurations
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@azure/mcp@latest", "server", "start"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available Azure MCP tools:")
            for tool in tools.tools: 
                print(f"  🔧 {tool.name} - {tool.description}")

            # Format tools for Azure OpenAI
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in tools.tools]

            print(f"\n🚀 Ready! You can now chat and use {len(available_tools)} Azure tools.")
            print("💡 Try: 'List my Azure resources' or 'Create a storage account'")

            # Start conversational loop - optimized for higher tier
            messages = []
            while True:
                try:
                    user_input = input("\n💬 You: ")
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                        
                    messages.append({"role": "user", "content": user_input})

                    # First API call - increased tokens for better responses
                    print("🤖 Assistant: ", end="", flush=True)
                    response = call_azure_openai_with_minimal_retry(
                        client,
                        model=AZURE_OPENAI_MODEL,
                        messages=messages,
                        tools=available_tools,
                        max_tokens=1000,  # Increased for better responses
                        temperature=0.1   # Lower temp for more consistent tool usage
                    )

                    # Process the model's response
                    response_message = response.choices[0].message
                    messages.append(response_message)

                    # Handle function calls
                    if response_message.tool_calls:
                        print("🔧 Executing Azure tasks...\n")
                        
                        for tool_call in response_message.tool_calls:
                            function_args = json.loads(tool_call.function.arguments)
                            print(f"  ⚡ {tool_call.function.name}({function_args})")
                            
                            result = await session.call_tool(tool_call.function.name, function_args)

                            # Add the tool response to the messages
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": result.content,
                            })

                        # Get the final response from the model after tool execution
                        final_response = call_azure_openai_with_minimal_retry(
                            client,
                            model=AZURE_OPENAI_MODEL,
                            messages=messages,
                            tools=available_tools,
                            max_tokens=1000,
                            temperature=0.1
                        )

                        assistant_response = final_response.choices[0].message.content
                        print(f"✅ {assistant_response}")
                        messages.append(final_response.choices[0].message)
                    else:
                        # No tool calls, just print the response
                        print(response_message.content)
                        
                    # No artificial delays needed with higher tier
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in conversation loop: {e}")
                    print(f"❌ Error: {e}")
                    if "rate limit" in str(e).lower():
                        print("⚠️  Unexpected rate limit. Checking pricing tier...")
                        time.sleep(5)

if __name__ == "__main__":
    print("🚀 Starting Azure AI Assistant with MCP...")
    print("💰 Using Azure credits - High performance mode")
    import asyncio
    asyncio.run(run())