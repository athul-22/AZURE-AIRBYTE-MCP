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

def call_azure_openai_with_retry(client, **kwargs):
    """Call Azure OpenAI with retry logic for rate limiting"""
    max_retries = 3
    base_delay = 60  # Start with 60 seconds as suggested
    
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"🚫 Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
            time.sleep(delay)
        except Exception as e:
            raise e

async def run():
    # Initialize Azure OpenAI client with appropriate auth
    if USE_API_KEY:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-08-01-preview"  # Updated API version
        )
    else:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT, 
            api_version="2024-08-01-preview",  # Updated API version
            azure_ad_token_provider=token_provider
        )

    # Test Azure OpenAI connection first with retry
    try:
        test_response = call_azure_openai_with_retry(
            client,
            model=AZURE_OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ Azure OpenAI connection successful!")
        print(f"Test response: {test_response.choices[0].message.content}")
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
            print("Available tools:")
            for tool in tools.tools: 
                print(f"  - {tool.name}")

            # Format tools for Azure OpenAI
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in tools.tools]

            # Start conversational loop with rate limiting
            messages = []
            while True:
                try:
                    user_input = input("\nPrompt: ")
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                        
                    messages.append({"role": "user", "content": user_input})

                    # First API call with tool configuration and retry logic
                    print("🤖 Thinking...")
                    response = call_azure_openai_with_retry(
                        client,
                        model=AZURE_OPENAI_MODEL,
                        messages=messages,
                        tools=available_tools,
                        max_tokens=300  # Reduced to save tokens
                    )

                    # Process the model's response
                    response_message = response.choices[0].message
                    messages.append(response_message)

                    # Handle function calls
                    if response_message.tool_calls:
                        print("🔧 Model is calling tools...")
                        for tool_call in response_message.tool_calls:
                            function_args = json.loads(tool_call.function.arguments)
                            print(f"Calling {tool_call.function.name} with args: {function_args}")
                            result = await session.call_tool(tool_call.function.name, function_args)

                            # Add the tool response to the messages
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": result.content,
                            })

                        # Get the final response from the model after tool execution
                        print("🤖 Processing results...")
                        final_response = call_azure_openai_with_retry(
                            client,
                            model=AZURE_OPENAI_MODEL,
                            messages=messages,
                            tools=available_tools,
                            max_tokens=300  # Reduced to save tokens
                        )

                        for item in final_response.choices:
                            print(item.message.content)
                    else:
                        # No tool calls, just print the response
                        print(response_message.content)
                        
                    # Add a small delay between requests to avoid rate limiting
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in conversation loop: {e}")
                    print(f"An error occurred: {e}")
                    if "rate limit" in str(e).lower():
                        print("💡 Tip: Try upgrading your Azure OpenAI pricing tier or wait a minute before continuing.")
                        time.sleep(10) 

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())