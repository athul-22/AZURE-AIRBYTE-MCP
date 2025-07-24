import os
import json
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import requests
import airbyte as ab
import subprocess
import asyncio
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AzureOpenAI
import openai

# --- ENV VARIABLES
load_dotenv()

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

# Updated MCP configuration
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "npx -y @azure/mcp@latest server start")
MCP_API_KEY = os.getenv("MCP_API_KEY")

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Azure Agentic Data Integrator", page_icon="🚀")
st.title("🚀 Azure Agentic Data Integrator with MCP")
st.caption("💰 Powered by Azure Credits - High Performance Mode")

# Initialize Azure OpenAI client
@st.cache_resource
def get_azure_openai_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-08-01-preview"
    )

def call_azure_openai_optimized(client, **kwargs):
    """Optimized Azure OpenAI calls for higher tier"""
    try:
        return client.chat.completions.create(**kwargs)
    except openai.RateLimitError as e:
        st.warning("Brief rate limit encountered. Retrying...")
        time.sleep(5)
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        raise e

# Credit usage indicator
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 Credits Available", "$1,000", "Azure Sponsorship")
with col2:
    st.metric("⚡ Performance", "High", "S1/S2 Tier")
with col3:
    st.metric("🔧 MCP Tools", "Azure", "Available")

with st.expander("🔑 Configuration", expanded=False):
    st.code(f"Azure Storage Account: {AZURE_STORAGE_ACCOUNT_NAME}")
    st.code(f"Azure Storage Container: {AZURE_STORAGE_CONTAINER_NAME}")
    st.code(f"MCP Command: {MCP_SERVER_COMMAND}")
    st.code(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    st.code(f"Azure OpenAI Model: {AZURE_OPENAI_MODEL}")

# --- AIRBYTE OPERATIONS ---
st.subheader("🛠 1. Ingest Data From Azure Blob using Airbyte")

if st.button("📥 Ingest Now using PyAirbyte", type="primary"):
    with st.spinner("🔄 Ingesting data..."):
        try:
            source = ab.get_source(
                connector_name="source-azure-blob-storage",
                config={
                    "azure_blob_storage_account_name": AZURE_STORAGE_ACCOUNT_NAME,
                    "azure_blob_storage_container_name": AZURE_STORAGE_CONTAINER_NAME,
                    "azure_blob_storage_account_key": AZURE_STORAGE_ACCOUNT_KEY,
                    "format": {"format_type": "jsonl"}
                },
                install_if_missing=True
            )
            check = source.check()
            st.success(f"✅ Airbyte Source Status: {check}")
            
            streams = source.get_available_streams()
            source.select_all_streams()
            st.success(f"✅ Streams Found: {streams}")

            cache = ab.get_default_cache()
            result = source.read(cache=cache)
            st.success("✅ Data cached via PyAirbyte!")

            # Display sample from first stream
            if streams:
                df = cache[streams[0]].to_pandas()
                st.dataframe(df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Airbyte Error: {str(e)}")

# --- AZURE AI CHAT SECTION ---
st.subheader("🤖 2. Chat with Azure AI + MCP Tools")

# Initialize session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Suggested prompts for Azure tasks
st.info("💡 **Try these commands:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("📋 List my Azure resources"):
        st.session_state.suggested_prompt = "List all my Azure resources"
    if st.button("💾 Create storage account"):
        st.session_state.suggested_prompt = "Create a new Azure storage account"
with col2:
    if st.button("🔍 Check resource group"):
        st.session_state.suggested_prompt = "Show me details of my resource groups"
    if st.button("📊 Get cost analysis"):
        st.session_state.suggested_prompt = "Show me Azure cost analysis"

async def get_mcp_tools():
    """Get available MCP tools"""
    try:
        cmd_parts = MCP_SERVER_COMMAND.split()
        server_params = StdioServerParameters(
            command=cmd_parts[0],
            args=cmd_parts[1:] if len(cmd_parts) > 1 else [],
            env=None
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return tools.tools
    except Exception as e:
        st.error(f"Failed to get MCP tools: {e}")
        return []

async def call_mcp_tool_async(tool_name, arguments):
    """Call MCP tool asynchronously"""
    try:
        cmd_parts = MCP_SERVER_COMMAND.split()
        server_params = StdioServerParameters(
            command=cmd_parts[0],
            args=cmd_parts[1:] if len(cmd_parts) > 1 else [],
            env=None
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result
    except Exception as e:
        return {"error": f"MCP Error: {str(e)}"}

def run_async_in_streamlit(coro):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        loop.close()
        return result
    except Exception as e:
        return {"error": f"Async Error: {str(e)}"}

# Chat interface
user_input = st.chat_input("Ask me anything about Azure or request a task...")

# Handle suggested prompts
if 'suggested_prompt' in st.session_state:
    user_input = st.session_state.suggested_prompt
    del st.session_state.suggested_prompt

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get MCP tools
    with st.spinner("🔧 Loading Azure tools..."):
        tools = run_async_in_streamlit(get_mcp_tools())
    
    if tools:
        st.success(f"✅ Loaded {len(tools)} Azure MCP tools")
    
    # Format tools for Azure OpenAI
    available_tools = [{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        }
    } for tool in tools] if tools else []
    
    # Call Azure OpenAI
    client = get_azure_openai_client()
    
    with st.spinner("🤖 Processing with Azure AI..."):
        try:
            response = call_azure_openai_optimized(
                client,
                model=AZURE_OPENAI_MODEL,
                messages=st.session_state.messages,
                tools=available_tools if available_tools else None,
                max_tokens=1500,  # Increased for detailed responses
                temperature=0.1
            )
            
            response_message = response.choices[0].message
            st.session_state.messages.append(response_message)
            
            # Handle tool calls
            if response_message.tool_calls:
                st.info("🔧 Executing Azure tasks...")
                
                progress_bar = st.progress(0)
                for i, tool_call in enumerate(response_message.tool_calls):
                    function_args = json.loads(tool_call.function.arguments)
                    
                    st.write(f"⚡ Calling: `{tool_call.function.name}`")
                    st.code(json.dumps(function_args, indent=2))
                    
                    # Call MCP tool
                    result = run_async_in_streamlit(
                        call_mcp_tool_async(tool_call.function.name, function_args)
                    )
                    
                    # Add tool response
                    st.session_state.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": json.dumps(result),
                    })
                    
                    progress_bar.progress((i + 1) / len(response_message.tool_calls))
                
                progress_bar.empty()
                
                # Get final response
                with st.spinner("🤖 Finalizing response..."):
                    final_response = call_azure_openai_optimized(
                        client,
                        model=AZURE_OPENAI_MODEL,
                        messages=st.session_state.messages,
                        tools=available_tools if available_tools else None,
                        max_tokens=1500,
                        temperature=0.1
                    )
                    
                    final_message = final_response.choices[0].message
                    st.session_state.messages.append(final_message)
            
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    elif message["role"] == "assistant" and hasattr(message, 'content') and message.content:
        with st.chat_message("assistant"):
            st.write(message.content)

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.markdown("---")
st.caption("🚀 High Performance Mode - Built with Azure Credits, OpenAI GPT-4o, Azure MCP & Streamlit")
