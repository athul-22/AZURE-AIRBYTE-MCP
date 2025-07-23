import os
import json
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import requests
import airbyte as ab
import subprocess
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- ENV VARIABLES
load_dotenv()

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "dotnet azmcp.dll server start")  # Command to start MCP server
MCP_API_KEY = os.getenv("MCP_API_KEY")

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Azure Agentic Data Integrator")
st.title("🔗 Azure Agentic Data Integrator with Airbyte & MCP")

with st.expander("🔑 Airbyte + Azure Blob Storage Configuration"):
    st.code(f"Account: {AZURE_STORAGE_ACCOUNT_NAME}")
    st.code(f"Container: {AZURE_STORAGE_CONTAINER_NAME}")

with st.expander("🤖 MCP Server Settings"):
    st.code(f"Command: {MCP_SERVER_COMMAND}")
    st.code(f"API Key: {'*' * len(MCP_API_KEY) if MCP_API_KEY else 'Not Set'}")

# --- AIRBYTE OPERATIONS ---
st.subheader("🛠 1. Ingest Data From Azure Blob using Airbyte")

if st.button("📥 Ingest Now using PyAirbyte"):
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
        st.success(f"Airbyte Source Status: {check}")
        
        streams = source.get_available_streams()
        source.select_all_streams()
        st.success(f"Streams Found: {streams}")

        cache = ab.get_default_cache()
        result = source.read(cache=cache)
        st.success("✅ Data cached via PyAirbyte!")

        # Display sample from first stream
        if streams:
            df = cache[streams[0]].to_pandas()
            st.dataframe(df.head())

    except Exception as e:
        st.error(f"Airbyte Error: {str(e)}")

# --- MCP SECTION ---
st.subheader("🤖 2. Trigger Azure MCP Agent")

# MCP Tool selection
available_tools = ["hello", "list_resources", "read_resource", "custom_tool"]
selected_tool = st.selectbox("🔧 Select MCP Tool", available_tools)

# Tool arguments input
tool_arguments = st.text_area(
    "📝 Tool Arguments (JSON)", 
    value='{}', 
    height=100,
    help="Enter JSON arguments for the selected tool"
)

async def call_mcp_tool(command, tool_name, arguments):
    """Call MCP tool using stdio transport"""
    try:
        # Parse command
        cmd_parts = command.split()
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=cmd_parts[0],
            args=cmd_parts[1:] if len(cmd_parts) > 1 else [],
            env=None
        )
        
        # Connect to MCP server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # Parse arguments
                try:
                    args = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON arguments"}
                
                # Call the tool
                result = await session.call_tool(tool_name, args)
                return result
                
    except Exception as e:
        return {"error": f"MCP Error: {str(e)}"}

def run_mcp_sync(command, tool_name, arguments):
    """Synchronous wrapper for MCP call"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_mcp_tool(command, tool_name, arguments))
        loop.close()
        return result
    except Exception as e:
        return {"error": f"Async Error: {str(e)}"}

if st.button("🚀 Call MCP Tool"):
    if not MCP_SERVER_COMMAND:
        st.error("❌ MCP_SERVER_COMMAND not configured in .env file")
    else:
        with st.spinner(f"Calling MCP tool '{selected_tool}'..."):
            try:
                result = run_mcp_sync(MCP_SERVER_COMMAND, selected_tool, tool_arguments)
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ MCP Tool Response Received")
                    st.json(result)
                    
            except Exception as e:
                st.error(f"❌ MCP Call Failed: {str(e)}")

# --- ALTERNATIVE: Manual MCP Server Management ---
st.subheader("🔧 3. Manual MCP Server Management")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start MCP Server"):
        if 'mcp_process' not in st.session_state:
            try:
                cmd_parts = MCP_SERVER_COMMAND.split()
                process = subprocess.Popen(
                    cmd_parts,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                st.session_state.mcp_process = process
                st.success("✅ MCP Server Started")
            except Exception as e:
                st.error(f"❌ Failed to start MCP server: {str(e)}")
        else:
            st.warning("⚠️ MCP Server already running")

with col2:
    if st.button("⏹️ Stop MCP Server"):
        if 'mcp_process' in st.session_state:
            try:
                st.session_state.mcp_process.terminate()
                del st.session_state.mcp_process
                st.success("✅ MCP Server Stopped")
            except Exception as e:
                st.error(f"❌ Failed to stop MCP server: {str(e)}")
        else:
            st.warning("⚠️ No MCP Server running")

# Server status
if 'mcp_process' in st.session_state:
    if st.session_state.mcp_process.poll() is None:
        st.success("🟢 MCP Server Status: Running")
    else:
        st.error("🔴 MCP Server Status: Stopped")
        del st.session_state.mcp_process

st.markdown("---")
st.caption("Built with 🤖 PyAirbyte, Azure Blob & MCP | Streamlit UI Demo")
