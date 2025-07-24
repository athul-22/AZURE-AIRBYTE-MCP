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
import google.generativeai as genai

# --- ENV VARIABLES
load_dotenv()

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

# Updated MCP configuration
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "npx -y @azure/mcp@latest server start")
MCP_API_KEY = os.getenv("MCP_API_KEY")

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Azure Agentic Data Integrator", page_icon="🚀")
st.title("🚀 Azure Agentic Data Integrator with MCP + Gemini")
st.caption("🤖 Powered by Google Gemini + Azure MCP Tools")

# Initialize Gemini client
@st.cache_resource
def get_gemini_client():
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY not found in environment variables")
        return None
    return genai.GenerativeModel(GEMINI_MODEL)

def create_tool_aware_prompt(user_input, mcp_tools):
    """Create a prompt that makes Gemini aware of available tools"""
    if not mcp_tools:
        return user_input
    
    tool_list = []
    for tool in mcp_tools:
        tool_info = f"**{tool.name}**: {tool.description}"
        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            properties = tool.inputSchema.get('properties', {})
            if properties:
                params = ", ".join(properties.keys())
                tool_info += f" (Parameters: {params})"
        tool_list.append(tool_info)
    
    tools_text = "\n".join(tool_list)
    
    return f"""
{user_input}

Available Azure MCP tools:
{tools_text}

Please provide a helpful response. If you think any of these Azure tools would be useful for the user's request, mention which tool(s) should be used and why. Be specific about which tool would help accomplish the task.
"""

def extract_tool_suggestions(response_text, mcp_tools):
    """Extract tool suggestions from Gemini's response"""
    if not mcp_tools:
        return []
    
    suggested_tools = []
    response_lower = response_text.lower()
    
    for tool in mcp_tools:
        tool_name_lower = tool.name.lower()
        # Check if tool name is mentioned in response
        if tool_name_lower in response_lower:
            suggested_tools.append(tool)
    
    return suggested_tools

# Credit usage indicator
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🤖 AI Model", "Gemini", "Google AI")
with col2:
    st.metric("⚡ Performance", "High", "No Rate Limits")
with col3:
    st.metric("🔧 MCP Tools", "Azure", "Available")

# API Key status
if not GOOGLE_API_KEY:
    st.error("❌ **Missing Google API Key!** Please add GOOGLE_API_KEY to your .env file")
    st.info("💡 Get your API key from: https://makersuite.google.com/app/apikey")
else:
    st.success("✅ Google API Key configured")

with st.expander("🔑 Configuration", expanded=False):
    st.code(f"Azure Storage Account: {AZURE_STORAGE_ACCOUNT_NAME}")
    st.code(f"Azure Storage Container: {AZURE_STORAGE_CONTAINER_NAME}")
    st.code(f"MCP Command: {MCP_SERVER_COMMAND}")
    st.code(f"Gemini Model: {GEMINI_MODEL}")
    st.code(f"Google API Key: {'✅ Set' if GOOGLE_API_KEY else '❌ Missing'}")

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
st.subheader("🤖 2. Chat with Gemini + Azure MCP Tools")

# Initialize session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Suggested prompts for Azure tasks
st.info("💡 **Try these Azure commands with Gemini:**")
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
user_input = st.chat_input("Ask me anything about Azure...")

# Handle suggested prompts
if 'suggested_prompt' in st.session_state:
    user_input = st.session_state.suggested_prompt
    del st.session_state.suggested_prompt

if user_input and GOOGLE_API_KEY:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get MCP tools (cache them for performance)
    if 'cached_mcp_tools' not in st.session_state:
        with st.spinner("🔧 Loading Azure tools..."):
            mcp_tools = run_async_in_streamlit(get_mcp_tools())
            st.session_state.cached_mcp_tools = mcp_tools
    else:
        mcp_tools = st.session_state.cached_mcp_tools
    
    if mcp_tools:
        st.success(f"✅ Loaded {len(mcp_tools)} Azure MCP tools")
    else:
        st.warning("⚠️ No MCP tools loaded")
    
    # Call Gemini
    client = get_gemini_client()
    
    if client:
        with st.spinner("🤖 Processing with Gemini..."):
            try:
                # Create tool-aware prompt
                enhanced_prompt = create_tool_aware_prompt(user_input, mcp_tools)
                
                # Call Gemini
                response = client.generate_content(enhanced_prompt)
                
                # Display response
                assistant_message = response.text
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                
                # Check for tool suggestions
                suggested_tools = extract_tool_suggestions(assistant_message, mcp_tools)
                
                if suggested_tools:
                    st.info(f"🔧 Gemini suggests using {len(suggested_tools)} Azure tool(s)")
                    
                    # Show suggested tools with execution buttons
                    for tool in suggested_tools:
                        with st.expander(f"🛠️ Execute: {tool.name}", expanded=False):
                            st.write(f"**Description**: {tool.description}")
                            
                            # Show parameters if available
                            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                                properties = tool.inputSchema.get('properties', {})
                                if properties:
                                    st.write("**Parameters needed:**")
                                    for param, schema in properties.items():
                                        param_type = schema.get('type', 'string')
                                        param_desc = schema.get('description', 'No description')
                                        st.code(f"{param} ({param_type}): {param_desc}")
                            
                            # Execute button
                            if st.button(f"▶️ Execute {tool.name}", key=f"exec_{tool.name}"):
                                with st.spinner(f"Executing {tool.name}..."):
                                    # For now, execute with empty parameters
                                    # In a full implementation, you'd want to collect parameters from user
                                    result = run_async_in_streamlit(
                                        call_mcp_tool_async(tool.name, {})
                                    )
                                    
                                    if "error" in str(result):
                                        st.error(f"❌ Tool execution failed: {result}")
                                    else:
                                        st.success("✅ Tool executed successfully!")
                                        st.json(result.content if hasattr(result, 'content') else result)
                                        
                                        # Add tool result to conversation
                                        tool_result_message = f"Tool {tool.name} executed successfully. Result: {result.content if hasattr(result, 'content') else result}"
                                        st.session_state.messages.append({
                                            "role": "assistant", 
                                            "content": tool_result_message
                                        })
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                if "API_KEY" in str(e):
                    st.info("💡 Check your Google API key configuration")

elif user_input and not GOOGLE_API_KEY:
    st.error("❌ Cannot process request: Google API Key is missing")

# Display chat messages
st.subheader("💬 Conversation")
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])

# Control buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("🔄 Refresh Tools"):
        if 'cached_mcp_tools' in st.session_state:
            del st.session_state.cached_mcp_tools
        st.rerun()

with col3:
    if st.button("🧪 Test Gemini"):
        if GOOGLE_API_KEY:
            try:
                client = get_gemini_client()
                test_response = client.generate_content("Say hello in one word")
                st.success(f"✅ Gemini test: {test_response.text}")
            except Exception as e:
                st.error(f"❌ Gemini test failed: {e}")
        else:
            st.error("❌ No API key to test")

st.markdown("---")
st.caption("🤖 Powered by Google Gemini + Azure MCP Tools + Streamlit")

# Debug info
if st.checkbox("🔍 Show Debug Info"):
    st.subheader("Debug Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Environment Variables:**")
        st.code(f"GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not Set'}")
        st.code(f"GEMINI_MODEL: {GEMINI_MODEL}")
        st.code(f"MCP_SERVER_COMMAND: {MCP_SERVER_COMMAND}")
    
    with col2:
        st.write("**Session State:**")
        st.code(f"Messages: {len(st.session_state.messages)}")
        st.code(f"Cached Tools: {'Yes' if 'cached_mcp_tools' in st.session_state else 'No'}")
        if 'cached_mcp_tools' in st.session_state:
            st.code(f"Tool Count: {len(st.session_state.cached_mcp_tools)}")
