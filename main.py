import os
import json
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import asyncio
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import google.generativeai as genai
import subprocess
import re

# --- ENV VARIABLES
load_dotenv()

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Azure configuration
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")

# MCP configuration
MCP_SERVER_COMMAND = "npx -y @azure/mcp@latest server start"

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Azure Assistant", page_icon="🚀")
st.title("🚀 Azure Assistant with Working MCP Integration")

# Check prerequisites
st.subheader("🔧 Prerequisites Check")
col1, col2, col3 = st.columns(3)

with col1:
    if GOOGLE_API_KEY:
        st.success("✅ Gemini API Key")
    else:
        st.error("❌ Missing Gemini API Key")

with col2:
    # Check if Azure CLI is logged in
    try:
        result = subprocess.run(['az', 'account', 'show'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Azure CLI Logged In")
        else:
            st.error("❌ Azure CLI Not Logged In")
    except:
        st.error("❌ Azure CLI Not Found")

with col3:
    # Check if NPX is available
    try:
        subprocess.run(['npx', '--version'], capture_output=True)
        st.success("✅ NPX Available")
    except:
        st.error("❌ NPX Not Found")

# Authentication instructions
with st.expander("🔑 Setup Instructions", expanded=False):
    st.markdown("""
    ### 1. Azure CLI Login
    ```bash
    az login
    az account set --subscription your-subscription-id
    ```
    
    ### 2. Google API Key
    Add to .env file:
    ```
    GOOGLE_API_KEY=your_gemini_api_key
    ```
    
    ### 3. Install Dependencies
    ```bash
    npm install -g @azure/mcp
    pip install google-generativeai streamlit python-dotenv
    ```
    """)

async def get_azure_tools():
    """Get and cache Azure MCP tools"""
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
                tools_result = await session.list_tools()
                return tools_result.tools
    except Exception as e:
        st.error(f"❌ Failed to load Azure tools: {e}")
        return []

async def execute_azure_tool(tool_name, arguments=None):
    """Execute an Azure MCP tool and return results"""
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
                result = await session.call_tool(tool_name, arguments or {})
                return result
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}

def run_async_in_streamlit(coro):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        loop.close()
        return result
    except Exception as e:
        st.error(f"Async error: {e}")
        return None

def parse_tool_suggestions(gemini_response, available_tools):
    """Parse Gemini's response to extract tool suggestions"""
    suggestions = []
    response_lower = gemini_response.lower()
    
    # Look for specific patterns
    tool_patterns = [
        r"use\s+(\w+)",
        r"run\s+(\w+)",
        r"execute\s+(\w+)",
        r"call\s+(\w+)"
    ]
    
    for tool in available_tools:
        tool_name = tool.name
        if tool_name.lower() in response_lower:
            # Try to extract arguments if mentioned
            suggestions.append({
                "tool": tool,
                "arguments": {}  # Could be enhanced to parse arguments
            })
    
    return suggestions

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'azure_tools' not in st.session_state:
    st.session_state.azure_tools = None

# Load Azure tools
if st.session_state.azure_tools is None:
    with st.spinner("🔧 Loading Azure MCP tools..."):
        tools = run_async_in_streamlit(get_azure_tools())
        st.session_state.azure_tools = tools or []

if st.session_state.azure_tools:
    st.success(f"✅ Loaded {len(st.session_state.azure_tools)} Azure tools")
    
    # Show available tools
    with st.expander("🛠️ Available Azure Tools", expanded=False):
        for tool in st.session_state.azure_tools:
            st.code(f"{tool.name}: {tool.description}")
else:
    st.error("❌ No Azure tools loaded")

# Chat interface
st.subheader("💬 Chat with Azure Assistant")

# Quick action buttons
st.info("💡 **Quick Actions:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📋 List Resources", key="list_resources"):
        st.session_state.pending_action = "list_resources"

with col2:
    if st.button("🏪 List Storage Accounts", key="list_storage"):
        st.session_state.pending_action = "list_storage"

with col3:
    if st.button("👥 List Resource Groups", key="list_rgs"):
        st.session_state.pending_action = "list_rgs"

# Handle quick actions
if 'pending_action' in st.session_state:
    action = st.session_state.pending_action
    del st.session_state.pending_action
    
    if action == "list_resources" and st.session_state.azure_tools:
        with st.spinner("🔍 Getting your Azure resources..."):
            # Find the right tool
            az_tool = next((t for t in st.session_state.azure_tools if t.name == "azmcp_extension_az"), None)
            if az_tool:
                result = run_async_in_streamlit(
                    execute_azure_tool("azmcp_extension_az", {"command": "resource list --output table"})
                )
                if result and hasattr(result, 'content'):
                    st.success("✅ Your Azure Resources:")
                    st.text(result.content)
                else:
                    st.error(f"❌ Failed to get resources: {result}")
    
    elif action == "list_storage" and st.session_state.azure_tools:
        with st.spinner("🔍 Getting your storage accounts..."):
            storage_tool = next((t for t in st.session_state.azure_tools if "storage" in t.name.lower() and "list" in t.name.lower()), None)
            if storage_tool:
                result = run_async_in_streamlit(
                    execute_azure_tool(storage_tool.name, {})
                )
                if result and hasattr(result, 'content'):
                    st.success("✅ Your Storage Accounts:")
                    st.text(result.content)
                else:
                    st.error(f"❌ Failed to get storage accounts: {result}")
    
    elif action == "list_rgs" and st.session_state.azure_tools:
        with st.spinner("🔍 Getting your resource groups..."):
            rg_tool = next((t for t in st.session_state.azure_tools if "resource_group" in t.name.lower() and "list" in t.name.lower()), None)
            if rg_tool:
                result = run_async_in_streamlit(
                    execute_azure_tool(rg_tool.name, {})
                )
                if result and hasattr(result, 'content'):
                    st.success("✅ Your Resource Groups:")
                    st.text(result.content)
                else:
                    st.error(f"❌ Failed to get resource groups: {result}")

# Text input for custom queries
user_input = st.chat_input("Ask me about your Azure resources...")

if user_input and GOOGLE_API_KEY and st.session_state.azure_tools:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create enhanced prompt with tools
    tools_info = "\n".join([f"- {tool.name}: {tool.description}" for tool in st.session_state.azure_tools])
    enhanced_prompt = f"""
{user_input}

Available Azure tools:
{tools_info}

Please provide a helpful response and suggest which specific tool(s) to use. If you suggest a tool, I will execute it automatically.
"""
    
    with st.spinner("🤖 Thinking..."):
        try:
            # Get Gemini response
            client = genai.GenerativeModel(GEMINI_MODEL)
            response = client.generate_content(enhanced_prompt)
            
            assistant_message = response.text
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            
            # Parse for tool suggestions and auto-execute
            suggestions = parse_tool_suggestions(assistant_message, st.session_state.azure_tools)
            
            if suggestions:
                st.info(f"🔧 Executing {len(suggestions)} suggested tool(s)...")
                
                for suggestion in suggestions:
                    tool = suggestion["tool"]
                    args = suggestion["arguments"]
                    
                    with st.spinner(f"⚡ Running {tool.name}..."):
                        result = run_async_in_streamlit(
                            execute_azure_tool(tool.name, args)
                        )
                        
                        if result and hasattr(result, 'content'):
                            st.success(f"✅ {tool.name} results:")
                            st.text(result.content)
                            
                            # Add result to conversation
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Tool {tool.name} executed. Results:\n{result.content}"
                            })
                        else:
                            st.error(f"❌ {tool.name} failed: {result}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Display conversation
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("🔄 Reload Tools"):
        st.session_state.azure_tools = None
        st.rerun()

st.markdown("---")
st.caption("🚀 Azure Assistant with Working MCP Integration")
