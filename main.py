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
from azure.identity import DeviceCodeCredential, ClientSecretCredential
from azure.core.exceptions import ClientAuthenticationError
import requests

# --- ENV VARIABLES
load_dotenv()

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Azure App Registration (you'll need to create this)
AZURE_CLIENT_ID = "your-app-client-id"  
AZURE_TENANT_ID = "your-tenant-id"    

# MCP configuration
MCP_SERVER_COMMAND = "npx -y @azure/mcp@latest server start"

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Azure Assistant", page_icon="🚀")
st.title("🚀 Azure Assistant with Web Authentication")

# Initialize session state for authentication
if 'azure_credentials' not in st.session_state:
    st.session_state.azure_credentials = None
if 'azure_token' not in st.session_state:
    st.session_state.azure_token = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

def device_code_auth():
    """Perform Azure device code authentication"""
    try:
        # Create device code credential
        credential = DeviceCodeCredential(
            client_id="04b07795-8ddb-461a-bbee-02f9e1bf7b46",  # Azure CLI client ID (public)
            tenant_id="common"  # Multi-tenant
        )
        
        # Get token
        token = credential.get_token("https://management.azure.com/.default")
        
        return credential, token
    
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None, None

def get_user_info(token):
    """Get user information using the token"""
    try:
        headers = {
            'Authorization': f'Bearer {token.token}',
            'Content-Type': 'application/json'
        }
        
        # Get user info
        response = requests.get(
            'https://graph.microsoft.com/v1.0/me',
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    
    except Exception as e:
        st.error(f"Failed to get user info: {e}")
        return None

def get_subscriptions(token):
    """Get user's Azure subscriptions"""
    try:
        headers = {
            'Authorization': f'Bearer {token.token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            'https://management.azure.com/subscriptions?api-version=2020-01-01',
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json().get('value', [])
        else:
            return []
    
    except Exception as e:
        st.error(f"Failed to get subscriptions: {e}")
        return []

# Authentication Section
st.subheader("🔐 Azure Authentication")

if st.session_state.azure_credentials is None:
    st.info("👋 **Welcome!** Please authenticate with Azure to get started.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔐 Login with Azure", type="primary", key="azure_login"):
            with st.spinner("🔄 Starting Azure authentication..."):
                try:
                    # Show device code instructions
                    st.info("""
                    **Device Code Authentication:**
                    1. Click the button below
                    2. A code will appear
                    3. Visit the Azure login page
                    4. Enter the code
                    5. Complete authentication in your browser
                    """)
                    
                    # Create a placeholder for the device code
                    device_code_placeholder = st.empty()
                    
                    # Custom device code flow with Streamlit integration
                    from azure.identity import DeviceCodeCredential
                    
                    def device_code_callback(verification_uri, user_code, expires_in):
                        device_code_placeholder.success(f"""
                        🔑 **Your Device Code:** `{user_code}`
                        
                        📱 **Next Steps:**
                        1. Click here: [{verification_uri}]({verification_uri})
                        2. Enter code: `{user_code}`
                        3. Complete login in browser
                        4. Return here after authentication
                        
                        ⏰ Code expires in {expires_in} seconds
                        """)
                    
                    credential = DeviceCodeCredential(
                        client_id="04b07795-8ddb-461a-bbee-02f9e1bf7b46",  # Azure CLI public client ID
                        prompt_callback=device_code_callback
                    )
                    
                    # Get token (this will trigger the callback)
                    token = credential.get_token("https://management.azure.com/.default")
                    
                    if token:
                        st.session_state.azure_credentials = credential
                        st.session_state.azure_token = token
                        
                        # Get user info
                        user_info = get_user_info(token)
                        if user_info:
                            st.session_state.user_info = user_info
                        
                        device_code_placeholder.empty()
                        st.success("✅ Authentication successful!")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Authentication failed: {e}")
    
    with col2:
        st.info("""
        **What happens during authentication:**
        
        🔐 **Secure**: Uses Azure's official device flow
        🌐 **Browser-based**: Login happens in your browser
        🔑 **Temporary**: Tokens are session-only
        🚫 **No storage**: No credentials saved locally
        """)

else:
    # User is authenticated
    st.success("✅ **Authenticated with Azure!**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.user_info:
            st.metric(
                "👤 User", 
                st.session_state.user_info.get('displayName', 'Unknown'),
                st.session_state.user_info.get('mail', '')
            )
    
    with col2:
        # Get and display subscriptions
        if 'subscriptions' not in st.session_state:
            with st.spinner("Getting subscriptions..."):
                subs = get_subscriptions(st.session_state.azure_token)
                st.session_state.subscriptions = subs
        
        if st.session_state.get('subscriptions'):
            selected_sub = st.selectbox(
                "🏢 Subscription",
                options=[sub['subscriptionId'] for sub in st.session_state.subscriptions],
                format_func=lambda x: next(sub['displayName'] for sub in st.session_state.subscriptions if sub['subscriptionId'] == x)
            )
            if selected_sub:
                st.session_state.selected_subscription = selected_sub
    
    with col3:
        if st.button("🚪 Logout", key="logout"):
            # Clear all authentication data
            for key in ['azure_credentials', 'azure_token', 'user_info', 'subscriptions', 'selected_subscription']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Set environment variables for MCP server based on authentication
if st.session_state.azure_credentials and st.session_state.azure_token:
    # Set environment variables that the MCP server can use
    os.environ['AZURE_ACCESS_TOKEN'] = st.session_state.azure_token.token
    if st.session_state.get('selected_subscription'):
        os.environ['AZURE_SUBSCRIPTION_ID'] = st.session_state.selected_subscription

# MCP Tools Section (only show if authenticated)
if st.session_state.azure_credentials:
    st.subheader("🛠️ Azure Tools")
    
    async def get_azure_tools():
        """Get and cache Azure MCP tools"""
        try:
            cmd_parts = MCP_SERVER_COMMAND.split()
            
            # Add environment variables for authentication
            env = os.environ.copy()
            env['AZURE_ACCESS_TOKEN'] = st.session_state.azure_token.token
            if st.session_state.get('selected_subscription'):
                env['AZURE_SUBSCRIPTION_ID'] = st.session_state.selected_subscription
            
            server_params = StdioServerParameters(
                command=cmd_parts[0],
                args=cmd_parts[1:] if len(cmd_parts) > 1 else [],
                env=env
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
            
            # Add environment variables for authentication
            env = os.environ.copy()
            env['AZURE_ACCESS_TOKEN'] = st.session_state.azure_token.token
            if st.session_state.get('selected_subscription'):
                env['AZURE_SUBSCRIPTION_ID'] = st.session_state.selected_subscription
            
            server_params = StdioServerParameters(
                command=cmd_parts[0],
                args=cmd_parts[1:] if len(cmd_parts) > 1 else [],
                env=env
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

    # Quick action buttons
    st.info("💡 **Quick Actions:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 List Resources", key="list_resources"):
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

    with col2:
        if st.button("🏪 List Storage Accounts", key="list_storage"):
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

    with col3:
        if st.button("👥 List Resource Groups", key="list_rgs"):
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

    # Chat interface
    st.subheader("💬 Chat with Azure Assistant")
    
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

else:
    st.info("🔐 Please authenticate with Azure to access Azure tools and chat functionality.")

st.markdown("---")
st.caption("🚀 Azure Assistant with Web Authentication")
