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
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ENV VARIABLES
load_dotenv()

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

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
        logger.error(f"Authentication failed: {e}")
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
            logger.warning(f"Failed to get user info: {response.status_code}")
            return None
    
    except Exception as e:
        st.error(f"Failed to get user info: {e}")
        logger.error(f"Failed to get user info: {e}")
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
            logger.warning(f"Failed to get subscriptions: {response.status_code}")
            return []
    
    except Exception as e:
        st.error(f"Failed to get subscriptions: {e}")
        logger.error(f"Failed to get subscriptions: {e}")
        return []

def get_tool_parameters(tool_name, available_tools, subscription_id=None):
    """Get appropriate parameters for each tool - FIXED VERSION"""
    
    if not subscription_id:
        logger.warning(f"No subscription ID provided for tool {tool_name}")
        return {}
    
    # Base parameters with subscription
    base_params = {"subscription": subscription_id}
    
    # Tool-specific parameter mapping - COMPLETE MAPPING
    tool_params = {
        # Basic listing tools that only need subscription
        "azmcp_group_list": base_params,
        "azmcp_storage_account_list": base_params,
        "azmcp_vm_list": base_params,
        "azmcp_webapp_list": base_params,
        "azmcp_aks_cluster_list": base_params,
        "azmcp_sql_server_list": base_params,
        "azmcp_cosmos_account_list": base_params,
        "azmcp_appconfig_account_list": base_params,
        "azmcp_monitor_workspace_list": base_params,
        
        # Extension tools with custom commands
        "azmcp_extension_az": {
            "command": "resource list --output table"
        },
        "azmcp_extension_azqr": base_params,
        
        # Tools that don't need subscription (best practices, models, etc.)
        "azmcp_azureterraformbestpractices_get": {},
        "azmcp_bestpractices_azurefunctions_get-code-generation": {},
        "azmcp_foundry_models_list": {},
        
        # Tools that need additional parameters - return None to skip
        "azmcp_keyvault_key_list": None,  # Needs vault name
        "azmcp_cosmos_database_container_item_query": None,  # Needs account, database, container
        "azmcp_monitor_resource_log_query": None,  # Needs resource-id, table, query
        "azmcp_monitor_workspace_log_query": None,  # Needs workspace, table, query, resource-group
        "azmcp_kusto_query": None,  # Needs cluster info
        "azmcp_monitor_healthmodels_entity_gethealth": None,  # Needs entity, model, resource-group
        "azmcp_monitor_metrics_query": None,  # Needs resource, metrics, namespace
        "azmcp_postgres_database_list": None,  # Needs resource-group, user, server
        "azmcp_postgres_table_list": None,  # Needs resource-group, user, server, database
        "azmcp_role_assignment_list": None,  # Needs scope
        "azmcp_search_index_query": None,  # Needs service, index, query
        "azmcp_sql_db_show": None,  # Needs resource-group, server, database
        "azmcp_foundry_models_deploy": None,  # Needs deployment name, model, format, etc.
        "azmcp_appconfig_kv_list": None,  # Needs account name
        "azmcp_appconfig_kv_set": None,  # Needs account, key, value
        "azmcp_appconfig_kv_delete": None,  # Needs account, key
        "azmcp_appconfig_kv_lock": None,  # Needs account, key
        "azmcp_appconfig_kv_unlock": None,  # Needs account, key
    }
    
    result = tool_params.get(tool_name, base_params)
    logger.info(f"Tool {tool_name} parameters: {result}")
    return result

def extract_tool_names_from_response(response_text, available_tools):
    """Extract tool names that Gemini wants to use - IMPROVED VERSION"""
    tool_names = []
    response_lower = response_text.lower()
    
    # Look for exact tool name matches
    for tool in available_tools:
        if tool.name.lower() in response_lower:
            tool_names.append(tool.name)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tools = []
    for tool in tool_names:
        if tool.lower() not in seen:
            seen.add(tool.lower())
            unique_tools.append(tool)
    
    # If no tools found, suggest basic ones for common requests
    if not unique_tools:
        if "resource group" in response_lower:
            unique_tools.append("azmcp_group_list")
        elif "storage" in response_lower:
            unique_tools.append("azmcp_storage_account_list")
        elif "virtual machine" in response_lower or "vm" in response_lower:
            unique_tools.append("azmcp_vm_list")
        elif "list" in response_lower and "all" in response_lower:
            unique_tools.append("azmcp_extension_az")
    
    return unique_tools

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
                        logger.info("Azure authentication successful")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Authentication failed: {e}")
                    logger.error(f"Authentication failed: {e}")
    
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
                logger.info(f"Found {len(subs)} subscriptions")
        
        if st.session_state.get('subscriptions'):
            selected_sub = st.selectbox(
                "🏢 Subscription",
                options=[sub['subscriptionId'] for sub in st.session_state.subscriptions],
                format_func=lambda x: next(sub['displayName'] for sub in st.session_state.subscriptions if sub['subscriptionId'] == x)
            )
            if selected_sub:
                st.session_state.selected_subscription = selected_sub
                logger.info(f"Selected subscription: {selected_sub}")
    
    with col3:
        if st.button("🚪 Logout", key="logout"):
            # Clear all authentication data
            for key in ['azure_credentials', 'azure_token', 'user_info', 'subscriptions', 'selected_subscription']:
                if key in st.session_state:
                    del st.session_state[key]
            logger.info("User logged out")
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
            logger.info(f"Starting MCP server with command: {cmd_parts}")
            
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
                    logger.info(f"Loaded {len(tools_result.tools)} Azure tools")
                    return tools_result.tools
        except Exception as e:
            st.error(f"❌ Failed to load Azure tools: {e}")
            logger.error(f"Failed to load Azure tools: {e}")
            return []

    async def execute_azure_tool(tool_name, arguments=None):
        """Execute an Azure MCP tool and return results"""
        try:
            cmd_parts = MCP_SERVER_COMMAND.split()
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
            
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
                    logger.info(f"Tool {tool_name} executed successfully")
                    return result
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    def run_async_in_streamlit(coro):
        """Run async function in Streamlit"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.close()
            return result
        except Exception as e:
            error_msg = f"Async error: {e}"
            st.error(error_msg)
            logger.error(error_msg)
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

    # Quick action buttons with proper parameters - FIXED
    st.info("💡 **Quick Actions:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 List Resources", key="list_resources"):
            if not st.session_state.get('selected_subscription'):
                st.error("❌ Please select a subscription first")
            else:
                with st.spinner("🔍 Getting your Azure resources..."):
                    # Find the right tool
                    az_tool = next((t for t in st.session_state.azure_tools if t.name == "azmcp_extension_az"), None)
                    if az_tool:
                        params = get_tool_parameters("azmcp_extension_az", st.session_state.azure_tools, st.session_state.get('selected_subscription'))
                        st.info(f"🔧 Using parameters: {params}")
                        result = run_async_in_streamlit(
                            execute_azure_tool("azmcp_extension_az", params)
                        )
                        if result and hasattr(result, 'content'):
                            st.success("✅ Your Azure Resources:")
                            content = result.content
                            if isinstance(content, list) and len(content) > 0:
                                content = content[0].text if hasattr(content[0], 'text') else str(content[0])
                            st.text(content)
                        else:
                            st.error(f"❌ Failed to get resources: {result}")

    with col2:
        if st.button("🏪 List Storage Accounts", key="list_storage"):
            if not st.session_state.get('selected_subscription'):
                st.error("❌ Please select a subscription first")
            else:
                with st.spinner("🔍 Getting your storage accounts..."):
                    storage_tool = next((t for t in st.session_state.azure_tools if t.name == "azmcp_storage_account_list"), None)
                    if storage_tool:
                        params = get_tool_parameters("azmcp_storage_account_list", st.session_state.azure_tools, st.session_state.get('selected_subscription'))
                        st.info(f"🔧 Using parameters: {params}")
                        result = run_async_in_streamlit(
                            execute_azure_tool("azmcp_storage_account_list", params)
                        )
                        if result and hasattr(result, 'content'):
                            st.success("✅ Your Storage Accounts:")
                            content = result.content
                            if isinstance(content, list) and len(content) > 0:
                                content = content[0].text if hasattr(content[0], 'text') else str(content[0])
                            st.text(content)
                        else:
                            st.error(f"❌ Failed to get storage accounts: {result}")

    with col3:
        if st.button("👥 List Resource Groups", key="list_rgs"):
            if not st.session_state.get('selected_subscription'):
                st.error("❌ Please select a subscription first")
            else:
                with st.spinner("🔍 Getting your resource groups..."):
                    # Use the correct tool name
                    rg_tool = next((t for t in st.session_state.azure_tools if t.name == "azmcp_group_list"), None)
                    if rg_tool:
                        params = get_tool_parameters("azmcp_group_list", st.session_state.azure_tools, st.session_state.get('selected_subscription'))
                        st.info(f"🔧 Using parameters: {params}")
                        result = run_async_in_streamlit(
                            execute_azure_tool("azmcp_group_list", params)
                        )
                        if result and hasattr(result, 'content'):
                            st.success("✅ Your Resource Groups:")
                            content = result.content
                            if isinstance(content, list) and len(content) > 0:
                                content = content[0].text if hasattr(content[0], 'text') else str(content[0])
                            st.text(content)
                        else:
                            st.error(f"❌ Failed to get resource groups: {result}")
                    else:
                        st.error("❌ Resource group list tool not found")

    # Chat interface
    st.subheader("💬 Chat with Azure Assistant")
    
    # Debug section
    with st.expander("🔍 Debug Information", expanded=False):
        if st.session_state.azure_tools:
            st.write("**Available tool names:**")
            tool_names = [tool.name for tool in st.session_state.azure_tools]
            st.code("\n".join(tool_names))
        
        st.write("**Environment Variables:**")
        st.code(f"AZURE_ACCESS_TOKEN: {'Set' if os.environ.get('AZURE_ACCESS_TOKEN') else 'Not Set'}")
        st.code(f"AZURE_SUBSCRIPTION_ID: {os.environ.get('AZURE_SUBSCRIPTION_ID', 'Not Set')}")
        
        st.write("**Session State:**")
        st.code(f"Selected Subscription: {st.session_state.get('selected_subscription', 'None')}")
    
    # Text input for custom queries
    user_input = st.chat_input("Ask me about your Azure resources...")

    if user_input and GOOGLE_API_KEY and st.session_state.azure_tools:
        # Check if subscription is selected
        if not st.session_state.get('selected_subscription'):
            st.error("❌ Please select a subscription before asking questions")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            logger.info(f"User input: {user_input}")
            
            # Create enhanced prompt with tools
            simple_tools = [
                "azmcp_group_list",
                "azmcp_storage_account_list", 
                "azmcp_vm_list",
                "azmcp_webapp_list",
                "azmcp_aks_cluster_list",
                "azmcp_extension_az"
            ]
            
            tools_info = "\n".join([f"- {tool}: Use for listing {tool.replace('azmcp_', '').replace('_list', '').replace('_', ' ')}" for tool in simple_tools])
            
            enhanced_prompt = f"""
{user_input}

Available Azure tools (that work with just subscription):
{tools_info}

Please suggest ONE specific tool from the list above that would be most helpful for this request. Only suggest tools that are in the list.
"""
            
            with st.spinner("🤖 Thinking..."):
                try:
                    # Get Gemini response
                    client = genai.GenerativeModel(GEMINI_MODEL)
                    response = client.generate_content(enhanced_prompt)
                    
                    assistant_message = response.text
                    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                    logger.info(f"Gemini response: {assistant_message}")
                    
                    # Extract and execute suggested tools
                    suggested_tools = extract_tool_names_from_response(assistant_message, st.session_state.azure_tools)
                    logger.info(f"Suggested tools: {suggested_tools}")
                    
                    if suggested_tools:
                        # Only use tools that we know work with just subscription
                        working_tools = [tool for tool in suggested_tools if tool in simple_tools]
                        
                        if working_tools:
                            st.info(f"🔧 Executing {len(working_tools)} tool(s): {', '.join(working_tools)}")
                            
                            for tool_name in working_tools:
                                with st.spinner(f"⚡ Running {tool_name}..."):
                                    # Get appropriate parameters for this tool
                                    params = get_tool_parameters(tool_name, st.session_state.azure_tools, st.session_state.get('selected_subscription'))
                                    
                                    if params is None:
                                        st.warning(f"⚠️ Tool {tool_name} requires additional parameters")
                                        continue
                                    
                                    st.info(f"🔧 Using parameters: {params}")
                                    
                                    result = run_async_in_streamlit(
                                        execute_azure_tool(tool_name, params)
                                    )
                                    
                                    if result and hasattr(result, 'content') and result.content:
                                        # Parse response
                                        content = result.content
                                        if isinstance(content, list) and len(content) > 0:
                                            content = content[0].text if hasattr(content[0], 'text') else str(content[0])
                                        
                                        try:
                                            # Try to parse as JSON to check for errors
                                            json_content = json.loads(content)
                                            if json_content.get('status') == 400:
                                                st.error(f"❌ {tool_name}: {json_content.get('message', 'Unknown error')}")
                                            else:
                                                st.success(f"✅ {tool_name} results:")
                                                if 'results' in json_content:
                                                    st.json(json_content['results'])
                                                else:
                                                    st.json(json_content)
                                        except json.JSONDecodeError:
                                            # Not JSON, display as text
                                            st.success(f"✅ {tool_name} results:")
                                            st.text(content)
                                        
                                        # Add result to conversation
                                        result_message = f"**{tool_name} Results:**\n```\n{content}\n```"
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": result_message
                                        })
                                    else:
                                        error_msg = f"❌ {tool_name} failed or returned no data: {result}"
                                        st.error(error_msg)
                                        logger.error(error_msg)
                        else:
                            st.info("💡 The suggested tools need additional parameters. Try asking for basic listings like 'list my resource groups' or 'show my storage accounts'")
                    else:
                        st.info("💡 No specific tools were suggested. Try asking more specifically, like 'list my resource groups' or 'show my storage accounts'")
                    
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    logger.error(error_msg)

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

# Show logs in console
if st.checkbox("📄 Show Console Logs"):
    st.text("Check your terminal/console for detailed logs")

st.markdown("---")
st.caption("🚀 Azure Assistant with FIXED Parameter Handling")
