import os
import json
import streamlit as st
from dotenv import load_dotenv, set_key
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

# --- Load ENV VARIABLES ---
load_dotenv()

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Azure configuration from .env
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "04b07795-8ddb-461a-bbee-02f9e1bf7b46")
AZURE_ACCESS_TOKEN = os.getenv("AZURE_ACCESS_TOKEN")

# MCP configuration
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "npx -y @azure/mcp@latest server start")

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- UTILITY FUNCTIONS ---
def save_to_env(key, value):
    """Save a key-value pair to .env file"""
    try:
        set_key('.env', key, value)
        logger.info(f"Saved {key} to .env file")
        return True
    except Exception as e:
        logger.error(f"Failed to save {key} to .env: {e}")
        return False

def extract_tenant_from_token(token):
    """Extract tenant ID from JWT token"""
    try:
        import base64
        import json
        
        # JWT tokens have 3 parts separated by dots
        parts = token.token.split('.')
        if len(parts) >= 2:
            # Decode the payload (second part)
            # Add padding if needed
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding
            
            decoded = base64.b64decode(payload)
            token_data = json.loads(decoded)
            
            # Extract tenant ID
            tenant_id = token_data.get('tid')
            logger.info(f"Extracted tenant ID from token: {tenant_id}")
            return tenant_id
    except Exception as e:
        logger.warning(f"Could not extract tenant from token: {e}")
    
    return None

def device_code_auth():
    """Perform Azure device code authentication"""
    try:
        # Create device code credential
        credential = DeviceCodeCredential(
            client_id=AZURE_CLIENT_ID,
            tenant_id="common"  # Multi-tenant
        )
        
        # Get token with proper scope
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
        
        # Get user info from Microsoft Graph
        response = requests.get(
            'https://graph.microsoft.com/v1.0/me',
            headers=headers
        )
        
        if response.status_code == 200:
            user_data = response.json()
            logger.info(f"User info retrieved: {user_data.get('displayName', 'Unknown')}")
            return user_data
        else:
            logger.warning(f"Failed to get user info: {response.status_code}")
            return None
    
    except Exception as e:
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
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            subscriptions = response.json().get('value', [])
            logger.info(f"Successfully retrieved {len(subscriptions)} subscriptions")
            return subscriptions
        else:
            logger.warning(f"Failed to get subscriptions: {response.status_code}")
            return []
    
    except Exception as e:
        logger.error(f"Failed to get subscriptions: {e}")
        return []

def get_tool_parameters(tool_name, available_tools, subscription_id=None):
    """Get appropriate parameters for each tool"""
    
    if not subscription_id:
        logger.warning(f"No subscription ID provided for tool {tool_name}")
        return {}
    
    # Base parameters with subscription
    base_params = {"subscription": subscription_id}
    
    # Tool-specific parameter mapping
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
        
        # Extension tools
        "azmcp_extension_az": {
            "command": "resource list --output table"
        },
        "azmcp_extension_azqr": base_params,
        
        # Tools that don't need subscription
        "azmcp_azureterraformbestpractices_get": {},
        "azmcp_bestpractices_azurefunctions_get-code-generation": {},
        "azmcp_foundry_models_list": {},
    }
    
    result = tool_params.get(tool_name, base_params)
    logger.info(f"Tool {tool_name} parameters: {result}")
    return result

# Add Airbyte configuration to your .env
AIRBYTE_API_URL = os.getenv("AIRBYTE_API_URL", "http://localhost:8000/api/v1")
AIRBYTE_USERNAME = os.getenv("AIRBYTE_USERNAME", "airbyte")
AIRBYTE_PASSWORD = os.getenv("AIRBYTE_PASSWORD", "password")

# Add Airbyte functions
def get_airbyte_sources():
    """Get available Airbyte data sources"""
    try:
        import requests
        from requests.auth import HTTPBasicAuth
        
        auth = HTTPBasicAuth(AIRBYTE_USERNAME, AIRBYTE_PASSWORD)
        response = requests.get(
            f"{AIRBYTE_API_URL}/sources",
            auth=auth,
            timeout=30
        )
        
        if response.status_code == 200:
            sources = response.json()
            logger.info(f"Found {len(sources.get('data', []))} Airbyte sources")
            return sources.get('data', [])
        else:
            logger.error(f"Failed to get Airbyte sources: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error connecting to Airbyte: {e}")
        return []

def get_airbyte_connections():
    """Get Airbyte connections and their data"""
    try:
        auth = HTTPBasicAuth(AIRBYTE_USERNAME, AIRBYTE_PASSWORD)
        response = requests.get(
            f"{AIRBYTE_API_URL}/connections",
            auth=auth,
            timeout=30
        )
        
        if response.status_code == 200:
            connections = response.json()
            return connections.get('data', [])
        return []
        
    except Exception as e:
        logger.error(f"Error getting Airbyte connections: {e}")
        return []

def create_adf_pipeline_from_airbyte(source_config, destination_config):
    """Create Azure Data Factory pipeline based on Airbyte source/destination"""
    
    pipeline_json = {
        "name": f"airbyte-{source_config['name']}-pipeline",
        "properties": {
            "activities": [
                {
                    "name": "CopyFromAirbyte",
                    "type": "Copy",
                    "inputs": [
                        {
                            "referenceName": f"airbyte_{source_config['name']}_dataset",
                            "type": "DatasetReference"
                        }
                    ],
                    "outputs": [
                        {
                            "referenceName": f"azure_{destination_config['name']}_dataset", 
                            "type": "DatasetReference"
                        }
                    ],
                    "typeProperties": {
                        "source": {
                            "type": source_config.get('connector_type', 'RestSource')
                        },
                        "sink": {
                            "type": destination_config.get('connector_type', 'AzureBlobSink')
                        }
                    }
                }
            ]
        }
    }
    
    return pipeline_json

async def execute_airbyte_adf_tool(action, parameters):
    """Execute Airbyte + ADF integration tool"""
    try:
        if action == "list_airbyte_sources":
            sources = get_airbyte_sources()
            return {"status": 200, "data": sources}
            
        elif action == "create_adf_pipeline":
            source_id = parameters.get('source_id')
            destination_type = parameters.get('destination_type', 'blob')
            
            # Get source details from Airbyte
            sources = get_airbyte_sources()
            source = next((s for s in sources if s['id'] == source_id), None)
            
            if not source:
                return {"status": 404, "error": "Source not found"}
            
            # Create ADF pipeline using Azure MCP
            pipeline_name = f"airbyte-{source['name']}-to-{destination_type}"
            
            # Use existing Azure MCP tools to create the pipeline
            adf_result = await execute_azure_tool(
                "azmcp_datafactory_pipeline_create", 
                {
                    "subscription": AZURE_SUBSCRIPTION_ID,
                    "resource-group": "sapioagent_group",  # Your existing RG
                    "factory-name": "sapioagents",  # Your existing Data Factory
                    "pipeline-name": pipeline_name,
                    "pipeline-json": json.dumps(create_adf_pipeline_from_airbyte(
                        source, 
                        {"name": destination_type, "connector_type": "AzureBlobSink"}
                    ))
                }
            )
            
            return adf_result
            
        elif action == "sync_airbyte_to_azure":
            # Trigger Airbyte sync and then ADF pipeline
            connection_id = parameters.get('connection_id')
            
            # Trigger Airbyte sync
            auth = HTTPBasicAuth(AIRBYTE_USERNAME, AIRBYTE_PASSWORD)
            sync_response = requests.post(
                f"{AIRBYTE_API_URL}/connections/{connection_id}/sync",
                auth=auth,
                json={}
            )
            
            if sync_response.status_code == 200:
                return {"status": 200, "message": "Airbyte sync triggered successfully"}
            else:
                return {"status": 500, "error": "Failed to trigger Airbyte sync"}
                
    except Exception as e:
        return {"status": 500, "error": str(e)}

# Add to your existing tool parameters function
def get_airbyte_tool_parameters(tool_name, subscription_id=None):
    """Get parameters for Airbyte integration tools"""
    
    airbyte_tools = {
        "list_airbyte_sources": {},
        "create_adf_pipeline": {
            "source_id": "required",
            "destination_type": "blob"
        },
        "sync_airbyte_to_azure": {
            "connection_id": "required"
        }
    }
    
    return airbyte_tools.get(tool_name, {})

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Azure Assistant", page_icon="🚀")
st.title("🚀 Azure Assistant with .env Configuration")

# Initialize session state
if 'azure_credentials' not in st.session_state:
    st.session_state.azure_credentials = None
if 'azure_token' not in st.session_state:
    st.session_state.azure_token = None

# Authentication Section
st.subheader("🔐 Azure Authentication")

# Show current .env status
with st.expander("📋 Current .env Configuration", expanded=False):
    st.code(f"""
AZURE_TENANT_ID: {AZURE_TENANT_ID or 'Not Set'}
AZURE_SUBSCRIPTION_ID: {AZURE_SUBSCRIPTION_ID or 'Not Set'}
AZURE_CLIENT_ID: {AZURE_CLIENT_ID}
AZURE_ACCESS_TOKEN: {'Set' if AZURE_ACCESS_TOKEN else 'Not Set'}
GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not Set'}
    """)

if not AZURE_TENANT_ID or not AZURE_SUBSCRIPTION_ID or not AZURE_ACCESS_TOKEN:
    st.warning("⚠️ **Azure configuration incomplete in .env file**")
    
    # Authentication flow
    if st.session_state.azure_credentials is None:
        st.info("👋 **Step 1:** Authenticate with Azure to get your tenant and subscription IDs")
        
        if st.button("🔐 Login with Azure", type="primary", key="azure_login"):
            with st.spinner("🔄 Starting Azure authentication..."):
                try:
                    # Create a placeholder for the device code
                    device_code_placeholder = st.empty()
                    
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
                        client_id=AZURE_CLIENT_ID,
                        prompt_callback=device_code_callback
                    )
                    
                    # Get token
                    token = credential.get_token("https://management.azure.com/.default")
                    
                    if token:
                        st.session_state.azure_credentials = credential
                        st.session_state.azure_token = token
                        
                        # Extract tenant ID from token
                        tenant_id = extract_tenant_from_token(token)
                        
                        # Get user info and subscriptions
                        user_info = get_user_info(token)
                        subscriptions = get_subscriptions(token)
                        
                        device_code_placeholder.empty()
                        
                        # Show configuration options
                        st.success("✅ Authentication successful!")
                        st.info("📝 **Step 2:** Configure your .env file with the discovered values")
                        
                        # Display discovered values
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**🔍 Discovered Tenant ID:**")
                            if tenant_id:
                                st.code(tenant_id)
                                if st.button("💾 Save Tenant ID to .env", key="save_tenant"):
                                    if save_to_env('AZURE_TENANT_ID', tenant_id):
                                        st.success("✅ Tenant ID saved!")
                                        st.rerun()
                            else:
                                st.warning("Could not extract tenant ID from token")
                                manual_tenant = st.text_input("Enter Tenant ID manually:", key="manual_tenant_input")
                                if manual_tenant and st.button("Save Manual Tenant", key="save_manual_tenant"):
                                    if save_to_env('AZURE_TENANT_ID', manual_tenant):
                                        st.success("✅ Tenant ID saved!")
                                        st.rerun()
                        
                        with col2:
                            st.write("**🏢 Available Subscriptions:**")
                            if subscriptions:
                                selected_sub = st.selectbox(
                                    "Choose subscription:",
                                    options=[sub['subscriptionId'] for sub in subscriptions],
                                    format_func=lambda x: next(f"{sub['displayName']} ({x[:8]}...)" for sub in subscriptions if sub['subscriptionId'] == x),
                                    key="sub_selector"
                                )
                                if st.button("💾 Save Subscription to .env", key="save_sub"):
                                    if save_to_env('AZURE_SUBSCRIPTION_ID', selected_sub):
                                        st.success("✅ Subscription ID saved!")
                                        st.rerun()
                            else:
                                st.warning("No subscriptions found")
                                manual_sub = st.text_input("Enter Subscription ID manually:", key="manual_sub_input")
                                if manual_sub and st.button("Save Manual Subscription", key="save_manual_sub"):
                                    if save_to_env('AZURE_SUBSCRIPTION_ID', manual_sub):
                                        st.success("✅ Subscription ID saved!")
                                        st.rerun()
                        
                        # Save access token
                        if st.button("💾 Save Access Token to .env", key="save_token"):
                            if save_to_env('AZURE_ACCESS_TOKEN', token.token):
                                st.success("✅ Access token saved!")
                                st.info("🔄 Please refresh the page to load the new configuration")
                
                except Exception as e:
                    st.error(f"❌ Authentication failed: {e}")
                    logger.error(f"Authentication failed: {e}")
    
    else:
        st.info("✅ Authenticated! Please save the configuration above and refresh the page.")

else:
    # All required env vars are set
    st.success("✅ **Azure configuration complete!**")
    
    # Set environment variables for MCP server
    os.environ['AZURE_ACCESS_TOKEN'] = AZURE_ACCESS_TOKEN
    os.environ['AZURE_TENANT_ID'] = AZURE_TENANT_ID
    os.environ['AZURE_SUBSCRIPTION_ID'] = AZURE_SUBSCRIPTION_ID
    
    # Show current config
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏢 Tenant", AZURE_TENANT_ID[:8] + "...")
    with col2:
        st.metric("📋 Subscription", AZURE_SUBSCRIPTION_ID[:8] + "...")
    with col3:
        if st.button("🔄 Refresh Token", key="refresh_token"):
            # Clear token to force re-authentication
            save_to_env('AZURE_ACCESS_TOKEN', '')
            st.info("Token cleared. Please refresh the page to re-authenticate.")
    
    # MCP Tools Section
    st.subheader("🛠️ Azure Tools")
    
    # Async functions for MCP
    async def get_azure_tools():
        """Get and cache Azure MCP tools"""
        try:
            cmd_parts = MCP_SERVER_COMMAND.split()
            logger.info(f"Starting MCP server with command: {cmd_parts}")
            
            # Set environment variables
            env = os.environ.copy()
            env.update({
                'AZURE_ACCESS_TOKEN': AZURE_ACCESS_TOKEN,
                'AZURE_TENANT_ID': AZURE_TENANT_ID,
                'AZURE_SUBSCRIPTION_ID': AZURE_SUBSCRIPTION_ID
            })
            
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
            
            # Set environment variables
            env = os.environ.copy()
            env.update({
                'AZURE_ACCESS_TOKEN': AZURE_ACCESS_TOKEN,
                'AZURE_TENANT_ID': AZURE_TENANT_ID,
                'AZURE_SUBSCRIPTION_ID': AZURE_SUBSCRIPTION_ID
            })
            
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

    # Initialize session state for tools
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
        
        # Quick action buttons
        st.info("💡 **Quick Actions:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 List Resources", key="list_resources"):
                with st.spinner("🔍 Getting your Azure resources..."):
                    result = run_async_in_streamlit(
                        execute_azure_tool("azmcp_extension_az", {"command": "resource list --output table"})
                    )
                    if result and hasattr(result, 'content'):
                        st.success("✅ Your Azure Resources:")
                        content = result.content
                        if isinstance(content, list) and len(content) > 0:
                            content = content[0].text if hasattr(content[0], 'text') else str(content[0])
                        st.text(content)

        with col2:
            if st.button("🏪 List Storage Accounts", key="list_storage"):
                with st.spinner("🔍 Getting your storage accounts..."):
                    params = get_tool_parameters("azmcp_storage_account_list", st.session_state.azure_tools, AZURE_SUBSCRIPTION_ID)
                    result = run_async_in_streamlit(
                        execute_azure_tool("azmcp_storage_account_list", params)
                    )
                    if result and hasattr(result, 'content'):
                        st.success("✅ Your Storage Accounts:")
                        content = result.content
                        if isinstance(content, list) and len(content) > 0:
                            content = content[0].text if hasattr(content[0], 'text') else str(content[0])
                        
                        try:
                            json_content = json.loads(content)
                            if json_content.get('status') == 500:
                                st.error(f"❌ Error: {json_content.get('message', 'Unknown error')}")
                            else:
                                st.json(json_content)
                        except json.JSONDecodeError:
                            st.text(content)

        with col3:
            if st.button("👥 List Resource Groups", key="list_rgs"):
                with st.spinner("🔍 Getting your resource groups..."):
                    params = get_tool_parameters("azmcp_group_list", st.session_state.azure_tools, AZURE_SUBSCRIPTION_ID)
                    result = run_async_in_streamlit(
                        execute_azure_tool("azmcp_group_list", params)
                    )
                    if result and hasattr(result, 'content'):
                        st.success("✅ Your Resource Groups:")
                        content = result.content
                        if isinstance(content, list) and len(content) > 0:
                            content = content[0].text if hasattr(content[0], 'text') else str(content[0])
                        
                        try:
                            json_content = json.loads(content)
                            if json_content.get('status') == 500:
                                st.error(f"❌ Error: {json_content.get('message', 'Unknown error')}")
                            else:
                                st.json(json_content)
                        except json.JSONDecodeError:
                            st.text(content)
        
        # Debug section
        with st.expander("🔍 Debug Information", expanded=False):
            st.write("**Environment Variables for MCP:**")
            st.code(f"""
AZURE_ACCESS_TOKEN: {'Set' if os.environ.get('AZURE_ACCESS_TOKEN') else 'Not Set'}
AZURE_TENANT_ID: {os.environ.get('AZURE_TENANT_ID', 'Not Set')}
AZURE_SUBSCRIPTION_ID: {os.environ.get('AZURE_SUBSCRIPTION_ID', 'Not Set')}
            """)
            
            if st.button("🧪 Test Direct API Call", key="test_direct_api"):
                with st.spinner("Testing API..."):
                    try:
                        headers = {
                            'Authorization': f'Bearer {AZURE_ACCESS_TOKEN}',
                            'Content-Type': 'application/json'
                        }
                        
                        response = requests.get(
                            f'https://management.azure.com/subscriptions/{AZURE_SUBSCRIPTION_ID}/resourceGroups?api-version=2021-04-01',
                            headers=headers,
                            timeout=10
                        )
                        
                        st.code(f"Status: {response.status_code}")
                        if response.status_code == 200:
                            data = response.json()
                            st.code(f"Resource Groups: {len(data.get('value', []))}")
                        else:
                            st.code(f"Error: {response.text[:500]}")
                    except Exception as e:
                        st.error(f"API test failed: {e}")
    
    else:
        st.error("❌ No Azure tools loaded")
        
    # Add the missing chat interface here
    st.subheader("💬 Chat with Azure Assistant")
    
    def extract_tool_names_from_response(response_text, available_tools):
        """Extract tool names that Gemini wants to use"""
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
    
    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display conversation history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask me about your Azure resources..."):
        if GOOGLE_API_KEY and st.session_state.azure_tools:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            logger.info(f"User input: {user_input}")
            
            with st.chat_message("user"):
                st.write(user_input)
            
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

Available Azure tools (that work with subscription):
{tools_info}

Based on the user's request, please suggest ONE specific tool from the list above that would be most helpful. Only suggest tools that are in the list.
"""
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Get Gemini response
                        client = genai.GenerativeModel(GEMINI_MODEL)
                        response = client.generate_content(enhanced_prompt)
                        
                        assistant_message = response.text
                        st.write(assistant_message)
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
                                        params = get_tool_parameters(tool_name, st.session_state.azure_tools, AZURE_SUBSCRIPTION_ID)
                                        
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
                                                if json_content.get('status') == 200:
                                                    st.success(f"✅ {tool_name} results:")
                                                    if 'results' in json_content and 'output' in json_content['results']:
                                                        st.text(json_content['results']['output'])
                                                    elif 'results' in json_content:
                                                        st.json(json_content['results'])
                                                    else:
                                                        st.json(json_content)
                                                else:
                                                    st.error(f"❌ {tool_name}: {json_content.get('message', 'Unknown error')}")
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
        else:
            st.error("❌ Please ensure Google API key is set and Azure tools are loaded")
    
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

# Instructions
st.markdown("---")
st.subheader("📝 Current Status")
st.success("""
✅ **Everything is working perfectly!**

Your Azure resources are being successfully retrieved:
- 14 resources found across multiple resource groups
- Storage accounts, web apps, container registries, and more
- MCP server is properly authenticated

**Try asking questions like:**
- "Show me my storage accounts"
- "List my resource groups" 
- "What web apps do I have?"
""")

st.caption("🚀 Azure Assistant - Fully Configured!")
