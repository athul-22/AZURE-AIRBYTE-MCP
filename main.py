import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# Airbyte/PyAirbyte
import airbyte as ab

# Networking for MCP client
import requests

# --- Load environment variables
load_dotenv()

# --- Azure & Airbyte Config
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

# MCP Config
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
MCP_EXTENSION_KEY = os.getenv("MCP_EXTENSION_KEY")

# ---- Streamlit UI ----

st.set_page_config(page_title="Azure Agentic Data Integrator", layout="wide")
st.title("Azure + Airbyte + MCP Data Integration Demo")

with st.expander("🔑 Azure Blob Storage Settings"):
    st.write(f"Account Name: `{AZURE_STORAGE_ACCOUNT_NAME}`")
    st.write(f"Container Name: `{AZURE_STORAGE_CONTAINER_NAME}`")

with st.expander("💬 MCP Server Settings"):
    st.write(f"Endpoint: {MCP_SERVER_URL}")

# --- Airbyte Pipeline Section ---

st.header("1. Extract & Cache Data with Airbyte (PyAirbyte)")

if st.button("Trigger Azure Blob Ingestion"):
    try:
        # Set up source connector (Azure Blob Storage as a source)
        source = ab.get_source(
            "source-azure-blob-storage",
            install_if_missing=True,
            config={
                "azure_blob_storage_account_name": AZURE_STORAGE_ACCOUNT_NAME,
                "azure_blob_storage_container_name": AZURE_STORAGE_CONTAINER_NAME,
                "azure_blob_storage_account_key": AZURE_STORAGE_ACCOUNT_KEY,
                "format": {"format_type": "jsonl"},  # Or 'csv' as needed
            }
        )
        # Validate connection
        check_result = source.check()
        st.success(f"Connection check: {check_result}")

        # List all available streams (folders/files)
        streams = source.get_available_streams()
        st.write("Available Streams:", streams)
        source.select_all_streams()  # Select everything for demonstration

        # Cache to DuckDB by default
        cache = ab.get_default_cache()
        result = source.read(cache=cache)
        st.success("Data cached from Azure Blob Storage!")

        # Display a DataFrame for the first stream
        main_stream = streams[0] if streams else None
        if main_stream:
            df = cache[main_stream].to_pandas()
            st.dataframe(df.head())

    except Exception as e:
        st.error(f"Airbyte ingestion error: {str(e)}")

# --- MCP Operations Section ---

st.header("2. Interact with Azure MCP Server")

input_payload = st.text_area(
    "Enter JSON request for MCP server",
    value='{"tool": "hello", "arguments": {}, "key": "' + (MCP_EXTENSION_KEY or "") + '"}'
)
trigger_mcp = st.button("Send to MCP Server")

if trigger_mcp:
    try:
        res = requests.post(
            MCP_SERVER_URL,
            headers={
                "Content-Type": "application/json",
                "x-functions-key": MCP_EXTENSION_KEY
            },
            data=input_payload
        )
        st.text("MCP Server Response:")
        st.code(res.text)
    except Exception as e:
        st.error(f"MCP server call failed: {str(e)}")

st.caption("© Your Company - Demo Agentic Data Integration Pipeline on Azure")

# --- End of File ---
