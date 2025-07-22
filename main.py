import os
import json
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import requests
import airbyte as ab

# --- ENV VARIABLES
load_dotenv()

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")  # Azure Function URL
MCP_EXTENSION_KEY = os.getenv("MCP_EXTENSION_KEY")  # x-functions-key

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Azure Agentic Data Integrator")
st.title("🔗 Azure Agentic Data Integrator with Airbyte & MCP")

with st.expander("🔑 Airbyte + Azure Blob Storage Configuration"):
    st.code(f"Account: {AZURE_STORAGE_ACCOUNT_NAME}")
    st.code(f"Container: {AZURE_STORAGE_CONTAINER_NAME}")

with st.expander("🤖 MCP Server Settings"):
    st.code(f"Endpoint: {MCP_SERVER_URL}")

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

default_payload = {
    "tool": "hello",  # Replace with your custom tool if available
    "arguments": {},
    "key": MCP_EXTENSION_KEY
}
json_input = st.text_area("✍️ MCP Request Payload (JSON)", value=json.dumps(default_payload, indent=2), height=200)
trigger = st.button("🚀 Send to MCP")

if trigger:
    try:
        res = requests.post(
            MCP_SERVER_URL,
            headers={
                "x-functions-key": MCP_EXTENSION_KEY,
                "Content-Type": "application/json"
            },
            data=json_input
        )
        st.success("✅ MCP Response Received")
        try:
            st.json(res.json())
        except:
            st.text(res.text)

    except Exception as e:
        st.error(f"❌ MCP Call Failed: {str(e)}")

st.markdown("---")
st.caption("Built with 🤖 PyAirbyte, Azure Blob & MCP | Streamlit UI Demo")
