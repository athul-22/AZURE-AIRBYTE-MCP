# 🚀 Azure Agentic Data Integrator

A robust, open-source solution for agentic data integration using **Azure Blob Storage**, **Airbyte (PyAirbyte)**, and **MCP server orchestration**—all managed through a modern Streamlit dashboard.

---

## ✨ Features

- 🔒 **Secure Azure Blob Ingestion**: Connect and ingest data from Azure Blob Storage with enterprise-grade security.
- ⚡ **Automated Data Caching**: Cache ingested data in DuckDB for rapid analytics.
- 🤖 **MCP Server Integration**: Orchestrate workflows and automation via MCP server endpoints.
- 🖥️ **Intuitive Streamlit UI**: Configure, trigger, and monitor data operations in a professional dashboard.

---

## 📦 Prerequisites

- Python 3.11+
- Azure Blob Storage account & container
- MCP server endpoint & extension key

**Required Python Packages:**
- `streamlit`
- `pyairbyte`
- `python-dotenv`
- `pandas`
- `requests`

---

## 🛠️ Installation

1. **Clone the Repository**
   ```sh
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Set Up Virtual Environment**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```sh
   pip install streamlit pyairbyte python-dotenv pandas requests
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the project root:
   ```
   AZURE_STORAGE_ACCOUNT_NAME=your_account_name
   AZURE_STORAGE_CONTAINER_NAME=your_container_name
   AZURE_STORAGE_ACCOUNT_KEY=your_account_key
   MCP_SERVER_URL=https://your-mcp-server-endpoint
   MCP_EXTENSION_KEY=your_mcp_extension_key
   ```

---

## 🚦 Quick Start

Launch the dashboard:
```sh
streamlit run main.py
```

**UI Widgets:**
- 🗄️ **Azure Blob Storage Settings**: Review your Azure configuration.
- 📥 **Trigger Azure Blob Ingestion**: Extract and preview data.
- 🌐 **MCP Server Settings**: Verify MCP endpoint connectivity.
- 📤 **Send to MCP Server**: Submit JSON payloads and view responses.

---

## 📁 Project Structure

```
├── main.py           # Streamlit application
├── .env              # Environment configuration
├── README.md         # Documentation
└── venv/             # Python virtual environment
```

---

## 🛡️ Security & Compliance

- Credentials managed via environment variables and `.env` files.
- All network operations use secure HTTPS endpoints.
- Designed for enterprise data governance standards.

---

## 🤝 Contributing

Contributions are welcome!  
Please open issues or submit pull requests to help improve this project.

---

## 📄 License

This project is provided for demonstration and prototyping purposes.  
© Your Company. All rights