# Parallel RAG Application with LangGraph and Azure Services

This application demonstrates a parallel RAG (Retrieval Augmented Generation) approach using LangGraph with Azure OpenAI and Azure Search services. It processes a user query through three parallel paths:

1. Direct query to Azure OpenAI (gpt-4o)
2. Semantic search using Azure Search Service
3. Vector search using Azure Search Service

The application then displays all three results, with the total latency being only as high as the slowest path.

## Setup

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure your `.env` file with the following variables:
   - `AZ_API_KEY`: Your Azure OpenAI API key
   - `AZ_BASE_URL`: Your Azure OpenAI base URL
   - `AZ_DEPLOYMENT_NAME`: Your Azure OpenAI deployment name
   - `AZ_SEARCH_KEY`: Your Azure Search service key
   - `AZ_SEARCH_ENDPOINT`: Your Azure Search service endpoint
   - `AZ_INDEX_NAME`: Your Azure Search index name for semantic search
   - `AZ_PM_VECTOR_INDEX_NAME`: Your Azure Search vector index name

## Usage

Run the application:
```
python app.py
```

When prompted, enter your question. The application will process it through all three paths in parallel and display the results.
