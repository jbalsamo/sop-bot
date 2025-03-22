# Parallel RAG Application with LangGraph and Azure Services

This application demonstrates a parallel RAG (Retrieval Augmented Generation) approach using LangGraph with Azure OpenAI and Azure Search services. It processes a user query through three parallel paths:

1. Direct query to Azure OpenAI (gpt-4o)
2. Semantic search using Azure Search Service
3. Vector search using Azure Search Service

The application then combines all three results into a comprehensive summary, with the total latency being only as high as the slowest path. The architecture is designed to be modular, maintainable, and robust with built-in rate limiting and caching.

## Features

### Modular Architecture
- **QueryProcessor Class**: Encapsulates all query-related functionality
- **CacheManager Class**: Handles caching of query results using SQLite with compression
- **RateLimiter Class**: Manages API call rate limiting to stay within Azure OpenAI limits

### Performance Optimizations
- **Parallel Execution**: All three search paths run concurrently
- **Caching System**: Previously processed queries are cached to reduce API calls
- **Rate Limiting**: Automatically manages API call frequency and token usage

### Error Handling
- Comprehensive error handling with retry mechanisms
- Fallback responses for various error scenarios
- Detailed logging for troubleshooting

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
   - `AZ_REQUESTS_PER_MINUTE`: (Optional) Maximum API requests per minute (default: 240)
   - `AZ_TOKENS_PER_MINUTE`: (Optional) Maximum tokens per minute (default: 40000)

## Usage

Run the application:
```
python app.py
```

When prompted, enter your question. The application will:
1. Check if the query has been cached
2. If not cached, process it through all three paths in parallel
3. Combine the results into a comprehensive summary
4. Cache the results for future use
5. Display all individual responses and the combined summary

Results are also saved to a text file (`diabetes_search_results.txt`) for reference.

### Cache Management

#### Resetting the Cache

To reset the cache and remove all stored queries, use the `--reset-cache` command-line option:

```
python app.py --reset-cache
```

This will delete all entries in the cache database (`cache.db`) and exit the application.

#### Viewing Cache Statistics

To check how many entries are currently in the cache:

```
sqlite3 cache.db "SELECT COUNT(*) FROM query_cache;"
```

To view the cached queries and when they were last accessed:

```
sqlite3 cache.db "SELECT query_hash, query, last_accessed FROM query_cache;"
```

To view the full details of cached entries (including compressed results):

```
sqlite3 cache.db "SELECT * FROM query_cache;"
```

### Batch Testing

The application includes a batch testing script that can run multiple medical questions through the system:

```
python batch_test.py [num_questions] [--reset-cache]
```

Options:
- `num_questions`: Number of questions to process (default: 5)
- `--reset-cache`: Reset the cache before running the batch test

The script randomly selects questions from a predefined pool of medical questions about ticks and diabetes. It provides detailed metrics on processing time and token usage for each question.

## Architecture

### Query Processing Flow
1. **Initialization**: Set up state and check cache
2. **Parallel Execution**: Run all three search methods concurrently
3. **Result Combination**: Merge results and generate a comprehensive summary
4. **Caching**: Store results for future queries

### Rate Limiting
The application includes a robust rate limiting system to prevent exceeding Azure OpenAI API limits:
- Tracks both request count and token usage per minute
- Automatically waits when approaching limits
- Configurable via environment variables
