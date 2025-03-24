"""
Parallel RAG Application using LangGraph with Azure OpenAI and Azure Search
This application sends a query to three different sources in parallel:
1. Azure OpenAI (gpt-4o)
2. Azure Search Service (semantic search)
3. Azure Search Service (vector search)
"""

import os
import time
from typing import Dict, Any, List, TypedDict
import asyncio
import tiktoken  # For token counting

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

# Azure Search imports
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# LangGraph imports
from langgraph.graph import StateGraph, END

# Define function to load environment variables manually
def load_env_vars():
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key] = value
                env_vars[key] = value
        print("Loaded environment variables manually")
        return env_vars
    except Exception as e:
        print(f"Error loading .env file: {e}")
        return {}

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    # Fallback method if python-dotenv is not available
    load_env_vars()

# Additional imports for caching
import json
import hashlib
import sqlite3
import zlib
from datetime import datetime, timedelta

class CacheManager:
    """Manages caching of query results to improve performance and reduce API calls
    
    This class handles storing and retrieving query results in a SQLite database,
    with compression to reduce storage requirements and hashing for efficient lookups.
    """
    
    def __init__(self, db_path="cache.db", cache_duration_days=7):
        """Initialize the cache manager
        
        Args:
            db_path: Path to the SQLite database file
            cache_duration_days: Number of days to keep cached results
        """
        self.db_path = db_path
        self.cache_duration_days = cache_duration_days
        self._create_tables_if_not_exist()
        
    def _create_tables_if_not_exist(self):
        """Create the necessary database tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create the query_cache table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT,
                result BLOB,
                created_at TEXT,
                last_accessed TEXT
            )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error creating cache tables: {str(e)}")
    
    def _hash_query(self, query: str) -> str:
        """Create a hash of the query for efficient lookup
        
        Args:
            query: The query string to hash
            
        Returns:
            MD5 hash of the query string
        """
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def get_cached_result(self, query: str) -> Dict[str, Any]:
        """Get cached result for a query if it exists
        
        Args:
            query: The query string to look up
            
        Returns:
            Cached result dictionary or None if not found
        """
        try:
            query_hash = self._hash_query(query)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the cached result
            cursor.execute(
                "SELECT result FROM query_cache WHERE query_hash = ?", 
                (query_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # Update last accessed time
                cursor.execute(
                    "UPDATE query_cache SET last_accessed = ? WHERE query_hash = ?",
                    (datetime.now().isoformat(), query_hash)
                )
                conn.commit()
                
                # Decompress and parse the result
                compressed_data = result[0]
                json_data = zlib.decompress(compressed_data).decode('utf-8')
                return json.loads(json_data)
            
            return None
        except Exception as e:
            print(f"Error retrieving from cache: {str(e)}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def cache_result(self, query: str, result: Dict[str, Any]) -> bool:
        """Cache the result for a query
        
        Args:
            query: The query string
            result: The result dictionary to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query_hash = self._hash_query(query)
            
            # Convert result to JSON string and compress
            result_json = json.dumps(result)
            compressed_data = zlib.compress(result_json.encode('utf-8'))
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store the result
            now = datetime.now().isoformat()
            cursor.execute(
                """INSERT OR REPLACE INTO query_cache 
                   (query_hash, query, result, created_at, last_accessed) 
                   VALUES (?, ?, ?, ?, ?)""",
                (query_hash, query, compressed_data, now, now)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error caching result: {str(e)}")
            return False
    
    def clean_old_entries(self) -> int:
        """Remove entries older than the specified cache duration
        
        Returns:
            Number of entries removed
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=self.cache_duration_days)).isoformat()
            
            # Delete old entries
            cursor.execute(
                "DELETE FROM query_cache WHERE last_accessed < ?",
                (cutoff_date,)
            )
            
            removed_count = cursor.rowcount
            conn.commit()
            conn.close()
            return removed_count
        except Exception as e:
            print(f"Error cleaning old cache entries: {str(e)}")
            return 0
            
    def reset_cache(self) -> bool:
        """Reset the cache by deleting all entries in the query_cache table
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete all entries
            cursor.execute("DELETE FROM query_cache")
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            print(f"Cache reset successfully. Deleted {deleted_count} entries.")
            return True
        except Exception as e:
            print(f"Error resetting cache: {str(e)}")
            return False

    load_env_vars()

system_prompt = """

"""

# Configuration class for API settings and environment variables
class AppConfig:
    """Configuration class for application settings and environment variables"""
    # Azure OpenAI configuration
    AZ_API_KEY = os.getenv("AZ_API_KEY")
    AZ_BASE_URL = os.getenv("AZ_BASE_URL")
    AZ_DEPLOYMENT_NAME = os.getenv("AZ_DEPLOYMENT_NAME")
    
    # Azure Search configuration
    AZ_SEARCH_KEY = os.getenv("AZ_SEARCH_KEY")
    AZ_SEARCH_ENDPOINT = os.getenv("AZ_SEARCH_ENDPOINT")
    AZ_INDEX_NAME = os.getenv("AZ_INDEX_NAME")
    AZ_PM_VECTOR_INDEX_NAME = os.getenv("AZ_PM_VECTOR_INDEX_NAME")
    
    # API rate limiting configuration
    MAX_RETRIES = 5  # Maximum number of retries for API calls
    BASE_RETRY_DELAY = 1  # Base delay in seconds before retrying
    MAX_CONCURRENT_REQUESTS = 3  # Maximum number of concurrent API requests
    
    # Azure OpenAI rate limits (adjust based on your tier)
    REQUESTS_PER_MINUTE = int(os.getenv("AZ_REQUESTS_PER_MINUTE", "240"))  # Default for standard tier
    TOKENS_PER_MINUTE = int(os.getenv("AZ_TOKENS_PER_MINUTE", "40000"))  # Default for standard tier
    
    # Model configuration
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    EMBEDDING_MODEL = "text-embedding-ada-002"
    API_VERSION = "2023-12-01-preview"

# Define the state schema
class State(TypedDict):
    query: str
    openai_response: str
    semantic_search_response: List[Dict[str, Any]]
    vector_search_response: List[Dict[str, Any]]
    combined_answer: str
    token_usage: Dict[str, int]

# RateLimiter class to handle API rate limiting
class RateLimiter:
    """Class to handle rate limiting for Azure OpenAI API calls
    
    This class tracks both request count and token usage per minute to ensure
    the application stays within Azure OpenAI's rate limits.
    """
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        """Initialize the rate limiter with configured limits
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
            tokens_per_minute: Maximum number of tokens allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps = []
        self.token_usage = []
        self.lock = asyncio.Lock()
        print(f"Rate limiter initialized with {requests_per_minute} requests/min and {tokens_per_minute} tokens/min")
    
    async def add_request(self, tokens: int) -> None:
        """Record a new request with its token usage
        
        Args:
            tokens: Number of tokens used in this request
        """
        current_time = time.time()
        
        async with self.lock:
            # Add the current request
            self.request_timestamps.append(current_time)
            self.token_usage.append(tokens)
            
            # Remove timestamps older than 1 minute
            cutoff_time = current_time - 60
            while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
                self.request_timestamps.pop(0)
                self.token_usage.pop(0)
    
    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if we're approaching rate limits
        
        Args:
            estimated_tokens: Estimated number of tokens for the upcoming request
        """
        async with self.lock:
            current_time = time.time()
            cutoff_time = current_time - 60
            
            # Clean up old entries
            while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
                self.request_timestamps.pop(0)
                self.token_usage.pop(0)
            
            # Check if we're approaching limits
            current_requests = len(self.request_timestamps)
            current_tokens = sum(self.token_usage)
            
            # If we're at 80% of either limit, wait until we're under the threshold
            if current_requests >= 0.8 * self.requests_per_minute or \
               current_tokens + estimated_tokens >= 0.8 * self.tokens_per_minute:
                
                # Calculate wait time (when oldest request will expire from the window)
                if self.request_timestamps:
                    wait_time = 60 - (current_time - self.request_timestamps[0]) + 1  # Add 1 second buffer
                    print(f"Rate limit approaching: {current_requests}/{self.requests_per_minute} requests, "
                          f"{current_tokens}/{self.tokens_per_minute} tokens. Waiting {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)

# Initialize rate limiter
rate_limiter = RateLimiter(AppConfig.REQUESTS_PER_MINUTE, AppConfig.TOKENS_PER_MINUTE)

# Initialize LangChain Azure OpenAI client
azure_llm = AzureChatOpenAI(
    azure_deployment=AppConfig.AZ_DEPLOYMENT_NAME,
    openai_api_version=AppConfig.API_VERSION,
    openai_api_key=AppConfig.AZ_API_KEY,
    azure_endpoint=AppConfig.AZ_BASE_URL,
    temperature=AppConfig.TEMPERATURE,
    max_tokens=AppConfig.MAX_TOKENS,
    verbose=True,  # Enable verbose mode for debugging
    streaming=False  # Disable streaming to ensure we get token counts
)

# Initialize Azure OpenAI Embeddings
azure_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AppConfig.EMBEDDING_MODEL,
    openai_api_version=AppConfig.API_VERSION,
    openai_api_key=AppConfig.AZ_API_KEY,
    azure_endpoint=AppConfig.AZ_BASE_URL
)

# Initialize Azure Search credential
search_credential = AzureKeyCredential(AppConfig.AZ_SEARCH_KEY)

# Initialize search clients
semantic_search_client = SearchClient(
    endpoint=AppConfig.AZ_SEARCH_ENDPOINT,
    index_name=AppConfig.AZ_INDEX_NAME,
    credential=search_credential
)

vector_search_client = SearchClient(
    endpoint=AppConfig.AZ_SEARCH_ENDPOINT,
    index_name=AppConfig.AZ_PM_VECTOR_INDEX_NAME,
    credential=search_credential
)

# Define the prompt template for OpenAI with more specific instructions for medical questions
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
    You are an experienced medical professional providing accurate, accessible health information. Follow these guidelines:

        RESPONSE CRITERIA:
        - Provide answers at a 7th-grade reading level (Flesch-Kincaid 70-80)
        - Detect and respond in the user's language
        - Focus exclusively on medical, health, legal, or psychological topics
        - ALWAYS end your response with this EXACT medical disclaimer: "**Medical Disclaimer**: Always seek professional medical advice for diagnosis and treatment. This information is not a substitute for medical advice from a healthcare provider."
        - Include sources, filename, or citations when available.

        LANGUAGE REQUIREMENTS:
        - Replace technical terms with plain language explanations
        - Use common drug names (both brand and generic) instead of drug classes
        - Define any necessary medical terms in parentheses
        - Censor inappropriate language with asterisks

        PROHIBITED CONTENT:
        - Investment or financial advice
        - Non-medical product recommendations
        - Diagnostic conclusions
        - Treatment prescriptions
        - Dosage recommendations

        UNSUPPORTED LANGUAGE RESPONSE:
        If the language is not recognized, respond with:
        "This language is not currently supported. Please try:
        English: Please use a supported language
        Español: Por favor, utilice un idioma compatible
        Français: Veuillez utiliser une langue prise en charge
        中文: 请使用支持的语言
        日本語: サポートされている言語を使用してください
        한국어: 지원되는 언어를 사용해주세요
        हिंदी: कृपया समर्थित भाषा का प्रयोग करें"
    """),
    ("user", "{query}")
])

# Create the LangChain chain
llm_chain = prompt_template | azure_llm | StrOutputParser()

# QueryProcessor class to encapsulate query-related functionalities
class QueryProcessor:
    """Class to encapsulate query-related functionalities
    
    This class provides methods for querying OpenAI, semantic search, and vector search,
    with proper error handling and rate limiting.
    """
    
    @staticmethod
    async def query_openai(state: State) -> State:
        """Query Azure OpenAI with the user's question using LangChain and a specialized medical prompt
        
        Args:
            state: The current application state containing the query
            
        Returns:
            Updated state with OpenAI response and token usage
        """
        try:
            print("Querying Azure OpenAI with specialized medical prompt")

            # Format the query for token counting
            query_text = state["query"]
            
            # Estimate token usage for rate limiting
            enc = tiktoken.encoding_for_model("gpt-4")
            system_prompt = "You are an experienced medical professional providing accurate, accessible health information."
            estimated_tokens = len(enc.encode(system_prompt + query_text)) * 2  # Rough estimate including response
            
            # Wait if we're approaching rate limits
            await rate_limiter.wait_if_needed(estimated_tokens)

            # Get response from Azure OpenAI
            response = await llm_chain.ainvoke({"query": query_text})

            # Track actual token usage
            prompt_tokens = len(enc.encode(system_prompt + query_text))
            completion_tokens = len(enc.encode(response))
            total_tokens = prompt_tokens + completion_tokens

            # Record the request and token usage for rate limiting
            await rate_limiter.add_request(total_tokens)
            
            # Update token usage in state
            state["token_usage"]["openai_query"] = total_tokens
            state["token_usage"]["total"] += total_tokens

            # Store the response
            state["openai_response"] = response
            print(f"Received response from Azure OpenAI (used {total_tokens} tokens)")
        except Exception as e:
            print(f"Error querying OpenAI: {str(e)}")
            state["openai_response"] = f"Error querying OpenAI: {str(e)}"

        return state

    @staticmethod
    async def query_semantic_search(state: State) -> State:
        """Perform semantic search using Azure Search Service and generate an answer using PromptTemplate
        
        Args:
            state: The current application state containing the query
            
        Returns:
            Updated state with semantic search results and token usage
        """
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                print(f"Performing semantic search using index: {AppConfig.AZ_INDEX_NAME} (attempt {attempt+1}/{max_retries})")

                # Use default semantic search configuration with top=3 as requested
                results = semantic_search_client.search(
                    search_text=state["query"],
                    query_type="semantic",
                    semantic_configuration_name="default",
                    top=3
                )

                # Convert results to a list for easier handling
                search_results = []
                context_docs = []

                for doc in results:
                    # Only extract fields that we know exist in the results
                    content = doc.get("content", "")
                    title = doc.get("title", "")

                    # Format document for context - only use fields that exist
                    if title and content:
                        context_docs.append(f"Document Title: {title}\n\nContent: {content}")
                    elif content:
                        context_docs.append(f"Content: {content}")

                    search_results.append({
                        "id": doc["id"],
                        "content": content,
                        "title": title,
                        "score": doc["@search.score"],
                        "search_type": "semantic"
                    })

                # Generate an answer using the retrieved context
                if context_docs:
                    try:
                        # Create a prompt template for answering the question with context
                        rag_prompt = ChatPromptTemplate.from_template("""
                        Medical question: {query}
                            Review these additional medical sources to provide an answer for the question above:
                            - Don't use data outside of the context to answer the question.
                            - Be VERY concise and brief in your answer (maximum 250 words)
                            - Focus only on the most important information from these categories:
                              - Patient-centered educational material
                              - Real-world applications and examples
                              - Prevention and self-care guidance
                              - Treatment options from authoritative sources
                            - Extract citation and filename for each source used
                            - ALWAYS end your response with this EXACT medical disclaimer: "**Medical Disclaimer**: Always seek professional medical advice for diagnosis and treatment. This information is not a substitute for medical advice from a healthcare provider."
                            Include citations and/or filenames of sources at the bottom if they have links or filenames.

                        CONTEXT INFORMATION:
                        {context}

                        ANSWER:
                        """)

                        # Join context documents with separators
                        context_text = "\n\n---\n\n".join(context_docs)

                        # Format the prompt with the query and context
                        formatted_prompt = rag_prompt.format(query=state["query"], context=context_text)

                        # Estimate token usage for rate limiting
                        enc = tiktoken.encoding_for_model("gpt-4")
                        estimated_tokens = len(enc.encode(str(formatted_prompt))) * 2  # Rough estimate
                        
                        # Wait if we're approaching rate limits
                        await rate_limiter.wait_if_needed(estimated_tokens)

                        # Get answer from Azure OpenAI
                        answer_response = await azure_llm.ainvoke(formatted_prompt)
                        answer_text = answer_response.content

                        # Track token usage using tiktoken
                        prompt_tokens = len(enc.encode(str(formatted_prompt)))
                        completion_tokens = len(enc.encode(answer_text))
                        total_tokens = prompt_tokens + completion_tokens

                        # Record the request and token usage for rate limiting
                        await rate_limiter.add_request(total_tokens)

                        # Update token usage in state
                        state["token_usage"]["semantic_search"] = total_tokens
                        state["token_usage"]["total"] += total_tokens
                        print(f"Semantic search answer generated using {total_tokens} tokens")

                        # Store both the raw results and the generated answer
                        state["semantic_search_response"] = {
                            "raw_results": search_results,
                            "answer": answer_text,
                            "context_used": True
                        }

                        print(f"Generated answer from semantic search results using {len(search_results)} documents")
                    except Exception as answer_error:
                        print(f"Error generating answer from semantic search results: {str(answer_error)}")
                        state["semantic_search_response"] = {
                            "raw_results": search_results,
                            "answer": f"Error generating answer: {str(answer_error)}",
                            "context_used": False
                        }
                else:
                    state["semantic_search_response"] = {
                        "raw_results": search_results,
                        "answer": "No relevant information found in the semantic search results to answer the question.",
                        "context_used": False
                    }

                # If we got here, the search was successful
                break

            except (ConnectionError, ConnectionResetError) as conn_error:
                if attempt < max_retries - 1:
                    error_msg = f"Connection error in semantic search (attempt {attempt+1}/{max_retries}): {str(conn_error)}"
                    print(error_msg)
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    error_msg = f"Error in semantic search after {max_retries} attempts: {str(conn_error)}"
                    print(error_msg)
                    state["semantic_search_response"] = {
                        "raw_results": [],
                        "answer": error_msg,
                        "context_used": False
                    }
            except Exception as e:
                # Handle other exceptions without retry
                error_msg = f"Error in semantic search: {str(e)}"
                print(error_msg)
                state["semantic_search_response"] = {
                    "raw_results": [],
                    "answer": error_msg,
                    "context_used": False
                }
                break  # Exit the retry loop for non-connection errors

        return state

    @staticmethod
    async def query_vector_search(state: State) -> State:
        """Perform vector search using Azure Search Service with the vector index and generate an answer using PromptTemplate
        
        Args:
            state: The current application state containing the query
            
        Returns:
            Updated state with vector search results and token usage
        """
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                print(f"Performing vector search using index: {AppConfig.AZ_PM_VECTOR_INDEX_NAME} (attempt {attempt+1}/{max_retries})")

                # Generate embedding for the query using LangChain
                query_embedding = await azure_embeddings.aembed_query(state["query"])
                print(f"Generated embedding with {len(query_embedding)} dimensions")

                # Based on the debug output, we know the correct vector field name is 'contentVector'
                vector_field_names = ["contentVector", "content_vector", "embedding", "embeddings", "vector", "text_embedding"]
                vector_results = []

                # Try vector search first
                for field_name in vector_field_names:
                    try:
                        print(f"Attempting vector search with field name: {field_name}")

                        # Create a vectorized query with the specified field
                        vector_query = VectorizedQuery(
                            vector=query_embedding,
                            k_nearest_neighbors=3,
                            fields=field_name
                        )

                        # Perform the vector search with top=3 as requested
                        results = vector_search_client.search(
                            search_text=None,
                            vector_queries=[vector_query],
                            select=["id", "content", "title"],  # Explicitly select fields
                            top=3
                        )

                        # Convert results to a list for easier handling
                        for doc in results:
                            content = doc.get("content", "")
                            vector_results.append({
                                "id": doc["id"],
                                "content": content,
                                "title": doc.get("title", ""),
                                "score": doc["@search.score"] if "@search.score" in doc else None,
                                "search_type": f"vector-{field_name}"
                            })

                        if vector_results:
                            print(f"Found {len(vector_results)} results using vector search with '{field_name}' field")
                            break
                    except Exception as e:
                        print(f"Error with field '{field_name}': {str(e)}")

                # If vector search fails, fall back to standard search with the vector index
                if not vector_results:
                    try:
                        print("Falling back to standard search with vector index")
                        results = vector_search_client.search(
                            search_text=state["query"],
                            top=3  # Use top=3 as requested
                        )

                        # Convert results to a list for easier handling
                        for doc in results:
                            content = doc.get("content", "")
                            vector_results.append({
                                "id": doc["id"],
                                "content": content,
                                "title": doc.get("title", ""),
                                "score": doc["@search.score"] if "@search.score" in doc else None,
                                "search_type": "standard-vector-index"
                            })

                        if vector_results:
                            print(f"Found {len(vector_results)} results using standard search with vector index")
                        else:
                            print("No results found using standard search with vector index")
                    except Exception as standard_error:
                        print(f"Error in standard search with vector index: {str(standard_error)}")

                # Generate an answer using the retrieved context
                context_docs = []
                for result in vector_results:
                    # Only extract fields that we know exist in the results
                    content = result.get("content", "")
                    title = result.get("title", "")

                    # Format document for context - only use fields that exist
                    if title and content:
                        context_docs.append(f"Document Title: {title}\n\nContent: {content}")
                    elif content:
                        context_docs.append(f"Content: {content}")

                if context_docs:
                    try:
                        # Create a prompt template for answering the question with context
                        rag_prompt = ChatPromptTemplate.from_template("""
                        Medical question: {query}
                Using only the provided medical reference documents, analyze and answer the question above with:
                  - Don't use trained data outside of the provided context.
                  - Be VERY concise and brief in your answer (maximum 250 words)
                  - Focus only on the most important information from:
                    - Peer-reviewed research and clinical guidelines
                    - Specific relevant data and statistics
                  - Note the title, publication dates, and authors of sources used
                  - Highlight any contradicting information
                  - Extract citation and filename for each source used
                  - ALWAYS end your response with this EXACT medical disclaimer: "**Medical Disclaimer**: Always seek professional medical advice for diagnosis and treatment. This information is not a substitute for medical advice from a healthcare provider."
                Include citations and/or filenames of sources at the bottom.

                        CONTEXT INFORMATION:
                        {context}

                        ANSWER:
                        """)

                        # Join context documents with separators
                        context_text = "\n\n---\n\n".join(context_docs)

                        # Format the prompt with the query and context
                        formatted_prompt = rag_prompt.format(query=state["query"], context=context_text)
                        
                        # Estimate token usage for rate limiting
                        enc = tiktoken.encoding_for_model("gpt-4")
                        estimated_tokens = len(enc.encode(str(formatted_prompt))) * 2  # Rough estimate
                        
                        # Wait if we're approaching rate limits
                        await rate_limiter.wait_if_needed(estimated_tokens)

                        # Get answer from Azure OpenAI
                        answer_response = await azure_llm.ainvoke(formatted_prompt)
                        answer_text = answer_response.content

                        # Track token usage using tiktoken
                        prompt_tokens = len(enc.encode(str(formatted_prompt)))
                        completion_tokens = len(enc.encode(answer_text))
                        total_tokens = prompt_tokens + completion_tokens
                        
                        # Record the request and token usage for rate limiting
                        await rate_limiter.add_request(total_tokens)

                        # Update token usage in state
                        state["token_usage"]["vector_search"] = total_tokens
                        state["token_usage"]["total"] += total_tokens
                        print(f"Vector search answer generated using {total_tokens} tokens")

                        # Store both the raw results and the generated answer
                        state["vector_search_response"] = {
                            "raw_results": vector_results,
                            "answer": answer_text,
                            "context_used": True
                        }

                        print(f"Generated answer from vector search results using {len(vector_results)} documents")
                    except Exception as answer_error:
                        print(f"Error generating answer from vector search results: {str(answer_error)}")
                        state["vector_search_response"] = {
                            "raw_results": vector_results,
                            "answer": f"Error generating answer: {str(answer_error)}",
                            "context_used": False
                        }
                else:
                    state["vector_search_response"] = {
                        "raw_results": vector_results,
                        "answer": "No relevant information found in the vector search results to answer the question.",
                        "context_used": False
                    }

                # If we got here, the search was successful
                break

            except (ConnectionError, ConnectionResetError) as conn_error:
                if attempt < max_retries - 1:
                    error_msg = f"Connection error in vector search (attempt {attempt+1}/{max_retries}): {str(conn_error)}"
                    print(error_msg)
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    error_msg = f"Error in vector search after {max_retries} attempts: {str(conn_error)}"
                    print(error_msg)
                    state["vector_search_response"] = {
                        "raw_results": [],
                        "answer": error_msg,
                        "context_used": False
                    }
            except Exception as e:
                # Handle other exceptions without retry
                error_msg = f"Error in vector search: {str(e)}"
                print(error_msg)
                state["vector_search_response"] = {
                    "raw_results": [],
                    "answer": error_msg,
                    "context_used": False
                }
                break  # Exit the retry loop for non-connection errors

        # Handle case where all approaches failed but no exception was thrown
        if "vector_search_response" not in state:
            state["vector_search_response"] = {
                "raw_results": [],
                "answer": "Vector search failed with all attempted approaches. Check the Azure Search index configuration to ensure it supports vector search.",
                "context_used": False
            }

        return state

    @staticmethod
    async def combine_answers(state: State) -> State:
        """Combine the answers from all three sources into a comprehensive summary
        
        Args:
            state: The current application state containing all search responses
            
        Returns:
            Updated state with combined answer and token usage
        """
        try:
            print("Generating comprehensive summary from all three sources")

            # Extract the answers from each source
            openai_answer = state["openai_response"]

            semantic_response = state["semantic_search_response"]
            semantic_answer = semantic_response["answer"] if isinstance(semantic_response, dict) and "answer" in semantic_response else "No answer available from semantic search."

            vector_response = state["vector_search_response"]
            vector_answer = vector_response["answer"] if isinstance(vector_response, dict) and "answer" in vector_response else "No answer available from vector search."

            # Create a prompt template for combining the answers
            combine_prompt = ChatPromptTemplate.from_template("""
            Given the following:

            Question: {question}
            Answer 1: {openai_answer}
            Answer 2: {semantic_answer}
            Answer 3: {vector_answer}

            Create a comprehensive summary by combining the three previous answers to the question above:
              - Resolve any conflicts between sources
              - Prioritize most recent and authoritative information
              - Include both clinical data and practical guidance
              - Maintain 7th-grade reading level throughout
              - End with this COMPLETE medical disclaimer: "**Medical Disclaimer**: Always seek professional medical advice for diagnosis and treatment. This information is not a substitute for medical advice from a healthcare provider."
            """)

            # Format the prompt with the query and answers
            formatted_prompt = combine_prompt.format(
                question=state["query"],
                openai_answer=openai_answer,
                semantic_answer=semantic_answer,
                vector_answer=vector_answer
            )
            
            # Estimate token usage for rate limiting
            enc = tiktoken.encoding_for_model("gpt-4")
            estimated_tokens = len(enc.encode(str(formatted_prompt))) * 2  # Rough estimate
            
            # Wait if we're approaching rate limits
            await rate_limiter.wait_if_needed(estimated_tokens)

            # Get combined answer from Azure OpenAI
            combined_response = await azure_llm.ainvoke(formatted_prompt)
            combined_answer = combined_response.content

            # Track token usage using tiktoken
            prompt_tokens = len(enc.encode(str(formatted_prompt)))
            completion_tokens = len(enc.encode(combined_answer))
            total_tokens = prompt_tokens + completion_tokens
            
            # Record the request and token usage for rate limiting
            await rate_limiter.add_request(total_tokens)

            # Update token usage in state
            state["token_usage"]["combined_summary"] = total_tokens
            state["token_usage"]["total"] += total_tokens
            print(f"Comprehensive summary generated using {total_tokens} tokens")

            # Store the combined answer in the state
            state["combined_answer"] = combined_answer
            print("Generated comprehensive summary successfully")

        except Exception as e:
            error_msg = f"Error generating comprehensive summary: {str(e)}"
            print(error_msg)
            state["combined_answer"] = error_msg

        return state

# Create the LangGraph workflow
async def run_parallel_rag(query: str) -> Dict[str, Any]:
    """Execute the parallel RAG workflow using the QueryProcessor class
    
    Args:
        query: The user's query string
        
    Returns:
        Dictionary containing all search results, answers, and performance metrics
    """
    # Initialize state
    state: State = {
        "query": query,
        "openai_response": "",
        "semantic_search_response": [],
        "vector_search_response": [],
        "combined_answer": "",
        "token_usage": {
            "openai_query": 0,
            "semantic_search": 0,
            "vector_search": 0,
            "combined_summary": 0,
            "total": 0
        }
    }

    # Check cache first
    cache_manager = CacheManager()
    cached_result = cache_manager.get_cached_result(query)
    if cached_result:
        print(f"Using cached result for query: {query}\n")
        # Update the result to indicate it's from cache
        cached_result["from_cache"] = True
        # Reset token usage to zero to indicate no API calls were made
        if "token_usage" in cached_result:
            cached_result["token_usage"] = {
                "openai_query": 0,
                "semantic_search": 0,
                "vector_search": 0,
                "combined_summary": 0,
                "total": 0,
                "note": "Retrieved from cache - no tokens used"
            }
        # Update execution time to current retrieval time (near zero)
        cached_result["execution_time"] = 0.001  # Negligible time for cache retrieval
        cached_result["cache_retrieval_time"] = time.time()
        return cached_result

    # Create a query processor instance for methods that require instance state
    # Note: query_openai is an instance method while query_semantic_search and query_vector_search are static methods
    query_processor = QueryProcessor()
    
    # Create tasks for parallel execution - use separate state copies to avoid race conditions
    tasks = [
        query_processor.query_openai(state.copy()),  # Instance method requires instance
        QueryProcessor.query_semantic_search(state.copy()),  # Static method called on class
        QueryProcessor.query_vector_search(state.copy())  # Static method called on class
    ]

    # Execute tasks in parallel
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    # Merge results from all tasks into a single state
    final_state = state.copy()
    for result_state in results:
        # Merge token usage
        for key, value in result_state["token_usage"].items():
            if key in final_state["token_usage"]:
                final_state["token_usage"][key] = value
        
        # Merge responses
        if "openai_response" in result_state and result_state["openai_response"]:
            final_state["openai_response"] = result_state["openai_response"]
            
        if "semantic_search_response" in result_state and result_state["semantic_search_response"]:
            final_state["semantic_search_response"] = result_state["semantic_search_response"]
            
        if "vector_search_response" in result_state and result_state["vector_search_response"]:
            final_state["vector_search_response"] = result_state["vector_search_response"]

    # Run the combine answers step sequentially after parallel tasks
    final_state = await QueryProcessor.combine_answers(final_state)

    # Add execution time
    execution_time = end_time - start_time
    final_state["execution_time"] = execution_time
    
    # Cache the result for future use
    cache_manager.cache_result(query, final_state)

    return final_state

# Define a function to print the results
def print_results(results: Dict[str, Any]) -> None:
    """Print the results from all three sources and save to a file"""
    # Create a file to save the complete results - append mode to keep history
    with open("diabetes_search_results.txt", "a", encoding="utf-8") as f:
        # Add clear separator and timestamp at the top of each query
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n\n{'#'*100}\n")
        f.write(f"TIMESTAMP: {current_time}\n")
        # Indicate if result is from cache
        if results.get("from_cache", False):
            f.write(f"SOURCE: Retrieved from cache\n")
        else:
            f.write(f"SOURCE: Fresh API call\n")
        f.write(f"{'#'*100}\n")
        f.write("\n" + "="*80 + "\n")
        f.write(f"QUERY: {results['query']}\n")
        f.write("="*80 + "\n")

        f.write("\n" + "-"*30 + " AZURE OPENAI DIRECT RESPONSE " + "-"*30 + "\n")
        f.write(results["openai_response"] + "\n")

        f.write("\n" + "-"*30 + " SEMANTIC SEARCH ANSWER (STANDARD INDEX) " + "-"*30 + "\n")
        semantic_response = results["semantic_search_response"]
        if isinstance(semantic_response, dict) and "answer" in semantic_response:
            f.write(semantic_response["answer"] + "\n")
        else:
            f.write("No answer generated from semantic search results.\n")

        f.write("\n" + "-"*30 + " VECTOR SEARCH ANSWER (VECTOR INDEX) " + "-"*30 + "\n")
        vector_response = results["vector_search_response"]
        if isinstance(vector_response, dict) and "answer" in vector_response:
            f.write(vector_response["answer"] + "\n")
        else:
            f.write("No answer generated from vector search results.\n")

        f.write("\n" + "-"*30 + " COMPREHENSIVE SUMMARY " + "-"*30 + "\n")
        if "combined_answer" in results and results["combined_answer"]:
            f.write(results["combined_answer"] + "\n")
        else:
            f.write("No comprehensive summary was generated.\n")

        f.write("\n" + "-"*30 + " PERFORMANCE " + "-"*30 + "\n")
        f.write(f"Total execution time: {results['execution_time']:.2f} seconds\n")

        if "token_usage" in results:
            token_usage = results["token_usage"]
            f.write("\n" + "-"*30 + " TOKEN USAGE " + "-"*30 + "\n")
            f.write(f"OpenAI Direct Query: {token_usage.get('openai_query', 0)} tokens\n")
            f.write(f"Semantic Search: {token_usage.get('semantic_search', 0)} tokens\n")
            f.write(f"Vector Search: {token_usage.get('vector_search', 0)} tokens\n")
            f.write(f"Comprehensive Summary: {token_usage.get('combined_summary', 0)} tokens\n")
            f.write(f"Total Tokens: {token_usage.get('total', 0)} tokens\n")

        f.write("="*80 + "\n")

    # Also print to console as before
    print("\n" + "="*80)
    print(f"QUERY: {results['query']}")
    print("="*80)

    print("\n" + "-"*30 + " AZURE OPENAI DIRECT RESPONSE " + "-"*30)
    # Print the full OpenAI response without truncation
    print(results["openai_response"])

    print("\n" + "-"*30 + " SEMANTIC SEARCH ANSWER (STANDARD INDEX) " + "-"*30)
    semantic_response = results["semantic_search_response"]
    if isinstance(semantic_response, dict):
        if "answer" in semantic_response:
            # Print the full semantic search answer without truncation
            print(semantic_response["answer"])
        else:
            print("No answer generated from semantic search results.")
    else:
        # Handle the case where it's still using the old format
        print("Semantic search did not generate an answer. Raw results:")
        for i, result in enumerate(semantic_response[:2]):
            print(f"\nResult {i+1}:")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Title: {result.get('title', 'N/A')}")
                content = result.get('content', 'N/A')
                print(f"Content: {content}")

    print("\n" + "-"*30 + " VECTOR SEARCH ANSWER (VECTOR INDEX) " + "-"*30)
    vector_response = results["vector_search_response"]
    if isinstance(vector_response, dict):
        if "answer" in vector_response:
            # Print the full vector search answer without truncation
            print(vector_response["answer"])
        else:
            print("No answer generated from vector search results.")
    else:
        # Handle the case where it's still using the old format
        print("Vector search did not generate an answer. Raw results:")
        for i, result in enumerate(vector_response[:2]):
            print(f"\nResult {i+1}:")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Title: {result.get('title', 'N/A')}")
                content = result.get('content', 'N/A')
                print(f"Content: {content}")

    print("\n" + "-"*30 + " COMPREHENSIVE SUMMARY " + "-"*30)
    if "combined_answer" in results and results["combined_answer"]:
        # Print the full combined answer without any truncation
        combined_answer = results["combined_answer"]
        print(combined_answer)
    else:
        print("No comprehensive summary was generated.")

    print("\n" + "-"*30 + " PERFORMANCE " + "-"*30)
    print(f"Total execution time: {results['execution_time']:.2f} seconds")

    # Print token usage statistics
    if "token_usage" in results:
        token_usage = results["token_usage"]
        print("\n" + "-"*30 + " TOKEN USAGE " + "-"*30)
        print(f"OpenAI Direct Query: {token_usage.get('openai_query', 0)} tokens")
        print(f"Semantic Search: {token_usage.get('semantic_search', 0)} tokens")
        print(f"Vector Search: {token_usage.get('vector_search', 0)} tokens")
        print(f"Comprehensive Summary: {token_usage.get('combined_summary', 0)} tokens")
        print(f"Total Tokens: {token_usage.get('total', 0)} tokens")

    print("="*80 + "\n")
    print("\nComplete results (including any truncated content) have been saved to 'diabetes_search_results.txt'")

async def main():
    """Main function to run the application"""
    import sys
    
    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--reset-cache":
        cache_manager = CacheManager()
        cache_manager.reset_cache()
        print("Cache has been reset. Exiting.")
        return
        
    print("\n" + "="*80)
    print("Welcome to the Parallel RAG Application with LangGraph and Azure Services")
    print("="*80)
    print("\nThis application processes your query through three parallel paths:")
    print("1. Direct query to Azure OpenAI (gpt-4o)")
    print("2. Semantic search using Azure Search Service (using standard index)")
    print("3. Vector search using Azure Search Service (using vector index)")

    try:
        # Check if running in a non-interactive environment
        import sys
        if not sys.stdin.isatty():
            # Use a default question in non-interactive mode
            print("\nRunning in non-interactive mode with default question.")
            default_question = "Can diabetes cause nerve damage?"
            print(f"\nProcessing default question: {default_question}")
            print("-"*80)

            # Run the parallel RAG workflow with the default question
            results = await run_parallel_rag(default_question)

            # Print the results
            print_results(results)
            return

        # Interactive mode
        continue_asking = True

        while continue_asking:
            # Ask the user for a medical question
            print("\n" + "-"*80)
            medical_question = ""
            while not medical_question.strip():
                try:
                    medical_question = input("Enter your medical question (or type 'exit' to quit): ")
                    if medical_question.lower().strip() == 'exit':
                        continue_asking = False
                        break
                    if not medical_question.strip():
                        print("Please enter a valid question.")
                except EOFError:
                    # Handle EOF error (non-interactive environment)
                    print("\nDetected non-interactive environment, using default question.")
                    medical_question = "Can diabetes cause nerve damage?"
                    continue_asking = False
                    break

            if not continue_asking and not medical_question.strip():
                break

            print("\nProcessing: " + medical_question)
            print("-"*80)

            # Run the parallel RAG workflow
            results = await run_parallel_rag(medical_question)

            # Print the results
            print_results(results)

            # Ask if the user wants to continue
            try:
                continue_response = input("\nDo you want to ask another question? (y/n): ").lower().strip()
                continue_asking = continue_response.startswith('y')
            except EOFError:
                # Handle EOF error (non-interactive environment)
                continue_asking = False

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

    print("\nThank you for using the Parallel RAG Application!")

if __name__ == "__main__":
    asyncio.run(main())
