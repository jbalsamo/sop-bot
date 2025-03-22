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

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    # Fallback method if python-dotenv is not available
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

    load_env_vars()

system_prompt = """

"""

# Azure OpenAI configuration
AZ_API_KEY = os.getenv("AZ_API_KEY")
AZ_BASE_URL = os.getenv("AZ_BASE_URL")
AZ_DEPLOYMENT_NAME = os.getenv("AZ_DEPLOYMENT_NAME")

# Azure Search configuration
AZ_SEARCH_KEY = os.getenv("AZ_SEARCH_KEY")
AZ_SEARCH_ENDPOINT = os.getenv("AZ_SEARCH_ENDPOINT")
AZ_INDEX_NAME = os.getenv("AZ_INDEX_NAME")
AZ_PM_VECTOR_INDEX_NAME = os.getenv("AZ_PM_VECTOR_INDEX_NAME")

# Define the state schema
class State(TypedDict):
    query: str
    openai_response: str
    semantic_search_response: List[Dict[str, Any]]
    vector_search_response: List[Dict[str, Any]]
    combined_answer: str
    token_usage: Dict[str, int]

# Initialize LangChain Azure OpenAI client
azure_llm = AzureChatOpenAI(
    azure_deployment=AZ_DEPLOYMENT_NAME,
    openai_api_version="2023-12-01-preview",
    openai_api_key=AZ_API_KEY,
    azure_endpoint=AZ_BASE_URL,
    temperature=0.7,
    max_tokens=1000,  # Increased to 1000 to prevent truncation
    verbose=True,  # Enable verbose mode for debugging
    streaming=False  # Disable streaming to ensure we get token counts
)

# Initialize Azure OpenAI Embeddings
azure_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",  # Adjust if using a different embedding model
    openai_api_version="2023-12-01-preview",
    openai_api_key=AZ_API_KEY,
    azure_endpoint=AZ_BASE_URL
)

# Initialize Azure Search credential
search_credential = AzureKeyCredential(AZ_SEARCH_KEY)

# Initialize search clients
semantic_search_client = SearchClient(
    endpoint=AZ_SEARCH_ENDPOINT,
    index_name=AZ_INDEX_NAME,
    credential=search_credential
)

vector_search_client = SearchClient(
    endpoint=AZ_SEARCH_ENDPOINT,
    index_name=AZ_PM_VECTOR_INDEX_NAME,
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

# Define the functions for each parallel task
async def query_openai(state: State) -> State:
    """Query Azure OpenAI with the user's question using LangChain and a specialized medical prompt"""
    try:
        print("Querying Azure OpenAI with specialized medical prompt")

        # Format the query for token counting
        query_text = state["query"]

        # Get response from Azure OpenAI
        response = await llm_chain.ainvoke({"query": query_text})

        # Track token usage using tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        # Count tokens in prompt (approximate using the system prompt + query)
        system_prompt = "You are an experienced medical professional providing accurate, accessible health information."
        prompt_tokens = len(enc.encode(system_prompt + query_text))
        # Count tokens in response
        completion_tokens = len(enc.encode(response))
        total_tokens = prompt_tokens + completion_tokens

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

async def query_semantic_search(state: State) -> State:
    """Perform semantic search using Azure Search Service and generate an answer using PromptTemplate"""
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            print(f"Performing semantic search using index: {AZ_INDEX_NAME} (attempt {attempt+1}/{max_retries})")

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

                    # Get answer from Azure OpenAI
                    answer_response = await azure_llm.ainvoke(formatted_prompt)
                    answer_text = answer_response.content

                    # Track token usage using tiktoken
                    enc = tiktoken.encoding_for_model("gpt-4")
                    # Count tokens in prompt
                    prompt_tokens = len(enc.encode(str(formatted_prompt)))
                    # Count tokens in response
                    completion_tokens = len(enc.encode(answer_text))
                    total_tokens = prompt_tokens + completion_tokens

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

async def query_vector_search(state: State) -> State:
    """Perform vector search using Azure Search Service with the vector index and generate an answer using PromptTemplate"""
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            print(f"Performing vector search using index: {AZ_PM_VECTOR_INDEX_NAME} (attempt {attempt+1}/{max_retries})")

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

                    # Get answer from Azure OpenAI
                    answer_response = await azure_llm.ainvoke(formatted_prompt)
                    answer_text = answer_response.content

                    # Track token usage using tiktoken
                    enc = tiktoken.encoding_for_model("gpt-4")
                    # Count tokens in prompt
                    prompt_tokens = len(enc.encode(str(formatted_prompt)))
                    # Count tokens in response
                    completion_tokens = len(enc.encode(answer_text))
                    total_tokens = prompt_tokens + completion_tokens

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

async def combine_answers(state: State) -> State:
    """Combine the answers from all three sources into a comprehensive summary"""
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

        # Get combined answer from Azure OpenAI
        combined_response = await azure_llm.ainvoke(formatted_prompt)
        combined_answer = combined_response.content

        # Track token usage using tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        # Count tokens in prompt
        prompt_tokens = len(enc.encode(str(formatted_prompt)))
        # Count tokens in response
        completion_tokens = len(enc.encode(combined_answer))
        total_tokens = prompt_tokens + completion_tokens

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
    """Execute the parallel RAG workflow"""
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

    # Create tasks for parallel execution
    tasks = [
        query_openai(state),
        query_semantic_search(state),
        query_vector_search(state)
    ]

    # Execute tasks in parallel
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    # Merge results (all tasks modify the same state object, but we'll take the last one to be safe)
    final_state = results[-1]

    # Run the combine answers step sequentially after parallel tasks
    final_state = await combine_answers(final_state)

    # Add execution time
    execution_time = end_time - start_time
    final_state["execution_time"] = execution_time

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
