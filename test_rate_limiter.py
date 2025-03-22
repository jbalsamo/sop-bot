#!/usr/bin/env python3
"""
Test script for rate limiting functionality in the diabetes search application.
This script sends multiple queries in parallel to test the rate limiter.
"""

import asyncio
import os
import time
from app import run_parallel_rag

# Test queries - using fewer queries for faster testing
TEST_QUERIES = [
    "What are the early symptoms of diabetes?",
    "Can diabetes cause heart problems?",
    "How does diabetes affect the kidneys?",
    "What is the relationship between diabetes and obesity?"
]

async def main():
    """Run multiple queries in parallel to test rate limiting"""
    print("Using existing Azure clients from app.py...")
    # The Azure clients are already initialized in app.py
    
    print(f"Starting test with {len(TEST_QUERIES)} queries...")
    print("This test will demonstrate rate limiting by sending multiple queries in parallel.")
    print("You should see the rate limiter pause execution when limits are reached.")
    
    # Process queries in parallel
    start_time = time.time()
    tasks = [run_parallel_rag(query) for query in TEST_QUERIES]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Print summary
    total_time = end_time - start_time
    total_tokens = sum(result["token_usage"]["total"] for result in results)
    
    print("\n" + "="*80)
    print(f"Test completed in {total_time:.2f} seconds")
    print(f"Total tokens used: {total_tokens}")
    print(f"Average time per query: {total_time/len(TEST_QUERIES):.2f} seconds")
    print(f"Average tokens per query: {total_tokens/len(TEST_QUERIES):.2f}")
    print("="*80)

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["AZ_REQUESTS_PER_MINUTE"] = "5"  # Very low limit for testing
    os.environ["AZ_TOKENS_PER_MINUTE"] = "5000"  # Very low limit for testing
    
    print("\nNOTE: Using artificially low rate limits for testing:")
    print("- 5 requests per minute")
    print("- 5000 tokens per minute")
    print("This will force the rate limiter to activate during our test.\n")
    
    # Run the test
    asyncio.run(main())
