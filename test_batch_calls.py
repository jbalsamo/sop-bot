#!/usr/bin/env python
"""
Batch Test Script for Parallel RAG Application
This script runs a batch of medical questions through the application to test functionality.
"""

import asyncio
import time
import sys
from app import run_parallel_rag, CacheManager

# List of medical questions about ticks and diabetes
MEDICAL_QUESTIONS = [
    # Diabetes questions
    "What are the early warning signs of diabetes?",
    "Can diabetes affect your vision?",
    "Is type 2 diabetes reversible with diet and exercise?",
    "What foods should diabetics avoid?",
    "How often should diabetics check their blood sugar?",
    
    # Tick questions
    "How do I safely remove a tick?",
    "What diseases can ticks transmit to humans?",
    "How long does a tick need to be attached to transmit Lyme disease?",
    "What are the symptoms of a tick-borne illness?",
    "Do all ticks carry diseases?"
]

async def run_batch_test(questions=None, num_questions=5):
    """Run a batch of questions through the application
    
    Args:
        questions: List of questions to process (default: MEDICAL_QUESTIONS)
        num_questions: Number of questions to process (default: 5)
    """
    if questions is None:
        questions = MEDICAL_QUESTIONS
    
    # Select a subset of questions if needed
    if num_questions < len(questions):
        import random
        test_questions = random.sample(questions, num_questions)
    else:
        test_questions = questions[:num_questions]
    
    print(f"\n{'='*80}")
    print(f"RUNNING BATCH TEST WITH {len(test_questions)} QUESTIONS")
    print(f"{'='*80}")
    
    total_start_time = time.time()
    
    # Process each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'-'*80}")
        print(f"QUESTION {i}/{len(test_questions)}: {question}")
        print(f"{'-'*80}")
        
        # Process the question
        start_time = time.time()
        result = await run_parallel_rag(question)
        end_time = time.time()
        
        # Print summary of results
        print(f"\nProcessed in {end_time - start_time:.2f} seconds")
        print(f"Token usage: {result['token_usage']['total']} tokens")
        
        # Add a delay between questions to avoid rate limiting issues
        if i < len(test_questions):
            print("\nWaiting 2 seconds before next question...")
            await asyncio.sleep(2)
    
    total_end_time = time.time()
    print(f"\n{'='*80}")
    print(f"BATCH TEST COMPLETED")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Average time per question: {(total_end_time - total_start_time) / len(test_questions):.2f} seconds")
    print(f"{'='*80}")

async def main():
    """Main function to run the batch test"""
    # Check for command-line arguments
    if len(sys.argv) > 1:
        try:
            num_questions = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of questions: {sys.argv[1]}")
            print(f"Using default: 5 questions")
            num_questions = 5
    else:
        num_questions = 5
    
    # Check if we should reset the cache first
    if "--reset-cache" in sys.argv:
        print("Resetting cache before running batch test...")
        cache_manager = CacheManager()
        cache_manager.reset_cache()
    
    # Run the batch test
    await run_batch_test(num_questions=num_questions)

if __name__ == "__main__":
    asyncio.run(main())
