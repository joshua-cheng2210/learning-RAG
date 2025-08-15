#!/usr/bin/env python3
"""
Main script to run the RAG system for Alice in Wonderland quiz.
This script demonstrates how to use the DatabaseManager and QueryEngine classes.
"""

import argparse
import json
import os
import pathlib
from database_manager import DatabaseManager
from query_engine import QueryEngine


# Centralized model options
EMBEDDING_MODEL_OPTIONS = [
    "sentence-transformers/all-MiniLM-L6-v2", # success
    "sentence-transformers/all-mpnet-base-v2", # success
    "BAAI/bge-large-en", # success
    "intfloat/e5-base-v2", # success
    "sentence-transformers/static-retrieval-mrl-en-v1", # success
    "sentence-transformers/all-MiniLM-L12-v2", # 5 # success # best one so far
    # "gemini/embedding-001",       # Older Gemini model
    "gemini/text-embedding-005",  # New Gemini model
]

# Add model types to distinguish between HuggingFace and Gemini
EMBEDDING_MODEL_TYPES = [
    "huggingface",  # 0
    "huggingface",  # 1
    "huggingface",  # 2
    "huggingface",  # 3
    "huggingface",  # 4
    "huggingface",  # 5
    "gemini",       # 6 - New
    # "gemini",       # 7 - New
]

TEXT_GENERATION_MODEL_OPTIONS = [
    "google/flan-t5-small",
    "google/flan-t5-base", # have been using this for default development testing
    "google/flan-t5-large",
    "google/flan-t5-xl", # 3
    "gpt2",
    "distilgpt2"
]


def list_models():
    """List all available models."""
    print("Available Embedding Models:")
    for i, (model, model_type) in enumerate(zip(EMBEDDING_MODEL_OPTIONS, EMBEDDING_MODEL_TYPES)):
        print(f"  {i}: {model} ({model_type})")
    
    print("\nAvailable Text Generation Models:")
    for i, model in enumerate(TEXT_GENERATION_MODEL_OPTIONS):
        print(f"  {i}: {model}")


def main():
    parser = argparse.ArgumentParser(description="RAG system for Alice in Wonderland quiz")
    parser.add_argument("--mode", choices=["create", "query", "quiz", "list-models"], default="quiz",
                       help="Mode: create database, query single question, run quiz, or list models")
    parser.add_argument("--question", type=str, help="Single question to query")
    parser.add_argument("--embedding-model", type=int, default=0, 
                       help=f"Embedding model index (0-{len(EMBEDDING_MODEL_OPTIONS)-1})")
    parser.add_argument("--text-model", type=int, default=1,
                       help=f"Text generation model index (0-{len(TEXT_GENERATION_MODEL_OPTIONS)-1})")
    parser.add_argument("--limit", type=int, help="Limit number of quiz questions")
    
    args = parser.parse_args()
    
    # Validate model indices
    if args.embedding_model < 0 or args.embedding_model >= len(EMBEDDING_MODEL_OPTIONS):
        print(f"Error: Embedding model index must be between 0 and {len(EMBEDDING_MODEL_OPTIONS)-1}")
        list_models()
        return
    
    if args.text_model < 0 or args.text_model >= len(TEXT_GENERATION_MODEL_OPTIONS):
        print(f"Error: Text generation model index must be between 0 and {len(TEXT_GENERATION_MODEL_OPTIONS)-1}")
        list_models()
        return
    
    # Get selected models
    embedding_model = EMBEDDING_MODEL_OPTIONS[args.embedding_model]
    embedding_model_type = EMBEDDING_MODEL_TYPES[args.embedding_model]
    text_model = TEXT_GENERATION_MODEL_OPTIONS[args.text_model]
    db_data_path = f"chroma/{embedding_model.split('/')[-1].replace('/', '_').replace('-', '_')}"
    result_file_path = f"quiz_results/{embedding_model.split('/')[-1].replace('/', '_').replace('-', '_')}--{text_model.split('/')[-1].replace('/', '_').replace('-', '_')}_quiz_results.json"
    raw_knowledge_directory = "books"

    if args.mode == "list-models":
        list_models()
        return
    
    print(f"Using embedding model: {embedding_model} ({embedding_model_type})")
    print(f"Using text generation model: {text_model}")

    def create_mode():
        print("Creating database...")
        db_manager = DatabaseManager(embedding_model_name=embedding_model, 
                                   embedding_model_type=embedding_model_type)
        success = db_manager.generate_data_store(data_path=raw_knowledge_directory, 
                                                persist_directory=db_data_path)
        
        if success:
            print("\n✓ Database created successfully!")
        else:
            print("\n✗ Failed to create database.")
    
    def query_mode():
        if not args.question:
            print("Please provide a question using --question")
            return
        
        print(f"Querying: {args.question}")
        query_engine = QueryEngine(persist_directory=db_data_path,
                                 embedding_model_name=embedding_model,
                                 embedding_model_type=embedding_model_type,
                                 text_model_name=text_model)
        
        result = query_engine.query_single_question(args.question, show_context=True)
        
        print("\n" + "="*50)
        print(f"Question: {result['question']}")
        print(f"Response: {result['response']}")
        print(f"Sources: {', '.join(result['sources'])}")
        if 'context' in result:
            print(f"\nContext:\n{result['context'][:500]}...")
    
    def quiz_mode():
        print("Running Alice in Wonderland quiz...")
        query_engine = QueryEngine(persist_directory=db_data_path,
                                 embedding_model_name=embedding_model,
                                 embedding_model_type=embedding_model_type,
                                 text_model_name=text_model)
        
        # Run the quiz
        results = query_engine.run_quiz("test_questions.json", limit=args.limit)
        
        # Rest of quiz_mode code remains the same...
        if results:
            print("\n" + "="*50)
            print("DETAILED ANALYSIS")
            print("="*50)
            print("Embedding Model:", embedding_model)
            print("Text Generation Model:", text_model)

            correct_results = [r for r in results if r['is_correct']]
            incorrect_results = [r for r in results if not r['is_correct']]
            
            if incorrect_results:
                print(f"\nIncorrect answers ({len(incorrect_results)}):")
                for result in incorrect_results:
                    print(f"Q{result['id']}: Expected {result['correct_answer']}, got {result['response']}")
            
            if correct_results:
                print(f"\nCorrect answers: {len(correct_results)}")

            with open(result_file_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Quiz results saved to {result_file_path}")
    
    if args.mode == "create":
        create_mode()

    if args.mode == "query":
        if os.path.exists(db_data_path):
            query_mode()
        else:
            create_mode()
            query_mode()
        
    if args.mode == "quiz":
        if os.path.exists(db_data_path):
            quiz_mode()
        else:
            create_mode()
            quiz_mode()


if __name__ == "__main__":
    main()