"""Main script to run the RAG system."""

import os
import sys
import argparse
from pathlib import Path
from rag_system import RAGSystem, run_evaluation


def detect_environment():
    """Detect if running on Kaggle and return appropriate model path."""
    # Check for Kaggle environment
    kaggle_dataset_path = "/kaggle/input/mistralai-mistral-7b-instruct-v02"
    
    if os.path.exists("/kaggle") and os.path.exists(kaggle_dataset_path):
        print("🔍 Detected: Kaggle environment")
        print(f"📂 Using dataset: {kaggle_dataset_path}")
        return kaggle_dataset_path
    else:
        print("🔍 Detected: Local environment")
        print("🌐 Will use HuggingFace cache/download (requires HF token)")
        return None

def main():
    """Main function to orchestrate the RAG system."""
    
    parser = argparse.ArgumentParser(description="RAG System for SEC filing analysis")
    parser.add_argument("--mode", choices=["index", "query", "evaluate"], default="evaluate",
                       help="Mode of operation")
    parser.add_argument("--query", type=str, help="Query for query mode")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="LLM model name",
    )
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="Path to local model directory. Auto-detects Kaggle dataset if not specified. "
             "On local: uses HF cache. On Kaggle: uses /kaggle/input/mistralai-mistral-7b-instruct-v02"
    )
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--index-dir", type=str, default="./rag_index",
                       help="Directory for saving/loading index")
    
    args = parser.parse_args()
    
    # Auto-detect environment if local_model_path not specified
    if args.local_model_path is None:
        detected_path = detect_environment()
        args.local_model_path = detected_path
    
    # Initialize RAG system
    print("\n🚀 Initializing RAG system...")
    print(f"   Model: {args.model}")
    
    if args.mode == "index":
        print(f"   📝 Mode: Index only")
        print(f"   ⚡ LLM: Will NOT be loaded (not needed for indexing)")
    else:
        if args.local_model_path:
            print(f"   📂 LLM will load from: {args.local_model_path}")
            print(f"   🔓 HF Token: NOT REQUIRED")
        else:
            print(f"   🌐 LLM will load from: HuggingFace")
            print(f"   🔑 HF Token: REQUIRED (from .env.txt)")
        print(f"   ⚡ LLM: Lazy loading (loads on first query)")
    
    rag = RAGSystem(
        model_name=args.model,
        embedding_model=args.embedding_model,
        use_reranker=True,
        local_model_path=args.local_model_path
    )
    
    # Define documents
    documents = [
        {
            "path": "10-Q4-2024-As-Filed.pdf",
            "name": "Apple 10-K"
        },
        {
            "path": "tsla-20231231-gen.pdf",
            "name": "Tesla 10-K"
        }
    ]
    
    # Check if documents exist
    for doc in documents:
        if not os.path.exists(doc["path"]):
            print(f"Warning: {doc['path']} not found")
    
    if args.mode == "index":
        # Index mode
        print("\nMode: Index and save")
        rag.ingest_documents(documents)
        rag.save_index(args.index_dir)
        print(f"Index saved to {args.index_dir}")
        
    elif args.mode == "query":
        # Query mode
        print("\nMode: Query")
        if not args.query:
            print("Error: --query required for query mode")
            sys.exit(1)
        
        # Load index if it exists
        if os.path.exists(args.index_dir):
            rag.load_index(args.index_dir)
        else:
            print("Index not found. Indexing documents...")
            rag.ingest_documents(documents)
            rag.save_index(args.index_dir)
        
        # Answer the query
        result = rag.answer_question(args.query)
        print(f"\nQuestion: {args.query}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        
    elif args.mode == "evaluate":
        # Evaluation mode (default)
        print("\nMode: Evaluate (answer all 13 test questions)")
        
        # Load index if it exists
        if os.path.exists(args.index_dir):
            print(f"Loading index from {args.index_dir}...")
            rag.load_index(args.index_dir)
        else:
            print("Index not found. Indexing documents...")
            rag.ingest_documents(documents)
            rag.save_index(args.index_dir)
        
        # Run evaluation
        results = run_evaluation(rag)
        
        return results

if __name__ == "__main__":
    main()
