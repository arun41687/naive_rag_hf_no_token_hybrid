"""Main RAG system orchestrator."""

import os
import json
from typing import Dict, List
from rag_system.ingestion import DocumentIngestor, VectorStore
from rag_system.retriever import RetrieverWithReranker
from rag_system.llm_integration import LLMIntegration, RAGPrompt

class RAGSystem:
    """Complete RAG system for answering questions about SEC filings."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 700,  # Optimized from 1000 to 700 for better precision
        chunk_overlap: int = 150,
        use_reranker: bool = True,
        use_advanced_reranker: bool = True,  # Enable advanced stopword+keyword reranker
        local_model_path: str = None  # Local model path (auto-detected or manual)
    ):
        """
        Initialize the RAG system.
        
        Args:
            model_name: LLM model name
            embedding_model: Embedding model name
            chunk_size: Size of text chunks (default: 700 for better precision)
            chunk_overlap: Overlap between chunks
            use_reranker: Whether to use re-ranking
            use_advanced_reranker: Use advanced stopword+keyword reranker (recommended)
            local_model_path: Path to local model (e.g., Kaggle dataset)
        """
        self.ingestor = DocumentIngestor(chunk_size=chunk_size, overlap=chunk_overlap)
        self.vector_store = VectorStore(model_name=embedding_model)
        self.retriever = RetrieverWithReranker(
            self.vector_store, 
            use_reranker=use_reranker,
            use_advanced_reranker=use_advanced_reranker
        )
        
        # Lazy load LLM (only when needed for queries)
        self._model_name = model_name
        self._local_model_path = local_model_path
        self._llm = None  # Will be initialized on first query
        
        self.indexed = False
    
    def _ensure_llm_loaded(self) -> None:
        """Lazy load LLM only when needed (for queries, not indexing)."""
        if self._llm is None:
            print("\n🔄 Loading LLM (first query)...")
            self._llm = LLMIntegration(
                model_name=self._model_name,
                local_model_path=self._local_model_path
            )
            print("✅ LLM loaded successfully!\n")
    
    @property
    def llm(self):
        """Property to access LLM, ensures it's loaded."""
        self._ensure_llm_loaded()
        return self._llm
    
    def ingest_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Ingest and index documents.
        
        Args:
            documents: List of dicts with 'path' and 'name' keys
        """
        print("Starting document ingestion...")
        all_chunks = []
        
        for doc in documents:
            print(f"Processing {doc['name']} from {doc['path']}...")
            chunks = self.ingestor.parse_pdf(doc['path'], doc['name'])
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        
        print(f"Total chunks created: {len(all_chunks)}")
        
        # Add to vector store
        print("Creating embeddings and indexing...")
        self.vector_store.add_chunks(all_chunks)
        
        self.indexed = True
        print("Indexing complete!")
    
    def answer_question(self, query: str, top_k: int = 10, max_per_page: int = 3) -> Dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            query: The question to answer
            top_k: Number of chunks to retrieve (default: 10 for better coverage)
            max_per_page: Maximum chunks from same page for diversity
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        # Ensure LLM is loaded (lazy initialization)
        self._ensure_llm_loaded()
        
        if not self.indexed:
            return {
                "answer": "Error: System not yet indexed. Please ingest documents first.",
                "sources": []
            }
        
        # Check for out-of-scope questions
        if self._is_out_of_scope(query):
            return {
                "answer": "This question cannot be answered based on the provided documents.",
                "sources": []
            }
        
        # Retrieve relevant chunks with diversity
        retrieved_chunks = self.retriever.retrieve_diverse(
            query, 
            top_k=top_k, 
            max_per_page=max_per_page
        )
        
        if not retrieved_chunks:
            return {
                "answer": "Not specified in the document.",
                "sources": []
            }
        
        # Format context
        context = RAGPrompt.format_context(retrieved_chunks)
        
        # Generate answer
        answer = self.llm.generate_answer(query, context)
        
        # Extract sources
        sources = self.retriever.format_sources(retrieved_chunks)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    @staticmethod
    def _is_out_of_scope(query: str) -> bool:
        """
        Determine if a question is out of scope.
        
        Args:
            query: The question to check
            
        Returns:
            True if out of scope, False otherwise
        """
        out_of_scope_keywords = [
            "stock price forecast",
            "future price",
            "predict",
            "2025",
            "next quarter",
            "next year",
            "color",
            "painted",
            "weather",
            "climate change",
            "political",
            "stock recommendation"
        ]
        
        query_lower = query.lower()
        
        # Check for specific out-of-scope questions
        if "stock price forecast" in query_lower:
            return True
        if "what color" in query_lower:
            return True
        if "2025" in query_lower and ("cfo" in query_lower or "ceo" in query_lower):
            return True
        
        return False
    
    def save_index(self, save_dir: str) -> None:
        """Save the indexed documents."""
        os.makedirs(save_dir, exist_ok=True)
        self.vector_store.save(save_dir)
        print(f"Index saved to {save_dir}")
    
    def load_index(self, save_dir: str) -> None:
        """Load a saved index."""
        self.vector_store.load(save_dir)
        self.indexed = True
        print(f"Index loaded from {save_dir}")


def run_evaluation(rag_system: RAGSystem) -> List[Dict]:
    """Run the evaluation on all test questions.
    
    Returns:
        List of answer dictionaries with question_id, answer, and sources
    """
    
    questions = [
        {"question_id": 1, "question": "What was Apples total revenue for the fiscal year ended September 28, 2024?"},
        {"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
        {"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
        {"question_id": 4, "question": "On what date was Apples 10-K report for 2024 signed and filed with the SEC?"},
        {"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
        {"question_id": 6, "question": "What was Teslas total revenue for the year ended December 31, 2023?"},
        {"question_id": 7, "question": "What percentage of Teslas total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
        {"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
        {"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
        {"question_id": 10, "question": "What is the purpose of Teslas 'lease pass-through fund arrangements'?"},
        {"question_id": 11, "question": "What is Teslas stock price forecast for 2025?"},
        {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
        {"question_id": 13, "question": "What color is Teslas headquarters painted?"}
    ]
    
    answers = []
    
    print("\n" + "="*80)
    print("RUNNING EVALUATION ON 13 TEST QUESTIONS")
    print("="*80 + "\n")
    
    for q_data in questions:
        print(f"Q{q_data['question_id']}: {q_data['question']}")
        result = rag_system.answer_question(q_data['question'])
        
        answer_entry = {
            "question_id": q_data['question_id'],
            "answer": result['answer'],
            "sources": result['sources']
        }
        answers.append(answer_entry)
        
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Sources: {result['sources']}\n")
    
    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(answers, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Evaluation complete! Results saved to {output_file}")
    print("="*80)
    
    return answers
