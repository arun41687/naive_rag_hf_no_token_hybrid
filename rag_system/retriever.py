"""Retrieval and re-ranking module."""

from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder
from rag_system.ingestion import VectorStore
from rag_system.advanced_reranker import StopwordKeywordReranker

class RetrieverWithReranker:
    """Retrieves and re-ranks relevant chunks with advanced keyword matching."""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        use_reranker: bool = True,
        use_advanced_reranker: bool = True
    ):
        """
        Initialize retriever with optional re-ranking.
        
        Args:
            vector_store: VectorStore instance
            use_reranker: Whether to use cross-encoder for re-ranking
            use_advanced_reranker: Whether to use advanced stopword+keyword reranker
        """
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        self.use_advanced_reranker = use_advanced_reranker
        
        if use_reranker:
            if use_advanced_reranker:
                # Advanced three-stage reranker with stopword removal
                print("Using advanced stopword+keyword reranker")
                self.advanced_reranker = StopwordKeywordReranker()
            else:
                # Original cross-encoder only
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True) -> List[Dict]:
        """
        Retrieve relevant chunks with optional reranking.
        
        Args:
            query: Search query
            top_k: Number of results to return
            rerank: Whether to apply re-ranking
            
        Returns:
            List of relevant chunks with scores
        """
        # Initial retrieval with more candidates for re-ranking
        # Use 6x prefetch for advanced reranker (matches rag_pipeline.py strategy)
        if rerank and self.use_reranker and self.use_advanced_reranker:
            initial_k = top_k * 6
        else:
            initial_k = top_k * 3 if rerank and self.use_reranker else top_k
        
        results = self.vector_store.search(query, k=initial_k)
        
        if rerank and self.use_reranker:
            if self.use_advanced_reranker:
                # Advanced three-stage reranking
                chunks = [r[0] for r in results]
                reranked_chunks = self.advanced_reranker.rerank(
                    query=query,
                    chunks=chunks,
                    top_k=top_k,
                    keyword_filter_threshold=0.05,
                    hybrid_weight=0.3  # 30% keyword, 70% cross-encoder
                )
                
                # Add 'score' field for consistency
                for chunk in reranked_chunks:
                    if 'score' not in chunk:
                        chunk['score'] = chunk.get('rerank_score', 0.0)
                
                return reranked_chunks
            else:
                # Original cross-encoder only reranking
                pairs = [[query, r[0]["text"]] for r in results]
                scores = self.reranker.predict(pairs)
                chunks = [r[0] for r in results]
                
                # Sort by cross-encoder scores (higher is better)
                ranked = sorted(
                    zip(chunks, scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Return top_k
                return [
                    {
                        **chunk,
                        "score": float(score)
                    }
                    for chunk, score in ranked[:top_k]
                ]
        else:
            # Return results sorted by distance
            return [
                {
                    **result[0],
                    "score": 1.0 / (1.0 + result[1])  # Convert distance to similarity
                }
                for result in results[:top_k]
            ]
    
    def retrieve_diverse(self, query: str, top_k: int = 5, max_per_page: int = 2) -> List[Dict]:
        """
        Retrieve diverse chunks with page diversity constraint.
        Prevents over-reliance on a single page by limiting chunks per page.
        
        Args:
            query: Search query
            top_k: Number of results to return
            max_per_page: Maximum chunks allowed from same page
            
        Returns:
            List of diverse relevant chunks with scores
        """
        # Get more candidates than needed for diversity filtering
        candidates = self.retrieve(query, top_k=top_k * 2, rerank=True)
        
        selected = []
        page_count = {}
        
        for chunk in candidates:
            page = chunk.get('page', 0)
            current_count = page_count.get(page, 0)
            
            # Add chunk if under page limit
            if current_count < max_per_page:
                selected.append(chunk)
                page_count[page] = current_count + 1
            
            # Stop when we have enough diverse chunks
            if len(selected) >= top_k:
                break
        
        return selected
    
    @staticmethod
    def format_sources(chunks: List[Dict]) -> List[str]:
        """Format chunks into source citations."""
        sources = []
        for chunk in chunks:
            sources.append(
                f"{chunk['document']}, p. {chunk['page']}"
            )
        return list(set(sources))  # Remove duplicates
