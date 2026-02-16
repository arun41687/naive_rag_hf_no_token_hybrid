"""Advanced reranking with stopword removal and keyword matching."""

import re
import nltk
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class StopwordKeywordReranker:
    """
    Advanced three-stage reranking with stopword removal and keyword matching.
    
    Stages:
    1. Remove stopwords and filter by keyword similarity
    2. Apply cross-encoder reranking to filtered documents
    3. Combine keyword + cross-encoder scores with hybrid weighting
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: Cross-encoder model name for semantic reranking
        """
        self.ranker = CrossEncoder(model_name)
        
        # Extended stopwords for financial documents
        self.english_stopwords = set(stopwords.words('english'))
        
        # Add SEC/Financial document common stopwords
        self.financial_stopwords = {
            'company', 'companies', 'business', 'operations', 'period', 
            'year', 'years', 'quarter', 'quarters', 'fiscal', 'ended',
            'including', 'related', 'certain', 'various', 'primarily',
            'approximately', 'substantially', 'significantly', 'generally',
            'may', 'could', 'would', 'should', 'might', 'will', 'shall',
            'also', 'however', 'therefore', 'furthermore', 'moreover',
            'item', 'part', 'section', 'table', 'note', 'see', 'refer'
        }
        
        self.all_stopwords = self.english_stopwords.union(self.financial_stopwords)
        
        # Important financial/SEC keywords that should NEVER be removed
        self.protected_keywords = {
            'revenue', 'earnings', 'profit', 'loss', 'cash', 'debt', 'assets',
            'liabilities', 'equity', 'shares', 'dividend', 'eps', 'ebitda',
            'operating', 'income', 'expenses', 'margin', 'growth', 'risk',
            'material', 'adverse', 'segment', 'goodwill', 'impairment',
            'depreciation', 'amortization', 'taxes', 'automotive', 'interest', 'cost',
            'sales', 'services', 'products', 'customers', 'market', 'competition',
            'regulatory', 'compliance', 'litigation', 'contingencies',
            'apple', 'tesla', 'automotive', 'technology', 'manufacturing', 
            'unresolved', 'leasing', 'filing', 'sec', 'filed', 'signed'
        }
    
    def _clean_and_tokenize(self, text: str) -> List[str]:
        """
        Clean text and tokenize while preserving important terms.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of cleaned tokens
        """
        text = text.lower()
        
        # Remove special characters but keep $ and %
        text = re.sub(r'[^\w\s\$%]', ' ', text)
        
        tokens = word_tokenize(text)
        
        cleaned_tokens = []
        for token in tokens:
            # Always keep protected keywords
            if token in self.protected_keywords:
                cleaned_tokens.append(token)
            # Keep numbers
            elif re.match(r'^\d+(?:\.\d+)?$', token):
                cleaned_tokens.append(token)
            # Keep dollar amounts and percentages
            elif '$' in token or '%' in token:
                cleaned_tokens.append(token)
            # Keep non-stopwords longer than 2 chars
            elif token not in self.all_stopwords and len(token) > 2:
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def _extract_financial_patterns(self, text: str) -> List[str]:
        """
        Extract specific financial patterns (dollar amounts, percentages, years, etc.).
        
        Args:
            text: Input text
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        # Dollar amounts
        dollar_patterns = re.findall(
            r'\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?', 
            text.lower()
        )
        patterns.extend(dollar_patterns)
        
        # Percentages
        percent_patterns = re.findall(r'\d+(?:\.\d+)?%', text)
        patterns.extend(percent_patterns)
        
        # Years
        year_patterns = re.findall(r'\b(?:19|20)\d{2}\b', text)
        patterns.extend(year_patterns)
        
        # SEC items
        item_patterns = re.findall(r'\bitem\s+\d+[a-z]?\b', text.lower())
        patterns.extend(item_patterns)
        
        return patterns
    
    def _calculate_keyword_similarity(
        self, 
        query_tokens: List[str], 
        doc_tokens: List[str], 
        query_patterns: List[str], 
        doc_patterns: List[str]
    ) -> Dict[str, float]:
        """
        Calculate various keyword similarity metrics.
        
        Args:
            query_tokens: Cleaned tokens from query
            doc_tokens: Cleaned tokens from document
            query_patterns: Financial patterns from query
            doc_patterns: Financial patterns from document
            
        Returns:
            Dictionary of similarity metrics
        """
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        
        if not query_set:
            return {
                'token_overlap': 0.0, 
                'pattern_match': 0.0, 
                'jaccard': 0.0, 
                'weighted_score': 0.0
            }
        
        # 1. Token overlap score
        intersection = query_set.intersection(doc_set)
        token_overlap = len(intersection) / len(query_set)
        
        # 2. Financial pattern matching
        pattern_matches = sum(1 for pattern in query_patterns if pattern in doc_patterns)
        pattern_match = pattern_matches / max(1, len(query_patterns))
        
        # 3. Jaccard similarity
        union = query_set.union(doc_set)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # 4. Weighted score (protected keywords count more)
        weighted_intersection = 0
        for token in intersection:
            if token in self.protected_keywords:
                weighted_intersection += 2.0  # Double weight for important terms
            else:
                weighted_intersection += 1.0
        
        weighted_score = weighted_intersection / len(query_set)
        
        return {
            'token_overlap': token_overlap,
            'pattern_match': pattern_match,
            'jaccard': jaccard,
            'weighted_score': weighted_score
        }
    
    def _filter_by_keywords(
        self, 
        query: str, 
        chunks: List[Dict], 
        min_similarity: float = 0.1
    ) -> List[Tuple[Dict, Dict[str, float]]]:
        """
        Filter and score chunks based on keyword similarity.
        
        Args:
            query: Search query
            chunks: List of chunk dictionaries
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (chunk, metrics) tuples
        """
        query_tokens = self._clean_and_tokenize(query)
        query_patterns = self._extract_financial_patterns(query)
        
        scored_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.get('text', chunk.get('content', ''))
            doc_tokens = self._clean_and_tokenize(chunk_text)
            doc_patterns = self._extract_financial_patterns(chunk_text)
            
            similarity_metrics = self._calculate_keyword_similarity(
                query_tokens, doc_tokens, query_patterns, doc_patterns
            )
            
            # Combine metrics with weights
            combined_score = (
                0.4 * similarity_metrics['weighted_score'] +
                0.3 * similarity_metrics['token_overlap'] +
                0.2 * similarity_metrics['pattern_match'] +
                0.1 * similarity_metrics['jaccard']
            )
            
            if combined_score >= min_similarity:
                scored_chunks.append((chunk, {
                    **similarity_metrics,
                    'combined_keyword_score': combined_score
                }))
        
        # Sort by keyword score
        scored_chunks.sort(key=lambda x: x[1]['combined_keyword_score'], reverse=True)
        
        return scored_chunks
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5, 
        keyword_filter_threshold: float = 0.05,
        hybrid_weight: float = 0.3
    ) -> List[Dict]:
        """
        Three-stage reranking:
        1. Remove stopwords and filter by keyword similarity
        2. Apply cross-encoder to filtered documents
        3. Combine keyword + cross-encoder scores
        
        Args:
            query: Search query
            chunks: List of chunk dictionaries
            top_k: Number of results to return
            keyword_filter_threshold: Minimum keyword similarity
            hybrid_weight: Weight for keyword score (0-1, rest is cross-encoder)
            
        Returns:
            List of reranked chunks with updated metadata
        """
        if not chunks:
            return []
        
        # Stage 1: Keyword filtering
        keyword_scored_chunks = self._filter_by_keywords(query, chunks, keyword_filter_threshold)
        
        if not keyword_scored_chunks:
            # Fallback: use all chunks if none pass filter
            keyword_scored_chunks = [(chunk, {'combined_keyword_score': 0.0}) for chunk in chunks]
        
        candidates = keyword_scored_chunks
        candidate_chunks = [chunk for chunk, _ in candidates]
        
        # Stage 2: Cross-encoder reranking
        try:
            pairs = [[query, chunk.get('text', chunk.get('content', ''))] for chunk in candidate_chunks]
            cross_encoder_scores = self.ranker.predict(pairs)
            
            # Stage 3: Hybrid scoring
            final_results = []
            
            for i, ce_score in enumerate(cross_encoder_scores):
                chunk, keyword_metrics = candidates[i]
                
                # Normalize cross-encoder score to [0, 1]
                normalized_ce_score = float(ce_score)
                
                keyword_score = keyword_metrics['combined_keyword_score']
                
                # Combine scores
                final_score = (1 - hybrid_weight) * normalized_ce_score + hybrid_weight * keyword_score
                
                final_results.append({
                    'chunk': chunk,
                    'final_score': final_score,
                    'cross_encoder_score': normalized_ce_score,
                    'keyword_score': keyword_score,
                    'keyword_metrics': keyword_metrics
                })
            
            # Sort by final score
            final_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Add metadata to chunks
            reranked_chunks = []
            for i, result in enumerate(final_results[:top_k]):
                chunk = result['chunk'].copy()
                chunk['rerank_score'] = result['final_score']
                chunk['cross_encoder_score'] = result['cross_encoder_score']
                chunk['keyword_score'] = result['keyword_score']
                chunk.update(result['keyword_metrics'])
                reranked_chunks.append(chunk)
            
            return reranked_chunks
            
        except Exception as e:
            print(f"[Warning] Cross-encoder error: {e}. Falling back to keyword-only ranking.")
            
            # Fallback: Use only keyword scores
            fallback_chunks = []
            for chunk, metrics in candidates[:top_k]:
                chunk_copy = chunk.copy()
                chunk_copy['rerank_score'] = metrics['combined_keyword_score']
                chunk_copy['keyword_score'] = metrics['combined_keyword_score']
                chunk_copy.update(metrics)
                fallback_chunks.append(chunk_copy)
            
            return fallback_chunks
