"""Document ingestion and indexing module for RAG system."""

import os
import json
import re
from typing import List, Dict, Tuple, Any
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentIngestor:
    """Handles PDF parsing and text chunking with SEC document awareness."""
    
    # SEC Item patterns for 10-K filings
    SEC_ITEM_PATTERNS = [
        r"Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
        r"ITEM\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
        r"Part\s+([IV]+)\s*[,\-]\s*Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
    ]
    
    def __init__(self, chunk_size: int = 700, overlap: int = 150):
        """
        Initialize the document ingestor.
        
        Args:
            chunk_size: Number of characters per chunk (default: 700 for better precision)
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @classmethod
    def _extract_sec_items(cls, text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract SEC item numbers and titles from text.
        
        Args:
            text: Page text to search
            page_num: Current page number
            
        Returns:
            List of items found with their metadata
        """
        items = []
        for pattern in cls.SEC_ITEM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                groups = [g for g in match.groups() if g is not None]
                if not groups:
                    continue
                
                item_num, item_title = None, ""
                
                if len(groups) == 1:
                    potential = groups[0].strip()
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', potential):
                        item_num = potential
                elif len(groups) == 2:
                    a, b = groups
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', a.strip()):
                        item_num = a.strip()
                        item_title = b.strip()[:100]
                    elif re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', b.strip()):
                        if re.match(r'^(Part\s+)?[IVX]+$', a.strip(), re.IGNORECASE):
                            item_num = f"{a.strip()}-{b.strip()}"
                        else:
                            item_num = b.strip()
                elif len(groups) == 3:
                    part, num, title = groups
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', num.strip()):
                        item_num = f"{part.strip()}-{num.strip()}"
                        item_title = (title or "").strip()[:100]
                
                if item_num:
                    items.append({
                        'item_number': item_num,
                        'item_title': item_title,
                        'page': page_num,
                        'position': match.start()
                    })
        
        # Remove duplicates
        seen, unique = set(), []
        for item in sorted(items, key=lambda x: x['position']):
            key = (item['item_number'], item['page'])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        
        return unique
    
    def parse_pdf(self, pdf_path: str, doc_name: str) -> List[Dict]:
        """
        Parse a PDF file into chunks with rich metadata including SEC items.
        
        Args:
            pdf_path: Path to the PDF file
            doc_name: Name of the document (e.g., "Apple 10-K")
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # First pass: extract page texts and SEC items
                pages_data = []
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    page_items = self._extract_sec_items(page_text, page_num)
                    pages_data.append({
                        'page_num': page_num,
                        'text': page_text,
                        'items': page_items
                    })
                
                # Build full text with page boundaries
                full_text = ""
                page_mapping = {}
                
                for page_data in pages_data:
                    start_pos = len(full_text)
                    full_text += page_data['text'] + "\n\n"
                    end_pos = len(full_text)
                    page_mapping[(start_pos, end_pos)] = page_data
                
                # Track current SEC item for inheritance
                last_item = None
                last_item_page = 0
                chunks = []
                chunk_counter = {}  # Track chunks per page
                
                # Split into overlapping chunks
                for i in range(0, len(full_text), self.chunk_size - self.overlap):
                    chunk_text = full_text[i:i + self.chunk_size]
                    
                    if len(chunk_text.strip()) < 50:  # Skip very small chunks
                        continue
                    
                    # Find which page this chunk is on
                    page_num = 1
                    page_data = None
                    for (start, end), pdata in page_mapping.items():
                        if start <= i < end:
                            page_num = pdata['page_num']
                            page_data = pdata
                            break
                    
                    # Determine SEC item for this chunk
                    primary_item = None
                    if page_data and page_data['items']:
                        # New item found on this page
                        last_item = page_data['items'][0]
                        last_item_page = page_num
                        primary_item = last_item
                    elif last_item:
                        # Inherit from previous item with 5-page limit for Item 16
                        if last_item['item_number'] == '16' and (page_num - last_item_page > 5):
                            primary_item = None
                        else:
                            primary_item = last_item
                    
                    # Generate hierarchical chunk ID
                    page_key = (doc_name, page_num)
                    seq = chunk_counter.get(page_key, 0)
                    chunk_id = f"{doc_name}|p{page_num}|c{seq}"
                    chunk_counter[page_key] = seq + 1
                    
                    chunks.append({
                        "id": chunk_id,
                        "chunk_id": chunk_id,  # Duplicate for compatibility
                        "text": chunk_text,
                        "document": doc_name,
                        "document_name": doc_name,
                        "page": page_num,
                        "position": i,
                        "item_number": primary_item['item_number'] if primary_item else '',
                        "item_title": primary_item['item_title'] if primary_item else '',
                        "has_sec_items": bool(page_data and page_data['items'])
                    })
            
            return chunks
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF {pdf_path}: {str(e)}") from e


class VectorStore:
    """Manages embeddings and vector search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries
        """
        self.chunks.extend(chunks)
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples
        """
        if self.index is None or not self.chunks:
            raise RuntimeError("Vector store is empty. Add chunks before searching.")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def save(self, save_dir: str) -> None:
        """Save the vector store."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        
        # Save chunks metadata
        with open(os.path.join(save_dir, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)
    
    def load(self, save_dir: str) -> None:
        """Load the vector store."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        
        # Load chunks
        with open(os.path.join(save_dir, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
