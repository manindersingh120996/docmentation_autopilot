"""
ChromaDB implementation of VectorStore (in-process mode).

This implementation uses ChromaDB in embedded/in-process mode, meaning the
database runs inside your Python process (no separate server needed).

ARCHITECTURE:
    - Embeddings: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
    - Storage: Local directory (./data/chromadb/) with DuckDB backend
    - Index: HNSW for fast approximate nearest neighbor search
    - Distance: Cosine similarity (standard for semantic text search)

WHY IN-PROCESS MODE:
    - Simpler setup (no Docker/server to manage)
    - Faster for local dev (no network overhead)
    - Perfect for moderate datasets (<1M documents)
    - Data persists to disk across restarts

PRODUCTION CONSIDERATIONS:
    In-process mode is fine for single-machine deployments. For distributed
    systems with multiple workers, consider ChromaDB's client-server mode
    or a managed vector database (Pinecone, Weaviate, Qdrant).

EMBEDDING MODEL CHOICE:
    We use 'all-MiniLM-L6-v2' because it's:
    - Fast (5-10ms per encoding on CPU)
    - Small (384 dimensions = compact storage)
    - Good quality for general text (trained on 1B+ pairs)
    - Open-source (no API costs, runs offline)
    
    For production, consider upgrading to 'all-mpnet-base-v2' (768 dims)
    or OpenAI's text-embedding-ada-002 for better quality.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
# print("import successfull")

from src.infrastructure.base.vector_store import (
    VectorStore,
    Document,
    SearchResult,
    DistanceMetric,
    VectorStoreError
)

logger = logging.getLogger(__name__)

class ChromaDBVectorStore(VectorStore):
    """
    VectorStore implementation using ChromaDB (in-process).
    
    Provides semantic search over documents using learned embeddings.
    Perfect for RAG systems, documentation search, and similarity matching.
    
    Thread Safety:
        ChromaDB is thread-safe for reads but not concurrent writes.
        Multiple processes should use client-server mode, not in-process.
    """

    def __init__(self,
                 persist_directory: str = "./data/chromadb",
                 collection_name: str = "documents",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 distance_metric: DistanceMetric = DistanceMetric.COSINE,
                 ):
        """
        Initialize ChromaDB vector store.
        
        Creates or loads a persistent ChromaDB collection from disk.
        Initializes the embedding model for converting text to vectors.
        
        Args:
            persist_directory: Directory for ChromaDB data files
            collection_name: Name of the collection (like a table name)
            embedding_model_name: SentenceTransformers model name
            distance_metric: Similarity metric (COSINE, EUCLIDEAN, DOT_PRODUCT)
        
        Raises:
            VectorStoreError: If initialization fails
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.distance_metric = distance_metric
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initializing ChromaDBVectorStore: "
            f"collection='{collection_name}', "
            f"model='{embedding_model_name}', "
            f"metric={distance_metric.value}, "
            f"persist_dir='{persist_directory}'"
        )

        # initialize CHROMADB client (in=process model)
        try:
            self._client = chromadb.PersistentClient(
                path = str(self.persist_directory),
                settings = Settings(
                    anonymized_telemetry = False, # Disable usage tracking
                    allow_reset = True, # allow reset() for testing
                )
            )
            logger.info(f"✓ ChromaDB client initialized at {self.persist_directory}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB client: {e}") from e
        
        # Initialize embedding model
        try:
            self._embedding_model = SentenceTransformer(embedding_model_name)
            self._embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
            
            logger.info(
                f"✓ Loaded embedding model '{embedding_model_name}' "
                f"(dimension: {self._embedding_dim})"
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to load embedding model: {e}") from e
        
        # Get or create collection
        try:
            # Map our distance metric enum to ChromDB's space names
            metric_map = {
                DistanceMetric.COSINE: "cosine",
                DistanceMetric.EUCLIDEAN: "l2",
                DistanceMetric.DOT_PRODUCT: "ip", # Inner Product
            }

            chroma_metric = metric_map.get(distance_metric, "cosine")

            self._collection = self._client.get_or_create_collection(
                name = collection_name,
                metadata = {"hnsw:space": chroma_metric}
            )

            doc_count = self._collection.count()
            logger.info(
                f"✓ Collection '{collection_name}' ready "
                f"({doc_count} documents)"
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get/create collection: {e}") from e
        
    def _generate_embeddings(
            self,
            texts: List[str],
            batch_size: int = 32,
    ) -> List[List[float]] :
        """
        Generate embeddings for a list of texts.
        
        Uses batching for efficiency (processes 32 texts at once instead of
        one at a time, which is 10-50× faster due to GPU parallelism).
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
        
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        try:
            # ecnode handles batching internally
            embeddings = self._embedding_model.encode(
                texts,
                batch_size = batch_size,
                show_progress_bar = False, # not to spaming logs
                convert_to_numpy = True, # Return numpy arrays
            )

            # Convert numpy arrays to lists (ChromaDB expects lists)
            return [emb.tolist() for emb in embeddings]
        
        except Exception as e:
            raise VectorStoreError(f"Failed to generate embeddings: {e}") from e
        
    def add_documents(self,
                    documents: List[Document],
                    batch_size: int = 100 
                    )-> List[str]:
        """
        Add documents to the vector store.
        
        Steps:
        1. Generate embeddings for document content (unless provided)
        2. Store documents with embeddings in ChromaDB
        3. Return document IDs
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process at once
        
        Returns:
            List of document IDs (in same order as input)
        
        Raises:
            VectorStoreError: If adding documents fails
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Cannot add empty list of documents")
        
        logger.info(f"Adding {len(documents)} documents to collection '{self.collection_name}'")
        
        # Separate documents with/without embeddings
        docs_need_embedding = []
        docs_have_embedding = []
        
        for doc in documents:
            if doc.embedding is None:
                docs_need_embedding.append(doc)
            else:
                docs_have_embedding.append(doc)
        
        # Generate embeddings for documents that need them
        if docs_need_embedding:
            logger.debug(f"Generating embeddings for {len(docs_need_embedding)} documents")
            
            texts = [doc.content for doc in docs_need_embedding]
            embeddings = self._generate_embeddings(texts, batch_size=batch_size)
            
            # Assign embeddings back to documents
            for doc, emb in zip(docs_need_embedding, embeddings):
                doc.embedding = emb
        
        # All documents now have embeddings, combine them
        all_docs = docs_need_embedding + docs_have_embedding
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in all_docs]
        embeddings = [doc.embedding for doc in all_docs]
        documents_text = [doc.content for doc in all_docs]
        metadatas = [doc.metadata or {'meta':'none'} for doc in all_docs]
        print(metadatas)
        
        # Add to ChromaDB in batches
        try:
            for i in range(0, len(all_docs), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_documents = documents_text[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                self._collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )
                
                logger.debug(f"Added batch {i//batch_size + 1} ({len(batch_ids)} docs)")
            
            logger.info(f"✓ Added {len(all_docs)} documents to '{self.collection_name}'")
            return ids
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents to ChromaDB: {e}") from e
    

    def search(
            self,
            query: str,
            top_k: int = 5,
            filters: Optional[Dict[str, Any]] = None,
            embedding_function: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Search for documents similar to query text.
        
        This is the PRIMARY search method for text queries.
        
        Flow:
        1. Convert query text to embedding vector
        2. Perform HNSW approximate nearest neighbor search
        3. Return top K most similar documents with scores
        
        Args:
            query: Text query to search for
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"language": "python"})
            embedding_function: Optional custom embedding function
        
        Returns:
            List of SearchResult objects, ranked by similarity
        
        Raises:
            VectorStoreError: If search fails

        """
        logger.debug(f"Searching for: '{query[:50]}...' (top_k={top_k})")
        
        # Generate query embedding
        try:
            if embedding_function:
                query_embedding = embedding_function(query)
            else:
                query_embedding = self._generate_embeddings([query])[0]
                
        except Exception as e:
            raise VectorStoreError(f"Failed to generate query embedding: {e}") from e
        
        # Perform vector search
        return self.search_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

    def search_by_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search using a pre-computed embedding vector.
        
        Use this when you already have the embedding (avoids re-computing it).
        
        Args:
            query_embedding: Pre-computed embedding vector
            top_k: Number of results to return
            filters: Metadata filters
        
        Returns:
            List of SearchResult objects
        """
        try:
            # ChromaDB query
            results = self._collection.query(
                query_embeddings = [query_embedding],
                n_results = top_k,
                where = filters, # Metadata pre-filtering
                include=["documents", "metadatas", "distances"]
            )     

            # Convert ChromaDB results to our SearchResult format
            search_results = []

            # ChromaDB returns nested lists: [[doc1, doc2,...]]
            # We extract the first (and only ) query's results
            ids = results['ids'][0]
            documents_text = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for doc_id, content, metadata, distance in zip(
                ids, documents_text, metadatas, distances
            ):
                # Convert distance to similarity score
                # ChromaDB returns L2 distance for cosine (1 - cosine_similarity)
                # Smaller distance = higher similarity
                if self.distance_metric == DistanceMetric.COSINE:
                    # Distance is 1 - cosine_similarity
                    # So similarity = 1 - distance
                    score = 1.0 - distance
                else:
                    # For other metrics, use inverse distance as score
                    score = 1.0 / (1.0 + distance)
                
                # Reconstruct Document
                doc = Document(
                    id=doc_id,
                    content=content,
                    embedding=None,  # Don't return embeddings (large, usually not needed)
                    metadata=metadata,
                )
                
                search_results.append(
                    SearchResult(
                        document=doc,
                        score=score,
                        distance=distance,
                    )
                )
            
            logger.debug(f"✓ Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}") from e

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity and keyword matching.
        
        NOTE: ChromaDB doesn't have built-in BM25, so this is a simplified
        implementation. For production, consider using Weaviate or Qdrant
        which have native hybrid search.
        
        Current implementation:
        - Pure vector search with alpha weighting of final scores
        - In production, you'd integrate with Elasticsearch for true BM25
        
        Args:
            query: Text query
            top_k: Number of results
            alpha: Weight for vector search (0.0 = pure keyword, 1.0 = pure vector)
            filters: Metadata filters
        
        Returns:
            List of SearchResult objects
        """
        logger.warning(
            "ChromaDB doesn't support true hybrid search. "
            "Falling back to pure vector search. "
            "For production hybrid search, consider Weaviate or Qdrant."
        )
        
        # For now, just do vector search
        # In production, you'd combine with Elasticsearch BM25 scores here
        results = self.search(query, top_k=top_k, filters=filters)
        
        # Scale scores by alpha (simulate weighting)
        for result in results:
            result.score *= alpha
        
        return results
    
    def update_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update metadata for a document without re-embedding.
        
        Useful for adding tags, categories, timestamps without expensive
        re-embedding.
        
        Args:
            document_id: ID of document to update
            metadata: New metadata (replaces existing)
        """
        try:
            self._collection.update(
                ids=[document_id],
                metadatas=[metadata],
            )
            
            logger.debug(f"✓ Updated metadata for document '{document_id}'")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to update metadata: {e}") from e
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents by ID."""
        try:
            self._collection.delete(ids=document_ids)
            logger.info(f"✓ Deleted {len(document_ids)} documents")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents: {e}") from e
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a single document by ID."""
        try:
            results = self._collection.get(
                ids=[document_id],
                include=["documents", "metadatas"],
            )
            
            if not results['ids']:
                return None
            
            return Document(
                id=results['ids'][0],
                content=results['documents'][0],
                embedding=None,
                metadata=results['metadatas'][0],
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get document: {e}") from e
    
    def count(self) -> int:
        """Return total number of documents in collection."""
        return self._collection.count()
    
    def clear(self) -> None:
        """Delete all documents from collection."""
        try:
            # Delete and recreate collection
            self._client.delete_collection(name=self.collection_name)
            
            metric_map = {
                DistanceMetric.COSINE: "cosine",
                DistanceMetric.EUCLIDEAN: "l2",
                DistanceMetric.DOT_PRODUCT: "ip",
            }
            
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": metric_map[self.distance_metric]},
            )
            
            logger.info(f"✓ Cleared collection '{self.collection_name}'")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to clear collection: {e}") from e

    

