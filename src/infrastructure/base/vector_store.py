"""
Docstring for doc-auto-pilot.src.infrastructure.base.vector_store
----------------------------------------------------------------------

Abstract base class for vector store implementations.

A VectorStore stores document embeddings and provides semantic similarity
serach. This is the backbone of RAG (Retrieval-Augmented Generation).

WHY THIS EXISTS:
    Layer 2 analysis needs to retrieve relevant context when analyzing
    code changes - commit messages, PR discussions, related documentations.
    This context is stored as vector embeddings and retrieved by semantic
    similarity (not keyword matching).

WHAT THIS ABSTRACTS:
    Local development: ChromaDB (in-process running)
    Production (GCP): Vertex AI Vector Search (or Weaviate)

HOW RAG USES THIS:
    1. At indexing time: embed documentation chunks -> store in vector DB
    2. At query time: embed the query -> search fopr similar chunks -> pass those
      chunks to the LLM as context in the prompt


"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

class DistanceMetric(Enum):
    """
    Distance metrics for measuring vector simillarity

    COSINE:             Measure angle between vectors, Best for text embeddgins
                        because it ignores magnitude (word count) and captures
                        semantic direction. Range: 0 (identical) -> 1 (no simillarity) to 2 (Opposite)'
    
    EUCLIDEAN:          Straight line distance in vector store. Sensitive to
                        magnitude. Range:[0 to infinity]
    
    DOT_PRODUCT:        Inner product. Fast to compute. Used when vectors are normalized (magnitude=1),
                        equivalent to cosine then.
    
    For our use case (text embeddings for documentation), COSINE is the standard choice. It's what all
    major embeddigns models optimized for.
    """

    COSINE= "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class Document:
    """
    A document stored in the vectore database.

    A 'document' in our system is a chunk of text with a unique ID and
    A "document" in our system is a chunk of text with a unique ID and
    optional metadata. We chunk documentation into smaller pieces (e.g.,
    one section per chunk) so the similarity search returns precise,
    relevant excerpts rather than entire documents.

    Attributes:
        id:        Unique identifier for this document chunk.
                   You assign this — use something meaningful like
                   "pytorch_docs/autograd/section_3" or a UUID.
        content:   The actual text content of this chunk.
        embedding: The vector representation of content.
                   None when creating (embedding generated on store).
                   Populated when retrieved.
        metadata:  Key-value data for filtering and context.
                   Example: {"source": "pytorch", "doc_type": "api_ref",
                             "last_updated": "2024-01-15", "section": "autograd"}
    
    """
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str,Any]] = None

@dataclass
class SearchResult:
    """
    A single result from a similarity search.

    Bundles the matching document with its similarity score, so callers
    can decide whether the result is relevant enough to use.

    Attributes:
        document: The matching document chunk.
        score:    Similarity score. Higher = more similar.
                  Range depends on distance metric and implementation.
                  For cosine: typically 0.0 to 1.0 (1.0 = identical).
        distance: Raw distance in vector space. Lower = more similar.
                  Inverse relationship to score.
    """
    document: Document
    score: float
    distance: float



# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class VectorStoreError(Exception):
    """BNase exception for all vectore store errors."""
    pass

class EmbeddingError(VectorStoreError):
    """
    Raised when generating an embedding fails.

    Common causes:
        - Embedding model API is unavailbale
        - Text exceeds the model's token limit
        - Invalid characters in input text

    """
    pass


class SearchError(VectorStoreError):
    """
    Raised when similarity search fails.

    Common causes:
        - Vector store is unavailable
        - Query embedding dimension doesn't match stored embeddings dimensions.
        - Invalid filter syntax
    """
    pass

class DocumentNotFoundError(VectorStoreError):
    """
    Raised when a document with the specified ID doesn't exist.

    Note: search() and hybrid_search() do NOT raise this, thewy simply
    return empty liss if no results match. This exception is only for
    get_document() and update_metadate() which trarget specific IDs.
    """
    pass

# ---------------------------------------------------------------------------
# The Abstract Base Class
# ---------------------------------------------------------------------------

class VectorStore(ABC):
    """
    Abstract interface for vector database implementations.

    CORE CONCEPT:
        You store documents (text chunks with optional metadata).
        You search by semantic similarity (not exact keyword matching).
        Results are ranked by how similar their meaning is to your query.

    HOW IT FITS IN OUR SYSTEM:
        Layer 2 uses this to find documentation context relevant to a
        code change. Instead of searching for exact keywords, we search
        for conceptually related documentation — even if it uses different
        terminology than the code.

    EMBEDDING FUNCTIONS:
        Several methods accept an optional `embedding_function` parameter.
        This lets you override the default embedding model per-call.
        Pass None to use the store's default model (configured at init time).
        This flexibility allows using different embedding models for
        different content types (code vs. prose vs. API docs).

    HYBRID SEARCH:
        Many methods support `hybrid_search()` which combines:
        - Dense retrieval: semantic similarity via embeddings
        - Sparse retrieval: keyword matching (like BM25)
        Hybrid search typically outperforms either alone, especially for
        technical content with specific terms (class names, method names).
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embedding_function: Optional[Callable[[str], List[float]]] = None,

    ) -> List[str]:
        """ 
        Add documents to the vector store.

        If documents already have embeddings (document.embedding is not None),
        those are used directly. Otherwise, the embedding_function (or the
        store's default model) generates embeddings from document.content.

        Documents with the same ID as existing documents overwrite them.

        Args:
            documents:          Documents to store. Must have at least id and content.
            embedding_function: Optional function: (text: str) -> List[float]
                                If None, the store's configured default is used.

        Returns:
            List of document IDs in the same order as the input list.

        Raises:
            EmbeddingError:   If embedding generation fails.
            VectorStoreError: If storage fails.

        Performance note:
            Pass all documents in one call when possible. Batch operations
            are significantly faster than N individual add_documents() calls.
        
        Example:
            docs = [
                Document(
                    id="pytorch_docs/linear/overview",
                    content = "The Linear layer applies a linear transformation...",
                    metadata={"source":"pytorch", "section": "linear"}
                    ),
                Document(
                    id="pytorch_docs/linear/parameters",
                    content="Parameters: in_features (int), out_features (int)...",
                    metadata={"source": "pytorch", "section": "linear"}
                ),
            ]
            ids = vector_store.add_documents(docs)
            # ids == ["pytorch_docs/linear/overview", "pytorch_docs/linear/parameters"]
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None
    ) -> List[SearchResult]:
        """
        Find documents semantically similar to a text query.

        Generates an embedding for `query` and returns the `top_k` most
        similar documents in the store.

        Args:
            query:              Text to search for.
            top_k:              Number of results to return (default 10).
            filters:            Optional metadata filters. Only return documents
                                where ALL filter conditions match.
                                Example: {"source": "pytorch", "doc_type": "api"}
                                Filter semantics: exact match on string values,
                                range support is implementation-specific.
            embedding_function: Optional embedding override. If None, use default.

        Returns:
            List of SearchResult objects sorted by similarity (best match first).
            May be empty if no results are found.
            Will contain at most top_k results (possibly fewer if store has less).

        Raises:
            EmbeddingError: If query embedding fails.
            SearchError:    If search execution fails.

        Example:
            results = vector_store.search(
                query="How do I implement backpropagation in PyTorch?",
                top_k=5,
                filters={"source": "pytorch", "doc_type": "tutorial"}
            )
            for r in results:
                print(f"Score {r.score:.3f}: {r.document.content[:100]}")

        
        """
        pass

    @abstractmethod
    def search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str,Any]] = None
    ) -> List[SearchResult]:
        """
        Docstring for search_by_vector      
        ---------------------------------------------

        Find documents similar to a pre-computed embedding vector.

        Use this when you've already computed the query embedding
        (e.g., it was cached, or computed by a different model).
        Avoids the cost of re-embedding the same query.

        Args:
            embedding: Pre-computed query embedding vector.
            top_k:     Number of results to return.
            filters:   Optional metadata filters.

        Returns:
            List of SearchResult objects sorted by similarity.

        Raises:
            SearchError: If search fails or embedding dimension mismatches.

        """
        pass

    @abstractmethod
    def hybrid_search(self,
                      query:str,
                      top_k: int=10,
                      alpha: float = 0.5,
                      filters: Optional[Dict[str,Any]] = None
                      ) -> List[SearchResult]:
        """
        Search combining dense vector similarity with sparse keyword matching.

        hybrind serach combines two retrieval strategies:
            - Dense: embeddigns similarity (sematic understanding)
            - Sparse: Keyword/BM25 matching (exact term matching)

        Results are fused by their weighted combination.

        WHY HYBRID mATTERS FOR OUR SYSTEM:
            If a code change affects the 'torch.nn.Linear' class, we want to
            find docs that mention 'Linear' exactlky (sparse match) AND docs
            that discuss "linear layers" conceptually (dense amtch). Pure
            Vector seach might miss the exact class name. Pure keyword search might
            miss semantically related docs, Hybrid gets both.

        Args:
            query:      Text query
            top_k:      Number of results
            alpha:      Balance between dense and sparse.
                        0.0 = Pure Sparse (Keyword matching only)
                        1.0 = Pure Dense (Vector similarity only)
                        0.5 = equal weight (default value)
                        Tune this based on your content type:
                     - Technical docs with specific terms → lower alpha (more sparse)
                     - Conceptual docs → higher alpha (more dense)
            filters: Optional metadata filters.

        Returns:
            List of SearchResult sorted by combined score.

        Raises:
            SearchError:          If search fails.
            NotImplementedError:  If this implementation doesn't support hybrid.
        
        """
        pass

    @abstractmethod
    def update_metadata(
        self,
        document_id: str,
        metadata_updates: Dict[str, Any]
    ) -> None:
        """
        Update metadata fields for a document without re-embedding.

        Merges metadata_updates into the document's existing metadata
        (like dict.update()). New keys are added. Existing keys are
        updated. Unmentioned keys are preserved unchanged.

        WHY THIS EXISTS:
            Re-embedding a document to change its metadata is wasteful.
            If we just want to add a "status: reviewed" tag or update
            "last_checked: 2024-01-20", we shouldn't regenerate the
            embedding. This method makes that efficient.

        Args:
            document_id:      ID of the document to update.
            metadata_updates: Fields to add or update in the metadata.
                              Example: {"status": "reviewed", "reviewed_by": "alice"}

        Returns:
            None (raises exception on failure, silence means success)

        Raises:
            DocumentNotFoundError: If no document with this ID exists.
            VectorStoreError:      If update fails.

        Example:
            vector_store.update_metadata(
                document_id="pytorch_docs/linear/overview",
                metadata_updates={"status": "outdated", "last_checked": "2024-01-15"}
            )
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> int:
        """
        Delete documents from the store by ID.

        Args:
            document_ids: IDs of documents to delete.

        Returns:
            Number of documents actually deleted.
            May be less than len(document_ids) if some IDs don't exist.

        Raises:
            VectorStoreError: If deletion fails.
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.

        Args:
            document_id: ID of the document to retrieve.

        Returns:
            Document object with content, embedding, and metadata.
            None if no document with this ID exists.

        Raises:
            VectorStoreError: If retrieval fails (not including "not found").
        """
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the store.

        Args:
            filters: Optional metadata filters. Count only matching documents.
                    None counts all documents.

        Returns:
            Number of documents matching the filter (or total if no filter).

        Example:
            total = vector_store.count()
            pytorch_count = vector_store.count({"source": "pytorch"})
            print(f"{pytorch_count}/{total} documents are from PyTorch")
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Delete ALL documents from the store.

        WARNING: Irreversible. Use only for testing or full re-indexing.
        In production, prefer targeted delete_documents() calls.

        Raises:
            VectorStoreError: If clearing fails.
        """
        pass

