"""
Unit tests for ChromaDBVectorStore.

These tests verify:
- Document storage and retrieval
- Semantic search accuracy
- Metadata filtering
- Edge cases and error handling

Note: These are integration tests (they use real ChromaDB), not pure unit tests.
"""

import sys
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.infrastructure.local.chromadb_vector_store import ChromaDBVectorStore
from src.infrastructure.base.vector_store import (
    Document,
    SearchResult,
    DistanceMetric,
    VectorStoreError,
)

@pytest.fixture
def temp_db_dir(tmp_path):
    """Create temporary directory for ChromaDB data."""
    return tmp_path / "chromadb_test"

@pytest.fixture
def vector_store(temp_db_dir):
    """Create ChromaDBVectoreStore with temporary Storage."""
    store = ChromaDBVectorStore(
        persist_directory=str(temp_db_dir),
        collection_name="test_collection",
        embedding_model_name="all-MiniLM-L6-v2",

    )
    yield store
    # Cleanup happens automatically( temp direcoryt deleted)

def test_initialization(vector_store, temp_db_dir):
    """Test that vector store initializes correctly."""
    assert vector_store.collection_name == 'test_collection'
    assert vector_store.embedding_model_name == "all-MiniLM-L6-v2"
    assert temp_db_dir.exists()
    assert vector_store.count() == 0
    print("✓ Initialization successful")

def test_add_documents_without_embeddings(vector_store):
    """Test adding documents (embeddings generated automatically)."""
    docs = [
        Document(
            id="doc1",
            content="Python is a programming language",
            metadata={"category": "programming", "language": "python"},
        ),
        Document(
            id="doc2",
            content="Java is used for enterprise applications",
            metadata={"category": "programming", "language": "java"},
        ),
        Document(
            id="doc3",
            content="The weather is sunny today",
            metadata={"category": "weather"},
        ),
    ]
    
    ids = vector_store.add_documents(docs)

    assert len(ids) == 3
    assert ids == ["doc1","doc2","doc3"]
    assert vector_store.count() == 3
    
    print("✓ Documents added successfully (auto-embedding)")

def test_add_documents_with_embeddings(vector_store):
    """Test adding documents with pre-computed embeddings."""
    # Pre-compute embedding
    embedding = vector_store._generate_embeddings(["Test content"])[0]
    
    doc = Document(
        id="doc_with_emb",
        content="Test content",
        embedding=embedding,  # Provide embedding
        metadata={"source": "test"},
    )
    
    ids = vector_store.add_documents([doc])
    
    assert ids == ["doc_with_emb"]
    assert vector_store.count() == 1
    
    print("✓ Documents with embeddings added successfully")

def test_semantic_search_basic(vector_store):
    """Test basic semantic search."""
    # Add documents
    docs = [
        Document(id="1", content="Python programming tutorial"),
        Document(id="2", content="Java enterprise development"),
        Document(id="3", content="Machine learning with Python"),
        Document(id="4", content="Sunny weather forecast"),
    ]
    vector_store.add_documents(docs)
    
    # Search for programming-related content
    results = vector_store.search("coding in Python", top_k=2)
    
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    
    # Should find Python-related docs (doc1 or doc3)
    top_result_ids = [r.document.id for r in results]
    assert "1" in top_result_ids or "3" in top_result_ids
    
    # Weather doc should NOT be in top 2
    assert "4" not in top_result_ids
    
    print(f"✓ Semantic search working: top results = {top_result_ids}")

def test_semantic_search_synonyms(vector_store):
    """Test that search understands synonyms."""
    docs = [
        Document(id="auth1", content="User authentication with JWT tokens"),
        Document(id="auth2", content="Login system using OAuth"),
        Document(id="db1", content="Database connection pooling"),
    ]
    vector_store.add_documents(docs)
    
    # Search with synonym "login" for "authentication"
    results = vector_store.search("login methods", top_k=2)
    
    # Should find auth-related docs even though "login" wasn't in doc1
    top_ids = [r.document.id for r in results]
    assert "auth1" in top_ids or "auth2" in top_ids
    
    # DB doc should rank lower
    assert results[0].document.id in ["auth1", "auth2"]
    
    print(f"✓ Synonym search working: found {top_ids}")

def test_search_with_metadata_filters(vector_store):
    """Test metadata filtering during search."""
    docs = [
        Document(
            id="py1",
            content="Python authentication guide",
            metadata={"language": "python", "category": "security"},
        ),
        Document(
            id="py2",
            content="Python database tutorial",
            metadata={"language": "python", "category": "database"},
        ),
        Document(
            id="js1",
            content="JavaScript authentication guide",
            metadata={"language": "javascript", "category": "security"},
        ),
    ]
    vector_store.add_documents(docs)
    
    # Search only Python docs
    results = vector_store.search(
        "authentication",
        top_k=5,
        filters={"language": "python"},
    )
    
    # Should only return Python docs
    assert len(results) <= 2  # Only 2 Python docs exist
    for result in results:
        assert result.document.metadata["language"] == "python"
    
    print("✓ Metadata filtering working correctly")

def test_search_by_vector(vector_store):
    """Test searching with pre-computed embedding."""
    docs = [
        Document(id="1", content="Python programming"),
        Document(id="2", content="Weather forecast"),
    ]
    vector_store.add_documents(docs)
    
    # Generate query embedding manually
    query_embedding = vector_store._generate_embeddings(["Python coding"])[0]
    
    # Search using the embedding
    results = vector_store.search_by_vector(query_embedding, top_k=1)
    
    assert len(results) == 1
    assert results[0].document.id == "1"  # Should find Python doc
    
    print("✓ Search by vector working")


def test_get_document(vector_store):
    """Test retrieving a single document by ID."""
    doc = Document(
        id="test123",
        content="Test content here",
        metadata={"key": "value"},
    )
    vector_store.add_documents([doc])
    
    # Retrieve it
    retrieved = vector_store.get_document("test123")
    
    assert retrieved is not None
    assert retrieved.id == "test123"
    assert retrieved.content == "Test content here"
    assert retrieved.metadata["key"] == "value"
    
    # Try non-existent ID
    missing = vector_store.get_document("nonexistent")
    assert missing is None
    
    print("✓ Get document working")


def test_update_metadata(vector_store):
    """Test updating document metadata without re-embedding."""
    doc = Document(
        id="update_test",
        content="Original content",
        metadata={"version": "1.0"},
    )
    vector_store.add_documents([doc])
    
    # Update metadata
    new_metadata = {"version": "2.0", "updated": True}
    vector_store.update_metadata("update_test", new_metadata)
    
    # Verify update
    retrieved = vector_store.get_document("update_test")
    assert retrieved.metadata["version"] == "2.0"
    assert retrieved.metadata["updated"] is True
    
    print("✓ Metadata update working")


def test_delete_documents(vector_store):
    """Test deleting documents."""
    docs = [
        Document(id="keep1", content="Keep this"),
        Document(id="delete1", content="Delete this"),
        Document(id="delete2", content="Delete this too"),
    ]
    vector_store.add_documents(docs)
    
    assert vector_store.count() == 3
    
    # Delete two documents
    vector_store.delete_documents(["delete1", "delete2"])
    
    assert vector_store.count() == 1
    assert vector_store.get_document("keep1") is not None
    assert vector_store.get_document("delete1") is None
    
    print("✓ Document deletion working")


def test_clear_collection(vector_store):
    """Test clearing all documents."""
    docs = [
        Document(id="1", content="Doc 1"),
        Document(id="2", content="Doc 2"),
        Document(id="3", content="Doc 3"),
    ]
    vector_store.add_documents(docs)
    
    assert vector_store.count() == 3
    
    # Clear everything
    vector_store.clear()
    
    assert vector_store.count() == 0
    
    print("✓ Clear collection working")


def test_persistence(temp_db_dir):
    """Test that data persists across store instances."""
    # Create store and add documents
    store1 = ChromaDBVectorStore(
        persist_directory=str(temp_db_dir),
        collection_name="persist_test",
    )
    
    docs = [
        Document(id="persist1", content="Persisted content"),
    ]
    store1.add_documents(docs)
    
    assert store1.count() == 1
    
    # Close first store, create new one (simulates restart)
    del store1
    
    store2 = ChromaDBVectorStore(
        persist_directory=str(temp_db_dir),
        collection_name="persist_test",
    )
    
    # Data should still be there
    assert store2.count() == 1
    retrieved = store2.get_document("persist1")
    assert retrieved is not None
    assert retrieved.content == "Persisted content"
    
    print("✓ Persistence working (data survives restart)")


def test_batch_processing(vector_store):
    """Test that large batches are handled efficiently."""
    # Create 100 documents
    docs = [
        Document(
            id=f"batch_{i}",
            content=f"Document number {i} about topic {i % 10}",
            metadata={"batch": i // 10},
        )
        for i in range(100)
    ]
    
    # Add all at once
    ids = vector_store.add_documents(docs, batch_size=25)
    
    assert len(ids) == 100
    assert vector_store.count() == 100
    
    # Search should work
    results = vector_store.search("topic 5", top_k=5)
    assert len(results) == 5
    
    print("✓ Batch processing working (100 docs)")


def test_score_ranges(vector_store):
    """Test that similarity scores are in expected range."""
    docs = [
        Document(id="exact", content="machine learning algorithms"),
        Document(id="similar", content="artificial intelligence methods"),
        Document(id="unrelated", content="cooking recipes for dinner"),
    ]
    vector_store.add_documents(docs)
    
    results = vector_store.search("machine learning", top_k=3)
    
    # Scores should be between 0 and 1 for cosine similarity
    for result in results:
        assert 0.0 <= result.score <= 1.0
    
    # Exact match should have highest score
    assert results[0].document.id == "exact"
    
    # Similar content should score higher than unrelated
    scores_by_id = {r.document.id: r.score for r in results}
    assert scores_by_id["similar"] > scores_by_id["unrelated"]
    
    print(f"✓ Score ranges correct: {[(r.document.id, r.score) for r in results]}")


def test_empty_query_handling(vector_store):
    """Test handling of edge cases."""
    docs = [Document(id="1", content="Test content")]
    vector_store.add_documents(docs)
    
    # Empty query should still work (though results may be random)
    results = vector_store.search("", top_k=1)
    assert len(results) <= 1
    
    print("✓ Empty query handled gracefully")


def test_embedding_dimension_consistency(vector_store):
    """Test that embeddings have consistent dimensions."""
    texts = ["Short text", "This is a much longer piece of text with many words"]
    embeddings = vector_store._generate_embeddings(texts)
    
    # All embeddings should have same dimension
    assert len(embeddings[0]) == len(embeddings[1])
    assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 dimension
    
    print(f"✓ Embeddings dimension consistent: {len(embeddings[0])}")


if __name__ == '__main__':
    # Run tests manually
    print("=" * 70)
    print("ChromaDBVectorStore Unit Tests")
    print("=" * 70)
    print()
    
    import tempfile
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_dir = tmp_path / "chromadb_test"
        
        # Initialize store for tests
        store = ChromaDBVectorStore(
            persist_directory=str(db_dir),
            collection_name="test_collection",
        )
        
        print("Running tests...")
        print()
        
        test_initialization(store, db_dir)
        
        # Clear for next test
        store.clear()
        test_add_documents_without_embeddings(store)
        
        store.clear()
        test_add_documents_with_embeddings(store)
        
        store.clear()
        test_semantic_search_basic(store)
        
        store.clear()
        test_semantic_search_synonyms(store)
        
        store.clear()
        test_search_with_metadata_filters(store)
        
        store.clear()
        test_search_by_vector(store)
        
        store.clear()
        test_get_document(store)
        
        store.clear()
        test_update_metadata(store)
        
        store.clear()
        test_delete_documents(store)
        
        store.clear()
        test_clear_collection(store)
        
        test_persistence(tmp_path / "persist_test")
        
        store.clear()
        test_batch_processing(store)
        
        store.clear()
        test_score_ranges(store)
        
        store.clear()
        test_empty_query_handling(store)
        
        test_embedding_dimension_consistency(store)
        
        print()
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        del store._client
        import gc
        gc.collect()
        print("=" * 70)
        print("✓ Client Deleted Successfully and force garbase collection performed!")
        print("=" * 70)        

