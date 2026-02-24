
#!/usr/bin/env python3
"""
Verify that all ABCs are correctly defined and importable.

This doesn't test implementations — it tests the contracts themselves:
- All files exist and have valid Python syntax
- All ABCs can be imported
- ABCs are truly abstract (can't be instantiated)
- All abstract methods are declared
- DTOs (dataclasses) work correctly
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv('.env.local')


def test_event_bus():
    print("Testing EventBus ABC...")
    from src.infrastructure.base.event_bus import (
        EventBus, EventMessage, PublishError, SubscriptionError, AcknowledgmentError
    )

    # Verify it's abstract — cannot instantiate directly
    try:
        EventBus()
        print("  ✗ EventBus should not be instantiatable")
        sys.exit(1)
    except TypeError:
        print("  ✓ EventBus correctly prevents direct instantiation")

    # Verify abstract methods
    abstract_methods = EventBus.__abstractmethods__
    expected = {"publish", "subscribe", "acknowledge", "nack", "close"}
    assert abstract_methods == expected, f"Expected {expected}, got {abstract_methods}"
    print(f"  ✓ Abstract methods: {sorted(abstract_methods)}")

    # Verify EventMessage dataclass works
    from datetime import datetime
    msg = EventMessage(
        message_id="test-123",
        topic="code-changes",
        payload={"repo": "pytorch/pytorch", "commit": "abc123"},
        timestamp=datetime.now(),
        delivery_attempt=1,
        attributes={"source": "test"}
    )
    assert msg.message_id == "test-123"
    assert msg.delivery_attempt == 1
    print(f"  ✓ EventMessage dataclass works: {msg.topic}")

    # Verify exceptions are distinct types
    assert issubclass(PublishError, Exception)
    assert issubclass(SubscriptionError, Exception)
    print("  ✓ Exception hierarchy correct")
    print("  ✓ EventBus: ALL CHECKS PASSED\n")


def test_vector_store():
    print("Testing VectorStore ABC...")
    from src.infrastructure.base.vector_store import (
        VectorStore, Document, SearchResult, DistanceMetric,
        EmbeddingError, SearchError, DocumentNotFoundError
    )

    try:
        VectorStore()
        print("  ✗ VectorStore should not be instantiatable")
        sys.exit(1)
    except TypeError:
        print("  ✓ VectorStore correctly prevents direct instantiation")

    expected = {"add_documents", "search", "search_by_vector", "hybrid_search",
                "update_metadata", "delete_documents", "get_document", "count", "clear"}
    abstract_methods = VectorStore.__abstractmethods__
    assert abstract_methods == expected, f"Expected {expected}, got {abstract_methods}"
    print(f"  ✓ Abstract methods: {sorted(abstract_methods)}")

    # Verify DTOs
    doc = Document(id="doc1", content="Test content", metadata={"source": "test"})
    assert doc.id == "doc1"
    assert doc.embedding is None  # Not set yet
    print(f"  ✓ Document dataclass works: id={doc.id}")

    result = SearchResult(document=doc, score=0.95, distance=0.05)
    assert result.score == 0.95
    print(f"  ✓ SearchResult dataclass works: score={result.score}")

    assert DistanceMetric.COSINE.value == "cosine"
    print(f"  ✓ DistanceMetric enum works: {DistanceMetric.COSINE}")
    print("  ✓ VectorStore: ALL CHECKS PASSED\n")


def test_graph_database():
    print("Testing GraphDatabase ABC...")
    from src.infrastructure.base.graph_db import (
        GraphDatabase, Node, Relationship, Path,
        NodeNotFoundError, QueryError
    )

    try:
        GraphDatabase()
        print("  ✗ GraphDatabase should not be instantiatable")
        sys.exit(1)
    except TypeError:
        print("  ✓ GraphDatabase correctly prevents direct instantiation")

    expected = {"create_node", "create_relationship", "find_nodes", "get_node",
                "get_relationships", "execute_query", "traverse", "shortest_path",
                "delete_node", "delete_relationship", "clear"}
    abstract_methods = GraphDatabase.__abstractmethods__
    assert abstract_methods == expected, f"Expected {expected}, got {abstract_methods}"
    print(f"  ✓ Abstract methods: {sorted(abstract_methods)}")

    # Verify DTOs
    node = Node(labels=["Function"], properties={"name": "authenticate"})
    assert node.id is None  # Not yet persisted
    assert "Function" in node.labels
    print(f"  ✓ Node dataclass works: labels={node.labels}")

    rel = Relationship(from_node_id="a", to_node_id="b", type="REFERENCES")
    assert rel.type == "REFERENCES"
    print(f"  ✓ Relationship dataclass works: type={rel.type}")
    print("  ✓ GraphDatabase: ALL CHECKS PASSED\n")


def test_blob_storage():
    print("Testing BlobStorage ABC...")
    from src.infrastructure.base.blob_storage import (
        BlobStorage, BlobMetadata,
        BlobNotFoundError, BlobAlreadyExistsError
    )

    try:
        BlobStorage()
        print("  ✗ BlobStorage should not be instantiatable")
        sys.exit(1)
    except TypeError:
        print("  ✓ BlobStorage correctly prevents direct instantiation")

    expected = {"store", "store_from_file", "retrieve", "retrieve_to_file",
                "retrieve_stream", "exists", "get_metadata", "delete",
                "list_blobs", "generate_signed_url", "copy"}
    abstract_methods = BlobStorage.__abstractmethods__
    assert abstract_methods == expected, f"Expected {expected}, got {abstract_methods}"
    print(f"  ✓ Abstract methods: {sorted(abstract_methods)}")
    print("  ✓ BlobStorage: ALL CHECKS PASSED\n")


def test_secrets_manager():
    print("Testing SecretsManager ABC...")
    from src.infrastructure.base.secrets import (
        SecretsManager, SecretMetadata,
        SecretNotFoundError, SecretAccessDeniedError
    )

    try:
        SecretsManager()
        print("  ✗ SecretsManager should not be instantiatable")
        sys.exit(1)
    except TypeError:
        print("  ✓ SecretsManager correctly prevents direct instantiation")

    expected = {"get_secret", "set_secret", "delete_secret", "list_secrets", "secret_exists"}
    abstract_methods = SecretsManager.__abstractmethods__
    assert abstract_methods == expected, f"Expected {expected}, got {abstract_methods}"
    print(f"  ✓ Abstract methods: {sorted(abstract_methods)}")
    print("  ✓ SecretsManager: ALL CHECKS PASSED\n")


def test_consolidated_imports():
    print("Testing consolidated imports from src.infrastructure.base...")
    from src.infrastructure.base import (
        EventBus, EventMessage,
        VectorStore, Document, SearchResult,
        GraphDatabase, Node, Relationship, Path,
        BlobStorage, BlobMetadata,
        SecretsManager, SecretMetadata,
    )
    print("  ✓ All classes importable from src.infrastructure.base")
    print("  ✓ Consolidated imports: ALL CHECKS PASSED\n")


def main():
    print("=" * 70)
    print("Infrastructure ABC Test Suite")
    print("=" * 70)
    print()

    test_event_bus()
    test_vector_store()
    test_graph_database()
    test_blob_storage()
    test_secrets_manager()
    test_consolidated_imports()

    print("=" * 70)
    print("✓ All ABC tests passed!")
    print("=" * 70)
    print()
    print("All five abstract base classes are correctly defined.")
    print("Ready to proceed to Phase 2D: Concrete Local Implementations.")


if __name__ == "__main__":
    main()
