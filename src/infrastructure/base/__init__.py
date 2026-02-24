"""
Infrastructure abstract base clases.

These define the CONTRACTS that all infrastructure implementations must fulfill.
Import from here and from the individual files so the import path is stable
even if we reorganize the files later.

Usage:
    from src.infrastructure.base import EventBus, VectorStore, GraphDatabase
    from src.infrastructure.base import BlobStorage, SecretsManager

    # or import specific types:
    from src.infrastructure.base import EventMessage, Document, SearchResult

"""

from src.infrastructure.base.event_bus import (
    EventBus,
    EventMessage,
    EventBusError,
    PublishError,
    SubscriptionError,
    AcknowledgmentError,
)

from src.infrastructure.base.vector_store import (
    VectorStore,
    Document,
    SearchResult,
    DistanceMetric,
    VectorStoreError,
    EmbeddingError,
    SearchError,
    DocumentNotFoundError,
)

from src.infrastructure.base.graph_db import (
    GraphDatabase,
    Node,
    Relationship,
    Path,
    GraphDatabaseError,
    NodeNotFoundError,
    QueryError,
)

from src.infrastructure.base.blob_storage import (
    BlobStorage,
    BlobMetadata,
    BlobStorageError,
    BlobNotFoundError,
    BlobAlreadyExistsError,
)

from src.infrastructure.base.secrets import (
    SecretsManager,
    SecretMetadata,
    SecretsManagerError,
    SecretNotFoundError,
    SecretAccessDeniedError,
)

__all__ = [
    # Event Bus
    "EventBus", "EventMessage",
    "EventBusError", "PublishError", "SubscriptionError", "AcknowledgmentError",

    # Vector Store
    "VectorStore", "Document", "SearchResult", "DistanceMetric",
    "VectorStoreError", "EmbeddingError", "SearchError", "DocumentNotFoundError",

    # Graph Database
    "GraphDatabase", "Node", "Relationship", "Path",
    "GraphDatabaseError", "NodeNotFoundError", "QueryError",

    # Blob Storage
    "BlobStorage", "BlobMetadata",
    "BlobStorageError", "BlobNotFoundError", "BlobAlreadyExistsError",

    # Secrets Manager
    "SecretsManager", "SecretMetadata",
    "SecretsManagerError", "SecretNotFoundError", "SecretAccessDeniedError",
]