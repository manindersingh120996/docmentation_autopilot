"""
Docstring for doc-auto-pilot.src.common.exceptions

Custom Excetpion hierarchy for the application.

All application-specific exceptions inherit from a base exception,
making it easy to catch all our errors vs. system errors.
"""

class DocumentationAutopilotError(Exception):
    """
    Base exception for all application errors.

    All custom exceptions should inherit from this so application code
    can distinguish between errors we explicitly raise (which are often
    recoverable or expected) vs. system errors (which might indicate bugs)
    """
    pass

class ConfigurationError(DocumentationAutopilotError):
    """
    Raised when configuration is invalid or cannot be loaded.

    This might happen if:
    - Required config files are missing
    - YAML syntax is invalid
    - Required environment vairables are not set
    - Config values are out of valid ranges
    """
    pass

class InfrastructureError(DocumentationAutopilotError):
    """
    Base exception for infrastructure-related errors:

    Covers errors from databases, message queues, storage, etc.
    """
    pass

class EventBusError(InfrastructureError):
    """Base exception for event bus erroors."""
    pass

class PublishError(EventBusError):
    """Raised when publishing a message fails."""
    pass

class SubscriptionError(EventBusError):
    """Raised when subscribing to a topic fails"""
    pass

class VectorStoreError(InfrastructureError):
    """Base exception for vector store errors."""
    pass


class SearchError(VectorStoreError):
    """Raised when vector search fails."""
    pass


class GraphDatabaseError(InfrastructureError):
    """Base exception for graph database errors."""
    pass


class NodeNotFoundError(GraphDatabaseError):
    """Raised when a referenced node doesn't exist."""
    pass


class QueryError(GraphDatabaseError):
    """Raised when a graph query fails."""
    pass


class BlobStorageError(InfrastructureError):
    """Base exception for blob storage errors."""
    pass


class BlobNotFoundError(BlobStorageError):
    """Raised when a blob doesn't exist."""
    pass


class SecretsManagerError(InfrastructureError):
    """Base exception for secrets management errors."""
    pass


class SecretNotFoundError(SecretsManagerError):
    """Raised when a secret doesn't exist."""
    pass


class SecretAccessDeniedError(SecretsManagerError):
    """Raised when access to a secret is denied."""
    pass


