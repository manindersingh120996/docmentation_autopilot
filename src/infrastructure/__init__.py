"""
Infrastructure layer ,external service abstractions and implementations.

Structure:
    base/   → Abstract interfaces (contracts)
    local/  → Concrete implementations for local development (RabbitMQ, ChromaDB, etc.)
    gcp/    → Concrete implementations for production (Pub/Sub, Vertex AI, etc.)

Application code should ONLY import from base/.
The factory (coming in Phase 2D) decides which concrete implementation to use.
"""