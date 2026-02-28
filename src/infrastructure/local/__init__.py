"""
Local/OSS infrastructure implementations.

These concrete classes implement the abstract base classes using
open-source, self-hosted technologies suitable for local development.

All implementations can run in Docker containers on a developer's laptop.
"""

from src.infrastructure.local.rabbitmq_event_bus import RabbitMQEventBus

__all__ = [
    'RabbitMQEventBus',
]