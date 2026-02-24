"""
Abstract base class for event bus implementations.

An EventBus provides publish-subscribe messaging between system components.
It decouples producers from consumers and provides reliable, asynchronous
message delivery with at-least-once guarantees.

WHY THIS EXISTS:
    Layer 1 (webhook receiver) publishes ChangeEvents.
    Layer 2 (analysis workers) consumes ChangeEvents.
    They must never be directly coupled — if the analysis worker is slow,
    the webhook receiver must not block. The EventBus sits between them,
    buffering events and ensuring delivery even if workers crash.

WHAT THIS ABSTRACTS:
    Local development: RabbitMQ (running in Docker)
    Production (GCP):  GCP Pub/Sub

    The webhook receiver code calls event_bus.publish() without knowing
    or caring which system is underneath. Config decides the implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Data Transfer Objects (DTOs)
# These are plain data containers — no logic, no methods (except dataclass
# defaults). They define the "shape" of data passed through the event bus.
# ---------------------------------------------------------------------------

@dataclass
class EventMessage:
    """
    A message travelling through the event bus.

    This wraps the raw payload with metadata needed for processing,
    tracking, and acknowledgment. Every message the event bus delivers
    to a consumer is wrapped in this structure.

    Why wrap at all? Because consumers need more than just the payload:
    - message_id: needed to acknowledge (confirm processing succeeded)
    - delivery_attempt: so consumers can handle retries differently
    - attributes: for correlation IDs, tracing, source tagging
    - timestamp: for latency measurement and ordering

    Attributes:
        message_id:       Unique ID assigned by the event bus (not us)
        topic:            Which topic this message came from
        payload:          The actual data (arbitrary JSON-serializable dict)
        timestamp:        When the message was published
        delivery_attempt: How many times delivery was attempted (1 = first try)
        attributes:       Optional key-value metadata (correlation IDs, etc.)
    """
    message_id: str
    topic: str
    payload: Dict[str, Any]
    timestamp: datetime
    delivery_attempt: int = 1
    attributes: Optional[Dict[str, str]] = None

# -------------------------------------------------------------------------
# Exception Hierarchy
# 
# WHY DEFINE EXCEPTIONS HERE (not in common/exceptions.py)
# These exceptionds are specific to the EventBus Contract. When you import
# EventBus, you get its exceptions too , they travel together. We donot
# wat to scatter event-bus specific errors into a global exceptions file.

# However, they INHERIT from the common base classes defined in 
# src/common/excetpions.py so callers can catch at any level

# -----------------------------------------------------------------------

class EventBusError(Exception):
    """
    Base exception for all event bus errors.

    Catch this to handle any event bus failure regardless of specifics
    Catch subclasses for fine-grained handling.
    """
    pass

class PublishError(EventBusError):
    """Raised when publishing a message to the event bus fails.
    
    Common causes:
    - Network connection to RabbitMQ/Pub/Sub is down.
    - Topic/exchange doesn't exist
    - Message payload is not JSON-Serializable
    - Message size exceeds limits

    Callers should retry PublishErrors with exponential backoff
    because they're often transient (temporary network blips)
    """
    pass

class SubscriptionError(EventBusError):
    """
    Raised when setting up a subscription leads to failure
    
    Common causes:
        - Topic doesn't exist
        - Insufficient permissions
        - Invalid subscriptions configuration


    SubscriptionErrors at startup usually indicate misconfiguration,
    not transient failures, fix the config rather than retrying
    """
    pass

class AcknowledgmentError(EventBusError):
    """
    Raised when acknowledging or negatively acknowledging a message fails.
    
    This is a rare but serious error, if you can't acknowledge, the
    message will be redelivered even though you processed it, This can
    cause duplicate processing, Log and alert when this occurs.
    """
    pass

# ---------------------------------------------------------------------------
# The Abstract Base Class
# ---------------------------------------------------------------------------


class EventBus(ABC):
    """
    Docstring for EventBus
    --------------------------

    Abstract interface for event bus(message queue) implementations.

    CORE CONCEPT:
        Publishers call publish() to send events.
        Consumers call subscribe() with a callback function.
        The event bus delivers messages to callbacks as they arrive.
        Messages are retained until explicitly acknowledged.

    DELIVERY GUARANTEE:
        At-least-once: a message is delivered at least once, possibly more
        if the consumer crashes before acknowledging. COnsumers must be
        idempotent (safe to prcoess the same message multiple times).

    THREAD SAFETY:
        Implementations must be thread-safe for publish(), multiple threads may publish
        concurrently. subscribe() typically runs in one thread.

    RESOURCE MAGANEMENT:
        Always close the event bus when done (or use as context manager).
        Unclosed connections leak file descriptors and keep queues open.


        Preffered usage:
            with event_bus:
                event_bus.publish(---)
                event_bus.subscribe(---)

        Or Manually:
            try:
                event_bus.publish(...)
            finally:
                event_bus.close(...)
    """

    @abstractmethod
    def publish(
        self,
        topic: str,
        message: Dict[str,Any],
        attributes: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Publish a message to a named topic.

        The message is durably persisted before this method returns.
        If publish() returns without raising, the message WILL be delivered
        to at least one consumer, even if consumers are offline right now.

        ARGs:
            topic:      Topic name (e.g., "code-changes","approval-requests")
            message:    Payload as a JSON-serializable dictionary.
                        Raise ValueError if not serializable.
            attributes: Optional metadata dict (string keys and values only).
                        Use for : correlations IDs, source identifiers, priority.
                        Example: {"correlation_id": "abc-123", "source":"webhook"}

        Retruns:
            message)_id: Unique ID assigned to this message by the bus.
                            Useful for logging, tracing, and deduplication.
        
        Raises:
            PublishError:   If the message could not be published
            ValueError :    If message payload is not JSON-serializable.

        Example:
            msg_id = event_bus.publish(
                        topic="code-changes",
                        message={
                            "repo":"pytorch/pytorch",
                            "commit_sha": "abs123456",
                            "changed_files": ["torch/nn/modules/linear.py"]
                },
                attributes={"source": "github-webhook", "correlation_id": "req-789"}
            )
            logger.info(f"Published change event: {msg_id}")
        """
        pass

    @abstractmethod
    def subscribe(
        self,
        topic: str,
        callback: Callable[[EventMessage], None],
        subscription_name: Optional[str] = None
    ) -> None:
        """
        Subscribe to a topic and receive messages via callback function.

        Tthis method blocks indefinitely, running a consumer loop:
            while True:
                message = pull_next_message()
                try:
                    callback(message)       # Your function is called here
                    acknowledge(message)    # Auto-ack on success
                except Exception:
                    nack(message)           # Auto-nack on failure -> redelivery

        The callback function must:
            - Process the message (or raise an exception if it can't)
            - Return normally on success (triggers auto-acknowledgement)
            - Raise an exception on failure (triggers redelivery)
            - Be idemptoent (safe to call multiple times for same message)

        Multiple workers can subscribe to the same topic simultaneously.
        The event bus distribtues messages across workers (load balancing).
        Each message goes to exactly ONE worker, not all of them.

        Args:
            topic:              Topic name to subscribe to.
            callback:           Function called for each message.
                                Signature: (message: EventMessage) -> None
            subscription_name:  Optional name for this subscription.
                                Required by some implementations (GCP Pub/Sub).
                                Allows multiple independent subscribers to the
                                same topic (each gets all messages).
        
        Raises:
            SubscriptionError: If subscription setup fails.

        Example:
            def handle_code_change(message: EventMessage) -> None:
                change_event = message.payload
                repo = change_event["repo"]
                commit = change_event["commit_sha"]
                logger.info(f"Processing change: {repo} @ {commit}")
                analysis = analyze_code_change(change_event)
                store_analysis(analysis)
                # Return normally -> message is auto-acknowledged

            event_bus.subscribe(
                topic="code-changes",
                callback = handle_code_change,
                subscription_name = "analysis-worker"
                )
            # THis line never executes, subscribe() blocks forever
        """
        pass
    
    @abstractmethod
    def acknowledge(self,message_id:str) -> None:
        """
        Docstring for acknowledge
        --------------------------------------------

        Manually acknowledge that a message was processed successfully.

        In most cases you do not needt this, subscribe() auto-acknowledges
        when your callback returns normally. Use this only when you need
        manual control over acknowledgement timing.

        Args:
            message_id: The message_id from the EventMessage you received.

        Riases:
            AcknowledgmentError: If acknowledgement fails.

        When to use manual acknowledgement:
            - Batch processing: you want to accumulate messages and
                acknowledge them all at once after processing the batch.
            - You need to do post-processing after returning from callback
                but before acknowledging.
        """
        pass

    @abstractmethod
    def nack(self,
             message_id: str,
             requeue: bool = True) -> None:
        """
        Docstring for nack
        ------------------------------

        Negatively acknowledge a message (signal that processing failed).

        Tells the event bus: "I couldn't process this message."
        The bus will redeliver it to another consumer (if requeue = True)
        or move it to a dead-letter queue (if requeue = False or max retries hit).

        Args:
            message_id: The message_id from the EventMessage.
            requeu:     True -> put back in queue for another consumer (default)
                        False -> send to dead-letter queue (give up on this message)

        Raises:
            AcknowledgmentError:    If negative acknowledgment fails.
        
        When to use:
            - Transient failures (DB down): requeue = True, will retry later.
            - Permanent failures (malformed message): requeue = False, give up
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Docstring for close
        ------------------------------------------------------------

        Close connections and release all resources.

        Must be called when shutting down to ensure:
            - Network connections are closed gracefully
            - Pending messages are flushed( not lost)
            - Subscriptions are cancelled cleanly
            - No resource leaks (dile descriptors, threads)

        After close() is called, this instance must not be used again.
        Create a new instance if you need to reconnect.


        """
        pass

    # ------------------------------------------------------------------
    # Context Manager Support
    # These two methods enable the "with" statement pattern.
    # They are NOT abstract — all subclasses inherit this behavior.
    # ------------------------------------------------------------------

    def __enter__(self) -> "EventBus":
        """Called when entring a 'with' block. Returns self."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Called when exiting a 'with' block. Ensures close() is alwyas called."""
        self.close()
        return False # False means: don't suppress exceptions
        

