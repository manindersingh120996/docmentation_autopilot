"""
┌─────────────────────────────────────────────────────────────┐
│ RabbitMQEventBus (implements EventBus ABC)                  │
│                                                              │
│  ┌────────────────────┐         ┌─────────────────────┐    │
│  │ Connection         │────────>│ RabbitMQ Server     │    │
│  │ (TCP socket)       │         │ (running in Docker) │    │
│  └────────────────────┘         └─────────────────────┘    │
│           │                                                  │
│           ├──> Channel 1 (for publishing)                   │
│           │                                                  │
│           └──> Channel 2 (for subscribing/consuming)        │
│                                                              │
│  publish() ──> Serialize → basic_publish() → Exchange       │
│                                               ↓              │
│                                            Routing           │
│                                               ↓              │
│                                            Queue             │
│                                               ↓              │
│  subscribe() <─ Deserialize <── Queue ← basic_consume()     │
│       │                                                      │
│       └──> callback(message) ──> ack/nack                   │
└─────────────────────────────────────────────────────────────┘


RabbitMQ implementattion of the EventBus Abstraction.

This implementation uses the pika library (RabbitMQ's Python client) to provide
reliable, asynchronous messaging between components of the Documentation Autopilot.

ARCHITECTURE:
    - One TCP connection per RabbitMQEventBus instance (expensive to create)
    - Multiple Channels on that connection (lightweight, one per operation)
    - Direct exchange for simple topic-based routing
    - Manual acknowledgment for reliability (at-least-once deliver)
    - Dead letter queue for messages that fails repeatedly.

CONFIGURATION:
    Reads connection parameters from config/local.yaml:
    - host: RabbitMQ server hostname (localhost for Docker)
    - port: AMQ port (5672 default)
    - username/password: Authentication credentials
    - exchange: Exchange name for routing
    - queue: Queue name for this topic.
    - durable: Whether queue survives broker restarts:

USAGE:
    This class is instantiated by the infrastructure factory based on
    configuration. Application code never imports this directly.

    from src.infrastructure.factory import get_factory
    event_bus = get_factory().get_event_bus() # Returns RabbitMQEventBus

    # Publishing
    event_bus.publish(topic="code-changes", message={"repo":"pytorch"})

    # subscribing
    def handle(msg):
        print(msg.payload)

    event_bus.subscribe(topic="code-chagnes", callback = handle)

"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from src.infrastructure.base.event_bus import (
    EventBus,
    EventMessage,
    PublishError,
    SubscriptionError,
    AcknowledgmentError
)

logger = logging.getLogger(__name__)

class RabbitMQEventBus(EventBus):
    """
    RabbitMQ implementation of EventBus.

    Provides reliable message delivery with at-elast-once gurantees.
    Messages are persisted to disk and survive broker restarts.

    Thread Safety:
        - publish() is thread-safe (uses a lock to protect channel acceess)
        - subscribe() should only be called once per topic per instance.
        - Connection is shared across threads but channels are not
    
    """

    def __init__(self,
                 host: str,
                 port: int,
                 username: str,
                 password: str,
                 exchange: str,
                 virtual_host: str = "/",
                 durable: bool = True,
                 connection_attempts: int = 3,
                 retry_delay_seconds: int = 2,
                 ):
        """
        Initialize RabbitMQ connection and declare exchange.

        This establishes the TCP connection to RabbitMQ and sets up the
        exchange (routing endpoint). Queues are created lazily when
        subscribe() is called for a topic.

        Args:
            host:                   RabbitMQ server hostname (e.g., "localhost")
            port:                   AMQP port (default 5672)
            username:               Authentication username (default "guest")
            password:               Authentication password (default "guest")
            exchange:               Exchange name for message routing
            virtual_host:           RabbitMQ virtual host (namespace for resources)
            durable:                If True, exchange/queues survive broker restart
            connections_attempts:   How many times to retry connection on failure.
            retry_delay_seconds:    Delay between connection retry attempts.

        Raises:
            AMQPConnectionError: If conection fails after all retries
        """

        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.exchange = exchange
        self.virtual_host = virtual_host
        self.durable = durable
        self.connection_attempts = connection_attempts
        self.retry_delay_seconds = retry_delay_seconds

        # Connection and channel management
        self._connection: Optional[pika.BlockingConnection] = None
        self._publish_channel: Optional[BlockingChannel] = None

        # Track active subscriptions (topic -> consuming state)
        self._subscriptions: Dict[str,bool] = {}

        logger.info(
            f"Initializing RabbitMQEventBus: {username}@{host}:{port}/{virtual_host},"
            f"exchange='{exchange}'"
        )

        # Establish connection on initialization
        self._connect()
        import threading

        self._stop_event = threading.Event()

        # Declare the exchange (crewates is if it doesn't exist)
        self._declare_exchange()

    def _connect(self) -> None:
        """
        Establish connection to RabbitMQ with retry logic.

        Creates a BlockingConnection whcih opens a TCP socket to RabbitMQ
        and performs authentication. This is an expensive operation, so we do
        it once and reuse the Connection/.

        Raises:
            AMQPConnectionError: If all connection attempts fail.
        """

        credentials = pika.PlainCredentials(self.username, self.password)

        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=credentials,
            # Heartbeat every 60 seconds to detect dead connections
            heartbeat=60,
            # Timeout for blocking operations
            blocked_connection_timeout=300,
        )

        last_error = None

        for attempt in range(1, self.connection_attempts + 1):
            try:
                logger.info(
                    f"Connecting to RabbitMQ at {self.host}:{self.port}"
                    f"(attempt {attempt}/{self.connection_attempts})"
                )
                self._connection = pika.BlockingConnection(parameters)

                logger.info(
                    f"✓ Successfully connected to RabbitMQ at {self.host}:{self.port}"
                )
                return

            except AMQPConnectionError as e:
                last_error = e
                logger.warning(
                    f"Connection attempt {attempt}/{self.connection_attempts} failed: {e}"
                )

                if attempt < self.connection_attempts:
                    logger.info(f"Retrying in {self.retry_delay_seconds} seconds...")
                    time.sleep(self.retry_delay_seconds)

        # All attempts failed
        error_msg = (
            f"Failed to connect to RabbitMQ after {self.connection_attempts} attempts. "
            f"Last error: {last_error}"
        )
        logger.error(error_msg)
        raise AMQPConnectionError(error_msg)
    
    def _declare_exchange(self) -> None:
        """
        Declare (create if needed) the exchange for routing messages.

        We use a 'direct' exchange, which routes messages to queues based on 
        exact routing key match, This is simple and suggicieant for our use case.

        The exchange is marked 'durable' so it survives broker restarts.
        
        Raises:
            AMQPChannelError: If exchange declation fails:

        """
        # Create a temporary channel just for this declaration
        # (We'll create dedicated channels for publishing and consuming later)
        # channel = self._connection.channel()
        # Create a NEW connection for this consumer thread
        try:
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=pika.PlainCredentials(self.username, self.password),
            )
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
        except Exception as e:
            raise SubscriptionError(f"Failed to create consumer connection: {e}") from e

        try:
            channel.exchange_declare(
                exchange=self.exchange,
                exchange_type='direct',  # Route by exact routing key match
                durable=self.durable,    # Survive broker restart
                auto_delete=False,       # Don't delete when no queues bound
            )
            
            logger.info(f"✓ Exchange '{self.exchange}' declared (type: direct, durable: {self.durable})")
            
        finally:
            # Close this temporary channel (connection stays open)
            # channel.close()
                try:
                    if channel.is_open:
                        channel.close()
                    if connection.is_open:
                        connection.close()
                except Exception:
                    pass

    def _get_publish_channel(self) -> BlockingChannel:
        """
        Get or create a channel for publishing messages.

        We use a dedicated channel for publishing and reuse it for all 
        publish operations. This is more efficient than creating a new 
        channel for each publish.

        Returns:
            BlockingChannel ready for publishing
        
        Raises:
            AMQPConnectionError: If connection is closed
        """
        if self._publish_channel is None or self._publish_channel.is_closed:
            if self._connection is None or self._connection.is_closed:
                # Connetion died, reconnect
                logger.warning("Connection closed, reconnecting...")
                self._connect()
                self._declare_exchange()
            
            self._publish_channel = self._connection.channel()
            logger.debug("Created new channel for publishing")
        
        return self._publish_channel
    
    def publish(
            self,
            topic: str,
            message: Dict[str, Any],
            attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Publish a message tyo a topic.

        The message is serialized to JSON and sent to exchange with
        the topic as the routing key. RabbitMQ routes it to queues bound
        to this topic.

        The message is marked 'persistent' so it's written to disk before
        RabbitMQ acknowledges receipt. This ensures the message survives
        broker restarts (at-least-once delivery guarantee).

        Args:
            topic:      Topic name (used as routing key). Example: "code-change"
            message:    Message payload as a JSON-serializable dictionary
            attributes: Optional metadata (correlation ID, source, etc.)

        RETURNS:
            Unique message ID (UUID) to this message
        
        Raises:
            PublishError:   If publishing fails
            ValueError:     If message is not JSON-serializable
        """

        # Generate the unique message ID
        message_id = str(uuid.uuid4())

        # wrap message with metadata
        envelope = {
            'message_id' : message_id,
            'topic' : topic,
            'payload' : message,
            'timestamp' : datetime.now(timezone.utc).isoformat(),
            'attributes' : attributes or {}
        }

        # Serialize to JSON
        try:
            body = json.dumps(envelope)
        except (TypeError,ValueError) as e:
            raise ValueError(f"Message payload is not JSON-serializable: {e}")
        
       
        # Prepare message properties
        properties = pika.BasicProperties(
            delivery_mode=2,           # 2 = persistent (survives broker restart)
            content_type='application/json',
            content_encoding='utf-8',
            message_id=message_id,
            timestamp=int(time.time()),
            headers=attributes or {},
        )
        
        # Publish to exchange
        try:
            channel = self._get_publish_channel()
            
            channel.basic_publish(
                exchange=self.exchange,
                routing_key=topic,      # Topic becomes routing key
                body=body.encode('utf-8'),
                properties=properties,
                mandatory=False,        # Don't return unroutable messages
            )
            
            logger.debug(
                f"Published message {message_id} to topic '{topic}' "
                f"({len(body)} bytes)"
            )
            
            return message_id
            
        except (AMQPConnectionError, AMQPChannelError) as e:
            error_msg = f"Failed to publish message to topic '{topic}': {e}"
            logger.error(error_msg)
            raise PublishError(error_msg) from e

    def subscribe(
        self,
        topic: str,
        callback: Callable[[EventMessage], None],
        subscription_name: Optional[str] = None
    ) -> None:
        """
        Subscribe to a topic and process messages via callback.
        
        This method BLOCKS indefinitely, running a consumer loop that:
        1. Pulls messages from the queue for this topic
        2. Deserializes each message into an EventMessage
        3. Calls your callback function with the message
        4. Acknowledges the message if callback succeeds
        5. Negatively acknowledges (requeues) if callback raises exception
        
        The queue is created if it doesn't exist, and bound to the exchange
        with the topic as the routing key.
        
        Args:
            topic:             Topic to subscribe to (becomes queue name)
            callback:          Function to call for each message
            subscription_name: Optional name (unused in RabbitMQ, for API compatibility)
        
        Raises:
            SubscriptionError: If subscription setup fails
        
        Note:
            This method never returns unless an unrecoverable error occurs.
            Run it in a separate thread or process if you need non-blocking behavior.
        """
        if topic in self._subscriptions:
            logger.warning(f"Already subscribed to topic '{topic}', ignoring duplicate subscription")
            return
        
        logger.info(f"Setting up subscription to topic '{topic}'")
        
        # Create a dedicated channel for consuming (separate from publishing channel)
        # try:
        #     channel = self._connection.channel()
        # except Exception as e:
        #     raise SubscriptionError(f"Failed to create channel for subscription: {e}") from e
        try:
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=pika.PlainCredentials(self.username, self.password),
                heartbeat=60,
                blocked_connection_timeout=300,
            )
            consumer_connection = pika.BlockingConnection(parameters)
            channel = consumer_connection.channel()
            
            logger.debug(f"Created dedicated connection for consumer thread (topic: {topic})")
            
        except Exception as e:
            raise SubscriptionError(f"Failed to create consumer connection: {e}") from e
    
        # Declare the queue (creates it if it doesn't exist)
        try:
            channel.queue_declare(
                queue=topic,
                durable=self.durable,     # Survive broker restart
                exclusive=False,          # Allow multiple consumers
                auto_delete=False,        # Don't delete when no consumers
                arguments={
                    # Dead letter exchange - messages that fail repeatedly go here
                    'x-dead-letter-exchange': f'{self.exchange}.dlx',
                    'x-dead-letter-routing-key': f'{topic}.dead',
                    # Maximum delivery attempts before dead-lettering
                    'x-max-length': 100000,  # Prevent unbounded queue growth
                }
            )
            
            logger.info(f"✓ Queue '{topic}' declared (durable: {self.durable})")
            
        except AMQPChannelError as e:
            raise SubscriptionError(f"Failed to declare queue '{topic}': {e}") from e
        
        # Bind the queue to the exchange with topic as routing key
        try:
            channel.queue_bind(
                queue=topic,
                exchange=self.exchange,
                routing_key=topic,
            )
            
            logger.info(f"✓ Queue '{topic}' bound to exchange '{self.exchange}'")
            
        except AMQPChannelError as e:
            raise SubscriptionError(f"Failed to bind queue '{topic}': {e}") from e
        
        # Set prefetch count (QoS) - how many unacked messages consumer can have
        # Setting this to 1 ensures fair distribution across multiple workers
        channel.basic_qos(prefetch_count=1)
        
        # Define the callback wrapper that handles acknowledgment
        def on_message_callback(ch, method, properties, body):
            """
            Internal callback invoked by pika when a message arrives.
            
            This wraps the user's callback to handle deserialization,
            acknowledgment, and error handling.
            """
            delivery_tag = method.delivery_tag
            
            try:
                # Deserialize the message envelope
                envelope = json.loads(body.decode('utf-8'))
                
                # Construct EventMessage from envelope
                event_message = EventMessage(
                    message_id=envelope['message_id'],
                    topic=envelope['topic'],
                    payload=envelope['payload'],
                    timestamp=datetime.fromisoformat(envelope['timestamp']),
                    delivery_attempt=1,  # RabbitMQ doesn't track this directly
                    attributes=envelope.get('attributes'),
                )
                
                logger.debug(f"Received message {event_message.message_id} from topic '{topic}'")
                
                # Call the user's callback
                try:
                    callback(event_message)
                    
                    # Callback succeeded - acknowledge the message
                    ch.basic_ack(delivery_tag=delivery_tag)
                    
                    logger.debug(f"✓ Acknowledged message {event_message.message_id}")
                    
                except Exception as callback_error:
                    # Callback raised an exception - nack with requeue
                    logger.error(
                        f"Callback failed for message {event_message.message_id}: {callback_error}",
                        exc_info=True
                    )
                    
                    ch.basic_nack(
                        delivery_tag=delivery_tag,
                        requeue=True  # Put back in queue for retry
                    )
                    
                    logger.warning(f"✗ Requeued message {event_message.message_id} after callback failure")
            
            except json.JSONDecodeError as e:
                # Message body is malformed JSON - can't be processed
                # Dead-letter it (don't requeue, it will fail again)
                logger.error(f"Message has invalid JSON, dead-lettering: {e}")
                ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
            
            except Exception as e:
                # Unexpected error in deserialization or message handling
                logger.error(f"Unexpected error processing message: {e}", exc_info=True)
                # Requeue in case it's a transient issue
                ch.basic_nack(delivery_tag=delivery_tag, requeue=True)
        
        # Start consuming messages
        try:
            logger.info(f"Starting consumer for topic '{topic}' (blocking call)")
            
            channel.basic_consume(
                queue=topic,
                on_message_callback=on_message_callback,
                auto_ack=False,  # Manual acknowledgment for reliability
            )
            
            self._subscriptions[topic] = True
            
            # This call BLOCKS FOREVER, processing messages as they arrive
            # channel.start_consuming()
            try:
                while not self._stop_event.is_set():
                    consumer_connection.process_data_events(time_limit=1)
            except KeyboardInterrupt:
                pass
            
        except KeyboardInterrupt:
            logger.info(f"Received interrupt signal, stopping consumer for topic '{topic}'")
            channel.stop_consuming()
            
        except Exception as e:
            error_msg = f"Subscription to topic '{topic}' failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise SubscriptionError(error_msg) from e
    def stop(self):
        """
        Signal all consumers to stop processing.
        """
        self._stop_event.set()
    
    def acknowledge(self, message_id: str) -> None:
        """
        Manually acknowledge a message.
        
        This is not typically used because subscribe() handles acknowledgment
        automatically. It's provided for API completeness.
        
        Args:
            message_id: ID of the message to acknowledge
        
        Raises:
            NotImplementedError: Manual ack requires tracking delivery tags,
                                which is complex with the current architecture
        """
        raise NotImplementedError(
            "Manual acknowledgment is not supported in this implementation. "
            "Use the subscribe() callback pattern, which handles acknowledgment automatically."
        )
    
    def nack(self, message_id: str, requeue: bool = True) -> None:
        """
        Manually negatively acknowledge a message.
        
        Not implemented for the same reason as acknowledge().
        
        Args:
            message_id: ID of the message to nack
            requeue:    Whether to requeue the message
        
        Raises:
            NotImplementedError: See acknowledge() for explanation
        """
        raise NotImplementedError(
            "Manual nack is not supported in this implementation. "
            "Raise an exception in your subscribe() callback to trigger automatic nack."
        )
    
    def close(self) -> None:
        """
        Close all channels and the connection to RabbitMQ.
        
        This releases network resources and ensures graceful shutdown.
        After calling close(), this instance should not be used again.
        """
        logger.info("Closing RabbitMQ connection...")
        
        # Close publish channel
        if self._publish_channel and not self._publish_channel.is_closed:
            try:
                self._publish_channel.close()
                logger.debug("✓ Closed publish channel")
            except Exception as e:
                logger.warning(f"Error closing publish channel: {e}")
        
        # Close connection (automatically closes all channels)
            try:
                self._stop_event.set()

                if self._connection and self._connection.is_open:
                    self._connection.close()

            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        # if self._connection and not self._connection.is_closed:
        #     try:
        #         self._connection.close()
        #         logger.info("✓ Closed RabbitMQ connection")
        #     except Exception as e:
        #         logger.warning(f"Error closing connection: {e}")
        
        self._connection = None
        self._publish_channel = None
        self._subscriptions.clear()

    