"""
Integration tests for RabbitMQEventBus.

These tests require RabbitMQ to be running (via docker-compose).
Run: docker compose up rabbitmq
"""

import sys
import time
from pathlib import Path
import threading
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.infrastructure.local.rabbitmq_event_bus import RabbitMQEventBus
from src.infrastructure.base.event_bus import EventMessage, PublishError


@pytest.fixture
def event_bus():
    """Create RabbitMQEventBus connected to local Docker instance."""
    bus = RabbitMQEventBus(
        host='localhost',
        port=5672,
        username='guest',
        password='guest',
        exchange='test_exchange',
        durable=False,  # Faster for tests
    )
    yield bus
    bus.close()


def test_connection(event_bus):
    """Test that connection to RabbitMQ succeeds."""
    assert event_bus._connection is not None
    assert not event_bus._connection.is_closed


def test_publish_single_message(event_bus):
    """Test publishing a single message."""
    message = {
        'repo': 'pytorch/pytorch',
        'commit': 'abc123',
        'files': ['torch/nn/modules/linear.py']
    }
    
    message_id = event_bus.publish(
        topic='test-topic',
        message=message,
        attributes={'source': 'test'}
    )
    
    assert message_id is not None
    assert isinstance(message_id, str)
    print(f"✓ Published message: {message_id}")


# def test_publish_and_receive(event_bus):
#     """Test end-to-end publish and subscribe."""
#     topic = 'test-publish-receive'
#     received_messages = []
    
#     def callback(message: EventMessage):
#         print(f"✓ Received message: {message.message_id}")
#         print(f"  Payload: {message.payload}")
#         received_messages.append(message)

#     # Subscribe (this would block forever, so we'll use a timeout trick)
#     # In a real test, you'd run subscribe in a thread
#     import threading
    
#     def subscribe_with_timeout():
#         try:
#             event_bus.subscribe(topic=topic, callback=callback)
#         except KeyboardInterrupt:
#             pass
    
#     subscriber_thread = threading.Thread(target=subscribe_with_timeout, daemon=True)
#     subscriber_thread.start()
#         # Give RabbitMQ a moment to route the message
#     time.sleep(0.5)
    
#     # Publish a message first
#     test_payload = {'test': 'data', 'value': 42}
#     message_id = event_bus.publish(topic=topic, message=test_payload)
#     print(f"✓ Published message: {message_id}")
    

#     # Wait for message to be received
#     timeout = 5
#     start = time.time()
#     while len(received_messages) == 0 and (time.time() - start) < timeout:
#         time.sleep(0.1)
    
#     # Verify we got the message
#     assert len(received_messages) == 1
#     assert received_messages[0].message_id == message_id
#     assert received_messages[0].payload == test_payload
#     assert received_messages[0].topic == topic
    
#     print("✓ End-to-end test passed!")

# # def test_publish_and_receive(event_bus):
#     """Test end-to-end publish and subscribe."""
#     topic = 'test-publish-receive'
#     received_messages = []
    
#     def callback(message: EventMessage):
#         print(f"✓ Received: {message.message_id}")
#         received_messages.append(message)
    
#     # Start subscriber in background thread
#     def subscribe_thread():
#         try:
#             event_bus.subscribe(topic=topic, callback=callback)
#         except Exception as e:
#             print(f"Subscriber error: {e}")
    
#     subscriber = threading.Thread(target=subscribe_thread, daemon=True)
#     subscriber.start()
    
#     # Give subscriber time to connect
#     time.sleep(0.5)
    
#     # Publish message
#     test_payload = {'test': 'data', 'value': 42}
#     message_id = event_bus.publish(topic=topic, message=test_payload)
#     print(f"✓ Published: {message_id}")
    
#     # Wait for message to be received
#     timeout = 5
#     start = time.time()
#     while len(received_messages) == 0 and (time.time() - start) < timeout:
#         time.sleep(0.1)
    
#     # Stop subscriber
#     event_bus.stop()
#     time.sleep(0.5)  # Let it clean up
    
#     # Verify
#     assert len(received_messages) == 1
#     assert received_messages[0].message_id == message_id
#     assert received_messages[0].payload == test_payload
    
#     print("✓ End-to-end test passed!")

def test_publish_and_receive(event_bus):
    topic = "test-publish-receive"
    received_messages = []

    ready_event = threading.Event()

    def callback(message: EventMessage):
        received_messages.append(message)

    def subscribe_thread():
        try:
            # Signal that thread started
            ready_event.set()
            event_bus.subscribe(topic=topic, callback=callback)
        except Exception as e:
            print(f"Subscriber error: {e}")

    subscriber = threading.Thread(target=subscribe_thread)
    subscriber.start()

    # Wait until subscriber thread starts
    ready_event.wait(timeout=5)

    # Give RabbitMQ a small deterministic setup window
    time.sleep(1)

    # Publish AFTER consumer is ready
    test_payload = {"test": "data", "value": 42}
    message_id = event_bus.publish(topic=topic, message=test_payload)

    timeout = 5
    start = time.time()
    while not received_messages and (time.time() - start) < timeout:
        time.sleep(0.1)

    event_bus.stop()
    subscriber.join(timeout=5)

    assert len(received_messages) == 1
    assert received_messages[0].message_id == message_id
    assert received_messages[0].payload == test_payload
    print("✓ End-to-end test passed!")


def test_publish_invalid_json(event_bus):
    """Test that non-serializable messages raise ValueError."""
    import datetime
    
    # datetime objects are not JSON-serializable by default
    message = {'timestamp': datetime.datetime.now()}
    
    with pytest.raises(ValueError, match="not JSON-serializable"):
        event_bus.publish(topic='test', message=message)
    print("✓ test_publish_invalid_json test passed!")


def test_callback_exception_handling(event_bus):
    topic = "test-callback-error"
    attempt_count = [0]

    def failing_callback(message: EventMessage):
        attempt_count[0] += 1
        print(f"Callback attempt {attempt_count[0]}")

        if attempt_count[0] < 3:
            raise ValueError("Simulated processing error")

    def subscribe_thread():
        event_bus.subscribe(topic=topic, callback=failing_callback)

    # Start subscriber FIRST
    thread = threading.Thread(target=subscribe_thread)
    thread.start()

    # Allow queue declaration + binding
    time.sleep(1)

    # Now publish
    event_bus.publish(topic=topic, message={"data": "test"})

    # Wait deterministically
    timeout = 6
    start = time.time()
    while attempt_count[0] < 3 and (time.time() - start) < timeout:
        time.sleep(0.1)

    print(attempt_count)
    assert attempt_count[0] == 3

    event_bus.stop()
    thread.join(timeout=5)
    event_bus.close()
    print("✓ Retry logic works correctly")


if __name__ == '__main__':
    # Run tests manually
    print("=" * 70)
    print("RabbitMQEventBus Integration Tests")
    print("=" * 70)
    print()
    
    print("Ensure RabbitMQ is running: docker compose up rabbitmq")
    print()
    
    bus = RabbitMQEventBus(
        host='localhost',
        port=5672,
        username='guest',
        password='guest',
        exchange='test_exchange',
    )
    
    try:
        test_connection(bus)
        print()
        
        test_publish_single_message(bus)
        print()
        
        test_publish_and_receive(bus)
        print()

        test_publish_invalid_json(bus)
        print()

        # test_callback_exception_handling(bus)
        # print()
        
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
    finally:
        # event_bus.stop()
        bus.close()
        
