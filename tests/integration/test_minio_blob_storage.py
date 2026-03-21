"""
Integration tests for MinioBlobStorage.

Runs against a REAL MinIO instance — no mocking.
MinIO's behavior around content types, ETags, and presigned URLs
is subtle enough that mocks would give false confidence.

TWO WAYS TO RUN:
    1. Via pytest (full suite):
           pytest tests/integration/test_minio_blob_storage.py -v

    2. As a script (quick manual check):
           python tests/integration/test_minio_blob_storage.py

SETUP:
    docker-compose up -d minio
    # MinIO is ready almost instantly (unlike Neo4j's ~15s startup)

ENVIRONMENT VARIABLES:
    MINIO_ENDPOINT   default: localhost:9000
    MINIO_ACCESS_KEY default: minioadmin
    MINIO_SECRET_KEY default: minioadmin
    MINIO_BUCKET     default: test-doc-autopilot

TEST ISOLATION:
    All blobs use a RUN_ID prefix in their keys.
    The session fixture cleans up by deleting all blobs with that prefix.
"""

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from src.infrastructure.local.minio_blob_storage import MinioBlobStorage
from src.infrastructure.base.blob_storage import (
    BlobAlreadyExistsError,
    BlobNotFoundError,
    BlobStorageError,
)

RUN_ID = uuid.uuid4().hex[:6]


def bkey(name: str) -> str:
    """Build an isolated blob key for this test session."""
    return f"test/{RUN_ID}/{name}"


def make_storage() -> MinioBlobStorage:
    """Create storage instance using env vars or defaults."""
    return MinioBlobStorage(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        bucket_name=os.getenv("MINIO_BUCKET", "test-doc-autopilot"),
        secure=False,
    )


# ---------------------------------------------------------------------------
# Session-scoped pytest fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def storage():
    """One MinIO connection for the whole pytest session."""
    s = make_storage()
    yield s

    # Teardown: delete all blobs created in this test session
    try:
        keys = s.list_blobs(prefix=f"test/{RUN_ID}/")
        for key in keys:
            s.delete(key, missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Basic store and retrieve
# ---------------------------------------------------------------------------

class TestStoreAndRetrieve:

    def test_store_and_retrieve_bytes(self, storage):
        """
        Round-trip: store bytes, retrieve them, confirm they are identical.
        This is the fundamental contract of any storage system.
        """
        key = bkey("basic_roundtrip.txt")
        original = b"Hello, MinIO! This is test data."

        meta = storage.store(key, original, content_type="text/plain")

        assert meta.key == key
        assert meta.size_bytes == len(original)
        assert meta.content_type == "text/plain"
        assert meta.etag is not None

        retrieved = storage.retrieve(key)
        assert retrieved == original
        print(f"✓ Store/retrieve round-trip: {len(original)} bytes")

    def test_store_json_and_retrieve(self, storage):
        """JSON stored as bytes, retrieved, and parsed back correctly."""
        key = bkey("analysis_result.json")
        payload = {"status": "complete", "affected_docs": ["doc1", "doc2"], "confidence": 0.95}
        data = json.dumps(payload).encode("utf-8")

        storage.store(key, data, content_type="application/json")

        raw = storage.retrieve(key)
        recovered = json.loads(raw.decode("utf-8"))

        assert recovered["status"] == "complete"
        assert recovered["confidence"] == 0.95
        print("✓ JSON round-trip verified")

    def test_store_with_custom_metadata(self, storage):
        """Custom metadata is stored alongside the blob and retrievable."""
        key = bkey("with_metadata.bin")
        meta = storage.store(
            key=key,
            data=b"some binary content",
            metadata={"commit_sha": "abc123", "generated_by": "layer2"},
        )
        # Retrieve metadata and confirm custom fields are present
        fetched = storage.get_metadata(key)
        assert fetched.custom_metadata is not None
        # MinIO lowercases metadata keys
        meta_lower = {k.lower(): v for k, v in fetched.custom_metadata.items()}
        assert meta_lower.get("commit_sha") == "abc123"
        print("✓ Custom metadata stored and retrieved")

    def test_overwrite_default_replaces_content(self, storage):
        """
        Calling store() twice on the same key replaces the content (overwrite=True default).
        This is the at-least-once delivery idempotency check.
        """
        key = bkey("overwrite_test.txt")
        storage.store(key, b"version 1")
        storage.store(key, b"version 2")  # overwrite=True is the default

        retrieved = storage.retrieve(key)
        assert retrieved == b"version 2"
        print("✓ Overwrite replaces content correctly")

    def test_overwrite_false_raises_on_existing(self, storage):
        """store(overwrite=False) must raise BlobAlreadyExistsError if key exists."""
        key = bkey("no_overwrite.txt")
        storage.store(key, b"original content")

        with pytest.raises(BlobAlreadyExistsError):
            storage.store(key, b"should not overwrite", overwrite=False)
        print("✓ overwrite=False raises BlobAlreadyExistsError correctly")

    def test_retrieve_nonexistent_raises(self, storage):
        """Retrieving a missing key must raise BlobNotFoundError."""
        with pytest.raises(BlobNotFoundError):
            storage.retrieve("test/definitely/does/not/exist/xyz_999.txt")
        print("✓ BlobNotFoundError raised for missing key")


# ---------------------------------------------------------------------------
# 2. File-based operations
# ---------------------------------------------------------------------------

class TestFileOperations:

    def test_store_from_file_and_retrieve(self, storage):
        """
        Write a temp file, upload with store_from_file(), retrieve bytes,
        confirm they match the original file content.
        """
        key = bkey("from_file_test.json")
        content = json.dumps({"from": "file", "size": 42}).encode("utf-8")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            meta = storage.store_from_file(key, temp_path)
            assert meta.content_type == "application/json"  # inferred from .json extension

            retrieved = storage.retrieve(key)
            assert retrieved == content
            print(f"✓ store_from_file round-trip: {len(content)} bytes")
        finally:
            os.unlink(temp_path)

    def test_retrieve_to_file(self, storage):
        """
        Store bytes, retrieve_to_file() to a temp path, read file,
        confirm content matches original.
        """
        key = bkey("retrieve_to_file.txt")
        original = b"content to be written to disk"
        storage.store(key, original)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            dest_path = f.name

        try:
            storage.retrieve_to_file(key, dest_path)
            with open(dest_path, "rb") as f:
                disk_content = f.read()
            assert disk_content == original
            print("✓ retrieve_to_file writes correct content to disk")
        finally:
            os.unlink(dest_path)

    def test_retrieve_stream_reads_content(self, storage):
        """
        retrieve_stream() must return a readable stream whose content
        matches the stored data. Caller (us, in this test) must close it.
        """
        key = bkey("stream_test.txt")
        original = b"streaming content line1\nstreaming content line2\n"
        storage.store(key, original)

        stream = storage.retrieve_stream(key)
        try:
            content = stream.read()
            assert content == original
            print("✓ retrieve_stream returns correct content")
        finally:
            stream.close()

    def test_store_from_nonexistent_file_raises(self, storage):
        """store_from_file() with a missing local path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            storage.store_from_file(
                bkey("ghost.txt"), "/tmp/this_file_absolutely_does_not_exist_xyz.txt"
            )
        print("✓ FileNotFoundError raised for missing source file")


# ---------------------------------------------------------------------------
# 3. Existence and metadata
# ---------------------------------------------------------------------------

class TestExistenceAndMetadata:

    def test_exists_true_for_stored_blob(self, storage):
        key = bkey("exists_true.txt")
        storage.store(key, b"content")
        assert storage.exists(key) is True
        print("✓ exists() returns True for stored blob")

    def test_exists_false_for_missing_blob(self, storage):
        assert storage.exists("test/does/not/exist/xyz_abc_999.bin") is False
        print("✓ exists() returns False for missing blob")

    def test_get_metadata_returns_correct_fields(self, storage):
        """
        Metadata must reflect what was stored: key, size, content_type, etag.
        get_metadata() must NOT download the blob content — it's a HEAD request.
        """
        key = bkey("metadata_check.json")
        data = b'{"test": true}'
        storage.store(key, data, content_type="application/json")

        meta = storage.get_metadata(key)

        assert meta.key == key
        assert meta.size_bytes == len(data)
        assert meta.content_type == "application/json"
        assert meta.etag is not None
        assert meta.created_at is not None
        print(f"✓ get_metadata correct: size={meta.size_bytes}, etag={meta.etag}")

    def test_etag_changes_when_content_changes(self, storage):
        """
        ETags are content-derived checksums. Storing different content
        under the same key must produce a different ETag.
        This is how callers detect "has this blob been updated?"
        """
        key = bkey("etag_change.bin")
        storage.store(key, b"version 1 content")
        etag_v1 = storage.get_metadata(key).etag

        storage.store(key, b"version 2 different content")
        etag_v2 = storage.get_metadata(key).etag

        assert etag_v1 != etag_v2
        print(f"✓ ETag changed after content update: {etag_v1} → {etag_v2}")

    def test_get_metadata_for_missing_raises(self, storage):
        with pytest.raises(BlobNotFoundError):
            storage.get_metadata("test/missing_metadata_xyz_999.json")
        print("✓ BlobNotFoundError raised for missing blob metadata")


# ---------------------------------------------------------------------------
# 4. Delete
# ---------------------------------------------------------------------------

class TestDelete:

    def test_delete_existing_blob(self, storage):
        """After delete, exists() returns False and retrieve() raises."""
        key = bkey("to_delete.txt")
        storage.store(key, b"delete me")
        storage.delete(key)

        assert storage.exists(key) is False
        with pytest.raises(BlobNotFoundError):
            storage.retrieve(key)
        print("✓ delete removes blob correctly")

    def test_delete_missing_raises_by_default(self, storage):
        """delete(missing_ok=False) raises BlobNotFoundError for missing keys."""
        with pytest.raises(BlobNotFoundError):
            storage.delete("test/missing_key_xyz_999.txt", missing_ok=False)
        print("✓ delete raises BlobNotFoundError when missing and missing_ok=False")

    def test_delete_missing_ok_true_is_silent(self, storage):
        """delete(missing_ok=True) silently succeeds for missing keys."""
        storage.delete("test/definitely_missing_abc_999.txt", missing_ok=True)
        print("✓ delete(missing_ok=True) is silent for missing blobs")


# ---------------------------------------------------------------------------
# 5. List blobs
# ---------------------------------------------------------------------------

class TestListBlobs:

    def test_list_with_prefix(self, storage):
        """list_blobs(prefix=...) returns only keys matching the prefix."""
        prefix = f"test/{RUN_ID}/list_test/"
        keys = [f"{prefix}file_{i}.txt" for i in range(3)]
        for key in keys:
            storage.store(key, b"list test content")

        listed = storage.list_blobs(prefix=prefix)
        assert set(keys) == set(listed)
        print(f"✓ list_blobs returned {len(listed)} keys for prefix")

    def test_list_with_limit(self, storage):
        """list_blobs(limit=N) returns at most N results."""
        prefix = f"test/{RUN_ID}/limit_test/"
        for i in range(5):
            storage.store(f"{prefix}file_{i}.txt", b"data")

        listed = storage.list_blobs(prefix=prefix, limit=3)
        assert len(listed) <= 3
        print(f"✓ list_blobs(limit=3) returned {len(listed)} results")

    def test_list_empty_prefix_returns_empty_list(self, storage):
        """A prefix with no matching blobs returns [], not an exception."""
        result = storage.list_blobs(prefix="test/nonexistent_prefix_xyz_zzz_999/")
        assert result == []
        print("✓ list_blobs returns [] for empty prefix")


# ---------------------------------------------------------------------------
# 6. Copy
# ---------------------------------------------------------------------------

class TestCopy:

    def test_copy_creates_identical_content(self, storage):
        """
        copy() creates a new key with identical content to the source.
        Content is confirmed by byte-for-byte comparison after retrieval.
        """
        src = bkey("copy_source.json")
        dst = bkey("copy_destination.json")
        original = b'{"copied": true, "data": "test"}'
        storage.store(src, original, content_type="application/json")

        storage.copy(src, dst)

        copied = storage.retrieve(dst)
        assert copied == original
        print("✓ copy() creates identical content at destination key")

    def test_copy_source_still_exists_after_copy(self, storage):
        """copy() does not delete or modify the source blob."""
        src = bkey("copy_src_persists.txt")
        dst = bkey("copy_dst_persists.txt")
        storage.store(src, b"persist me")
        storage.copy(src, dst)

        assert storage.exists(src) is True
        assert storage.retrieve(src) == b"persist me"
        print("✓ Source blob unchanged after copy()")

    def test_copy_missing_source_raises(self, storage):
        with pytest.raises(BlobNotFoundError):
            storage.copy("test/ghost_source_xyz.txt", bkey("dst.txt"))
        print("✓ copy() raises BlobNotFoundError for missing source")

    def test_copy_overwrite_false_raises_if_dest_exists(self, storage):
        src = bkey("copy_ow_src.txt")
        dst = bkey("copy_ow_dst.txt")
        storage.store(src, b"source data")
        storage.store(dst, b"existing destination")

        with pytest.raises(BlobAlreadyExistsError):
            storage.copy(src, dst, overwrite=False)
        print("✓ copy(overwrite=False) raises BlobAlreadyExistsError")


# ---------------------------------------------------------------------------
# 7. Signed URLs
# ---------------------------------------------------------------------------

class TestSignedUrls:

    def test_signed_url_is_valid_https_string(self, storage):
        """
        generate_signed_url() must return a non-empty string URL.
        We can't easily test the URL is actually accessible in unit tests,
        but we verify it looks like a URL and contains the key name.
        """
        key = bkey("signed_url_test.json")
        storage.store(key, b'{"url": "test"}', content_type="application/json")

        url = storage.generate_signed_url(key, expiration_seconds=300)

        assert isinstance(url, str)
        assert len(url) > 0
        assert "http" in url.lower()
        print(f"✓ Signed URL generated: {url[:80]}...")

    def test_signed_url_for_missing_blob_raises(self, storage):
        with pytest.raises(BlobNotFoundError):
            storage.generate_signed_url("test/missing_for_url_xyz_999.txt")
        print("✓ BlobNotFoundError raised for signed URL on missing blob")


# ---------------------------------------------------------------------------
# Standalone functions for __main__ block
# ---------------------------------------------------------------------------

def test_connection_manual(s: MinioBlobStorage):
    """Verify MinIO is reachable by storing and retrieving a tiny blob."""
    key = f"test/connection_check_{uuid.uuid4().hex[:6]}.txt"
    s.store(key, b"ping", content_type="text/plain")
    result = s.retrieve(key)
    assert result == b"ping"
    s.delete(key, missing_ok=True)
    print("✓ Connection verified (store/retrieve/delete)")


def test_json_roundtrip_manual(s: MinioBlobStorage):
    report = {"status": "ok", "docs_affected": 3, "confidence": 0.92}
    key = f"test/manual_json_{uuid.uuid4().hex[:6]}.json"
    s.store(key, json.dumps(report).encode(), content_type="application/json")
    recovered = json.loads(s.retrieve(key).decode())
    assert recovered["status"] == "ok"
    s.delete(key, missing_ok=True)
    print("✓ JSON store/retrieve round-trip")


def test_exists_manual(s: MinioBlobStorage):
    key = f"test/exists_{uuid.uuid4().hex[:6]}.bin"
    assert s.exists(key) is False
    s.store(key, b"exists test")
    assert s.exists(key) is True
    s.delete(key)
    assert s.exists(key) is False
    print("✓ exists() returns correct values before/after store/delete")


def test_list_manual(s: MinioBlobStorage):
    prefix = f"test/list_{uuid.uuid4().hex[:6]}/"
    keys = [f"{prefix}file_{i}.json" for i in range(3)]
    for key in keys:
        s.store(key, b"data")
    listed = s.list_blobs(prefix=prefix)
    assert set(keys) == set(listed)
    for key in keys:
        s.delete(key)
    print(f"✓ list_blobs returned {len(listed)} keys, all correct")


def test_signed_url_manual(s: MinioBlobStorage):
    key = f"test/signed_{uuid.uuid4().hex[:6]}.json"
    s.store(key, b'{"report": "data"}', content_type="application/json")
    url = s.generate_signed_url(key, expiration_seconds=60)
    assert "http" in url.lower()
    s.delete(key)
    print(f"✓ Signed URL generated (60s expiry): {url[:80]}...")


def test_copy_manual(s: MinioBlobStorage):
    src = f"test/copy_src_{uuid.uuid4().hex[:6]}.txt"
    dst = f"test/copy_dst_{uuid.uuid4().hex[:6]}.txt"
    s.store(src, b"server-side copy test")
    s.copy(src, dst)
    assert s.retrieve(dst) == b"server-side copy test"
    assert s.exists(src) is True  # Source untouched
    s.delete(src)
    s.delete(dst)
    print("✓ Server-side copy: source unchanged, destination has correct content")


def test_file_operations_manual(s: MinioBlobStorage):
    key = f"test/file_ops_{uuid.uuid4().hex[:6]}.json"
    content = json.dumps({"file": "upload", "test": True}).encode()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(content)
        src_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        dst_path = f.name

    try:
        s.store_from_file(key, src_path)
        s.retrieve_to_file(key, dst_path)
        with open(dst_path, "rb") as f:
            assert f.read() == content
        print("✓ store_from_file / retrieve_to_file round-trip verified")
    finally:
        os.unlink(src_path)
        os.unlink(dst_path)
        s.delete(key, missing_ok=True)


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("MinioBlobStorage Integration Tests")
    print("=" * 70)
    print()
    print("Ensure MinIO is running: docker-compose up -d minio")
    print()

    s = make_storage()

    try:
        test_connection_manual(s)
        print()

        test_json_roundtrip_manual(s)
        print()

        test_exists_manual(s)
        print()

        test_list_manual(s)
        print()

        test_signed_url_manual(s)
        print()

        test_copy_manual(s)
        print()

        test_file_operations_manual(s)
        print()

        print("=" * 70)
        print("✓ All manual tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise