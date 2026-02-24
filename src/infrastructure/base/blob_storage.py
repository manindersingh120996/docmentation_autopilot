"""
Abstract base class for blob (object) storage implementations.

BlobStorage provides simple, scalable storagee for Binary Data (files).
Think of it as an infinitely large, flat file system optimized for
reading and writing large objects.

WHY THIS EXISTS:
    Our system needs to store things that don't belong in databases:
    - Code repository snapshots (ZIP archives, potentially hundreds of MB)
    - Generated documentation artifacts (HTML, PDF reports)
    - Exported data for external consumption
    - Cached LLM responses (large JSON objects)

    Databases are optimized for structured, queryable data — not for
    storing large binary files efficiently. Blob storage is optimized
    exactly for this use case.

WHAT THIS ABSTRACTS:
    Local development: MinIO (S3-compatible, running in Docker)
    Production (GCP):  Google Cloud Storage (GCS)
    Also compatible with: AWS S3, Azure Blob Storage

KEY CONCEPT — Flat Namespace:
    Blob storage has NO real directories. Keys are just strings.
    "reports/2024/january/analysis.json" is ONE key, not a path.
    The slashes are a naming convention, not actual folder structure.
    This is why listing requires a prefix filter (not directory listing).

"""

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import BinaryIO, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class BlobMetadata:
    """
    Metadata about a stored blob, without the blob's actual content

    Useful for checking blob properties (size, modification time) wihtout
    downloading the (potentially large) blob data. The get_metadata()
    method returns this without fetching the blob content.

    Attributes:
        key:                The blob's unique identifier (the "filename")
        size_bytes:         Size of the stored content in bytes.
        content_type:       MIME type (e.g., "application/json", "text/plain")
        created_at:         When this blob was first stored
        updated_at:         Ehen this blob was last modified (or created_at if never)
        custom_metadata:    User-defined key-value metadata attached to this blob
                            Example: {"commit_sha": "abc123", "souce":"analysis_worker"}
        etag:               Checksum/fingerprint. Changes when content changes
                            Use for cache invalidation or change detection.
    """
    key: str
    size_bytes: int
    content_type: str
    created_at: datetime
    updated_at: datetime
    custom_metadata: Optional[Dict[str,str]] = None
    etag: Optional[str] = None

# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------


class BlobStorageError(Exception):
    """Base exception for all blob storage errors."""
    pass

class BlobNotFoundError(BlobStorageError):
    """
    Raised when trying to access a blob that doesn't exist.

    Note: exists() does NOT raise this — it returns False.
    This exception is raised by retrieve(), get_metadata(), etc.
    when the caller assumes the blob exists but it doesn't.
    """
    pass

class BlobAlreadyExistsError(BlobStorageError):
    """
    Raised when store() or copy() would overwrite an existing blob
    and overwrite=False was specified.

    By default, overwrite=True so this is not raised. Opt into
    protection by setting overwrite=False when you want to detect
    accidental overwrites.
    """
    pass


# ---------------------------------------------------------------------------
# The Abstract Base Class
# ---------------------------------------------------------------------------

class BlobStorage(ABC):
    """
    Abstract interface for object/blob storage implementations.

    CORE CONCEPT:
        A blob storage is essentially a key-value store where:
            - Key are strings (can contain slashes to simulate folder structure)
            - Values are arbitrary binary data (bytes)

        Simple operations: store, retrieve, delete, list, check existence.

    MEMORY CONSIDERATIONS:
        Some blobs can be gigabytes. This interface provides three retrieval
        patterns for different memory constraints:
        - retrieve():           Loads entire blob into memory (bytes). Simple but
                               dangerous for large files.
        - retrieve_to_file():   Streams to local disk. Memory-efficient.
        - retrieve_stream():    Returns streaming file-like object. Flexible.

        Use retrieve() only for blobs you know are small (< a few MB).
        Use retrieve_to_file() or retrieve_stream() for large files.

    KEY NAMING CONVENTION:
        Use forward slashes to create logical groupings:
        - "snapshots/{repo_name}/{commit_sha}.zip"
        - "reports/{date}/{report_type}.json"
        - "cache/llm_responses/{hash}.json"
        - "artifacts/{workflow_id}/{output_name}.html"

        This makes list_blobs(prefix="snapshots/pytorch/") work like
        listing a folder, even though it's really just prefix filtering.

    """

    @abstractmethod
    def store(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True
    ) -> BlobMetadata:
        """
        Store binary data as a blob.

        The data is persisted durably before this method returns. If this
        method returns without raising, the data is safely stored.

        Args:
            key:          Unique identifier for this blob.
                          Use slash-separated naming: "reports/2024-01/analysis.json"
            data:         Binary content to store.
            content_type: MIME type. Affects how browsers handle the file
                          when accessed via signed URL.
                          Common values: "application/json", "text/plain",
                          "application/octet-stream" (generic binary, the default),
                          "application/zip", "image/png"
            metadata:     Optional custom key-value metadata (strings only).
                          Values must be strings (not int, not dict).
                          Max ~8KB per blob across all metadata.
            overwrite:    True (default): replace if key already exists.
                          False: raise BlobAlreadyExistsError if key exists.

        Returns:
            BlobMetadata with size, timestamps, and etag of the stored blob.

        Raises:
            BlobAlreadyExistsError: If key exists and overwrite=False.
            BlobStorageError:       If storage fails.

        Example:
            import json
            report = {"status": "complete", "affected_docs": ["doc1", "doc2"]}
            blob_storage.store(
                key="reports/2024-01-15/impact_analysis.json",
                data=json.dumps(report).encode("utf-8"),
                content_type="application/json",
                metadata={"commit_sha": "abc123", "generated_by": "layer2"}
            )
        """
        pass

    @abstractmethod
    def store_from_file(
        self,
        key: str,
        file_path: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True
    ) -> BlobMetadata:
        """
        Upload a local file to blob storage.

        More memory-efficient than store() for large files because it
        streams from disk rather than loading the entire file into RAM.

        Args:
            key:          Unique identifier for the blob.
            file_path:    Absolute or relative path to the local file.
            content_type: MIME type. If None, inferred from file extension.
            metadata:     Optional custom metadata.
            overwrite:    Whether to replace existing blob.

        Returns:
            BlobMetadata of the stored blob.

        Raises:
            FileNotFoundError:     If file_path doesn't exist locally.
            BlobAlreadyExistsError: If key exists and overwrite=False.
            BlobStorageError:      If upload fails.

        Example:
            # Upload a code snapshot without loading it into memory
            blob_storage.store_from_file(
                key="snapshots/pytorch/abc123def456.zip",
                file_path="/tmp/pytorch_snapshot.zip",
                content_type="application/zip",
                metadata={"commit_sha": "abc123def456", "repo": "pytorch/pytorch"}
            )
        """
        pass



    @abstractmethod
    def retrieve(self, key: str) -> bytes:
        """
        Retrieve a blob's content as bytes.

        WARNING: Loads the ENTIRE blob into memory. Only use for blobs
        you know are small (rule of thumb: < 50 MB). For large blobs,
        use retrieve_to_file() or retrieve_stream().

        Args:
            key: Unique identifier of the blob.

        Returns:
            The blob's binary content.

        Raises:
            BlobNotFoundError: If key doesn't exist.
            BlobStorageError:  If retrieval fails.

        Example:
            import json
            data = blob_storage.retrieve("reports/2024-01-15/impact_analysis.json")
            report = json.loads(data.decode("utf-8"))
        """
        pass

    @abstractmethod
    def retrieve_to_file(self, key: str, file_path: str) -> None:
        """
        Download a blob to a local file.

        Memory-efficient: streams data from blob storage to disk without
        loading the entire blob into RAM. Use for large files.

        Args:
            key:       Unique identifier of the blob.
            file_path: Local path to write the file to.
                       Parent directories must exist.

        Raises:
            BlobNotFoundError: If key doesn't exist.
            BlobStorageError:  If download fails.
            IOError:           If writing to file_path fails (permissions, disk full, etc.)

        Example:
            blob_storage.retrieve_to_file(
                key="snapshots/pytorch/abc123def456.zip",
                file_path="/tmp/repo_snapshot.zip"
            )
            # Now process /tmp/repo_snapshot.zip without it being in RAM
        """
        pass

    @abstractmethod
    def retrieve_stream(self, key: str) -> BinaryIO:
        """
        Retrieve a blob as a streaming file-like object.

        Returns an object that behaves like an open file in binary read mode.
        Useful for processing large blobs incrementally (line by line, chunk
        by chunk) without loading everything into memory.

        The caller is responsible for closing the stream.

        Args:
            key: Unique identifier of the blob.

        Returns:
            File-like object supporting .read(), .readline(), .readlines(),
            and iteration. Must be closed by the caller.

        Raises:
            BlobNotFoundError: If key doesn't exist.
            BlobStorageError:  If retrieval fails.

        Example:
            stream = blob_storage.retrieve_stream("logs/analysis_worker.log")
            try:
                for line in stream:
                    process_log_line(line.decode("utf-8"))
            finally:
                stream.close()  # Always close!
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check whether a blob exists.

        Faster than retrieve() for existence checks — only fetches metadata,
        not the blob content.

        Args:
            key: Unique identifier to check.

        Returns:
            True if blob exists. False if it doesn't.

        Raises:
            BlobStorageError: If the existence check itself fails (e.g.,
                             network error). NOT raised for "doesn't exist".

        Example:
            cache_key = f"cache/llm_responses/{query_hash}.json"
            if blob_storage.exists(cache_key):
                cached_response = blob_storage.retrieve(cache_key)
            else:
                response = call_llm(query)
                blob_storage.store(cache_key, response.encode("utf-8"))
        """
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> BlobMetadata:
        """
        Get metadata about a blob without downloading its content.

        Efficient way to check size, modification time, content type, etc.
        Does NOT transfer the blob's binary content over the network.

        Args:
            key: Unique identifier of the blob.

        Returns:
            BlobMetadata with all information about the blob.

        Raises:
            BlobNotFoundError: If key doesn't exist.
            BlobStorageError:  If metadata retrieval fails.

        Example:
            meta = blob_storage.get_metadata("snapshots/pytorch/abc123.zip")
            if meta.size_bytes > 500 * 1024 * 1024:  # 500 MB
                logger.warning("Large snapshot — processing may be slow")
            if meta.etag != cached_etag:
                logger.info("Snapshot changed, need to reprocess")
        """
        pass

    @abstractmethod
    def delete(self, key: str, missing_ok: bool = False) -> None:
        """
        Delete a blob.

        Args:
            key:        Unique identifier of the blob to delete.
            missing_ok: False (default): raise BlobNotFoundError if key doesn't exist.
                        True: silently succeed if key doesn't exist.
                        Use missing_ok=True for cleanup operations where you don't
                        care whether the blob was there or not.

        Raises:
            BlobNotFoundError: If key doesn't exist and missing_ok=False.
            BlobStorageError:  If deletion fails.

        Example:
            # Clean up temporary processing artifacts (might or might not exist)
            blob_storage.delete(f"temp/processing_{job_id}.tmp", missing_ok=True)
        """
        pass

    @abstractmethod
    def list_blobs(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        List blob keys in the storage bucket.

        Since blob storage is a flat namespace, "listing a folder" is
        implemented as prefix filtering. All keys starting with the prefix
        are returned — there's no real directory structure.

        Args:
            prefix: If provided, only return keys starting with this string.
                    None returns ALL keys (be careful with large buckets).
                    Example: prefix="reports/2024-01/" lists January 2024 reports.
            limit:  Maximum number of keys to return. None = no limit.
                    Use limit for pagination or safety with large buckets.

        Returns:
            List of blob keys as strings. Empty list if none match.

        Raises:
            BlobStorageError: If listing fails.

        Performance note:
            Listing can be slow for large buckets. ALWAYS use a prefix
            to narrow the search space in production code. Only use
            prefix=None for development/debugging.

        Example:
            # List all snapshots for a specific repository
            pytorch_snapshots = blob_storage.list_blobs(
                prefix="snapshots/pytorch_pytorch/"
            )
            print(f"Found {len(pytorch_snapshots)} snapshots")
        """
        pass

    @abstractmethod
    def generate_signed_url(
        self,
        key: str,
        expiration_seconds: int = 3600,
        method: str = "GET"
    ) -> str:
        """
        Generate a temporary URL for unauthenticated access to a blob.

        Signed URLs embed access permissions and expiration time in the
        URL itself (cryptographically signed). They allow anyone with the
        URL to access the specific blob for a limited time — without
        needing your storage credentials.

        Use case: share a generated report with a user via email or Slack
        without giving them access to your entire storage bucket.

        Args:
            key:                Unique identifier of the blob.
            expiration_seconds: How long the URL remains valid.
                               Default: 3600 seconds (1 hour).
                               Maximum: typically 7 days (implementation-specific).
            method:             HTTP method the URL permits.
                               "GET"  — download the blob (most common)
                               "PUT"  — upload to this key (for pre-signed uploads)

        Returns:
            HTTPS URL string. Anyone with this URL can access the blob
            until it expires.

        Raises:
            BlobNotFoundError: If key doesn't exist (for GET URLs).
            BlobStorageError:  If URL generation fails.

        Security note:
            Don't log signed URLs — they're effectively credentials.
            Set the shortest practical expiration time.

        Example:
            # Share analysis report with engineering team via Slack
            url = blob_storage.generate_signed_url(
                key="reports/2024-01-15/impact_analysis.pdf",
                expiration_seconds=86400  # 24 hours
            )
            slack.post(f"Today's analysis report: {url}")
        """
        pass

    @abstractmethod
    def copy(
        self,
        source_key: str,
        destination_key: str,
        overwrite: bool = True
    ) -> None:
        """
        Copy a blob to a new key (server-side, without downloading).

        This is a server-side operation — the data doesn't travel from
        the storage service to your application and back. It's fast even
        for gigabyte-sized blobs.

        Compare to the naive approach (download + re-upload), which is:
        - Slow (data travels over the network twice)
        - Memory-intensive (entire blob in RAM)
        - Costly (double network egress fees)

        server-side copy avoids all of these problems.

        Args:
            source_key:      Key of the blob to copy from.
            destination_key: Key for the new copy.
            overwrite:       True (default): replace destination if exists.
                             False: raise BlobAlreadyExistsError if destination exists.

        Raises:
            BlobNotFoundError:     If source_key doesn't exist.
            BlobAlreadyExistsError: If destination_key exists and overwrite=False.
            BlobStorageError:      If copy fails.

        Example:
            # Backup before modifying an important blob
            blob_storage.copy(
                source_key="config/production_settings.json",
                destination_key=f"backups/production_settings_{timestamp}.json"
            )
            # Now safely modify the original
            blob_storage.store("config/production_settings.json", new_config_data)
        """
        pass
