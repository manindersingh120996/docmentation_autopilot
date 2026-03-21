"""
MinIO implementation of the BlobStorage ABC.

MinIO is an open-source S3-compatible object storage server. We run it
locally in Docker during development. In production, the same interface
is satisfied by Google Cloud Storage (via its S3-compatible API) or AWS S3.

WHY S3-COMPATIBLE MATTERS:
    The entire cloud storage industry converged on AWS S3's API as the
    de-facto standard. MinIO, GCS, Cloudflare R2, DigitalOcean Spaces —
    they all speak S3. Learning this API once transfers everywhere.
    Our BlobStorage ABC abstracts over it, but the concepts are universal.

CLIENT LIFECYCLE:
    The Minio client is constructed once and reused. It is thread-safe —
    it manages an internal urllib3 connection pool and creates new HTTP
    connections as needed. Unlike the Neo4j driver, there's no explicit
    connect() step: the client is ready to use immediately after construction
    and discovers connection errors on the first actual operation.

BUCKET MANAGEMENT:
    S3/MinIO organizes objects into buckets (top-level namespaces).
    Our implementation takes a bucket_name in the constructor. The
    ensure_bucket() helper creates it if it doesn't exist — called
    lazily on first use so we don't fail at construction time if
    MinIO hasn't started yet.

ERROR TRANSLATION:
    The MinIO client raises minio.error.S3Error for all storage errors.
    S3Error has an .code attribute (e.g., "NoSuchKey", "NoSuchBucket").
    We translate these into our custom exception hierarchy so application
    code only ever sees BlobStorageError and its subclasses.
"""

import logging
import mimetypes
from datetime import datetime, timezone
from io import BytesIO
from typing import BinaryIO, Dict, List, Optional

from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error

from src.infrastructure.base.blob_storage import (
    BlobAlreadyExistsError,
    BlobMetadata,
    BlobNotFoundError,
    BlobStorage,
    BlobStorageError,
)

logger = logging.getLogger(__name__)

class MinioBlobStorage(BlobStorage):
    """
    MinIO (S3-compatible) implementation of BlobStorage.

    Usage:
        storage = MinioBlobStorage(
            endpoint="localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket_name="doc-autopilot",
        )
        # No explicit connect() needed — client is ready immediately.
        storage.store("reports/analysis.json", data, content_type="application/json")
    """

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket_name: str = "doc-autopilot",
        secure: bool = False,
    ):
        """
        Initialise the MinIO client and store configuration.

        Args:
            endpoint:    Host:port of the MinIO server. No http:// prefix —
                         the 'secure' flag controls http vs https.
            access_key:  MinIO access key (equivalent to AWS_ACCESS_KEY_ID).
            secret_key:  MinIO secret key (equivalent to AWS_SECRET_ACCESS_KEY).
            bucket_name: The bucket all operations use. Created automatically
                         on first use via _ensure_bucket().
            secure:      False for local development (http).
                         True for production MinIO with TLS or GCS/S3.
        """
        self._bucket_name = bucket_name
        self._endpoint = endpoint
        self._secure = secure

        # The Minio client is constructed synchronously and is immediately
        # ready to use. It manages its own urllib3 connection pool internally.
        self._client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._bucket_ensured = False  # Lazy bucket creation flag
        logger.info(
            f"MinioBlobStorage initialised: endpoint={endpoint} "
            f"bucket={bucket_name} secure={secure}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_bucket(self) -> None:
        """
        Create the bucket if it doesn't exist. Called lazily on first use.

        WHY LAZY:
            If we created the bucket in __init__, construction would fail
            if MinIO isn't running yet. By deferring to first use, the
            application can start and only fail when it actually tries to
            store or retrieve something — a friendlier failure mode that
            gives retry logic a chance to work.

        WHY IDEMPOTENT:
            bucket_exists() + make_bucket() is safe to call multiple times.
            We use self._bucket_ensured as a fast-path flag so after the
            first successful check we skip the network round-trip entirely.
        """
        if self._bucket_ensured:
            return
        try:
            if not self._client.bucket_exists(self._bucket_name):
                self._client.make_bucket(self._bucket_name)
                logger.info(f"Created MinIO bucket: {self._bucket_name}")
            self._bucket_ensured = True
        except S3Error as e:
            raise BlobStorageError(
                f"Failed to ensure bucket '{self._bucket_name}' exists: {e}"
            ) from e

    def _metadata_from_stat(self, stat, key: str) -> BlobMetadata:
        """
        Convert a MinIO stat_object() result into our BlobMetadata DTO.

        stat_object() returns a minio.datatypes.Object with:
            .size           — content size in bytes
            .content_type   — MIME type string
            .last_modified  — datetime (timezone-aware)
            .etag           — MD5 hash of content (without quotes)
            .metadata       — dict of custom metadata headers

        MinIO stores custom metadata with an "x-amz-meta-" prefix on headers.
        The client strips this prefix, so .metadata gives us back exactly
        what we passed as the metadata dict when storing.
        """
        # last_modified is a timezone-aware datetime from MinIO.
        # We normalise to UTC and make it timezone-aware for consistency.
        last_modified = stat.last_modified
        if last_modified and last_modified.tzinfo is None:
            last_modified = last_modified.replace(tzinfo=timezone.utc)

        # MinIO doesn't separately track created_at — we use last_modified
        # for both fields. For our use case (append-mostly, rarely overwritten)
        # this is acceptable. A full audit trail lives in PostgreSQL anyway.
        return BlobMetadata(
            key=key,
            size_bytes=stat.size or 0,
            content_type=stat.content_type or "application/octet-stream",
            created_at=last_modified or datetime.now(timezone.utc),
            updated_at=last_modified or datetime.now(timezone.utc),
            custom_metadata=dict(stat.metadata) if stat.metadata else None,
            etag=stat.etag.strip('"') if stat.etag else None,
        )

    def _infer_content_type(self, file_path: str) -> str:
        """
        Infer MIME type from file extension using Python's mimetypes module.

        Returns "application/octet-stream" (generic binary) as the safe
        fallback when the extension is unknown. This is the S3 convention
        for "I don't know what this is, treat it as raw bytes."
        """
        content_type, _ = mimetypes.guess_type(file_path)
        return content_type or "application/octet-stream"

    # ------------------------------------------------------------------
    # Core storage operations
    # ------------------------------------------------------------------

    def store(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> BlobMetadata:
        """
        Store bytes as a blob in MinIO.

        HOW put_object WORKS:
            MinIO's put_object() requires a file-like object (not raw bytes)
            and the content length upfront. We wrap the bytes in BytesIO to
            create a seekable, file-like wrapper. BytesIO is in-memory — no
            temporary files on disk.

            The length parameter (len(data)) tells MinIO how many bytes to
            read from the stream. Without it, MinIO would have to buffer
            everything to determine size, which defeats the purpose of
            streaming. We know the size upfront so we always provide it.

        OVERWRITE CHECK:
            We check existence before storing when overwrite=False.
            There's a TOCTOU (time-of-check-time-of-use) race condition here:
            another process could store the same key between our exists()
            check and our put_object() call. For our single-pipeline use case
            this is acceptable. A true atomic "store if not exists" would
            require object locking, which MinIO supports but adds complexity.
        """
        self._ensure_bucket()

        if not overwrite and self.exists(key):
            raise BlobAlreadyExistsError(
                f"Blob '{key}' already exists and overwrite=False"
            )

        try:
            self._client.put_object(
                bucket_name=self._bucket_name,
                object_name=key,
                data=BytesIO(data),          # Wrap bytes in file-like object
                length=len(data),            # Required: exact byte count
                content_type=content_type,
                metadata=metadata,
            )
            logger.debug(f"Stored blob: key={key} size={len(data)} bytes")
            # Fetch and return metadata so the caller gets etag, timestamps etc.
            return self.get_metadata(key)

        except S3Error as e:
            raise BlobStorageError(f"Failed to store blob '{key}': {e}") from e

    def store_from_file(
        self,
        key: str,
        file_path: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> BlobMetadata:
        """
        Upload a local file to MinIO, streaming from disk.

        HOW fput_object DIFFERS FROM put_object:
            put_object() takes a file-like object (in-memory).
            fput_object() takes a file PATH and handles everything:
            - Opens the file itself
            - Reads size from filesystem (no need to load into memory)
            - Streams in chunks to MinIO
            - Closes the file when done

            For large files (repository ZIPs, multi-MB artifacts), this is
            critical. A 500MB ZIP loaded with put_object() would consume
            500MB of RAM. fput_object() uses O(chunk_size) memory regardless
            of file size.
        """
        import os
        self._ensure_bucket()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local file not found: {file_path}")

        if not overwrite and self.exists(key):
            raise BlobAlreadyExistsError(
                f"Blob '{key}' already exists and overwrite=False"
            )

        # Infer content type from extension if not provided
        resolved_content_type = content_type or self._infer_content_type(file_path)

        try:
            self._client.fput_object(
                bucket_name=self._bucket_name,
                object_name=key,
                file_path=file_path,
                content_type=resolved_content_type,
                metadata=metadata,
            )
            logger.debug(f"Uploaded file to blob: key={key} from={file_path}")
            return self.get_metadata(key)

        except S3Error as e:
            raise BlobStorageError(
                f"Failed to upload file '{file_path}' to blob '{key}': {e}"
            ) from e

    def retrieve(self, key: str) -> bytes:
        """
        Download a blob's entire content into memory as bytes.

        get_object() returns an HTTPResponse object. We call .read() to
        consume all bytes, then .close() to release the HTTP connection
        back to the urllib3 pool. The try/finally ensures the connection
        is always released even if .read() raises.

        WHEN TO USE THIS VS retrieve_to_file():
            Use retrieve() when:
            - You know the blob is small (< 50MB rule of thumb)
            - You need the content as bytes in memory (JSON parsing, etc.)

            Use retrieve_to_file() when:
            - File could be large (repository snapshots, etc.)
            - You need it on disk anyway (passing path to another process)
        """
        self._ensure_bucket()
        response = None
        try:
            response = self._client.get_object(
                bucket_name=self._bucket_name,
                object_name=key,
            )
            return response.read()

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise BlobNotFoundError(f"Blob '{key}' not found") from e
            raise BlobStorageError(f"Failed to retrieve blob '{key}': {e}") from e

        finally:
            # Always release the HTTP connection back to the pool
            if response:
                response.close()
                response.release_conn()

    def retrieve_to_file(self, key: str, file_path: str) -> None:
        """
        Download a blob directly to a local file path.

        fget_object() handles the streaming internally:
        1. Sends GET request to MinIO
        2. Reads response in chunks
        3. Writes each chunk to the local file
        4. Closes both the HTTP response and the file handle

        The caller just provides a path — no file handling needed.
        Parent directory of file_path must already exist.
        """
        self._ensure_bucket()
        try:
            self._client.fget_object(
                bucket_name=self._bucket_name,
                object_name=key,
                file_path=file_path,
            )
            logger.debug(f"Downloaded blob to file: key={key} dest={file_path}")

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise BlobNotFoundError(f"Blob '{key}' not found") from e
            raise BlobStorageError(
                f"Failed to download blob '{key}' to '{file_path}': {e}"
            ) from e

    def retrieve_stream(self, key: str) -> BinaryIO:
        """
        Return a streaming file-like object for the blob.

        The HTTPResponse returned by get_object() supports:
            .read(size)   — read up to N bytes
            .readline()   — read one line
            iteration     — for line in stream: ...
            .close()      — release connection

        CRITICAL: The caller MUST call .close() when done.
        An unclosed response holds an HTTP connection in the urllib3 pool.
        If all pool connections are held by unclosed responses, new requests
        will block waiting for a connection — effectively a connection leak.

        Pattern the caller should always use:
            stream = storage.retrieve_stream(key)
            try:
                data = stream.read()
            finally:
                stream.close()

        Or with context manager (if the returned object supports it):
            with storage.retrieve_stream(key) as stream:
                data = stream.read()
        """
        self._ensure_bucket()
        try:
            response = self._client.get_object(
                bucket_name=self._bucket_name,
                object_name=key,
            )
            return response  # Caller is responsible for .close()

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise BlobNotFoundError(f"Blob '{key}' not found") from e
            raise BlobStorageError(
                f"Failed to open stream for blob '{key}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Metadata and existence checks
    # ------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        """
        Check existence by attempting stat_object() and catching NoSuchKey.

        WHY EXCEPTION-BASED EXISTENCE CHECK:
            The S3 API has no dedicated "does this key exist?" endpoint.
            The standard pattern is: call stat_object() (a HEAD request),
            which returns metadata if the object exists, or raises S3Error
            with code "NoSuchKey" if it doesn't.

            This is counterintuitive coming from a filesystem background
            (where os.path.exists() is the norm), but it's idiomatic S3.
            The HEAD request is cheap — it doesn't transfer the object data.
        """
        self._ensure_bucket()
        try:
            self._client.stat_object(
                bucket_name=self._bucket_name,
                object_name=key,
            )
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            # Any other S3Error is a genuine failure, not just "not found"
            raise BlobStorageError(
                f"Failed to check existence of blob '{key}': {e}"
            ) from e

    def get_metadata(self, key: str) -> BlobMetadata:
        """
        Fetch metadata without downloading content.

        stat_object() sends a HEAD request — response contains only headers
        (size, content-type, etag, last-modified), not the object body.
        This is O(1) regardless of object size — fetching metadata for a
        10GB blob is as fast as for a 10-byte blob.
        """
        self._ensure_bucket()
        try:
            stat = self._client.stat_object(
                bucket_name=self._bucket_name,
                object_name=key,
            )
            return self._metadata_from_stat(stat, key)

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise BlobNotFoundError(f"Blob '{key}' not found") from e
            raise BlobStorageError(
                f"Failed to get metadata for blob '{key}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Delete and list
    # ------------------------------------------------------------------

    def delete(self, key: str, missing_ok: bool = False) -> None:
        """
        Delete a blob by key.

        WHY EXISTENCE CHECK BEFORE DELETE:
            MinIO's remove_object() silently succeeds even if the key doesn't
            exist — it doesn't raise S3Error for missing keys on delete.
            To implement the ABC's missing_ok=False behaviour (raise on missing),
            we must explicitly check existence first.

            This introduces a TOCTOU race (another process could delete between
            our check and our delete), but for our use case that's acceptable.
        """
        self._ensure_bucket()

        if not missing_ok and not self.exists(key):
            raise BlobNotFoundError(
                f"Blob '{key}' not found and missing_ok=False"
            )

        try:
            self._client.remove_object(
                bucket_name=self._bucket_name,
                object_name=key,
            )
            logger.debug(f"Deleted blob: key={key}")

        except S3Error as e:
            raise BlobStorageError(f"Failed to delete blob '{key}': {e}") from e

    def list_blobs(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        List blob keys, optionally filtered by prefix.

        list_objects() returns a lazy iterator — MinIO pages through results
        internally and fetches the next page only when the iterator advances.
        This is memory-efficient for buckets with millions of objects.

        We collect results into a list (breaking the laziness) because the
        ABC returns List[str]. If you ever need to handle millions of objects,
        you'd want a generator-based interface instead — worth noting for
        future extension.

        WHY recursive=True:
            Without recursive=True, list_objects() uses "delimiter" mode:
            it treats slashes as directory separators and only returns
            "virtual directory" entries for prefixes, not the objects inside them.
            recursive=True lists ALL objects with the prefix, which is what
            our callers expect ("give me every key under this prefix").
        """
        self._ensure_bucket()
        try:
            objects = self._client.list_objects(
                bucket_name=self._bucket_name,
                prefix=prefix,
                recursive=True,   # Return actual objects, not virtual directories
            )
            keys = []
            for obj in objects:
                keys.append(obj.object_name)
                if limit and len(keys) >= limit:
                    break
            return keys

        except S3Error as e:
            raise BlobStorageError(
                f"Failed to list blobs (prefix={prefix!r}): {e}"
            ) from e

    # ------------------------------------------------------------------
    # Signed URLs and server-side copy
    # ------------------------------------------------------------------

    def generate_signed_url(
        self,
        key: str,
        expiration_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """
        Generate a presigned URL for temporary unauthenticated access.

        HOW PRESIGNED URLS WORK:
            The MinIO client takes your secret key and uses it to create
            an HMAC signature of the request parameters (bucket, key,
            expiration time, allowed method). This signature is embedded
            in the URL's query parameters.

            When someone accesses the URL, MinIO re-computes the signature
            and compares it. If it matches and the URL hasn't expired,
            access is granted — without any database lookup or session check.
            It's pure cryptography.

        SECURITY IMPLICATIONS:
            - Anyone with the URL has access. Don't log signed URLs.
            - Use the shortest practical expiration (1 hour for downloads,
              minutes for uploads).
            - The URL grants access to ONE specific key, not the whole bucket.

        timedelta is required by the MinIO client for the expires parameter.
        We convert expiration_seconds to a timedelta object.
        """
        from datetime import timedelta
        self._ensure_bucket()

        # Verify the blob exists before generating a URL for it
        if not self.exists(key):
            raise BlobNotFoundError(
                f"Cannot generate URL for non-existent blob '{key}'"
            )

        try:
            if method.upper() == "GET":
                url = self._client.presigned_get_object(
                    bucket_name=self._bucket_name,
                    object_name=key,
                    expires=timedelta(seconds=expiration_seconds),
                )
            elif method.upper() == "PUT":
                url = self._client.presigned_put_object(
                    bucket_name=self._bucket_name,
                    object_name=key,
                    expires=timedelta(seconds=expiration_seconds),
                )
            else:
                raise BlobStorageError(
                    f"Unsupported method for signed URL: '{method}'. Use 'GET' or 'PUT'."
                )

            logger.debug(
                f"Generated signed URL: key={key} method={method} "
                f"expires_in={expiration_seconds}s"
            )
            return url

        except S3Error as e:
            raise BlobStorageError(
                f"Failed to generate signed URL for '{key}': {e}"
            ) from e

    def copy(
        self,
        source_key: str,
        destination_key: str,
        overwrite: bool = True,
    ) -> None:
        """
        Copy a blob server-side — no data travels to our application.

        HOW SERVER-SIDE COPY WORKS:
            Instead of: GET source → receive bytes → PUT destination
            MinIO does:  internal copy within storage cluster → done

            For a 500MB blob:
            - Naive approach: 500MB download + 500MB upload = 1GB network traffic
            - Server-side copy: ~0 bytes to our application (just two small HTTP requests)

            CopySource is a MinIO object that specifies what to copy from.
            copy_object() is the single API call that does the whole operation.
        """
        self._ensure_bucket()

        if not self.exists(source_key):
            raise BlobNotFoundError(f"Source blob '{source_key}' not found")

        if not overwrite and self.exists(destination_key):
            raise BlobAlreadyExistsError(
                f"Destination blob '{destination_key}' exists and overwrite=False"
            )

        try:
            self._client.copy_object(
                bucket_name=self._bucket_name,
                object_name=destination_key,
                source=CopySource(self._bucket_name, source_key),
            )
            logger.debug(f"Copied blob: {source_key} → {destination_key}")

        except S3Error as e:
            raise BlobStorageError(
                f"Failed to copy blob '{source_key}' to '{destination_key}': {e}"
            ) from e

