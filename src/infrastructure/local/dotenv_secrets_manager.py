"""
Dotenv-based implementations of SecretsManager.

This implementation reads secrets from a .env file (typically .env.local)
and provides programmatic access to them. It's a suitable ONLY for local
development, and never use this in production.

WHY THIS EXISTS:
    Local development needs secrets (API keys, passwords) but shouldn't
    use cloud services (GCP Secrets Manager costs money, requires AUTH).
    Reading from .env files is the industry-standard local dev pattern.

SECURITY WARNING:
    - .env files are PLAIN TEXT on disk (no encryption)
    - They provide NO access control (anyone with file access can read)
    - They provide NO audit logging (you can't see who accessed what)
    - They have NO automatic rotation
    
    This is acceptable for local dev where your laptop is (hopefully) secure.
    NEVER use DotenvSecretsManager in production. Use GCPSecretsManager instead.

THREAD SAFETY:
    Uses file locking to prevent concurrent writes from corrupting the file.
    Multiple processes can read simultaneously, but writes are serialized.

FILE FORMAT:
    Standard dotenv format:
        # Comments are preserved
        KEY_NAME=value
        ANOTHER_KEY="value with spaces"
    
    We use python-dotenv library for parsing (handles all format quirks).
"""

import os
# import fcntl  # POSIX file locking (Linux/Mac)
import portalocker
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import dotenv_values, set_key, unset_key

from src.infrastructure.base.secrets import (
    SecretsManager,
    SecretMetadata,
    SecretNotFoundError,
    SecretsManagerError
)

logger = logging.getLogger(__name__)
class DotenvSecretsManager(SecretsManager):
    """
    SecretsManager implementation that reads from .env files.
    
    CROSS-PLATFORM FILE LOCKING:
        Uses a separate .lock file to coordinate access. This works on both
        Windows and POSIX systems because we never lock the file we're reading.
        
        Pattern:
            .env.local        ← Data file (never locked directly)
            .env.local.lock   ← Lock file (locked during writes)
        
    WHY SEPARATE LOCK FILE:
        - Windows: Exclusive locks prevent ANY access to locked file
        - POSIX: Advisory locks allow same-process access
        - Separate lock file works on BOTH platforms
    
    CONCURRENCY SAFETY:
        Multiple processes can read simultaneously (no lock needed for reads).
        Writes acquire exclusive lock on .lock file, preventing concurrent writes.
    """
    
    def __init__(self, file_path: str = ".env.local"):
        """Initialize dotenv secrets manager."""
        self.file_path = Path(file_path)
        self.lock_path = Path(str(file_path) + ".lock")
        
        # Create data file if it doesn't exist
        if not self.file_path.exists():
            logger.warning(
                f"Secrets file {self.file_path} doesn't exist, creating empty file"
            )
            self.file_path.touch(mode=0o600)
        
        # Create lock file if it doesn't exist
        if not self.lock_path.exists():
            self.lock_path.touch(mode=0o600)
        
        # Verify we can read the data file
        if not os.access(self.file_path, os.R_OK):
            raise SecretsManagerError(
                f"Cannot read secrets file: {self.file_path}. "
                f"Check file permissions."
            )
        
        logger.info(
            f"Initialized DotenvSecretsManager with file: {self.file_path} "
            f"(lock file: {self.lock_path})"
        )
    
    def _acquire_lock(self) -> portalocker.Lock:
        """
        Acquire exclusive lock on the lock file.
        
        This prevents concurrent writes from corrupting the data file.
        The lock file is separate from the data file for Windows compatibility.
        
        Returns:
            Locked file handle (must be released by caller)
        
        Example:
            lock = self._acquire_lock()
            try:
                # Do write operations on self.file_path
                set_key(self.file_path, key, value)
            finally:
                lock.close()  # Releases lock
        """
        try:
            # Open lock file and acquire exclusive lock
            # This blocks if another process holds the lock
            lock = portalocker.Lock(
                self.lock_path,
                mode='a',  # Append mode (doesn't truncate)
                timeout=10,  # Wait up to 10 seconds for lock
                flags=portalocker.LOCK_EX,  # Exclusive lock
            )
            lock.acquire()
            
            logger.debug(f"Acquired lock: {self.lock_path}")
            return lock
            
        except portalocker.LockException as e:
            raise SecretsManagerError(
                f"Failed to acquire lock on {self.lock_path}: {e}. "
                f"Another process may be writing to secrets file."
            ) from e
    
    def _load_all_secrets(self) -> Dict[str, str]:
        """
        Load all secrets from the .env file.
        
        No locking needed for reads — multiple processes can read simultaneously.
        """
        try:
            secrets = dotenv_values(self.file_path)
            return secrets
        except Exception as e:
            raise SecretsManagerError(
                f"Failed to load secrets from {self.file_path}: {e}"
            ) from e
    
    def get_secret(self, name: str, version: str = "latest") -> str:
        """Retrieve a secret value by name."""
        if version != "latest":
            logger.warning(
                f"DotenvSecretsManager doesn't support versioning. "
                f"Ignoring version='{version}', returning latest."
            )
        
        secrets = self._load_all_secrets()
        
        if name not in secrets:
            available = ', '.join(secrets.keys()) if secrets else 'none'
            raise SecretNotFoundError(
                f"Secret '{name}' not found in {self.file_path}. "
                f"Available secrets: {available}"
            )
        
        value = secrets[name]
        
        logger.debug(
            f"Retrieved secret '{name}' from {self.file_path} "
            f"(length: {len(value)} chars)"
        )
        
        return value
    
    def set_secret(
        self,
        name: str,
        value: str,
        labels: Optional[Dict[str, str]] = None,
        overwrite: bool = True
    ) -> str:
        """
        Store a secret in the .env file.
        
        Uses lock file to prevent concurrent writes (cross-platform safe).
        """
        if labels:
            logger.warning("DotenvSecretsManager doesn't support labels.")
        
        # Check if secret exists (if overwrite=False)
        if not overwrite:
            try:
                existing_value = self.get_secret(name)
                raise SecretsManagerError(
                    f"Secret '{name}' already exists and overwrite=False. "
                    f"Set overwrite=True to update it."
                )
            except SecretNotFoundError:
                pass  # Good - doesn't exist, we can create it
        
        # Acquire lock on lock file (NOT the data file)
        lock = self._acquire_lock()
        
        try:
            # Now we can safely write to the data file
            # (it's unlocked, so set_key can open it)
            success = set_key(
                dotenv_path=str(self.file_path),
                key_to_set=name,
                value_to_set=value,
                quote_mode="auto",
            )
            
            if not success:
                raise SecretsManagerError(
                    f"Failed to set secret '{name}'. set_key() returned False."
                )
            
            logger.info(
                f"✓ Set secret '{name}' in {self.file_path} "
                f"(overwrite={overwrite})"
            )
            
            return "latest"
            
        finally:
            # Release lock (even if exception occurred)
            # lock.close()
            lock.release()
            # portalocker.unlock(lock)
    
    def delete_secret(self, name: str, permanent: bool = False) -> None:
        """Delete a secret from the .env file."""
        if not permanent:
            logger.warning(
                "DotenvSecretsManager doesn't support soft-delete. "
                "Deletion is always permanent."
            )
        
        # Verify secret exists
        self.get_secret(name)  # Raises SecretNotFoundError if missing
        
        # Acquire lock
        lock = self._acquire_lock()
        
        try:
            # Delete from file
            # success, _, message = unset_key(
            #     dotenv_path=str(self.file_path),
            #     key_to_unset=name,
            # )
            result = unset_key(
            dotenv_path=str(self.file_path),
            key_to_unset=name,
        )

            success = result[0]
            message = result[2] if len(result) > 2 else ""
            
            if not success:
                raise SecretsManagerError(
                    f"Failed to delete secret '{name}': {message}"
                )
            
            logger.info(f"✓ Deleted secret '{name}' from {self.file_path}")
            
        finally:
            lock.release()
    
    def list_secrets(
        self,
        prefix: Optional[str] = None
    ) -> List[SecretMetadata]:
        """List all secrets (metadata only, no values)."""
        secrets = self._load_all_secrets()
        
        # Get file modification time
        file_mtime = datetime.fromtimestamp(self.file_path.stat().st_mtime)
        
        metadata_list = []
        
        for secret_name in secrets.keys():
            # Filter by prefix if provided
            if prefix and not secret_name.startswith(prefix):
                continue
            
            metadata = SecretMetadata(
                name=secret_name,
                version="latest",
                created_at=file_mtime,
                updated_at=file_mtime,
                labels=None,
            )
            
            metadata_list.append(metadata)
        
        logger.debug(
            f"Listed {len(metadata_list)} secrets from {self.file_path} "
            f"(prefix: {prefix or 'none'})"
        )
        
        return metadata_list
    
    def secret_exists(self, name: str) -> bool:
        """Check if a secret exists."""
        try:
            self.get_secret(name)
            return True
        except SecretNotFoundError:
            return False


# class DotenvSecretsManager(SecretsManager):
#     """
#     SecretsManager implementation that reads from .env files.
    
#     Suitable ONLY for local development. Provides no encryption,
#     access control, or audit logging.
    
#     Design Decisions:
#         - Uses python-dotenv library for parsing (battle-tested)
#         - File locking prevents concurrent write corruption
#         - Preserves comments and formatting when editing
#         - No versioning support (limitation of .env format)
#     """

#     def __init__(self,
#                  file_path: str = ".env.local"):
#         """
#         Initialize dotenv secrets manager.
        
#         Args:
#             file_path: Path to .env file (default: .env.local)
        
#         Raises:
#             SecretsManagerError: If file doesn't exist or isn't readable
#         """
#         self.file_path = Path(file_path)
        
#         # Create file if it doesn't exist (with warning)
#         if not self.file_path.exists():
#             logger.warning(
#                 f"Secrets file {self.file_path} doesn't exist, creating empty file"
#             )
#             self.file_path.touch(mode=0o600)  # Create with restricted permissions
        
#         # Verify we can read it
#         if not os.access(self.file_path, os.R_OK):
#             raise SecretsManagerError(
#                 f"Cannot read secrets file: {self.file_path}. "
#                 f"Check file permissions."
#             )
        
#         logger.info(f"Initialized DotenvSecretsManager with file: {self.file_path}")
    
#     def _load_all_secrets(self) -> Dict[str, str]:
#         """
#         Load all secrets from the .env file.
        
#         Uses python-dotenv's dotenv_values() which:
#         - Handles comments (ignores them)
#         - Handles quoted values ("value" and 'value')
#         - Handles values with spaces
#         - Handles = signs in values (KEY=val=ue)
        
#         Returns:
#             Dictionary mapping secret names to values
        
#         Raises:
#             SecretsManagerError: If file can't be read
#         """
#         try:
#             # dotenv_values returns dict, doesn't modify os.environ
#             secrets = dotenv_values(self.file_path)
#             return secrets
#         except Exception as e:
#             raise SecretsManagerError(
#                 f"Failed to load secrets from {self.file_path}: {e}"
#             ) from e

#     def get_secret(self, name: str, version: str = "latest") -> str:
#         """
#         Retrieve a secret value by name.
        
#         Args:
#             name: Secret name (the KEY in KEY=value)
#             version: Ignored (dotenv doesn't support versioning)
        
#         Returns:
#             Secret value as string
        
#         Raises:
#             SecretNotFoundError: If secret doesn't exist
#             SecretsManagerError: If file read fails
#         """
#         if version != "latest":
#             logger.warning(
#                 f"DotenvSecretsManager doesn't support versioning. "
#                 f"Ignoring version='{version}', returning latest."
#             )
        
#         secrets = self._load_all_secrets()
        
#         if name not in secrets:
#             raise SecretNotFoundError(
#                 f"Secret '{name}' not found in {self.file_path}. "
#                 f"Available secrets: {', '.join(secrets.keys())}"
#             )
        
#         value = secrets[name]
        
#         logger.debug(
#             f"Retrieved secret '{name}' from {self.file_path} "
#             f"(length: {len(value)} chars)"
#         )
        
#         return value
    
#     def set_secret(
#         self,
#         name: str,
#         value: str,
#         labels: Optional[Dict[str, str]] = None,
#         overwrite: bool = True
#     ) -> str:
#         """
#         Store a secret in the .env file.
        
#         This is more complex than get_secret because we need to:
#         1. Check if key already exists
#         2. Either add new line or update existing line
#         3. Preserve comments and formatting
#         4. Use file locking to prevent concurrent writes
        
#         We use python-dotenv's set_key() which handles all the complexity.
        
#         Args:
#             name: Secret name (becomes KEY in file)
#             value: Secret value (becomes VALUE in KEY=VALUE)
#             labels: Ignored (dotenv doesn't support labels)
#             overwrite: If False, raise error if secret exists
        
#         Returns:
#             Version identifier (always "latest" for dotenv)
        
#         Raises:
#             SecretsManagerError: If secret exists and overwrite=False
#                                 or if file write fails
#         """
#         if labels:
#             logger.warning(
#                 "DotenvSecretsManager doesn't support labels. "
#                 "Labels will be ignored."
#             )
        
#         # Check if secret exists (if overwrite=False)
#         if not overwrite:
#             try:
#                 existing_value = self.get_secret(name)
#                 raise SecretsManagerError(
#                     f"Secret '{name}' already exists and overwrite=False. "
#                     f"Set overwrite=True to update it."
#                 )
#             except SecretNotFoundError:
#                 # Good - secret doesn't exist, we can create it
#                 pass
        
#         # Acquire file lock for writing
#         # This prevents concurrent writes from different processes
#         try:
#             with open(self.file_path, 'a') as lockfile:
#                 # Acquire exclusive lock (blocks until available)
#                 # fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
#                 portalocker.lock(lockfile, portalocker.LOCK_EX)
                
#                 try:
#                     # set_key handles the complexity:
#                     # - Finds existing key (if it exists) and updates value
#                     # - Appends new key=value (if key doesn't exist)
#                     # - Preserves comments and blank lines
#                     # - Handles quoting (adds quotes if value has spaces)
#                     success = set_key(
#                         dotenv_path=str(self.file_path),
#                         key_to_set=name,
#                         value_to_set=value,
#                         quote_mode="auto",  # Add quotes only if needed
#                     )
                    
#                     if not success:
#                         raise SecretsManagerError(
#                             f"Failed to set secret '{name}'. "
#                             f"set_key() returned False."
#                         )
                    
#                     logger.info(
#                         f"✓ Set secret '{name}' in {self.file_path} "
#                         f"(overwrite={overwrite})"
#                     )
                    
#                     return "latest"  # Dotenv doesn't have versions
                    
#                 finally:
#                     # Release lock (happens automatically on close, but explicit is better)
#                     # fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
#                     portalocker.unlock(lockfile)
                    
#         except Exception as e:
#             raise SecretsManagerError(
#                 f"Failed to set secret '{name}': {e}"
#             ) from e
    
#     def delete_secret(self, name: str, permanent: bool = False) -> None:
#         """
#         Delete a secret from the .env file.
        
#         Args:
#             name: Secret name to delete
#             permanent: Ignored (dotenv deletion is always permanent)
        
#         Raises:
#             SecretNotFoundError: If secret doesn't exist
#             SecretsManagerError: If file write fails
#         """
#         if not permanent:
#             logger.warning(
#                 "DotenvSecretsManager doesn't support soft-delete. "
#                 "Deletion is always permanent."
#             )
        
#         # Verify secret exists (better error message if it doesn't)
#         self.get_secret(name)  # Raises SecretNotFoundError if missing
        
#         # Acquire file lock for writing
#         try:
#             with open(self.file_path, 'a') as lockfile:
#                 fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
                
#                 try:
#                     # unset_key removes the line from the file
#                     # Returns tuple: (success, key, message)
#                     success, _, message = unset_key(
#                         dotenv_path=str(self.file_path),
#                         key_to_unset=name,
#                     )
                    
#                     if not success:
#                         raise SecretsManagerError(
#                             f"Failed to delete secret '{name}': {message}"
#                         )
                    
#                     logger.info(f"✓ Deleted secret '{name}' from {self.file_path}")
                    
#                 finally:
#                     fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
                    
#         except SecretNotFoundError:
#             raise  # Re-raise as-is
#         except Exception as e:
#             raise SecretsManagerError(
#                 f"Failed to delete secret '{name}': {e}"
#             ) from e
    
#     def list_secrets(
#         self,
#         prefix: Optional[str] = None
#     ) -> List[SecretMetadata]:
#         """
#         List all secrets in the .env file (metadata only, no values).
        
#         Returns name, creation time (file mtime), but NO values.
#         Safe for auditing without exposing sensitive data.
        
#         Args:
#             prefix: If provided, only return secrets starting with this prefix
        
#         Returns:
#             List of SecretMetadata (without secret values)
        
#         Raises:
#             SecretsManagerError: If file read fails
#         """
#         secrets = self._load_all_secrets()
        
#         # Get file modification time (best we can do for "created_at")
#         file_mtime = datetime.fromtimestamp(self.file_path.stat().st_mtime)
        
#         metadata_list = []
        
#         for secret_name in secrets.keys():
#             # Filter by prefix if provided
#             if prefix and not secret_name.startswith(prefix):
#                 continue
            
#             # Create metadata WITHOUT the secret value
#             metadata = SecretMetadata(
#                 name=secret_name,
#                 version="latest",  # Dotenv doesn't have versions
#                 created_at=file_mtime,  # Approximation
#                 updated_at=file_mtime,  # We don't track per-key updates
#                 labels=None,  # Dotenv doesn't support labels
#             )
            
#             metadata_list.append(metadata)
        
#         logger.debug(
#             f"Listed {len(metadata_list)} secrets from {self.file_path} "
#             f"(prefix: {prefix or 'none'})"
#         )
        
#         return metadata_list
    
#     def secret_exists(self, name: str) -> bool:
#         """
#         Check if a secret exists in the .env file.
        
#         Args:
#             name: Secret name to check
        
#         Returns:
#             True if secret exists, False otherwise
#         """
#         try:
#             self.get_secret(name)
#             return True
#         except SecretNotFoundError:
#             return False



