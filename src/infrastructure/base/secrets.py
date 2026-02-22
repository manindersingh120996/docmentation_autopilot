"""
Abstract base class for secrets management implementations.

A SecretManager provides secure storage and retrievalof sensitive values:
API keys, passwords, tokens, and encryption keys.

WHY THIS EXISTS:
    Secrets must NEVER appear in:
        - Source code (commited to Git, accessible to abyone with repo access)
        - Config YAML fiels (also commited to GIT)
        - Log files (often stored insecurely)
        - Environment variables in plain text in production systems

    But the application needs thses values at runtime. SecretsManager
    bridges this gap: secrets are stored in a secure, access-controlled,
    eccrypted store and retrived only at runtime when needed.

WHAT THIS ABSTRACTS:    
    Local development: python-dotenv reading .env.local
                        NOT SECURE, but acceptable for local dev because:
                        (a) .env.local is gitignored (never committed)
                        (b) it contains only dev/test credentials
                        (c) your local machine is (hopefully) secure
    
    Production (GCP): GCP secret Manager
                        Encrypted at rest and in transit
                        Fine-grained IAM access control
                        Complete audit log of all access
                        Automatic secret rotation support

IMPORTANT DISTINCTION:
    Our config system ALREADY handles one form of secret injection:
    the `password_env_var: "DB_PASSWORD"` pattern in YAML files.
    That pattern reads from environment variables (set by .env.local).

    SecretsManager is for when you need to retrieve secrets PROGRAMMATICALLY
    at runtime , not just at startup config loading. For example:
    - Rotating credentials mid-run
    - Retrieving secrets that change frequently
    - Accessing secrets that shouldn't be in environment variables at all
    - In production where env vars aren't appropriate

    In practice, for our system, the config-based env var substitution
    handles most cases. SecretsManager handles the rest.

"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class SecretMetadata:
    """
    Metadata about a secret — without the secret's value.

    Used by list_secrets() to provide information about what secrets
    exist without exposing their values. Useful for auditing and
    management operations.

    Attributes:
        name:       Name/identifier of the secret.
        version:    Which version this metadata describes.
        created_at: When this version was created.
        updated_at: When this version was last modified (if ever).
        labels:     Optional key-value labels for organization.
                    Example: {"environment": "production", "service": "doc-autopilot"}    
    
    """
    name: str
    version: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    labels: Optional[Dict[str, str]] = None

# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class SecretsManagerError(Exception):
    """Base exception for all secrets manager errors."""
    pass

class SecretNotFoundError(SecretsManagerError):
    """
    Raised when the requested secret doesn't exist.

    This is distinct from access denial , the secret simply isn't there.
    Check the secret name for typos and ensure it was set up in the
    correct environment.
    """
    
    
class SecretAccessDeniedError(SecretsManagerError):
    """
    Raised when the current credentials lack permission to access the secret.

    This means the secret EXISTS but you're not allowed to read it.
    In production this indicates an IAM/permission misconfiguration.
    Fix the service account permissions, not the code.
    """
    pass


# ---------------------------------------------------------------------------
# The Abstract Base Class
# ---------------------------------------------------------------------------

class SecretsManager(ABC):
    """
    Abstract interface for secrets management implementations.

    INTENTIONAL MINIMALISM:
        This interface is deliberately small. Secrets management is
        conceptually simple: store secrets, retrieve secrets, manage them.
        The complexity (encryption, access control, audit trails, rotation)
        is entirely in the implementation, hidden behind this interface.

    SECURITY PRINCIPLES enforced by this interface:
        1. list_secrets() returns METADATA ONLY — never values.
           You cannot accidentally log all secret values by listing them.

        2. get_secret() must be called explicitly for each secret.
           There's no "get_all_secrets()" that dumps everything.

        3. set_secret() has a note in its docstring warning about
           hardcoding values — the interface itself is a teaching tool.

    SECRET VERSIONING:
        Production secrets managers (GCP, AWS, Vault) support versioning.
        When you rotate a credential, the old version isn't destroyed —
        it becomes "version 1" while the new one is "version 2" (latest).
        This allows rollback if the new credential has issues.
        The version parameter in get_secret() exposes this capability.
        Implementations that don't support versioning should return
        "latest" for get_metadata().version and ignore the version param.
    """

    @abstractmethod
    def get_secret(self, name: str, version: str = "latest") -> str:
        """
        Retrieve a secret's value.

        This is the primary operation — called at runtime to get the
        actual credential value for authentication to external services.

        Args:
            name:    Name/identifier of the secret.
                     Use consistent naming: lowercase with underscores.
                     Examples: "openrouter_api_key", "github_webhook_secret",
                               "neo4j_password", "postgres_password"
            version: Which version to retrieve.
                     "latest" (default) gets the current active version.
                     Pass a specific version ID to retrieve historical versions
                     (for rollback or debugging).

        Returns:
            The secret value as a plain string.

        Raises:
            SecretNotFoundError:    If the secret doesn't exist.
            SecretAccessDeniedError: If access is denied.
            SecretsManagerError:    If retrieval fails for other reasons.

        SECURITY CONTRACT:
            The returned string is sensitive. You MUST:
            - NOT log it (even at DEBUG level)
            - NOT store it in a variable longer than necessary
            - NOT include it in exception messages
            - NOT serialize it to disk or databases
            Use it only for its intended purpose, then let Python's
            garbage collector clean it up.

        Example:
            # Good: use immediately, don't store longer than needed
            api_client = OpenRouterClient(
                api_key=secrets_manager.get_secret("openrouter_api_key")
            )

            # Bad: stored in a variable and potentially logged
            api_key = secrets_manager.get_secret("openrouter_api_key")
            logger.debug(f"Using config: {locals()}")  # LEAKS the key!
        """
        pass


    @abstractmethod
    def set_secret(
        self,
        name: str,
        value: str,
        labels: Optional[Dict[str, str]] = None,
        overwrite: bool = True
    ) -> str:
        """
        Store a secret value.

        In production, secrets are typically set by administrators or
        deployment automation — not by application code during normal
        operation. This method exists for initialization scripts and tests.

        If a secret with this name already exists:
        - overwrite=True (default): creates a new version (old version preserved)
        - overwrite=False: raises an error

        Args:
            name:      Name/identifier for the secret.
            value:     The secret value to store.
            labels:    Optional organizational labels.
                       Example: {"environment": "prod", "rotation": "quarterly"}
            overwrite: Whether to create a new version if secret exists.

        Returns:
            Version identifier of the newly created version.
            Use this version ID with get_secret(version=...) to retrieve
            this specific version later.

        Raises:
            SecretAccessDeniedError: If lacking write permission.
            SecretsManagerError:     If storage fails.

        !! SECURITY WARNING !!
            NEVER call this with a hardcoded secret value in source code:
                secrets_manager.set_secret("api_key", "sk-abc123...")  # WRONG

            The value would be committed to Git. Instead, read values
            from secure sources (stdin, HSMs, other secrets managers):
                api_key = input("Enter API key: ")  # or from deployment system
                secrets_manager.set_secret("api_key", api_key)

        Example (in a one-time setup script):
            # Initialize secrets during first deployment
            import getpass
            api_key = getpass.getpass("Enter OpenRouter API key: ")
            version = secrets_manager.set_secret(
                name="openrouter_api_key",
                value=api_key,
                labels={"service": "doc-autopilot", "env": "production"}
            )
            print(f"Secret stored as version {version}")
        """
        pass

    @abstractmethod
    def delete_secret(self, name: str, permanent: bool = False) -> None:
        """
        Delete a secret.

        Args:
            name:      Name/identifier of the secret to delete.
            permanent: False (default): soft-delete — secret can be recovered
                       for a grace period (typically ~30 days in GCP/AWS).
                       True: permanently and immediately destroy — cannot be recovered.
                       NEVER use permanent=True in production without absolute certainty.

        Raises:
            SecretNotFoundError:    If secret doesn't exist.
            SecretAccessDeniedError: If lacking delete permission.
            SecretsManagerError:    If deletion fails.

        Example:
            # Decommissioning an old service — remove its credentials
            # Soft delete first to allow recovery window
            secrets_manager.delete_secret("deprecated_service_api_key")

            # Only use permanent=True when you're certain the secret
            # should never exist again (regulatory compliance, etc.)
        """
        pass

    @abstractmethod
    def list_secrets(
        self,
        prefix: Optional[str] = None
    ) -> List[SecretMetadata]:
        """
        List secrets in the store — metadata only, NEVER values.

        Returns information ABOUT secrets (names, versions, timestamps,
        labels) but never the actual secret values. This makes it safe
        to call for auditing, inventory, or management without risk of
        exposing credentials.

        Args:
            prefix: If provided, only return secrets whose names start
                    with this string. Useful for namespacing:
                    - prefix="prod_"  → all production secrets
                    - prefix="test_"  → all test secrets
                    None returns all secrets visible to current credentials.

        Returns:
            List of SecretMetadata objects. Empty list if none found or none match.

        Raises:
            SecretAccessDeniedError: If lacking list permission.
            SecretsManagerError:     If listing fails.

        Example:
            # Audit all secrets, check for stale ones
            all_secrets = secrets_manager.list_secrets()
            for secret in all_secrets:
                age_days = (datetime.now() - secret.created_at).days
                if age_days > 90:
                    print(f"WARNING: {secret.name} is {age_days} days old — rotate?")
        """
        pass

    @abstractmethod
    def secret_exists(self, name: str) -> bool:
        """
        Check if a secret exists.

        More efficient than get_secret() for existence checks because it
        doesn't retrieve or decrypt the actual value.

        Args:
            name: Name/identifier to check.

        Returns:
            True if secret exists and is accessible.
            False if it doesn't exist.

        Raises:
            SecretAccessDeniedError: If you lack even the permission to check
                                    existence (rare, but possible in some systems).
            SecretsManagerError:     If the check fails.

        Example:
            # Initialize default secrets if they don't exist yet (first run)
            if not secrets_manager.secret_exists("github_webhook_secret"):
                import secrets
                webhook_secret = secrets.token_urlsafe(32)
                secrets_manager.set_secret("github_webhook_secret", webhook_secret)
                print("Generated new webhook secret — configure this in GitHub settings")
        """
        pass


