"""
Unit tests for DotenvSecretsManager.

These tests create temporary .env files and verify secret operations.
No external dependencies required.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.infrastructure.local.dotenv_secrets_manager import DotenvSecretsManager
from src.infrastructure.base.secrets import SecretNotFoundError, SecretsManagerError


@pytest.fixture
def temp_env_file(tmp_path):
    """Create a temporary .env file for testing."""
    env_file = tmp_path / "test.env"
    
    # Create with some initial secrets
    env_file.write_text(
        "# Test secrets file\n"
        "TEST_KEY_1=value1\n"
        "TEST_KEY_2=value_with_spaces\n"
        "DB_PASSWORD=secret123\n"
    )
    
    return env_file


@pytest.fixture
def secrets_manager(temp_env_file):
    """Create DotenvSecretsManager with temp file."""
    return DotenvSecretsManager(file_path=str(temp_env_file))


def test_get_secret_success(secrets_manager):
    """Test retrieving an existing secret."""
    value = secrets_manager.get_secret("TEST_KEY_1")
    assert value == "value1"
    print("✓ get_secret() works for existing secret")


def test_get_secret_not_found(secrets_manager):
    """Test that missing secrets raise SecretNotFoundError."""
    with pytest.raises(SecretNotFoundError, match="Secret 'NONEXISTENT' not found"):
        secrets_manager.get_secret("NONEXISTENT")
    
    print("✓ get_secret() raises SecretNotFoundError for missing secret")


def test_secret_exists(secrets_manager):
    """Test checking if secrets exist."""
    assert secrets_manager.secret_exists("TEST_KEY_1") is True
    assert secrets_manager.secret_exists("NONEXISTENT") is False
    print("✓ secret_exists() works correctly")


def test_set_secret_new(secrets_manager, temp_env_file):
    """Test adding a new secret."""
    version = secrets_manager.set_secret("NEW_SECRET", "new_value")
    
    assert version == "latest"
    
    # Verify it was written
    assert secrets_manager.get_secret("NEW_SECRET") == "new_value"
    
    # Verify file contains it
    content = temp_env_file.read_text()
    assert "NEW_SECRET=" in content
    
    print("✓ set_secret() adds new secret correctly")


def test_set_secret_overwrite(secrets_manager):
    """Test updating an existing secret."""
    # Update existing secret
    secrets_manager.set_secret("TEST_KEY_1", "updated_value", overwrite=True)
    
    # Verify it was updated
    assert secrets_manager.get_secret("TEST_KEY_1") == "updated_value"
    
    print("✓ set_secret() overwrites existing secret")


def test_set_secret_no_overwrite(secrets_manager):
    """Test that overwrite=False prevents updating existing secrets."""
    with pytest.raises(SecretsManagerError, match="already exists and overwrite=False"):
        secrets_manager.set_secret("TEST_KEY_1", "new_value", overwrite=False)
    
    print("✓ set_secret() respects overwrite=False")


def test_set_secret_with_spaces(secrets_manager):
    """Test that values with spaces are handled correctly."""
    secrets_manager.set_secret("KEY_WITH_SPACES", "value with multiple spaces")
    
    # Should retrieve with spaces intact
    assert secrets_manager.get_secret("KEY_WITH_SPACES") == "value with multiple spaces"
    
    print("✓ set_secret() handles values with spaces")


def test_delete_secret(secrets_manager, temp_env_file):
    """Test deleting a secret."""
    # Verify secret exists
    assert secrets_manager.secret_exists("TEST_KEY_1")
    
    # Delete it
    secrets_manager.delete_secret("TEST_KEY_1")
    
    # Verify it's gone
    assert not secrets_manager.secret_exists("TEST_KEY_1")
    
    # Verify file doesn't contain it
    content = temp_env_file.read_text()
    assert "TEST_KEY_1" not in content
    
    print("✓ delete_secret() removes secret correctly")


def test_delete_nonexistent(secrets_manager):
    """Test deleting a non-existent secret raises error."""
    with pytest.raises(SecretNotFoundError):
        secrets_manager.delete_secret("NONEXISTENT")
    
    print("✓ delete_secret() raises error for missing secret")


def test_list_secrets(secrets_manager):
    """Test listing all secrets."""
    metadata_list = secrets_manager.list_secrets()
    
    # Should have 3 secrets from fixture
    # print(metadata_list)
    assert len(metadata_list) == 3
    
    # Check names (values should NOT be in metadata)
    names = [m.name for m in metadata_list]
    assert "TEST_KEY_1" in names
    assert "TEST_KEY_2" in names
    assert "DB_PASSWORD" in names
    
    # Verify metadata structure
    for metadata in metadata_list:
        assert metadata.version == "latest"
        assert metadata.created_at is not None
        # Values should NOT be in metadata!
        assert not hasattr(metadata, 'value')
    
    print("✓ list_secrets() returns metadata without values")


def test_list_secrets_with_prefix(secrets_manager):
    """Test filtering secrets by prefix."""
    metadata_list = secrets_manager.list_secrets(prefix="TEST_")
    
    # Should only get secrets starting with TEST_
    assert len(metadata_list) == 2
    
    names = [m.name for m in metadata_list]
    assert "TEST_KEY_1" in names
    assert "TEST_KEY_2" in names
    assert "DB_PASSWORD" not in names  # Doesn't match prefix
    
    print("✓ list_secrets() filters by prefix correctly")


def test_file_preservation(secrets_manager, temp_env_file):
    """Test that comments and formatting are preserved."""
    # Initial content has a comment
    initial_content = temp_env_file.read_text()
    assert "# Test secrets file" in initial_content
    
    # Update a secret
    secrets_manager.set_secret("TEST_KEY_1", "new_value")
    
    # Comment should still be there
    updated_content = temp_env_file.read_text()
    assert "# Test secrets file" in updated_content
    
    print("✓ Comments are preserved when editing secrets")


def test_empty_file_creation(tmp_path):
    """Test that manager creates file if it doesn't exist."""
    nonexistent_file = tmp_path / "new.env"
    
    # File doesn't exist yet
    assert not nonexistent_file.exists()
    
    # Create manager (should create file)
    manager = DotenvSecretsManager(file_path=str(nonexistent_file))
    
    # File should now exist
    assert nonexistent_file.exists()
    
    # Should be empty but functional
    assert manager.list_secrets() == []
    
    # Should be able to add secrets
    manager.set_secret("FIRST_KEY", "first_value")
    assert manager.get_secret("FIRST_KEY") == "first_value"
    
    print("✓ Manager creates file if it doesn't exist")


if __name__ == '__main__':
    # Run tests manually
    print("=" * 70)
    print("DotenvSecretsManager Unit Tests")
    print("=" * 70)
    print()
    
    import tempfile
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create temp env file
        env_file = tmp_path / "test.env"
        env_file.write_text(
            "# Test secrets file\n"
            "TEST_KEY_1=value1\n"
            "TEST_KEY_2=value_with_spaces\n"
            "DB_PASSWORD=secret123\n"
        )
        
        manager = DotenvSecretsManager(file_path=str(env_file))
        
        # Run tests
        test_get_secret_success(manager)
        test_get_secret_not_found(manager)
        test_secret_exists(manager)
        test_set_secret_new(manager, env_file)
        
        # Recreate manager for fresh state
        manager = DotenvSecretsManager(file_path=str(env_file))
        test_set_secret_overwrite(manager)
        
        manager = DotenvSecretsManager(file_path=str(env_file))
        test_set_secret_no_overwrite(manager)
        
        manager = DotenvSecretsManager(file_path=str(env_file))
        test_set_secret_with_spaces(manager)
        
        manager = DotenvSecretsManager(file_path=str(env_file))
        test_delete_secret(manager, env_file)
        
        manager = DotenvSecretsManager(file_path=str(env_file))
        test_delete_nonexistent(manager)
        
        # manager = DotenvSecretsManager(file_path=str(env_file))
        # test_list_secrets(manager)
        # Reset file to original state
        env_file.write_text(
            "# Test secrets file\n"
            "TEST_KEY_1=value1\n"
            "TEST_KEY_2=value_with_spaces\n"
            "DB_PASSWORD=secret123\n"
        )

        manager = DotenvSecretsManager(file_path=str(env_file))
        test_list_secrets(manager)
        
        manager = DotenvSecretsManager(file_path=str(env_file))
        test_list_secrets_with_prefix(manager)
        
        manager = DotenvSecretsManager(file_path=str(env_file))
        test_file_preservation(manager, env_file)
        
        test_empty_file_creation(tmp_path)
        
        print()
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
