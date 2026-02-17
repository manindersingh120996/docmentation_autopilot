#!/usr/bin/env python3
"""
Test script for configuration system.

Run this to verify configuration loads correctly and all required
environment variables are set.
"""

import sys
from pathlib import Path

# Add src/ to Python path so we can import from it
project_root = Path(__file__).parent.parent
sys.path.insert(0,str(project_root))


# Load environment variables from .env.local
from dotenv import load_dotenv
load_dotenv('.env.local')

# Now test configuration
from src.common.config import get_config, validate_config
from src.common.exceptions import ConfigurationError


def test_config_loading():
    """Test that configuration loads without errors."""
    print("Testing configuration loading...")
    
    try:
        config = get_config()
        print(f"✓ Configuration loaded successfully")
        print(f"  Environment: {config.system.environment}")
        print(f"  Version: {config.system.version}")
        return config
    except ConfigurationError as e:
        print(f"✗ Configuration loading failed: {e}")
        sys.exit(1)


def test_config_access():
    """Test various ways of accessing configuration."""
    print("Testing configuration access patterns...")
    
    config = get_config()
    
    # Test dot notation
    try:
        log_level = config.application.logging.level
        print(f"✓ Dot notation access works: logging.level = {log_level}")
    except Exception as e:
        print(f"✗ Dot notation failed: {e}")
        sys.exit(1)
    
    # Test dict-style access
    try:
        log_level = config['application']['logging']['level']
        print(f"✓ Dict-style access works: logging.level = {log_level}")
    except Exception as e:
        print(f"✗ Dict-style access failed: {e}")
        sys.exit(1)
    
    # Test accessing infrastructure config
    try:
        provider = config.infrastructure.event_bus.provider
        print(f"✓ Infrastructure config accessible: event_bus.provider = {provider}")
    except Exception as e:
        print(f"✗ Infrastructure access failed: {e}")
        sys.exit(1)


def test_env_var_substitution():
    """Test that environment variables were substituted correctly."""
    print("Testing environment variable substitution...")
    
    config = get_config()
    
    # Check that _env_var keys were replaced with actual values
    try:
        # These should have been substituted from .env.local
        neo4j_config = config.infrastructure.graph_database.neo4j
        
        # Check password was substituted (should have 'password', not 'password_env_var')
        if 'password_env_var' in neo4j_config:
            print(f"✗ Environment variable not substituted: password_env_var still present")
            sys.exit(1)
        
        if 'password' not in neo4j_config:
            print(f"✗ Environment variable substitution failed: password key missing")
            sys.exit(1)
        
        password = neo4j_config.password
        print(f"✓ Environment variables substituted correctly")
        print(f"  Neo4j password: {'*' * len(password)} (hidden for security)")
        
    except Exception as e:
        print(f"✗ Environment variable test failed: {e}")
        sys.exit(1)


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    config = get_config()
    
    try:
        validate_config(config)
        # validate_config prints its own success message
    except ConfigurationError as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)


def test_llm_api_key():
    """Test that LLM API key is configured."""
    print("Testing LLM API key configuration...")
    
    import os
    
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print(f"✗ OPENROUTER_API_KEY not set in environment")
        print(f"  Make sure you've set it in .env.local")
        sys.exit(1)
    
    if api_key == 'your_openrouter_api_key_here':
        print(f"✗ OPENROUTER_API_KEY is still the placeholder value")
        print(f"  Replace it with your actual API key in .env.local")
        sys.exit(1)
    
    print(f"✓ OPENROUTER_API_KEY is configured")
    print(f"  Key: {api_key[:10]}... (showing first 10 characters)")


def main():
    """Run all configuration tests."""
    print("=" * 70)
    print("Configuration System Test Suite")
    print("=" * 70)
    
    test_config_loading()
    test_config_access()
    test_env_var_substitution()
    test_config_validation()
    test_llm_api_key()
    
    print("" + "=" * 70)
    print("✓ All configuration tests passed!")
    print("=" * 70)
    print("Your configuration system is working correctly.")
    print("You're ready to proceed with implementing infrastructure components.")


if __name__ == '__main__':
    main()
