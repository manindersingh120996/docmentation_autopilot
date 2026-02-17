"""
Configuration management system

Loads configuration from YAML files with hierarchical overrides:
1. base.yaml (defaults for all env)
2. {environment}.yaml (environment-specific overrides)
3. Environment Variables (runtime secrets and overrides)

The configuration system follows these principles:
- Separation of Converns (config separate from code)
- Environmennt -specific overrides (one codebase, many deployments)
- Secure secrets management (secrets from env vars, never in files)
- Validation (fail fast if config is invalid)
- Caching (load once, reuse)

Usage: 
    from src.common.config import get_config

    config = get_config()
    log_level = config.application.logging.level
    db_host = config.infrastructure.relational_database.postgresql.host
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

from src.common.exceptions import ConfigurationError

def deep_merge(base:Dict[str, Any],
               override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.
    
    This is more sophesticated then dict.update() because it merges
    nested dictionaries recursively instead of replacing them entirely.

    Example:
        base = {'a': {'b': 1, 'c': 2}, 'd': 3}
        override = {'a': {'c': 99, 'e': 4}, 'f': 5}
        result = {'a': {'b': 1, 'c': 99, 'e': 4}, 'd': 3, 'f': 5}
    
    Args:
        base: Base dictionary (lower priority)
        override: Override dictionary (higher priority)
    
    Returns:
        Merged dictionary

    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key],dict) and isinstance(value,dict):
            # Both are dicts - merge recursively
            result[key] = deep_merge(result[key],value)
        else:
            # Override wins (not both dicts, or key not in base)
            result[key] = value
    
    return result

def substitute_env_vars(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Recursively substitute environment variables in config.
    
    Any config key ending with "_env_var" is treated as an environment
    variable name. We read that variable and replace the config entry.
    
    Example:
        config = {'database': {'password_env_var': 'DB_PASSWORD'}}
        os.environ['DB_PASSWORD'] = 'secret123'
        result = {'database': {'password': 'secret123'}}
    
    This pattern keeps secrets out of config files while making it
    clear in the config where secrets come from.
    
    Args:
        config: Configuration dictionary
        prefix: Key path prefix for error messages
    
    Returns:
        Config with environment variables substituted
    
    Raises:
        ConfigurationError: If required env var is not set
    """
    result = {}
    
    for key, value in config.items():
        current_path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recurse into nested dicts
            result[key] = substitute_env_vars(value, current_path)
            
        elif isinstance(value, str) and key.endswith('_env_var'):
            # This is an environment variable reference
            env_var_name = value
            env_var_value = os.environ.get(env_var_name)
            
            if env_var_value is None:
                raise ConfigurationError(
                    f"Required environment variable '{env_var_name}' not set "
                    f"(referenced by config key '{current_path}'). "
                    f"Did you create .env.local and set this variable?"
                )
            
            # Replace the "_env_var" key with actual key and value
            # e.g., "password_env_var" becomes "password"
            actual_key = key.replace('_env_var', '')
            result[actual_key] = env_var_value
            
        else:
            # Regular value, pass through unchanged
            result[key] = value
    
    return result


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return as dictionary.
    
    Args:
        file_path: Path to YAML file
    
    Returns:
        Parsed YAML as dictionary
    
    Raises:
        ConfigurationError: If file doesn't exist or can't be parsed
    """
    if not file_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {file_path}"
            f"Make sure you're running from the project root directory."
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML syntax in {file_path}: {e}"
            f"Check for indentation errors, invalid characters, or unclosed quotes."
        )
    except Exception as e:
        raise ConfigurationError(
            f"Error reading {file_path}: {e}"
        )


class DotDict(dict):
    """
    Dictionary that allows dot notation access to nested keys.
    
    This is purely for convenience - makes config access cleaner:
        config.application.logging.level
    instead of:
        config['application']['logging']['level']
    
    Implements both dict-style and attribute-style access so you can
    use whichever is more convenient in context.
    """
    
    def __getattr__(self, key: str) -> Any:
        """
        Allow attribute-style access: config.application
        
        Raises AttributeError if key doesn't exist (standard Python behavior).
        """
        try:
            value = self[key]
            # If value is a dict, wrap it too for chained access
            if isinstance(value, dict) and not isinstance(value, DotDict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(
                f"Configuration key '{key}' not found. "
                f"Available keys: {', '.join(self.keys())}"
            )
    
    def __setattr__(self, key: str, value: Any) -> None:
        """Allow attribute-style setting: config.new_key = value"""
        self[key] = value
    
    def __delattr__(self, key: str) -> None:
        """Allow attribute-style deletion: del config.key"""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found")


@lru_cache(maxsize=1)
def get_config(environment: Optional[str] = None) -> DotDict:
    """
    Load and return application configuration.
    
    This is the main entry point for configuration access. It:
    1. Determines which environment to use (local, production, etc.)
    2. Loads base.yaml (default configuration)
    3. Loads environment-specific overrides (local.yaml, production.yaml, etc.)
    4. Deep merges them with proper precedence
    5. Substitutes environment variables for secrets
    6. Returns a DotDict for convenient dot-notation access
    
    The result is cached (@lru_cache) so subsequent calls are instant.
    Configuration is loaded once at startup and reused throughout
    the application's lifecycle.
    
    Args:
        environment: Override the environment (defaults to APP_ENV env var
                    or 'local' if not set). Valid values: 'local', 'production'
    
    Returns:
        Configuration as a DotDict with dot notation access
    
    Raises:
        ConfigurationError: If configuration is invalid or can't be loaded
    
    Example:
        config = get_config()
        
        # Access with dot notation
        log_level = config.application.logging.level
        
        # Or dict-style
        log_level = config['application']['logging']['level']
        
        # Both work interchangeably
    """
    # Determine which environment to use
    if environment is None:
        environment = os.environ.get('APP_ENV', 'local')
    
    # Determine paths to config files
    # Look for config/ relative to this file's parent directory
    config_dir = Path(__file__).parent.parent.parent / 'config'
    base_config_path = config_dir / 'base.yaml'
    env_config_path = config_dir / f'{environment}.yaml'
    
    # Load base configuration (always required)
    base_config = load_yaml_file(base_config_path)
    
    # Load environment-specific overrides (optional, but usually exists)
    if env_config_path.exists():
        env_config = load_yaml_file(env_config_path)
        # Deep merge: env_config overrides base_config
        merged_config = deep_merge(base_config, env_config)
    else:
        # No environment-specific overrides, just use base
        merged_config = base_config
    
    # Substitute environment variables (for secrets)
    try:
        final_config = substitute_env_vars(merged_config)
    except ConfigurationError as e:
        # Add helpful context to the error
        raise ConfigurationError(
            f"Failed to load configuration for environment '{environment}': {e}"
            f"If you're running locally, make sure:"
            f"1. You've copied .env.example to .env.local"
            f"2. You've filled in all required values in .env.local"
            f"3. You've activated your Python virtual environment (source venv/bin/activate)"
        )
    
    # Wrap in DotDict for convenient access
    return DotDict(final_config)


def reload_config() -> None:
    """
    Clear the config cache and force reload on next get_config() call.
    
    This is primarily useful for testing when you want to simulate
    different configurations. In production, configuration is loaded
    once and never reloaded (reloading would require app restart anyway
    to propagate changes to all components).
    
    Example in tests:
        # Test with one config
        config = get_config()
        assert config.application.logging.level == "INFO"
        
        # Change environment and reload
        os.environ['APP_ENV'] = 'production'
        reload_config()
        
        # Get new config
        config = get_config()
        assert config.application.logging.level == "WARNING"
    """
    get_config.cache_clear()


def validate_config(config: DotDict) -> None:
    """
    Validate that configuration is complete and values are valid.
    
    This performs sanity checks on the loaded configuration to fail
    fast at startup rather than encountering errors later during runtime.
    
    Called automatically by application entry points (workers, API server, etc.)
    after loading configuration.
    
    Args:
        config: Configuration to validate
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Check required top-level sections exist
    required_sections = ['system', 'application', 'infrastructure']
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(
                f"Missing required config section: '{section}'"
                f"This indicates a problem with base.yaml. "
                f"Please ensure base.yaml is complete."
            )
    
    # Validate application limits are sensible
    limits = config.application.limits
    
    if limits.max_webhook_processing_time_seconds <= 0:
        raise ConfigurationError(
            "max_webhook_processing_time_seconds must be positive"
        )
    
    if limits.max_concurrent_workers < 1:
        raise ConfigurationError(
            "max_concurrent_workers must be at least 1"
        )
    
    if limits.max_llm_retries < 0:
        raise ConfigurationError(
            "max_llm_retries must be non-negative"
        )
    
    # Validate LLM temperature is in valid range
    for layer in ['layer2_analysis', 'layer3_orchestration']:
        if layer in config:
            layer_config = config[layer]
            for component in layer_config.values():
                if isinstance(component, dict) and 'temperature' in component:
                    temp = component['temperature']
                    if not (0 <= temp <= 2):
                        raise ConfigurationError(
                            f"LLM temperature must be between 0 and 2, got {temp}"
                        )
    
    # Validate infrastructure providers are recognized
    valid_event_bus_providers = ['local_rabbitmq', 'gcp_pubsub']
    if config.infrastructure.event_bus.provider not in valid_event_bus_providers:
        raise ConfigurationError(
            f"Invalid event_bus provider: '{config.infrastructure.event_bus.provider}'. "
            f"Must be one of: {', '.join(valid_event_bus_providers)}"
        )
    
    valid_vector_store_providers = ['chromadb', 'vertex_ai', 'weaviate']
    if config.infrastructure.vector_store.provider not in valid_vector_store_providers:
        raise ConfigurationError(
            f"Invalid vector_store provider: '{config.infrastructure.vector_store.provider}'. "
            f"Must be one of: {', '.join(valid_vector_store_providers)}"
        )
    
    # All checks passed
    print(f"✓ Configuration validated successfully for environment: {config.system.environment}")


# For convenience, export commonly used functions at module level
__all__ = [
    'get_config',
    'reload_config',
    'validate_config',
    'ConfigurationError',
    'DotDict'
]