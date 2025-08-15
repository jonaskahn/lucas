"""Plugin helper utilities for dynamic plugin loading and interfaces.

This module provides small, focused helpers used by the Lucas plugin system:

- `PluginLoader`: dynamically loads plugin modules from file paths with a
  simple in-memory cache.
- `PluginInterfaceBuilder`: normalizes optional plugin methods (e.g.,
  `get_config`, `validate`, `health_check`) so callers can rely on a
  consistent interface.
- `BasePluginHelper`/`PluginHelper`: convenience classes that combine loader
  and interface construction, and perform any environment setup needed by
  Lucas plugins.

These helpers are used by higher-level managers during plugin discovery and
assembly; documentation changes here do not alter runtime behavior.
"""

import importlib.util
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, Dict, Optional


class PluginLoader:
    """Handle dynamic loading of plugin modules with basic caching.

    Each loaded module is cached using a key derived from the module name and
    file path, avoiding repeated import work during discovery.
    """

    def __init__(self):
        """Initialize the plugin loader."""
        self._loaded_modules: Dict[str, Any] = {}
        self._lucas_path: Optional[Path] = None

    def setup_lucas_path(self) -> Path:
        """Add Lucas's source directory to `sys.path` if not already present.

        Returns:
            Path: The path that was ensured to be on `sys.path`.
        """
        if not self._lucas_path:
            self._lucas_path = Path(__file__).resolve().parents[1] / "src"
            if str(self._lucas_path) not in sys.path:
                sys.path.insert(0, str(self._lucas_path))
        return self._lucas_path

    def load_module(self, module_name: str, file_path: Path) -> Any:
        """Dynamically load a Python module from a file path.

        The loader uses `importlib.util.spec_from_file_location` and caches
        results to avoid duplicate loading.

        Args:
            module_name: Name to assign to the loaded module.
            file_path: Filesystem path to the module source.

        Returns:
            Any: The loaded module object.

        Raises:
            ImportError: If a spec cannot be created or executed.
        """
        # Check cache first
        cache_key = f"{module_name}_{file_path}"
        if cache_key in self._loaded_modules:
            return self._loaded_modules[cache_key]

        # Load module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Cache the loaded module
        self._loaded_modules[cache_key] = module
        return module

    def clear_cache(self):
        """Clear the loaded modules cache."""
        self._loaded_modules.clear()


class PluginInterfaceBuilder:
    """Build standard plugin interfaces from plugin classes.

    Ensures optional methods exist by providing sensible defaults so callers
    can treat all plugins uniformly during discovery and orchestration.
    """

    @staticmethod
    def create_interface(plugin_class: Type) -> Dict[str, Any]:
        """Create a normalized interface for a plugin class.

        Args:
            plugin_class: The plugin class to build an interface for.

        Returns:
            Dict[str, Any]: A mapping of interface function names to callables.
                Includes defaults for `get_config`, `validate`, and
                `health_check` when not provided by the class.
        """
        interface = {
            "get_metadata": plugin_class.get_metadata,
            "create_agent": plugin_class.create_agent,
        }

        # Add optional methods if they exist
        if hasattr(plugin_class, "get_config"):
            interface["get_config"] = plugin_class.get_config
        else:
            interface["get_config"] = lambda: {"enabled": True}

        if hasattr(plugin_class, "validate"):
            interface["validate"] = plugin_class.validate
        else:
            interface["validate"] = lambda: True

        if hasattr(plugin_class, "health_check"):
            interface["health_check"] = plugin_class.health_check
        else:
            interface["health_check"] = interface["validate"]

        return interface


class BasePluginHelper(ABC):
    """Abstract base for plugin helpers that combine loader and interface."""

    def __init__(self):
        """Initialize the base plugin helper."""
        self.loader = PluginLoader()
        self.interface_builder = PluginInterfaceBuilder()
        self.setup_environment()

    @abstractmethod
    def setup_environment(self):
        """Perform any environment setup required by the helper.

        Subclasses must implement this to ensure paths, configuration, or
        other prerequisites are set before loading plugins.
        """
        pass

    def load_agent_module(self, plugin_dir: Path) -> Any:
        """Load the `agent.py` module for a plugin.

        Args:
            plugin_dir: Directory containing the plugin sources.

        Returns:
            Any: The loaded agent module.

        Raises:
            ImportError: If the module cannot be loaded.
        """
        return self.loader.load_module(
            f"{plugin_dir.name}_agent", plugin_dir / "agent.py"
        )

    def load_tools_module(self, plugin_dir: Path) -> Any:
        """Load the `tools.py` module for a plugin.

        Args:
            plugin_dir: Directory containing the plugin sources.

        Returns:
            Any: The loaded tools module.

        Raises:
            ImportError: If the module cannot be loaded.
        """
        return self.loader.load_module(
            f"{plugin_dir.name}_tools", plugin_dir / "tools.py"
        )


class PluginHelper(BasePluginHelper):
    """Concrete plugin helper used by Lucas to load and normalize plugins."""

    def setup_environment(self):
        """Setup the Lucas plugin environment."""
        self.loader.setup_lucas_path()

    def create_plugin_class(self, plugin_class: Type) -> Dict[str, Any]:
        """Create a complete plugin class descriptor with a normalized interface.

        Args:
            plugin_class: The plugin class to enhance.

        Returns:
            Dict[str, Any]: A dictionary containing the original plugin class
            and the constructed interface mapping.
        """
        return {
            "plugin": plugin_class,
            "interface": self.interface_builder.create_interface(plugin_class),
        }


# Singleton instance for convenience
_helper_instance: Optional[PluginHelper] = None


def get_plugin_helper() -> PluginHelper:
    """Get or create the singleton plugin helper instance.

    Returns:
        The plugin helper instance
    """
    global _helper_instance
    if _helper_instance is None:
        _helper_instance = PluginHelper()
    return _helper_instance
