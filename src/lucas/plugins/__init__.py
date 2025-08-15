"""Plugin system exports.

Exposes:
- `BasePlugin`, `BasePluginAgent`, `PluginMetadata`: plugin interfaces
- `PluginManager`, `PluginBundle`: plugin discovery, loading, and lifecycle
"""

from .base import BasePlugin, BasePluginAgent, PluginMetadata
from .manager import PluginManager, PluginBundle

__all__ = [
    "BasePlugin",
    "BasePluginAgent",
    "PluginMetadata",
    "PluginManager",
    "PluginBundle",
]
