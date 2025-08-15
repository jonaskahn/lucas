"""Security-related exports.

Exposes:
- `PluginValidator`: basic plugin validation and static scanning utilities
"""

from lucas.plugins.validator import PluginValidator

__all__ = ["PluginValidator"]
