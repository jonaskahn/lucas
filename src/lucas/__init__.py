"""Lucas package public API and version.

Exposes convenient imports for external consumers:
- `Settings`: application configuration
- `MultiAgentOrchestrator`: conversation orchestrator
- `PluginManager`: plugin discovery and lifecycle manager
- `LLMModelFactory`: model provider and caching facade
"""

__version__ = "1.0.0"

from .config.settings import Settings
from .core.orchestrator import MultiAgentOrchestrator
from .llm.factory import LLMModelFactory
from .plugins.manager import PluginManager

__all__ = ["Settings", "MultiAgentOrchestrator", "PluginManager", "LLMModelFactory"]
