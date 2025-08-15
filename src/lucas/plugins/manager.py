"""Plugin management for Lucas plugin bundles.

This module discovers plugin bundles (agent + tools + metadata), validates
their structure, binds tools to models via the LLM factory, and exposes
helpers for adding nodes and edges to the LangGraph-based orchestrator.

Documentation updates only; no behavior is changed.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from langchain_core.tools import Tool, tool
from langgraph.prebuilt import ToolNode

from .base import BasePluginAgent, PluginMetadata
from ..base.loggable import Loggable
from ..llm.factory import LLMModelFactory

# Optional import with fallback
try:
    from lucas.plugins.validator import PluginValidator
except ImportError:
    PluginValidator = None


class PluginBundle(Loggable):
    """A complete plugin bundle: agent, tools, and metadata.

    Instances materialize both the agent node callable and the LangGraph
    `ToolNode`, and provide helpers to retrieve graph nodes and edges for
    integration into the orchestrator.
    """

    def __init__(
        self,
        metadata: PluginMetadata,
        agent: BasePluginAgent,
        bound_model,
        tools: List[Tool],
    ):
        super().__init__()
        self.metadata = metadata
        self.agent = agent  # The agent node
        self.bound_model = bound_model  # Model with tools bound
        self.tools = tools  # The tools
        self.tool_node = ToolNode(tools)  # LangGraph tool node
        self.agent_node = agent.create_agent_node()  # LangGraph agent node

    def get_graph_nodes(self) -> Dict[str, Any]:
        """Return LangGraph node callables for this plugin bundle.

        Returns:
            Dict[str, Any]: Mapping of node names to callables for the agent
            and tool nodes.
        """
        agent_name = self.metadata.name
        return {
            f"{agent_name}_agent": self.agent_node,
            f"{agent_name}_tools": self.tool_node,
        }

    def get_graph_edges(self) -> Dict[str, Any]:
        """Return LangGraph edge definitions for this plugin bundle.

        Returns:
            Dict[str, Any]: Edge definitions with a conditional edge from the
            agent to either tools or the coordinator, and a direct edge from
            tools back to the agent.
        """
        agent_name = self.metadata.name
        return {
            "conditional_edges": {
                f"{agent_name}_agent": {
                    "condition": self.agent.should_continue,
                    "mapping": {
                        "continue": f"{agent_name}_tools",
                        "back": "coordinator",
                    },
                }
            },
            "direct_edges": [(f"{agent_name}_tools", f"{agent_name}_agent")],
        }


class PluginManager(Loggable):
    """Manage discovery, validation, loading, and health of plugin bundles.

    Responsibilities:
    - Discover plugin directories containing `plugin.py`
    - Validate plugin structure (optionally via `PluginValidator`)
    - Create and bind agent models using `LLMModelFactory`
    - Provide graph nodes/edges and routing tool helpers
    - Track healthy/failed plugins and expose simple health checks
    """

    def __init__(self, plugins_dir: str, llm_factory: LLMModelFactory):
        super().__init__()
        self.plugins_dir = Path(plugins_dir)
        self.llm_factory = llm_factory

        self.plugin_bundles: Dict[str, PluginBundle] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}

        self.validator = PluginValidator() if PluginValidator else None

        self.healthy_plugins: Set[str] = set()
        self.failed_plugins: Set[str] = set()

    def discover_plugin_directories(self) -> List[Path]:
        """Discover plugin directories containing `plugin.py`.

        Returns:
            List[Path]: Directories recognized as plugin bundles.
        """
        plugins = []
        if not self.plugins_dir.exists():
            self.logger.warning(f"Plugins directory not found: {self.plugins_dir}")
            return plugins

        for plugin_path in self.plugins_dir.iterdir():
            if plugin_path.is_dir() and (plugin_path / "plugin.py").exists():
                plugins.append(plugin_path)

        self.logger.info(f"Discovered {len(plugins)} plugin directories")
        return plugins

    def load_plugin_bundle(self, plugin_path: Path) -> bool:
        """Load a plugin bundle, validate it, and bind its model/tools.

        Args:
            plugin_path: Path to the plugin directory containing `plugin.py`.

        Returns:
            bool: True if loaded successfully; False otherwise.
        """
        try:
            self.logger.info(f"Loading plugin bundle from: {plugin_path}")

            if self.validator:
                validation_errors = self.validator.validate_plugin_directory(
                    plugin_path
                )
                if validation_errors:
                    self.logger.error(f"Plugin validation failed: {validation_errors}")
                    return False

            plugin_module = self._load_plugin_module(plugin_path)
            if not plugin_module:
                return False

            metadata = plugin_module.get_metadata()
            agent = plugin_module.create_agent()

            model_config = self._create_model_config(metadata)
            base_model = self.llm_factory.create_base_model(model_config)

            tools = agent.get_tools()
            bound_model = agent.bind_model(base_model)

            agent.initialize()

            bundle = PluginBundle(
                metadata=metadata, agent=agent, bound_model=bound_model, tools=tools
            )

            self.plugin_bundles[metadata.name] = bundle
            self.plugin_metadata[metadata.name] = metadata
            self.healthy_plugins.add(metadata.name)

            self.logger.info(
                f"Successfully loaded plugin bundle: {metadata.name} v{metadata.version}"
            )
            self.logger.info(f"  - Agent: {type(agent).__name__}")
            self.logger.info(f"  - Tools: {len(tools)} tools")
            self.logger.info(f"  - Capabilities: {metadata.capabilities}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load plugin bundle {plugin_path}: {e}")
            self.failed_plugins.add(plugin_path.name)
            return False

    def _load_plugin_module(self, plugin_path: Path):
        """Load a plugin module from disk using importlib.

        Args:
            plugin_path: Path to the plugin directory.

        Returns:
            Any | None: The loaded module or None on failure (errors are logged).
        """
        plugin_file = plugin_path / "plugin.py"
        module_name = f"plugin_{plugin_path.name}"

        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if not spec or not spec.loader:
            self.logger.error(f"Could not create spec for {plugin_file}")
            return None

        module = importlib.util.module_from_spec(spec)

        sys.path.insert(0, str(plugin_path))
        try:
            spec.loader.exec_module(module)

            if not hasattr(module, "get_metadata") or not hasattr(
                module, "create_agent"
            ):
                self.logger.error(f"Plugin missing required functions: {plugin_path}")
                return None

            return module
        except Exception as e:
            self.logger.error(f"Error executing plugin module {plugin_path}: {e}")
            return None
        finally:
            if str(plugin_path) in sys.path:
                sys.path.remove(str(plugin_path))

    @staticmethod
    def _create_model_config(metadata: PluginMetadata):
        """Create a model configuration from plugin metadata.

        Args:
            metadata: The plugin's `PluginMetadata`.

        Returns:
            ModelConfig: Provider/model parameters for the LLM factory.
        """
        from ..llm.providers import ModelConfig

        # Use plugin's LLM requirements or defaults
        llm_reqs = metadata.llm_requirements or {}

        return ModelConfig(
            provider=llm_reqs.get("provider", "openai"),
            model_name=llm_reqs.get("model", "gpt-4o"),
            temperature=llm_reqs.get("temperature", 0.0),
            max_tokens=llm_reqs.get("max_tokens", 1000),
        )

    def load_all_plugin_bundles(self) -> None:
        """Discover and load all plugin bundles.

        Logs a summary of successes and failures. Does not raise.
        """
        plugin_paths = self.discover_plugin_directories()

        loaded_count = 0
        for plugin_path in plugin_paths:
            if self.load_plugin_bundle(plugin_path):
                loaded_count += 1

        self.logger.info(f"Loaded {loaded_count}/{len(plugin_paths)} plugin bundles")

        if self.failed_plugins:
            self.logger.warning(f"Failed to load plugins: {list(self.failed_plugins)}")

    def get_plugin_bundle(self, name: str) -> Optional[PluginBundle]:
        """Get a previously loaded plugin bundle by name.

        Args:
            name: Plugin name as defined in `PluginMetadata.name`.

        Returns:
            Optional[PluginBundle]: The bundle if available, else None.
        """
        return self.plugin_bundles.get(name)

    def get_available_plugins(self) -> List[str]:
        """List names of successfully loaded plugin bundles.

        Returns:
            List[str]: Plugin names.
        """
        return list(self.plugin_bundles.keys())

    def get_all_plugin_tools(self) -> Dict[str, List[Tool]]:
        """Return all registered tools organized by plugin name.

        Returns:
            Dict[str, List[Tool]]: Mapping of plugin name to its tools.
        """
        result = {}
        for name, bundle in self.plugin_bundles.items():
            result[name] = bundle.tools
        return result

    def get_plugin_routing_info(self) -> Dict[str, str]:
        """Return short routing descriptions for coordinator prompts.

        Returns:
            Dict[str, str]: Mapping of plugin name to a concise description
            of capabilities.
        """
        result = {}
        for name, bundle in self.plugin_bundles.items():
            result[name] = (
                f"{bundle.metadata.description}. Capabilities only for: {', '.join(bundle.metadata.capabilities)}"
            )
        return result

    def perform_health_checks(self) -> Dict[str, bool]:
        """Perform basic health checks on all plugin bundles.

        Returns:
            Dict[str, bool]: Mapping of plugin name to health status.
        """
        results = {}

        for plugin_name, bundle in self.plugin_bundles.items():
            try:
                # Basic health check
                is_healthy = (
                    bundle.agent is not None
                    and bundle.bound_model is not None
                    and len(bundle.tools) > 0
                    and bundle.metadata is not None
                )

                results[plugin_name] = is_healthy

                if is_healthy:
                    self.healthy_plugins.add(plugin_name)
                    self.failed_plugins.discard(plugin_name)
                else:
                    self.failed_plugins.add(plugin_name)
                    self.healthy_plugins.discard(plugin_name)

            except Exception as e:
                self.logger.error(f"Health check failed for {plugin_name}: {e}")
                results[plugin_name] = False
                self.failed_plugins.add(plugin_name)

        return results

    def get_coordinator_tools(self) -> List[Tool]:
        """Create routing tools used by the coordinator for control flow.

        Returns:
            List[Tool]: A list of routing tools such as dynamic `goto_<name>`
            functions and a `finalize` tool to end the flow.
        """

        control_tools: List[Tool] = []

        # Create goto tools for each plugin
        for plugin_name in self.get_available_plugins():
            metadata = self.plugin_metadata[plugin_name]

            # Create closure to capture plugin_name
            def make_goto_tool(name: str, desc: str) -> Tool:
                # Create a function with dynamic name using exec
                func_name = f"goto_{name}"
                func_code = f'''
def {func_name}():
    """Route to {name} plugin for {desc}"""
    return "{name}"
'''
                local_vars = {}
                exec(func_code, {}, local_vars)
                return tool(local_vars[func_name])

            control_tools.append(make_goto_tool(plugin_name, metadata.description))

        # Add finalize tool
        @tool
        def finalize():
            """Signal that the task is complete and provide final response."""
            return "final"

        control_tools.append(finalize)

        return control_tools
