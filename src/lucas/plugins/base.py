"""Plugin base interfaces for Lucas.

This module defines the foundational interfaces used by the Lucas plugin system:

- `PluginMetadata`: a structured descriptor for plugin bundles, including
  capabilities and LLM requirements.
- `BasePluginAgent`: a base class for plugin agents that act as LangGraph
  agent nodes, encapsulating tool provisioning, system prompt management,
  model binding, and simple routing.
- `BasePlugin`: a minimal interface that each plugin bundle should implement.

No runtime behavior is changed by documentation updates. The classes here are
consumed by the orchestrator to assemble agent and tool nodes in the sequential
LangGraph flow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool


@dataclass
class PluginMetadata:
    """Comprehensive metadata for plugin bundles.

    Args:
        name: Unique plugin name. Used to name graph nodes and for routing.
        version: Semantic version of the plugin implementation.
        description: Human-readable summary of what the plugin does.
        capabilities: Focus areas this plugin can handle (used for routing UX).
        dependencies: Optional list of library/package prerequisites.
        llm_requirements: Optional provider/model hints for the agent's model
            (e.g., {"provider": "openai", "model": "gpt-4o"}).
        agent_type: Informational categorization: "specialized", "general",
            or "utility".
        system_prompt: Default system prompt used by the agent implementation.
        tool_categories: Optional classification for the provided tools.
    """

    name: str
    version: str
    description: str
    capabilities: List[str]
    dependencies: List[str] = field(default_factory=list)
    llm_requirements: Dict[str, Any] = field(default_factory=dict)
    agent_type: str = "specialized"
    system_prompt: str = ""
    tool_categories: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Plugin name cannot be empty")
        if not self.version or not self.version.strip():
            raise ValueError("Plugin version cannot be empty")
        if self.agent_type not in {"specialized", "general", "utility"}:
            raise ValueError(f"Invalid agent_type: {self.agent_type}")

    @property
    def is_specialized_agent(self) -> bool:
        """Check if this is a specialized agent."""
        return self.agent_type == "specialized"

    @property
    def is_general_agent(self) -> bool:
        """Check if this is a general agent."""
        return self.agent_type == "general"

    @property
    def is_utility_agent(self) -> bool:
        """Check if this is a utility agent."""
        return self.agent_type == "utility"


class BasePluginAgent(ABC):
    """Base class for plugin agents used as LangGraph agent nodes.

    Implementations must provide tools and a system prompt. During graph
    construction, a base chat model is bound to the agent's tools via
    `bind_model()`. The `create_agent_node()` helper returns a callable that
    the orchestrator adds to the graph. Simple routing is supported through
    `should_continue()` which signals whether to call tools or return to the
    coordinator.
    """

    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self._tools = None
        self._bound_model = None
        self._initialized = False

    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """Return the tools that this agent exposes.

        Returns:
            List of LangChain `Tool` instances to be bound to the LLM.
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt used by this agent."""
        pass

    def bind_model(self, model: BaseChatModel) -> BaseChatModel:
        """Bind the agent's tools to the provided chat model.

        Args:
            model: Base chat model to be specialized with this agent's tools.

        Returns:
            The tool-bound chat model to use within the agent node.
        """
        tools = self.get_tools()
        self._bound_model = model.bind_tools(tools)
        return self._bound_model

    def should_continue(self, state: Dict[str, Any]) -> str:
        """Decide whether to call tools or return to the coordinator.

        Args:
            state: The current graph state (expects a `messages` list).

        Returns:
            "continue" if the last assistant message contains tool calls;
            otherwise "back" to return control to the coordinator.
        """
        last_msg = state.get("messages", [])[-1] if state.get("messages") else None
        if not last_msg:
            return "back"

        tool_calls = getattr(last_msg, "tool_calls", None)
        return "continue" if tool_calls else "back"

    def create_agent_node(self):
        """Create the callable used as this plugin's agent node.

        The returned function consumes the current state, prepends the agent's
        system prompt, invokes the bound model, and returns a partial state
        update suitable for LangGraph state merging.

        Returns:
            A callable with signature `fn(state: Dict[str, Any]) -> Dict[str, Any]`.
        """
        import logging

        logger = logging.getLogger(__name__)

        def agent_node(state):
            """Invoke the bound model and return state updates.

            Expects `state["messages"]` to be a list of LangChain messages.
            Returns a dict containing new messages and metadata fields the
            orchestrator relies on (e.g., `hops`, `current_agent`).
            """
            try:
                logger.info(f"Agent node invoked for {self.metadata.name}")
                logger.info(f"State messages: {len(state.get('messages', []))}")

                if not hasattr(self, "_bound_model"):
                    logger.error(f"No bound model for agent {self.metadata.name}")
                    raise RuntimeError(f"No bound model for agent {self.metadata.name}")

                system = SystemMessage(content=self.get_system_prompt())
                response = self._bound_model.invoke([system] + state["messages"])

                logger.info(f"Agent response: {response}")

                # Track agent usage
                agents_used = state.get("agents_used", [])
                if self.metadata.name not in agents_used:
                    agents_used = agents_used + [self.metadata.name]

                # Update plugin context
                plugin_context = state.get("plugin_context", {})
                plugin_context["last_plugin"] = self.metadata.name
                if "routing_history" not in plugin_context:
                    plugin_context["routing_history"] = []
                plugin_context["routing_history"] = plugin_context[
                    "routing_history"
                ] + [self.metadata.name]

                return {
                    "messages": [response],
                    "hops": state.get("hops", 0) + 1,
                    "current_agent": self.metadata.name,
                    "agents_used": agents_used,
                    "plugin_context": plugin_context,
                }
            except Exception as e:
                logger.error(f"Error in agent node for {self.metadata.name}: {e}")
                raise

        return agent_node

    def initialize(self) -> None:
        """Initialize agent resources (e.g., cache tools)."""
        self._tools = self.get_tools()
        self._initialized = True

    def cleanup(self) -> None:
        """Cleanup agent resources (override if needed)."""
        pass


class BasePlugin(ABC):
    """Minimal interface that a Lucas plugin bundle must implement.

    A plugin bundle provides metadata and a concrete agent implementation.
    Optional hooks allow dependency validation and configuration schema
    exposure for UIs.
    """

    @staticmethod
    @abstractmethod
    def get_metadata() -> PluginMetadata:
        """Return plugin metadata used for discovery and routing."""
        pass

    @staticmethod
    @abstractmethod
    def create_agent() -> BasePluginAgent:
        """Create and return the plugin agent instance."""
        pass

    @staticmethod
    def validate_dependencies() -> List[str]:
        """Validate plugin dependencies.

        Returns:
            A list of error messages; empty if all checks pass.
        """
        return []

    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        """Return a JSON-serializable configuration schema for this plugin."""
        return {}
