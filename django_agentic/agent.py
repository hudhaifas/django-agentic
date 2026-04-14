"""ModelAgent — abstract base for entity-aware AI agents.

Mirrors silverstripe-ai's DataObjectAgent. Consuming app subclasses this
and implements get_static_instructions(), get_dynamic_context(), get_tools().
django_ai handles provider, prompt caching, chat history, logging, credits, HITL.

Usage:
    from django_agentic.agent import ModelAgent

    class MyModelAgent(ModelAgent):
        def get_static_instructions(self) -> str: ...
        def get_dynamic_context(self) -> str: ...
        def get_tools(self) -> list: ...  # optional

Register in settings:
    DJANGO_AGENTIC = {
        "AGENT_MAPPINGS": {
            "myapp.MyModel": "myapp.agents.MyModelAgent",
        }
    }
"""

import abc

from django.db.models import Model


class ModelAgent(abc.ABC):

    def __init__(self, user, model_config: "AIModel", entity: Model | None, thread_id: str):
        self.user = user
        self.model_config = model_config
        self.entity = entity
        self.thread_id = thread_id

    @property
    def is_collection_mode(self) -> bool:
        return self.entity is None or not getattr(self.entity, "pk", None)

    @abc.abstractmethod
    def get_static_instructions(self) -> str:
        """Cacheable system prompt. Cached by Anthropic for 5 min."""

    @abc.abstractmethod
    def get_dynamic_context(self) -> str:
        """Ephemeral context — entity state, date. Rebuilt per request."""

    def get_tools(self) -> list:
        """LangChain tools the agent can call. Default: no tools (chat-only).

        Return a list of LangChain tool instances (functions decorated with
        @tool, or BaseTool subclasses). Tools are passed to create_agent
        which binds them to the LLM automatically.
        """
        return []

    def get_tools_requiring_approval(self) -> list[str]:
        """Tool names requiring HITL approval before execution.

        Return tool names (strings) that should trigger an interrupt before
        the tool node runs. The frontend receives the pending tool calls and
        must resume with an approval/rejection decision.

        Maps to silverstripe-ai's ToolApproval middleware concept.
        """
        return []

    def summarise_action(self, tool_name: str, tool_input: dict) -> str:
        """Human-readable summary for a pending tool action (HITL card).

        Override in subclasses for domain-specific summaries.
        """
        return f"Execute: {tool_name}"


class AgentRegistry:
    """Maps Django model → agent class via DJANGO_AGENTIC['AGENT_MAPPINGS']."""

    @classmethod
    def get_agent_class(cls, entity_class_path: str) -> type[ModelAgent]:
        from django.conf import settings
        from django.utils.module_loading import import_string
        mappings = getattr(settings, "DJANGO_AGENTIC", {}).get("AGENT_MAPPINGS", {})
        if entity_class_path in mappings:
            return import_string(mappings[entity_class_path])
        try:
            entity_class = import_string(entity_class_path)
            for parent in entity_class.__mro__:
                if hasattr(parent, "_meta"):
                    short = f"{parent._meta.app_label}.{parent.__name__}"
                    if short in mappings:
                        return import_string(mappings[short])
        except (ImportError, AttributeError):
            pass
        raise ValueError(f"No agent mapping for '{entity_class_path}'.")

    @classmethod
    def create_for_entity(cls, user, model_config, entity, thread_id: str) -> ModelAgent:
        path = f"{entity._meta.app_label}.{entity.__class__.__name__}"
        agent_class = cls.get_agent_class(path)
        return agent_class(user=user, model_config=model_config, entity=entity, thread_id=thread_id)
