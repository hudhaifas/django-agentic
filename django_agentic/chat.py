"""Agent chat — LangGraph-based conversational agents with HITL.

Extracted from service.py. Handles agent graph building, chat/resume
invocation, interrupt handling, and conversation history.

All provider instantiation goes through providers.py.
All usage recording goes through service._record_usage().
"""

import functools
import logging
import time
from dataclasses import dataclass, field

from django.db.models import Model
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_anthropic.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from .providers import _get_setting, create_chat_model

logger = logging.getLogger(__name__)


# ── Response dataclass ────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """Response from agent chat or resume."""
    success: bool = True
    message: str = ""
    usage: dict = field(default_factory=dict)
    interrupt: dict | None = None

    def to_dict(self) -> dict:
        data = {"success": self.success, "message": self.message,
                "usage": self.usage}
        if self.interrupt is not None:
            data["interrupt"] = self.interrupt
        return data

    @property
    def has_interrupt(self) -> bool:
        return self.interrupt is not None


# ── Helpers ───────────────────────────────────────────────────────────

def _normalize_content(content) -> str:
    """Ensure message content is a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content) if content else ""


def _extract_chat_history(messages: list) -> list[dict]:
    """Convert LangGraph message list to serializable chat history."""
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": _normalize_content(m.content)})
        elif isinstance(m, AIMessage):
            content = _normalize_content(m.content)
            if not content and getattr(m, "tool_calls", None):
                continue
            if content:
                history.append({"role": "assistant", "content": content})
    return history


def _usage_from_callback(cb: UsageMetadataCallbackHandler) -> dict:
    """Extract usage from the built-in UsageMetadataCallbackHandler."""
    if not cb.usage_metadata:
        return {}
    total_in, total_out, cache_read, cache_create = 0, 0, 0, 0
    for model_usage in cb.usage_metadata.values():
        total_in += model_usage.get("input_tokens", 0)
        total_out += model_usage.get("output_tokens", 0)
        details = model_usage.get("input_token_details", {})
        cache_read += details.get("cache_read", 0)
        cache_create += (
            details.get("cache_creation", 0)
            + details.get("ephemeral_5m_input_tokens", 0)
            + details.get("ephemeral_1h_input_tokens", 0)
        )
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_create,
    }


def _count_interrupt_actions(state) -> int:
    """Count pending HITL actions from a graph state's interrupt tasks."""
    if not state or not hasattr(state, "tasks") or not state.tasks:
        return 1
    for task in state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            val = task.interrupts[0].value
            if isinstance(val, dict) and "action_requests" in val:
                return len(val["action_requests"])
    return 1


def build_system_message(static_text: str, dynamic_text: str = "") -> str:
    """Combine static instructions and dynamic context into a system prompt."""
    if dynamic_text:
        return static_text + "\n\n" + dynamic_text
    return static_text


@functools.lru_cache(maxsize=1)
def _get_checkpointer():
    custom = _get_setting("CHECKPOINTER")
    if custom is not None:
        return custom
    return InMemorySaver()


def _build_history_key(user, entity: Model | None, thread_id: str) -> str:
    """Build a deterministic thread key for the checkpointer."""
    if entity and getattr(entity, "pk", None):
        return f"{user.pk}_{entity._meta.label}_{entity.pk}"
    return thread_id


# ── Agent graph builder ──────────────────────────────────────────────

def build_agent_graph(chat_model: BaseChatModel, tools: list,
                      system_msg: str, context_window: int,
                      checkpointer, tools_requiring_approval: list[str]):
    """Build a LangGraph agent using langchain's create_agent.

    Uses create_agent for ALL agents — with or without tools.
    Prompt caching via AnthropicPromptCachingMiddleware.
    HITL via HumanInTheLoopMiddleware for selective tool approval.
    """
    middleware = [AnthropicPromptCachingMiddleware()]

    if tools and tools_requiring_approval:
        interrupt_on = {}
        approval_set = set(tools_requiring_approval)
        for t in tools:
            name = t.name if hasattr(t, "name") else t.__name__
            interrupt_on[name] = True if name in approval_set else False
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model=chat_model,
        tools=tools or None,
        system_prompt=system_msg,
        middleware=middleware,
        checkpointer=checkpointer,
    )


# ── Agent session management ─────────────────────────────────────────

def prepare_agent_session(user, entity: Model, thread_id: str = ""):
    """Shared setup for chat() and resume().

    Returns (agent, model, compiled_graph, config).
    """
    from .agent import AgentRegistry
    from .credits import resolve_model_for_user

    selection = resolve_model_for_user(user)
    model = selection.model
    agent = AgentRegistry.create_for_entity(
        user=user, model_config=model, entity=entity, thread_id=thread_id)

    history_key = _build_history_key(user, entity, thread_id)
    system_msg = build_system_message(
        agent.get_static_instructions(), agent.get_dynamic_context())
    chat_model = create_chat_model(model.name)
    checkpointer = _get_checkpointer()
    config = {"configurable": {"thread_id": history_key}}

    compiled = build_agent_graph(
        chat_model, agent.get_tools(), system_msg,
        model.context_window or 200_000,
        checkpointer, agent.get_tools_requiring_approval())

    return agent, model, compiled, config


def run_agent_turn(*, invoke_fn, agent, model, compiled, config,
                   user, entity, node_name: str, input_summary: str,
                   record_usage_fn) -> AgentResponse:
    """Execute one agent turn with shared error/usage handling.

    Args:
        invoke_fn: Callable(compiled, config, usage_cb) -> graph result.
        record_usage_fn: The service._record_usage method for logging.
    """
    start = time.time()
    error_text, usage, content = "", {}, ""
    interrupted = False
    usage_cb = UsageMetadataCallbackHandler()

    try:
        result = invoke_fn(compiled, config, usage_cb)

        if hasattr(result, "interrupts") and result.interrupts:
            interrupted = True
            return _build_interrupt_response(
                result, agent, user, model, entity, start, usage_cb, record_usage_fn)

        messages = result.value["messages"] if hasattr(result, "value") else result["messages"]
        content = messages[-1].content
        usage = _usage_from_callback(usage_cb)
    except Exception as exc:
        error_text = str(exc)
        logger.error("[agent.%s] %s", node_name, exc)
        raise
    finally:
        if not interrupted:
            record_usage_fn(
                user=user, model_name=model.name, workflow="agent_chat",
                node=node_name, usage=usage, output_json={}, entity=entity,
                input_summary=input_summary,
                duration_ms=int((time.time() - start) * 1000),
                error_text=error_text)

    return AgentResponse(success=True, message=_normalize_content(content), usage=usage)


def _build_interrupt_response(result, agent, user, model, entity,
                              start_time, usage_cb, record_usage_fn) -> AgentResponse:
    """Build an AgentResponse with interrupt data for the frontend."""
    if hasattr(result, "value"):
        interrupts = result.interrupts if hasattr(result, "interrupts") else ()
    else:
        interrupts = result.get("__interrupt__", [])

    usage = _usage_from_callback(usage_cb)

    actions = []
    for intr in interrupts:
        val = intr.value if hasattr(intr, "value") else intr
        if isinstance(val, dict):
            for req in val.get("action_requests", []):
                actions.append({
                    "name": req.get("name", ""),
                    "args": req.get("args", {}),
                    "description": agent.summarise_action(
                        req.get("name", ""), req.get("args", {})),
                })

    action_count = len(actions)
    interrupt_message = (
        f"Confirm {action_count} action{'s' if action_count != 1 else ''}"
        if actions else "Awaiting approval"
    )

    record_usage_fn(
        user=user, model_name=model.name, workflow="agent_chat", node="interrupt",
        usage=usage, output_json={}, entity=entity,
        input_summary=f"interrupt:{','.join(a['name'] for a in actions)}",
        duration_ms=int((time.time() - start_time) * 1000), error_text="")

    return AgentResponse(
        success=True, message=interrupt_message, usage=usage,
        interrupt={"message": interrupt_message, "actions": actions})


# ── History retrieval ─────────────────────────────────────────────────

def get_chat_history(*, user, entity: Model, thread_id: str = "") -> list[dict]:
    """Retrieve conversation history from the checkpointer."""
    history_key = _build_history_key(user, entity, thread_id)
    checkpointer = _get_checkpointer()

    graph = StateGraph(MessagesState)
    graph.add_node("noop", lambda state: state)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    compiled = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": history_key}}
    try:
        state = compiled.get_state(config)
        if state and state.values and "messages" in state.values:
            return _extract_chat_history(state.values["messages"])
    except Exception as exc:
        logger.debug("get_history failed for thread %s: %s", history_key, exc)
    return []
