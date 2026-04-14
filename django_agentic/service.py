"""Centralized AI service — single entry point for all LLM calls.

Usage:
    from django_agentic.service import ai_service
    result = ai_service.invoke(schema=MySchema, system_prompt="...", ...)
    response = ai_service.chat(message="...", user=user, entity=opp)
    response = ai_service.resume(thread_id="...", user=user, entity=opp, approved=True)
"""

import functools
import logging
import time
import uuid as _uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Type

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.db.models import Model
from django.utils import timezone
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_anthropic.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from .context import current_ai_model_name, current_ai_user
from .credits import deduct_credits, resolve_model_for_user, CreditLimitExceeded, AIServiceUnavailable
from .models import AIModel, AIUsageLog, SiteAIConfig

logger = logging.getLogger(__name__)


class ai_context:
    """Context manager for AI workflow execution.

    Handles credit pre-check, model selection, and context var setup/teardown.
    Every consuming app that runs an AI workflow needs this boilerplate —
    this context manager eliminates it.

    Usage:
        with ai_context(user) as ctx:
            # ctx.model_name, ctx.model, ctx.is_free_tier available
            my_workflow.invoke({"input": data})
        # context vars auto-reset on exit
    """

    def __init__(self, user):
        self.user = user
        self.model = None
        self.model_name: str = ""
        self.is_free_tier: bool = True
        self._user_token = None
        self._model_token = None

    def __enter__(self):
        selection = resolve_model_for_user(self.user)
        self.model = selection.model
        self.model_name = selection.model_name
        self.is_free_tier = selection.is_free_tier
        self._user_token = current_ai_user.set(self.user)
        self._model_token = current_ai_model_name.set(self.model_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._user_token is not None:
            current_ai_user.reset(self._user_token)
        if self._model_token is not None:
            current_ai_model_name.reset(self._model_token)
        return False


def _get_setting(key: str, default: Any = None) -> Any:
    return getattr(settings, "DJANGO_AGENTIC", {}).get(key, default)


@dataclass
class AgentResponse:
    """Response from agent chat or resume.

    Returns only the new assistant message — not the full history.
    History is managed server-side by the LangGraph checkpointer and
    loaded separately via get_history().
    """
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


ERROR_TYPE_PATTERNS: dict[str, list[str]] = {
    "credit_limit": ["credit", "insufficient"],
    "context_overflow": ["context overflow", "context length", "too long"],
    "validation": ["validation", "invalid"],
    "api_error": ["api", "rate", "timeout", "500", "503"],
}


def _classify_error(error_text: str) -> str:
    if not error_text:
        return ""
    lower = error_text.lower()
    for error_type, patterns in ERROR_TYPE_PATTERNS.items():
        if any(p in lower for p in patterns):
            return error_type
    return "unknown"


_rate_limiter = InMemoryRateLimiter(
    requests_per_second=_get_setting("REQUESTS_PER_SECOND", 0.8),
    check_every_n_seconds=0.1, max_bucket_size=5,
)


def _resolve_provider(model_name: str) -> str:
    try:
        record = AIModel.objects.filter(name=model_name, active=True).only("provider").first()
        if record:
            return record.provider
    except Exception as exc:
        logger.debug("Provider lookup failed for %s: %s", model_name, exc)
    return "openai" if model_name.startswith(("gpt-", "o1-", "o3-")) else "anthropic"


def _create_chat_model(model_name: str) -> BaseChatModel:
    provider = _resolve_provider(model_name)
    max_retries = _get_setting("MAX_RETRIES", 8)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name,
                          api_key=_get_setting("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", ""),
                          max_retries=max_retries, rate_limiter=_rate_limiter)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model_name,
                         api_key=_get_setting("ANTHROPIC_API_KEY") or getattr(settings, "ANTHROPIC_API_KEY", ""),
                         max_retries=max_retries, rate_limiter=_rate_limiter)


def _build_system_message(static_text: str, dynamic_text: str = "") -> str:
    """Combine static instructions and dynamic context into a system prompt string.

    Prompt caching is handled by AnthropicPromptCachingMiddleware — no manual
    cache_control blocks needed.
    """
    if dynamic_text:
        return static_text + "\n\n" + dynamic_text
    return static_text


def _extract_usage(raw_msg: BaseMessage | None) -> dict:
    """Extract usage from a single LLM response message. Used by invoke()."""
    if not raw_msg:
        return {}
    meta = getattr(raw_msg, "response_metadata", {}) or {}
    raw_usage = meta.get("usage", {})
    token_usage = meta.get("token_usage", {})
    return {
        "input_tokens": raw_usage.get("input_tokens") or token_usage.get("prompt_tokens"),
        "output_tokens": raw_usage.get("output_tokens") or token_usage.get("completion_tokens"),
        "cache_read_input_tokens": raw_usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": raw_usage.get("cache_creation_input_tokens"),
    }


def _usage_from_callback(cb: UsageMetadataCallbackHandler) -> dict:
    """Extract usage from the built-in UsageMetadataCallbackHandler.

    The callback automatically aggregates across all LLM calls in a turn.
    Anthropic reports cache tokens under input_token_details with keys
    like ephemeral_5m_input_tokens and cache_read.
    """
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


@functools.lru_cache(maxsize=1)
def _get_checkpointer():
    custom = _get_setting("CHECKPOINTER")
    if custom is not None:
        return custom
    return InMemorySaver()


def _build_history_key(user, entity: Model | None, thread_id: str) -> str:
    """Build a deterministic thread key for the checkpointer.

    Entity-scoped by default: user + entity type + entity PK.
    Falls back to thread_id for collection mode.
    """
    if entity and getattr(entity, "pk", None):
        return f"{user.pk}_{entity._meta.label}_{entity.pk}"
    return thread_id


def _normalize_content(content) -> str:
    """Ensure message content is a plain string.

    LangChain messages can have content as str or list[dict] (Anthropic
    structured content blocks). The frontend expects strings.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content) if content else ""


def _extract_chat_history(messages: list) -> list[dict]:
    """Convert LangGraph message list to serializable chat history.

    Filters out internal agent mechanics (tool calls, tool results).
    Only returns human messages and assistant messages with actual content.
    """
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": _normalize_content(m.content)})
        elif isinstance(m, AIMessage):
            content = _normalize_content(m.content)
            # Skip AI messages that only contain tool calls with no user-facing text
            if not content and getattr(m, "tool_calls", None):
                continue
            if content:
                history.append({"role": "assistant", "content": content})
        # ToolMessages are internal — never expose to frontend
    return history


class AIService:

    def resolve_model_name(self, node: str | None = None) -> str:
        if node:
            try:
                step_config = SiteAIConfig.load().step_model_config or {}
                if node in step_config:
                    return step_config[node]
            except Exception:
                pass
        ctx_model = current_ai_model_name.get()
        if ctx_model:
            return ctx_model
        return _get_setting("DEFAULT_MODEL", "claude-sonnet-4-20250514")

    # ── Structured invocation (workflows) ─────────────────────────────────

    def invoke(self, *, schema: Type[BaseModel], system_prompt: str,
               human_content: str | list[dict], workflow: str, node: str,
               dynamic_context: str = "", entity: Model | None = None,
               related_object_id: str | None = None, input_summary: str = "",
               model_name: str | None = None, idempotency_key: str | None = None) -> BaseModel:
        resolved_name = model_name or self.resolve_model_name(node)
        user = current_ai_user.get()
        chat_model = _create_chat_model(resolved_name)
        system_msg = SystemMessage(content=_build_system_message(system_prompt, dynamic_context))
        structured = chat_model.with_structured_output(schema, include_raw=True)

        request_time = timezone.now()
        start = time.time()
        error_text, parsed, output_json, usage, provider_request_id = "", None, {}, {}, ""

        try:
            result = structured.invoke([system_msg, HumanMessage(content=human_content)])
            if isinstance(result, dict) and "raw" in result:
                parsed, raw_msg = result["parsed"], result["raw"]
                usage = _extract_usage(raw_msg)
                meta = getattr(raw_msg, "response_metadata", {}) or {}
                provider_request_id = meta.get("id", "") or meta.get("system_fingerprint", "")
            else:
                parsed = result
            if parsed and hasattr(parsed, "model_dump"):
                output_json = parsed.model_dump()
        except Exception as exc:
            error_text = str(exc)
            logger.error("LLM %s/%s [%s] failed: %s", workflow, node, resolved_name, exc)
            raise
        finally:
            self._record_usage(
                user=user, model_name=resolved_name, workflow=workflow, node=node,
                usage=usage, output_json=output_json, entity=entity,
                related_object_id=related_object_id, input_summary=input_summary,
                duration_ms=int((time.time() - start) * 1000), error_text=error_text,
                request_time=request_time, response_time=timezone.now(),
                provider_request_id=provider_request_id, idempotency_key=idempotency_key)
        return parsed

    # ── Agent chat (with tools + HITL) ────────────────────────────────────

    def chat(self, *, message: str, user, entity: Model, thread_id: str = "") -> AgentResponse:
        """Process a chat message via create_agent with HumanInTheLoopMiddleware.

        Read tools execute immediately. Write tools pause for human approval
        and return an AgentResponse with interrupt data for the frontend.
        """
        from .agent import AgentRegistry
        from .credits import resolve_model_for_user

        selection = resolve_model_for_user(user)
        model = selection.model
        agent = AgentRegistry.create_for_entity(
            user=user, model_config=model, entity=entity, thread_id=thread_id)

        history_key = _build_history_key(user, entity, thread_id)
        system_msg = _build_system_message(
            agent.get_static_instructions(), agent.get_dynamic_context())
        chat_model = _create_chat_model(model.name)
        checkpointer = _get_checkpointer()
        config = {"configurable": {"thread_id": history_key}}

        tools = agent.get_tools()
        compiled = self._build_agent(
            chat_model, tools, system_msg, model.context_window or 200_000,
            checkpointer, agent.get_tools_requiring_approval())

        start = time.time()
        error_text, usage, content = "", {}, ""
        interrupted = False
        usage_cb = UsageMetadataCallbackHandler()

        try:
            # If the thread has a stale interrupt (user refreshed during HITL),
            # auto-reject it before sending the new message.
            state = compiled.get_state(config)
            if state and state.next:
                logger.info("[AIService.chat] Stale interrupt found, auto-rejecting")
                num_actions = 1
                if state.tasks:
                    for task in state.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            val = task.interrupts[0].value
                            if isinstance(val, dict) and "action_requests" in val:
                                num_actions = len(val["action_requests"])
                            break
                decisions = [{"type": "reject", "message": "Cancelled."}] * num_actions
                compiled.invoke(
                    Command(resume={"decisions": decisions}),
                    {**config, "callbacks": [usage_cb]}, version="v2")

            result = compiled.invoke(
                {"messages": [HumanMessage(content=message)]},
                {**config, "callbacks": [usage_cb]}, version="v2")

            # HumanInTheLoopMiddleware sets .interrupts when a write tool needs approval
            if hasattr(result, "interrupts") and result.interrupts:
                interrupted = True
                return self._build_interrupt_response(
                    result, agent, user, model, entity, start, usage_cb)

            # Normal completion
            messages = result.value["messages"] if hasattr(result, "value") else result["messages"]
            ai_msg = messages[-1]
            content = ai_msg.content
            usage = _usage_from_callback(usage_cb)
        except Exception as exc:
            error_text = str(exc)
            logger.error("[AIService.chat] %s", exc)
            raise
        finally:
            if not interrupted:
                self._record_usage(
                    user=user, model_name=model.name, workflow="agent_chat", node="chat",
                    usage=usage, output_json={}, entity=entity,
                    input_summary=message[:200], duration_ms=int((time.time() - start) * 1000),
                    error_text=error_text)

        return AgentResponse(success=True, message=_normalize_content(content), usage=usage)

    def resume(self, *, user, entity: Model, thread_id: str = "",
               approved: bool = True) -> AgentResponse:
        """Resume an interrupted agent workflow after user approval/rejection.

        Uses Command(resume={"decisions": [...]}) — the HumanInTheLoopMiddleware
        prescribed pattern for resuming from an interrupt.
        """
        from .agent import AgentRegistry
        from .credits import resolve_model_for_user

        selection = resolve_model_for_user(user)
        model = selection.model
        agent = AgentRegistry.create_for_entity(
            user=user, model_config=model, entity=entity, thread_id=thread_id)

        history_key = _build_history_key(user, entity, thread_id)
        system_msg = _build_system_message(
            agent.get_static_instructions(), agent.get_dynamic_context())
        chat_model = _create_chat_model(model.name)
        checkpointer = _get_checkpointer()
        config = {"configurable": {"thread_id": history_key}}

        tools = agent.get_tools()
        compiled = self._build_agent(
            chat_model, tools, system_msg, model.context_window or 200_000,
            checkpointer, agent.get_tools_requiring_approval())

        start = time.time()
        error_text, usage, content = "", {}, ""
        interrupted = False
        usage_cb = UsageMetadataCallbackHandler()

        try:
            # Count pending actions from the interrupt state
            state = compiled.get_state(config)
            num_actions = 1
            if hasattr(state, "tasks") and state.tasks:
                for task in state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        val = task.interrupts[0].value
                        if isinstance(val, dict) and "action_requests" in val:
                            num_actions = len(val["action_requests"])
                        break

            decision_type = "approve" if approved else "reject"
            decisions = [{"type": decision_type}] * num_actions
            if not approved:
                decisions = [{"type": "reject", "message": "User declined."} for _ in range(num_actions)]

            result = compiled.invoke(
                Command(resume={"decisions": decisions}),
                {**config, "callbacks": [usage_cb]}, version="v2")

            # Check for chained interrupt
            if hasattr(result, "interrupts") and result.interrupts:
                interrupted = True
                return self._build_interrupt_response(
                    result, agent, user, model, entity, start, usage_cb)

            messages = result.value["messages"] if hasattr(result, "value") else result["messages"]
            ai_msg = messages[-1]
            content = ai_msg.content
            usage = _usage_from_callback(usage_cb)
        except Exception as exc:
            error_text = str(exc)
            logger.error("[AIService.resume] %s", exc)
            raise
        finally:
            if not interrupted:
                self._record_usage(
                    user=user, model_name=model.name, workflow="agent_chat", node="resume",
                    usage=usage, output_json={}, entity=entity,
                    input_summary="resume:approved" if approved else "resume:rejected",
                    duration_ms=int((time.time() - start) * 1000),
                    error_text=error_text)

        return AgentResponse(success=True, message=_normalize_content(content), usage=usage)

    # ── Graph builder ────────────────────────────────────────────────────

    def _build_agent(self, chat_model: BaseChatModel, tools: list,
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

    # ── HITL interrupt handling ───────────────────────────────────────────

    def _build_interrupt_response(self, result, agent,
                                  user, model, entity, start_time,
                                  usage_cb: UsageMetadataCallbackHandler) -> AgentResponse:
        """Build an AgentResponse with interrupt data for the frontend."""
        if hasattr(result, "value"):
            interrupts = result.interrupts if hasattr(result, "interrupts") else ()
        else:
            interrupts = result.get("__interrupt__", [])

        usage = _usage_from_callback(usage_cb)

        # Extract action_requests from the HITL middleware interrupt
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

        self._record_usage(
            user=user, model_name=model.name, workflow="agent_chat", node="interrupt",
            usage=usage, output_json={}, entity=entity,
            input_summary=f"interrupt:{','.join(a['name'] for a in actions)}",
            duration_ms=int((time.time() - start_time) * 1000), error_text="")

        return AgentResponse(
            success=True,
            message=interrupt_message,
            usage=usage,
            interrupt={
                "message": interrupt_message,
                "actions": actions,
            },
        )

    # ── Chat history retrieval ───────────────────────────────────────────

    def get_history(self, *, user, entity: Model, thread_id: str = "") -> list[dict]:
        """Retrieve existing conversation history from the checkpointer.

        Uses graph.get_state() — the LangGraph-prescribed way to read
        checkpoint state. Any graph compiled with the same checkpointer
        can read any thread's state; only thread_id matters.

        Returns a serializable list of chat messages (same format as
        AgentResponse.history).
        """
        history_key = _build_history_key(user, entity, thread_id)
        checkpointer = _get_checkpointer()

        # Minimal graph — get_state only needs thread_id, not matching structure
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

    # ── External usage logging (for non-LLM calls like Whisper) ──────────

    def log_usage(self, *, workflow: str, node: str, model_name: str,
                  entity: Model | None = None, related_object_id: str | None = None,
                  user=None, duration_ms: int = 0, success: bool = True,
                  error: str = "", input_summary: str = "",
                  output_json: dict | None = None) -> None:
        self._record_usage(
            user=user or current_ai_user.get(), model_name=model_name,
            workflow=workflow, node=node, usage={}, output_json=output_json or {},
            entity=entity, related_object_id=related_object_id, input_summary=input_summary,
            duration_ms=duration_ms, error_text=error if not success else "")

    # ── Usage recording (internal) ────────────────────────────────────────

    def _record_usage(self, *, user, model_name: str, workflow: str, node: str,
                      usage: dict, output_json: dict, entity: Model | None = None,
                      related_object_id: str | None = None, input_summary: str = "",
                      duration_ms: int = 0, error_text: str = "",
                      request_time=None, response_time=None,
                      provider_request_id: str = "", idempotency_key: str | None = None) -> None:
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        cache_read_tokens = usage.get("cache_read_input_tokens")
        cache_creation_tokens = usage.get("cache_creation_input_tokens")
        idempotency = idempotency_key or str(_uuid.uuid4())
        used_free, used_paid = Decimal(0), Decimal(0)

        if user and not error_text and (prompt_tokens or completion_tokens):
            try:
                ai_model = AIModel.objects.filter(name=model_name, active=True).first()
                if ai_model:
                    cost = Decimal(str(ai_model.calculate_cost({
                        "input_tokens": prompt_tokens or 0, "output_tokens": completion_tokens or 0,
                        "cache_creation_input_tokens": cache_creation_tokens or 0,
                        "cache_read_input_tokens": cache_read_tokens or 0})))
                    split = deduct_credits(user, cost, ai_model, idempotency_key=idempotency)
                    used_free, used_paid = split["free_deduct"], split["purchased_deduct"]
            except Exception as exc:
                logger.warning("Credit deduction failed: %s", exc)

        now = timezone.now()
        try:
            fields: dict = {
                "idempotency_key": idempotency, "workflow": workflow, "node": node,
                "model_name": model_name, "input_summary": (input_summary or "")[:500],
                "output_json": output_json, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0) or None,
                "cache_read_tokens": cache_read_tokens, "cache_creation_tokens": cache_creation_tokens,
                "used_free_credits": used_free, "used_paid_credits": used_paid,
                "provider_request_id": provider_request_id, "error_type": _classify_error(error_text),
                "request_time": request_time or now, "response_time": response_time or now,
                "duration_ms": duration_ms, "success": not error_text, "error": error_text}
            if user:
                fields["user"] = user
            if entity and hasattr(entity, "pk") and entity.pk:
                fields["entity_type"] = ContentType.objects.get_for_model(entity)
                fields["entity_id"] = str(entity.pk)
            elif related_object_id:
                self._resolve_entity_by_id(related_object_id, fields)
            AIUsageLog.objects.create(**fields)
        except Exception as exc:
            logger.warning("Failed to save AIUsageLog: %s", exc)

        logger.info("LLM %s/%s [%s] — %dms, in=%s, out=%s, cache_r=%s, cache_w=%s",
                     workflow, node, model_name, duration_ms,
                     prompt_tokens, completion_tokens, cache_read_tokens, cache_creation_tokens)

    @staticmethod
    def _resolve_entity_by_id(related_object_id: str, fields: dict) -> None:
        from django.apps import apps as django_apps
        for model_path in _get_setting("ENTITY_MODELS", []):
            try:
                app_label, mn = model_path.split(".")
                model_cls = django_apps.get_model(app_label, mn)
                if model_cls.objects.filter(pk=related_object_id).exists():
                    fields["entity_type"] = ContentType.objects.get_for_model(model_cls)
                    fields["entity_id"] = str(related_object_id)
                    return
            except (ValueError, LookupError) as exc:
                logger.debug("Entity resolution failed for %s: %s", model_path, exc)


ai_service = AIService()
