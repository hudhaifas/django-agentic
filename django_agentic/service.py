"""Centralized AI service — single entry point for all AI provider calls.

Supports chat/structured LLM calls, agent conversations, and audio transcription.
All calls go through the same credit-check and usage-logging pipeline.

Provider instantiation is delegated to providers.py (extensible registry).
Agent chat logic is delegated to chat.py (LangGraph agents with HITL).

Usage:
    from django_agentic.service import ai_service, ai_context

    # Structured LLM call (workflows)
    result = ai_service.invoke(schema=MySchema, system_prompt="...", ...)

    # Agent chat
    response = ai_service.chat(message="...", user=user, entity=opp)
    response = ai_service.resume(thread_id="...", user=user, entity=opp, approved=True)

    # Audio transcription
    text = ai_service.transcribe(file_path="/tmp/audio.m4a", ...)

    # Workflow context manager (credit pre-check + model selection)
    with ai_context(user) as ctx:
        my_workflow.invoke({"input": data})
"""

import logging
import time
import uuid as _uuid
from decimal import Decimal
from typing import Any, Type

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.db.models import Model
from django.utils import timezone
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from .context import current_ai_model_name, current_ai_user
from .credits import deduct_credits, resolve_model_for_user, CreditLimitExceeded, AIServiceUnavailable
from .models import AIModel, AIUsageLog, SiteAIConfig
from .providers import _get_setting, create_chat_model, get_api_key, create_transcription_client

logger = logging.getLogger(__name__)


# ── Workflow context manager ──────────────────────────────────────────

class ai_context:
    """Context manager for AI workflow execution.

    Handles credit pre-check, model selection, and context var setup/teardown.

    Usage:
        with ai_context(user) as ctx:
            # ctx.model_name, ctx.model, ctx.is_free_tier available
            my_workflow.invoke({"input": data})
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


# ── Error classification ──────────────────────────────────────────────

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


# ── Usage extraction helpers ──────────────────────────────────────────

def _extract_usage(raw_msg: BaseMessage | None) -> dict:
    """Extract usage from a single LLM response message."""
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


# ── AIService ─────────────────────────────────────────────────────────

class AIService:

    def resolve_model_name(self, node: str | None = None) -> str:
        """Resolve model name from step config, context var, or default."""
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

    # ── Structured invocation (workflows) ─────────────────────────────

    def invoke(self, *, schema: Type[BaseModel], system_prompt: str,
               human_content: str | list[dict], workflow: str, node: str,
               dynamic_context: str = "", entity: Model | None = None,
               related_object_id: str | None = None, input_summary: str = "",
               model_name: str | None = None, idempotency_key: str | None = None) -> BaseModel:
        """Single structured LLM call with automatic usage logging and credit deduction."""
        from .chat import build_system_message

        resolved_name = model_name or self.resolve_model_name(node)
        user = current_ai_user.get()
        chat_model = create_chat_model(resolved_name)
        system_msg = SystemMessage(content=build_system_message(system_prompt, dynamic_context))
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

    # ── Agent chat (delegates to chat.py) ─────────────────────────────

    def chat(self, *, message: str, user, entity: Model, thread_id: str = ""):
        """Process a chat message via LangGraph agent with HITL."""
        from . import chat as chat_mod
        from langgraph.types import Command

        agent, model, compiled, config = chat_mod.prepare_agent_session(
            user, entity, thread_id)

        def _invoke(compiled, config, usage_cb):
            state = compiled.get_state(config)
            if state and state.next:
                logger.info("[AIService.chat] Stale interrupt found, auto-rejecting")
                num_actions = chat_mod._count_interrupt_actions(state)
                decisions = [{"type": "reject", "message": "Cancelled."}] * num_actions
                compiled.invoke(
                    Command(resume={"decisions": decisions}),
                    {**config, "callbacks": [usage_cb]}, version="v2")
            return compiled.invoke(
                {"messages": [HumanMessage(content=message)]},
                {**config, "callbacks": [usage_cb]}, version="v2")

        return chat_mod.run_agent_turn(
            invoke_fn=_invoke, agent=agent, model=model,
            compiled=compiled, config=config, user=user, entity=entity,
            node_name="chat", input_summary=message[:200],
            record_usage_fn=self._record_usage)

    def resume(self, *, user, entity: Model, thread_id: str = "",
               approved: bool = True):
        """Resume an interrupted agent workflow after user approval/rejection."""
        from . import chat as chat_mod
        from langgraph.types import Command

        agent, model, compiled, config = chat_mod.prepare_agent_session(
            user, entity, thread_id)

        def _invoke(compiled, config, usage_cb):
            state = compiled.get_state(config)
            num_actions = chat_mod._count_interrupt_actions(state)
            if approved:
                decisions = [{"type": "approve"}] * num_actions
            else:
                decisions = [{"type": "reject", "message": "User declined."}] * num_actions
            return compiled.invoke(
                Command(resume={"decisions": decisions}),
                {**config, "callbacks": [usage_cb]}, version="v2")

        return chat_mod.run_agent_turn(
            invoke_fn=_invoke, agent=agent, model=model,
            compiled=compiled, config=config, user=user, entity=entity,
            node_name="resume",
            input_summary="resume:approved" if approved else "resume:rejected",
            record_usage_fn=self._record_usage)

    def get_history(self, *, user, entity: Model, thread_id: str = "") -> list[dict]:
        """Retrieve conversation history from the checkpointer."""
        from .chat import get_chat_history
        return get_chat_history(user=user, entity=entity, thread_id=thread_id)

    # ── Audio transcription ───────────────────────────────────────────

    def transcribe(self, *, file_path: str, workflow: str = "transcribe",
                   node: str = "whisper", model_name: str = "whisper-1",
                   language: str = "en", entity: Model | None = None,
                   related_object_id: str | None = None,
                   user=None, file_name: str = "") -> str:
        """Transcribe an audio file via the registered transcription provider.

        Handles chunking for files >25MB, timestamps, credit deduction,
        and usage logging — all through the same pipeline as LLM calls.

        Returns:
            Timestamped transcript text.
        """
        import math
        import os
        import subprocess
        import tempfile

        from .providers import resolve_provider

        WHISPER_MAX_BYTES = 25 * 1024 * 1024
        CHUNK_DURATION_SECONDS = 600
        COST_PER_MINUTE = Decimal("0.006")

        resolved_user = user or current_ai_user.get()
        display_name = file_name or os.path.basename(file_path)
        provider = resolve_provider(model_name)

        def _get_duration(path: str) -> float:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())

        def _split_chunks(input_path: str) -> list[str]:
            if os.path.getsize(input_path) <= WHISPER_MAX_BYTES:
                return [input_path]
            duration = _get_duration(input_path)
            num = math.ceil(duration / CHUNK_DURATION_SECONDS)
            logger.info("Audio %.0fs (%.1fMB) -> %d chunks",
                        duration, os.path.getsize(input_path) / 1024 / 1024, num)
            paths = []
            for i in range(num):
                start = i * CHUNK_DURATION_SECONDS
                chunk = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                chunk.close()
                subprocess.run(
                    ["ffmpeg", "-i", input_path,
                     "-ss", str(start), "-t", str(CHUNK_DURATION_SECONDS),
                     "-ac", "1", "-b:a", "96k", "-y", chunk.name],
                    capture_output=True, check=True, timeout=300)
                paths.append(chunk.name)
            return paths

        def _fmt_ts(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

        chunk_paths = _split_chunks(file_path)
        created_paths = [p for p in chunk_paths if p != file_path]
        request_time = timezone.now()
        start = time.time()
        error_text = ""
        lines: list[str] = []
        total_audio_seconds = 0.0

        try:
            client = create_transcription_client(provider)

            for i, cp in enumerate(chunk_paths):
                offset = i * CHUNK_DURATION_SECONDS if len(chunk_paths) > 1 else 0
                with open(cp, "rb") as f:
                    resp = client.audio.transcriptions.create(
                        model=model_name, file=f,
                        response_format="verbose_json", language=language,
                        timestamp_granularities=["segment"])
                if hasattr(resp, "duration") and resp.duration:
                    total_audio_seconds += resp.duration
                for seg in (resp.segments or []):
                    abs_time = seg.start + offset
                    if seg.text.strip():
                        lines.append(f"{_fmt_ts(abs_time)}: {seg.text.strip()}")

            if total_audio_seconds == 0:
                try:
                    total_audio_seconds = _get_duration(file_path)
                except Exception:
                    pass

        except Exception as exc:
            error_text = str(exc)
            logger.error("Transcription %s/%s [%s] failed: %s",
                         workflow, node, model_name, exc)
            raise
        finally:
            for p in created_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

            full_text = "\n".join(lines)
            duration_ms = int((time.time() - start) * 1000)
            audio_minutes = Decimal(str(total_audio_seconds)) / 60
            cost = (audio_minutes * COST_PER_MINUTE).quantize(Decimal("0.000001"))

            self._record_usage(
                user=resolved_user, model_name=model_name,
                workflow=workflow, node=node, usage={},
                output_json={
                    "text_length": len(full_text), "lines": len(lines),
                    "chunks": len(chunk_paths),
                    "audio_seconds": round(total_audio_seconds, 1),
                },
                entity=entity, related_object_id=related_object_id,
                input_summary=f"Audio: {display_name} ({len(chunk_paths)} chunk(s), "
                              f"{round(total_audio_seconds)}s)",
                duration_ms=duration_ms, error_text=error_text,
                request_time=request_time, response_time=timezone.now(),
                explicit_cost_usd=cost)

        logger.info("Transcribed %s: %d lines, %.0fs audio, $%.4f in %dms",
                     display_name, len(lines), total_audio_seconds, cost, duration_ms)
        return full_text

    # ── External usage logging ────────────────────────────────────────

    def log_usage(self, *, workflow: str, node: str, model_name: str,
                  entity: Model | None = None, related_object_id: str | None = None,
                  user=None, duration_ms: int = 0, success: bool = True,
                  error: str = "", input_summary: str = "",
                  output_json: dict | None = None) -> None:
        """Log usage for external/non-LLM calls (backward compat)."""
        self._record_usage(
            user=user or current_ai_user.get(), model_name=model_name,
            workflow=workflow, node=node, usage={}, output_json=output_json or {},
            entity=entity, related_object_id=related_object_id, input_summary=input_summary,
            duration_ms=duration_ms, error_text=error if not success else "")

    # ── Usage recording (internal) ────────────────────────────────────

    def _record_usage(self, *, user, model_name: str, workflow: str, node: str,
                      usage: dict, output_json: dict, entity: Model | None = None,
                      related_object_id: str | None = None, input_summary: str = "",
                      duration_ms: int = 0, error_text: str = "",
                      request_time=None, response_time=None,
                      provider_request_id: str = "", idempotency_key: str | None = None,
                      explicit_cost_usd: Decimal | None = None) -> None:
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        cache_read_tokens = usage.get("cache_read_input_tokens")
        cache_creation_tokens = usage.get("cache_creation_input_tokens")
        idempotency = idempotency_key or str(_uuid.uuid4())
        used_free, used_paid = Decimal(0), Decimal(0)

        if user and not error_text:
            try:
                ai_model = AIModel.objects.filter(name=model_name, active=True).first()
                if ai_model:
                    if explicit_cost_usd is not None:
                        cost = explicit_cost_usd
                    elif prompt_tokens or completion_tokens:
                        cost = Decimal(str(ai_model.calculate_cost({
                            "input_tokens": prompt_tokens or 0, "output_tokens": completion_tokens or 0,
                            "cache_creation_input_tokens": cache_creation_tokens or 0,
                            "cache_read_input_tokens": cache_read_tokens or 0})))
                    else:
                        cost = None
                    if cost is not None:
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
            if explicit_cost_usd is not None:
                fields["cost_usd"] = explicit_cost_usd
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

        logger.info("AI %s/%s [%s] — %dms, in=%s, out=%s, cache_r=%s, cache_w=%s",
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
