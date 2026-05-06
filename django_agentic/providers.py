"""Provider registry — single source of truth for AI provider instantiation.

Built-in providers: anthropic, openai.
Consuming apps register additional providers at startup (e.g. in AppConfig.ready()).

Usage:
    from django_agentic.providers import create_chat_model, get_api_key, register_provider

    # Built-in — just works
    model = create_chat_model("claude-sonnet-4-20250514")

    # Custom provider
    register_provider("google", google_factory)
    model = create_chat_model("gemini-2.0-flash")

    # Transcription client
    from django_agentic.providers import create_transcription_client
    client = create_transcription_client("openai")
"""

import logging
from typing import Any, Callable, Protocol

from django.conf import settings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from .models import AIModel

logger = logging.getLogger(__name__)


def _get_setting(key: str, default: Any = None) -> Any:
    return getattr(settings, "DJANGO_AGENTIC", {}).get(key, default)


# ── Rate limiter (shared across all providers) ───────────────────────

_rate_limiter = InMemoryRateLimiter(
    requests_per_second=_get_setting("REQUESTS_PER_SECOND", 0.8),
    check_every_n_seconds=0.1, max_bucket_size=5,
)


# ── API key resolution ───────────────────────────────────────────────

def get_api_key(provider: str) -> str:
    """Resolve API key for a provider. Single source of truth.

    Checks DJANGO_AGENTIC settings first, then Django settings.
    Convention: {PROVIDER}_API_KEY (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY).
    """
    setting_name = f"{provider.upper()}_API_KEY"
    return _get_setting(setting_name) or getattr(settings, setting_name, "")


# ── Provider registry ────────────────────────────────────────────────

class ChatModelFactory(Protocol):
    """Protocol for chat model factory functions."""
    def __call__(self, model_name: str, api_key: str, **kwargs) -> BaseChatModel: ...


_chat_providers: dict[str, ChatModelFactory] = {}


def register_provider(name: str, factory: ChatModelFactory) -> None:
    """Register a chat model factory for a provider.

    Args:
        name: Provider name (lowercase). Must match AIModel.provider values.
        factory: Callable(model_name, api_key, **kwargs) -> BaseChatModel.

    Example:
        from langchain_google_genai import ChatGoogleGenerativeAI

        def google_factory(model_name, api_key, **kwargs):
            return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, **kwargs)

        register_provider("google", google_factory)
    """
    _chat_providers[name.lower()] = factory
    logger.debug("Registered AI provider: %s", name)


def get_registered_providers() -> list[str]:
    """Return list of registered provider names."""
    return list(_chat_providers.keys())


# ── Built-in provider factories ──────────────────────────────────────

def _anthropic_factory(model_name: str, api_key: str, **kwargs) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model_name, api_key=api_key, **kwargs)


def _openai_factory(model_name: str, api_key: str, **kwargs) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)


# Register built-ins
register_provider("anthropic", _anthropic_factory)
register_provider("openai", _openai_factory)


# ── Provider resolution ──────────────────────────────────────────────

def resolve_provider(model_name: str) -> str:
    """Resolve provider name from AIModel DB record, with fallback heuristic."""
    try:
        record = AIModel.objects.filter(name=model_name, active=True).only("provider").first()
        if record:
            return record.provider.lower()
    except Exception as exc:
        logger.debug("Provider lookup failed for %s: %s", model_name, exc)
    # Fallback heuristic
    if model_name.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    return "anthropic"


def create_chat_model(model_name: str) -> BaseChatModel:
    """Create a LangChain chat model for the given model name.

    Resolves provider from DB, looks up the registered factory, and
    instantiates with the correct API key and shared rate limiter.

    Raises:
        ValueError: If no factory is registered for the provider.
    """
    provider = resolve_provider(model_name)
    factory = _chat_providers.get(provider)
    if factory is None:
        available = ", ".join(_chat_providers.keys())
        raise ValueError(
            f"No provider registered for '{provider}'. "
            f"Available: {available}. "
            f"Use register_provider('{provider}', factory) in AppConfig.ready()."
        )
    api_key = get_api_key(provider)
    max_retries = _get_setting("MAX_RETRIES", 8)
    return factory(model_name, api_key, max_retries=max_retries, rate_limiter=_rate_limiter)


# ── Transcription client ─────────────────────────────────────────────

class TranscriptionClientFactory(Protocol):
    """Protocol for transcription client factory functions."""
    def __call__(self, api_key: str) -> Any: ...


_transcription_providers: dict[str, TranscriptionClientFactory] = {}


def register_transcription_provider(name: str, factory: TranscriptionClientFactory) -> None:
    """Register a transcription client factory.

    Example:
        register_transcription_provider("openai", lambda key: OpenAI(api_key=key))
    """
    _transcription_providers[name.lower()] = factory


def create_transcription_client(provider: str = "openai"):
    """Create a transcription API client for the given provider."""
    factory = _transcription_providers.get(provider.lower())
    if factory is None:
        raise ValueError(f"No transcription provider registered for '{provider}'.")
    return factory(get_api_key(provider))


def _openai_transcription_factory(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)


register_transcription_provider("openai", _openai_transcription_factory)


# ── Prompt caching capability registry ──────────────────────────────
# Providers that support content-block-level prompt caching.
# Add a provider name here when it gains cache_control support.
# The consuming app never needs to know — service.py reads this set.

_CONTENT_CACHING_PROVIDERS: set[str] = {"anthropic"}


def register_content_caching_provider(name: str) -> None:
    """Declare that a provider supports cache_control content blocks.

    Call this in AppConfig.ready() when registering a new provider
    that supports prompt caching (e.g. Google Gemini when it ships it).

    Example:
        register_content_caching_provider("google")
    """
    _CONTENT_CACHING_PROVIDERS.add(name.lower())


def supports_content_caching(provider: str) -> bool:
    """Return True if this provider supports cache_control content blocks."""
    return provider.lower() in _CONTENT_CACHING_PROVIDERS
