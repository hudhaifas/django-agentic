# django-agentic

AI framework for Django: entity-aware chatbot agents with tools, human-in-the-loop approval,
credit management, usage logging, and audio transcription. Built on LangGraph + LangChain.

Supports multiple AI providers (Anthropic, OpenAI, and any custom provider) through an
extensible registry. All calls — LLM, transcription, or custom — go through the same
credit-check and usage-logging pipeline.

```bash
pip install django-agentic[anthropic]
```

---

## What you get out of the box

- **Agent chatbot** — a chat API that knows which Django model the user is viewing, can call
  tools to read or write data, and asks for confirmation before making changes
- **REST API** — namespaced endpoints under `agentic/` (credits, agent chat, history)
- **Human-in-the-Loop (HITL)** — write tools pause for user approval via
  `HumanInTheLoopMiddleware`; read tools execute immediately
- **Prompt caching** — automatic via `AnthropicPromptCachingMiddleware` (up to 90% cost
  reduction on repeated requests)
- **Provider registry** — OpenAI and Anthropic built-in; add Google, Bedrock, or any
  provider with `register_provider()` in your `AppConfig.ready()`
- **Audio transcription** — `ai_service.transcribe()` handles chunking, Whisper API,
  duration-based cost tracking, and credit deduction
- **Credit system** — free monthly credits + purchased credits per user, with automatic
  model tier selection (paid model when credits allow, free model as fallback)
- **Usage logging** — every AI call logged with token counts, cost, cache metrics, and
  credit deductions. Non-token models (Whisper) use duration-based cost
- **Structured invocation** — `ai_service.invoke()` for workflow nodes that need structured
  Pydantic output (extraction, classification, generation)
- **Configurable checkpointer** — InMemorySaver by default, swap to PostgresSaver or
  RedisSaver for production persistence
- **Django admin** — all models registered with Chart.js usage dashboards

---

## Architecture

```
django_agentic/
  providers.py   — Provider registry (register_provider, create_chat_model, create_transcription_client)
  service.py     — AIService facade (invoke, chat, resume, transcribe, log_usage)
  chat.py        — Agent chat/resume (LangGraph agents, HITL, conversation history)
  credits.py     — Credit system (resolve_model_for_user, deduct_credits)
  context.py     — Context vars (current_ai_user, current_ai_model_name)
  agent.py       — Agent base class + registry (ModelAgent, AgentRegistry)
  models.py      — Django models (AIModel, AIUsageLog, SiteAIConfig, UserAIProfile)
  views.py       — REST API endpoints
  admin.py       — Django admin with usage charts
```

**Data flow:**

```
Browser -> POST /api/agentic/agent/chat
            |
       agent_chat view (DRF)
            |
       AIService.chat()  <-> resolve_model_for_user (credit check)
            |
       AgentRegistry  ->  YourCustomAgent (extends ModelAgent)
            |
       providers.create_chat_model()  ->  registered factory (Anthropic / OpenAI / custom)
            |
       create_agent()  <-> HumanInTheLoopMiddleware + AnthropicPromptCachingMiddleware
            |
       LLM call  ->  _record_usage()  ->  deduct_credits()  ->  AIUsageLog
```

---

## Quick start

### 1. Install

```bash
pip install django-agentic[anthropic]  # or django-agentic[openai]
```

### 2. Configure

```python
# settings.py
INSTALLED_APPS = [
    ...
    "rest_framework",
    "django.contrib.contenttypes",
    "django_agentic",
]

DJANGO_AGENTIC = {
    "DEFAULT_MODEL": "claude-sonnet-4-20250514",
    "ANTHROPIC_API_KEY": "sk-ant-...",
    "OPENAI_API_KEY": "sk-...",  # needed for Whisper transcription
}
```

### 3. Add URLs

```python
# urls.py
urlpatterns = [
    path("api/", include("django_agentic.urls")),
]
```

### 4. Migrate

```bash
python manage.py migrate
```

This creates the tables and seeds default AI models with current pricing.
Go to Django admin > AI Configuration to pick your free-tier and paid-tier defaults.

---

## Creating an agent

Subclass `ModelAgent` and implement three methods. django_agentic handles everything else:
provider selection, prompt caching, chat history, usage logging, credits, and HITL.

```python
# myapp/agents.py
from django_agentic.agent import ModelAgent

class ProductAgent(ModelAgent):

    def get_static_instructions(self) -> str:
        """Cacheable system prompt. Cached by Anthropic for 5 min."""
        return "You are a product assistant. Use tools to read and update data."

    def get_dynamic_context(self) -> str:
        """Ephemeral context — entity state, rebuilt per request."""
        p = self.entity
        return f"Product: {p.name} (ID: {p.pk})\nPrice: {p.price}"

    def get_tools(self) -> list:
        return [get_product_details, update_price]

    def get_tools_requiring_approval(self) -> list[str]:
        return ["update_price"]
```

Register in settings:

```python
DJANGO_AGENTIC = {
    "AGENT_MAPPINGS": {
        "catalog.Product": "catalog.agents.ProductAgent",
    },
}
```

---

## Provider registry

Built-in providers (anthropic, openai) are registered at import time. Add custom
providers in your `AppConfig.ready()`:

```python
# myapp/apps.py
from django.apps import AppConfig

class MyAppConfig(AppConfig):
    def ready(self):
        from django_agentic.providers import register_provider

        def google_factory(model_name, api_key, **kwargs):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, **kwargs)

        register_provider("google", google_factory)
```

Then create an AIModel record in admin with `provider="google"` and it just works.

---

## Audio transcription

`ai_service.transcribe()` handles the full Whisper pipeline: chunking files >25MB,
calling the OpenAI API, calculating cost from audio duration ($0.006/min), deducting
credits, and logging usage — all through the same pipeline as LLM calls.

```python
from django_agentic.service import ai_service

transcript = ai_service.transcribe(
    file_path="/tmp/interview.m4a",
    workflow="my_workflow",
    node="transcribe",
    file_name="interview.m4a",
    user=request.user,
)
```

Custom transcription providers (e.g. Deepgram) can be registered:

```python
from django_agentic.providers import register_transcription_provider

register_transcription_provider("deepgram", my_deepgram_factory)
```

---

## Structured invocation (workflows)

For LangGraph workflow nodes that need structured Pydantic output:

```python
from django_agentic.service import ai_service
from pydantic import BaseModel, Field

class ProductAnalysis(BaseModel):
    category: str = Field(description="Product category")
    sentiment: float = Field(description="Sentiment score 0-1")

result = ai_service.invoke(
    schema=ProductAnalysis,
    system_prompt="Analyze this product review.",
    human_content="Great product, fast shipping!",
    workflow="reviews",
    node="analyze",
)
```

Every `invoke()` call is automatically logged, costed, and credit-deducted.

---

## Running workflows (ai_context)

When running a LangGraph workflow that calls `ai_service.invoke()` internally:

```python
from django_agentic.service import ai_context

with ai_context(user) as ctx:
    # ctx.model_name, ctx.model, ctx.is_free_tier available
    my_workflow.invoke({"input": data})
# context vars auto-reset on exit, even if an exception occurs
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `agentic/credits/` | Credit status, current model, available models |
| PATCH | `agentic/credits/model-override/` | Set per-user model override |
| GET | `agentic/credits/usage/` | Usage statistics (daily, by model, by workflow) |
| POST | `agentic/agent/chat` | Send a chat message |
| POST | `agentic/agent/resume` | Approve/reject HITL interrupt |
| GET | `agentic/agent/history` | Load conversation history for an entity |

---

## Credit system

Each user gets a configurable free monthly credit allowance (default $2.00). When free
credits run out, the system falls back to the free-tier model. Users with purchased
credits get the paid-tier model.

- Admins see unlimited credits (staff bypass)
- Per-user model override in admin
- Atomic credit deduction with idempotency keys (no double-charging on retries)
- Non-token models (Whisper) use `explicit_cost_usd` for duration-based billing

```bash
python manage.py reset_free_credits  # run monthly via cron
```

---

## Checkpointer (chat history storage)

Default: `InMemorySaver` (works out of the box, loses history on server restart).

### PostgreSQL (production)

```python
from langgraph.checkpoint.postgres import PostgresSaver
DJANGO_AGENTIC = {
    "CHECKPOINTER": PostgresSaver.from_conn_string("postgresql://user:pass@host/db"),
}
```

### Redis

```python
from langgraph.checkpoint.redis import RedisSaver
DJANGO_AGENTIC = {"CHECKPOINTER": RedisSaver(redis_url="redis://localhost:6379")}
```

---

## Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Fallback LLM model name |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API key |
| `OPENAI_API_KEY` | `""` | OpenAI API key (also used for Whisper) |
| `MAX_RETRIES` | `8` | LLM retry count |
| `REQUESTS_PER_SECOND` | `0.8` | Rate limiter for LLM calls |
| `CHECKPOINTER` | `InMemorySaver()` | LangGraph checkpointer instance |
| `AGENT_MAPPINGS` | `{}` | Maps `"app.Model"` to `"app.agents.AgentClass"` |
| `ENTITY_MODELS` | `[]` | Model paths for entity resolution by ID |

---

## Requirements

- Python 3.11+
- Django 4.2+
- Django REST Framework 3.14+
- LangChain 0.3+
- LangGraph 1.0+

**Optional:**

- `langchain-anthropic` — Anthropic models + prompt caching
- `langchain-openai` — OpenAI models
- `openai` — Whisper transcription
- `langgraph-checkpoint-postgres` — PostgreSQL checkpointer
- `langgraph-checkpoint-redis` — Redis checkpointer

---

## License

MIT
