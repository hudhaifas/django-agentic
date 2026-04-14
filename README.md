# django-agentic

AI framework for Django: entity-aware chatbot agents with tools, human-in-the-loop approval,
credit management, and usage logging. Built on LangGraph + LangChain.

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
- **Multi-provider** — OpenAI and Anthropic, selected per-user based on credit tier
- **Credit system** — free monthly credits + purchased credits per user, with automatic
  model tier selection (paid model when credits allow, free model as fallback)
- **Usage logging** — every LLM call logged with token counts, cost, cache metrics, and
  credit deductions
- **Structured invocation** — `ai_service.invoke()` for workflow nodes that need structured
  Pydantic output (extraction, classification, generation)
- **Configurable checkpointer** — InMemorySaver by default, swap to PostgresSaver or
  RedisSaver for production persistence
- **Django admin** — all models registered with Chart.js usage dashboards

---

## How it works

```
Browser -> POST /api/agentic/agent/chat
            |
       agent_chat view (DRF)
            |
       AIService.chat()  <-> resolve_model_for_user (credit check + model selection)
            |
       AgentRegistry  ->  YourCustomAgent (extends ModelAgent)
            |
       create_agent()  <-> HumanInTheLoopMiddleware + AnthropicPromptCachingMiddleware
            |
       LLM  <->  OpenAI / Anthropic
            |
       [write tool called]
            |
       HumanInTheLoopMiddleware -> interrupt -> returns actions to browser
            |
       User approves/rejects in UI
            |
       POST /api/agentic/agent/resume -> Command(resume=decisions) -> tool executes -> LLM responds
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

This creates the tables and seeds 9 default AI models (GPT-4.1 family + Claude family)
with current pricing. Go to Django admin > AI Configuration to pick your free-tier and
paid-tier defaults.

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
        return (
            "You are a product assistant. Use tools to read and update data. "
            "Write tools require user approval -- call them immediately, "
            "the system handles confirmation."
        )

    def get_dynamic_context(self) -> str:
        """Ephemeral context -- entity state, rebuilt per request."""
        p = self.entity
        return f"Product: {p.name} (ID: {p.pk})\nPrice: {p.price}\nStock: {p.stock}"

    def get_tools(self) -> list:
        """Tools the LLM can call. Passed to create_agent which binds them
        to the LLM automatically -- no need to describe them in the prompt."""
        return [get_product_details, update_price, delete_product]

    def get_tools_requiring_approval(self) -> list[str]:
        """Tool names that trigger HITL interrupt before execution."""
        return ["update_price", "delete_product"]

    def summarise_action(self, tool_name: str, tool_input: dict) -> str:
        """Human-readable summary for the HITL confirmation card."""
        if tool_name == "update_price":
            return f"Update price to ${tool_input.get('new_price', '?')}"
        return f"Execute: {tool_name}"
```

Register the mapping in settings:

```python
DJANGO_AGENTIC = {
    "AGENT_MAPPINGS": {
        "catalog.Product": "catalog.agents.ProductAgent",
    },
}
```

---

## Creating tools

Tools are plain Python functions decorated with `@tool` from `langchain_core.tools`.
The consuming app imports only `@tool` -- no other langchain imports needed.

**Read tool** (runs immediately, no approval):

```python
from langchain_core.tools import tool
import json

@tool
def get_product_details(product_id: str) -> str:
    """Get full details of a product."""
    from myapp.models import Product
    p = Product.objects.get(pk=product_id)
    return json.dumps({"name": p.name, "price": str(p.price), "stock": p.stock})
```

**Write tool** (requires HITL approval):

```python
@tool
def update_price(product_id: str, new_price: float) -> str:
    """Update the price of a product."""
    from myapp.models import Product
    p = Product.objects.get(pk=product_id)
    old = p.price
    p.price = new_price
    p.save(update_fields=["price"])
    return json.dumps({"success": True, "old_price": str(old), "new_price": str(new_price)})
```

Tools listed in `get_tools_requiring_approval()` are gated by `HumanInTheLoopMiddleware`.
Read tools not in the approval list execute immediately.

---

## Human-in-the-Loop (HITL)

HITL is driven by LangChain's `HumanInTheLoopMiddleware` -- no custom interrupt code needed.

When the LLM calls a write tool, the middleware pauses execution and returns the pending
actions to the frontend. The frontend shows a confirmation card. The user approves or
rejects. A POST to `/ai/agent/resume` continues the workflow using
`Command(resume={"decisions": [{"type": "approve"}]})`.

If the user refreshes the page during an interrupt, the next chat message auto-rejects
the stale interrupt before processing the new message.

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
# result is a ProductAnalysis instance
```

Every `invoke()` call is automatically logged, costed, and credit-deducted.

---

## Running workflows (ai_context)

When running a LangGraph workflow that calls `ai_service.invoke()` internally, use
`ai_context` to handle credit pre-check, model selection, and context var setup:

```python
from django_agentic.service import ai_context

with ai_context(user) as ctx:
    # ctx.model_name, ctx.model, ctx.is_free_tier available
    # context vars (current_ai_user, current_ai_model_name) are set
    my_workflow.invoke({"input": data})
# context vars auto-reset on exit, even if an exception occurs
```

This replaces the manual pattern of calling `resolve_model_for_user()`, setting context
vars, and resetting them in a `finally` block.

---

## API endpoints

All endpoints are namespaced under `agentic/`. When you include the URLs with
`path("api/", include("django_agentic.urls"))`, the full paths become `/api/agentic/...`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `agentic/credits/` | Credit status, current model, available models |
| PATCH | `agentic/credits/model-override/` | Set per-user model override |
| GET | `agentic/credits/usage/` | Usage statistics (daily, by model, by workflow) |
| POST | `agentic/agent/chat` | Send a chat message |
| POST | `agentic/agent/resume` | Approve/reject HITL interrupt |
| GET | `agentic/agent/history` | Load conversation history for an entity |

### Request/Response formats

**Chat:**

```json
POST /api/agentic/agent/chat
{
  "message": "What is the current price?",
  "context": {"entity_class": "catalog.Product", "entity_id": "123"}
}

// Normal response
{"success": true, "message": "The price is $29.99.", "usage": {"input_tokens": 150, "output_tokens": 25}}

// HITL interrupt response
{"success": true, "message": "Confirm 1 action", "usage": {...},
 "interrupt": {"message": "Confirm 1 action", "actions": [{"name": "update_price", "args": {"new_price": 39.99}, "description": "Update price to $39.99"}]}}
```

**Resume:**

```json
POST /api/agentic/agent/resume
{"approved": true, "context": {"entity_class": "catalog.Product", "entity_id": "123"}}
```

**History:**

```
GET /api/agentic/agent/history?entity_class=catalog.Product&entity_id=123

{"history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## Checkpointer (chat history storage)

Default: `InMemorySaver` (works out of the box, loses history on server restart).

### PostgreSQL (production)

```bash
pip install django-agentic[postgres]
```

```python
from langgraph.checkpoint.postgres import PostgresSaver
DJANGO_AGENTIC = {
    "CHECKPOINTER": PostgresSaver.from_conn_string("postgresql://user:pass@host/db"),
}
# Call DJANGO_AGENTIC["CHECKPOINTER"].setup() once to create checkpoint tables.
```

### Redis

```bash
pip install django-agentic[redis]
```

```python
from langgraph.checkpoint.redis import RedisSaver
DJANGO_AGENTIC = {"CHECKPOINTER": RedisSaver(redis_url="redis://localhost:6379")}
```

---

## Credit system

Each user gets a configurable free monthly credit allowance (default $2.00). When free
credits run out, the system falls back to the free-tier model. Users with purchased
credits get the paid-tier model.

- Admins see unlimited credits (staff bypass)
- Per-user model override in admin
- Atomic credit deduction with idempotency keys (no double-charging on retries)

```bash
python manage.py reset_free_credits  # run monthly via cron
```

---

## Prompt caching (Anthropic)

When using an Anthropic model, `AnthropicPromptCachingMiddleware` automatically caches
the system prompt. On repeated requests within 5 minutes, cache hits cost ~90% less than
regular input tokens. No configuration needed -- the middleware is applied automatically.

---

## Configuration reference

All settings go in the `DJANGO_AGENTIC` dict in your Django settings:

| Key | Default | Description |
|-----|---------|-------------|
| `DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Fallback LLM model name |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API key |
| `OPENAI_API_KEY` | `""` | OpenAI API key |
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

- `langchain-anthropic` -- for Anthropic models + prompt caching middleware
- `langchain-openai` -- for OpenAI models
- `langgraph-checkpoint-postgres` -- PostgreSQL checkpointer
- `langgraph-checkpoint-redis` -- Redis checkpointer

---

## License

MIT
