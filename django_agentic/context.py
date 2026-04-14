"""Context vars for passing user/model through LangGraph workflows."""
import contextvars

current_ai_user = contextvars.ContextVar("current_ai_user", default=None)
current_ai_model_name = contextvars.ContextVar("current_ai_model_name", default=None)
