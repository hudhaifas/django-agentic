from django.urls import path
from . import views

app_name = "django_agentic"

urlpatterns = [
    path("agentic/credits/", views.credit_status, name="credit-status"),
    path("agentic/credits/model-override/", views.model_override, name="model-override"),
    path("agentic/credits/usage/", views.usage_stats, name="usage-stats"),
    path("agentic/agent/chat", views.agent_chat, name="agent-chat"),
    path("agentic/agent/resume", views.agent_resume, name="agent-resume"),
    path("agentic/agent/history", views.agent_history, name="agent-history"),
]
