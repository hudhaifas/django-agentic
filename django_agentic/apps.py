from django.apps import AppConfig


class DjangoAgenticConfig(AppConfig):
    """Django app configuration for django-agentic.

    Add ``'django_agentic'`` to ``INSTALLED_APPS`` to enable AI model management,
    credit tracking, usage logging, and agent chat endpoints.
    """
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_agentic"
    label = "django_agentic"
    verbose_name = "AI"
