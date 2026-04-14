"""Test URL configuration for django_agentic standalone tests."""
from django.urls import include, path

urlpatterns = [
    path("api/", include("django_agentic.urls")),
]
