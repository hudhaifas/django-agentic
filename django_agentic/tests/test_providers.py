"""Tests for django_agentic provider registry."""
from decimal import Decimal
from unittest.mock import MagicMock

from django.test import TestCase
from django_agentic.models import AIModel
from django_agentic.providers import (
    register_provider, get_registered_providers, resolve_provider,
    create_chat_model, get_api_key, _get_setting,
    register_transcription_provider, create_transcription_client,
)


class BuiltinProvidersTest(TestCase):
    """Built-in anthropic and openai providers are registered at import time."""

    def test_anthropic_registered(self):
        self.assertIn("anthropic", get_registered_providers())

    def test_openai_registered(self):
        self.assertIn("openai", get_registered_providers())


class RegisterProviderTest(TestCase):
    """Custom providers can be registered and used."""

    def test_register_and_list(self):
        mock_factory = MagicMock(return_value="fake_model")
        register_provider("test_custom", mock_factory)
        self.assertIn("test_custom", get_registered_providers())

    def test_registered_factory_is_called(self):
        mock_factory = MagicMock(return_value="fake_model")
        register_provider("test_call", mock_factory)
        AIModel.objects.create(
            name="test-call-model", provider="test_call", active=True,
            input_cost_per_1m=Decimal("1.00"), output_cost_per_1m=Decimal("1.00"),
        )
        result = create_chat_model("test-call-model")
        mock_factory.assert_called_once()
        self.assertEqual(result, "fake_model")

    def test_case_insensitive(self):
        register_provider("MixedCase", lambda n, k, **kw: None)
        self.assertIn("mixedcase", get_registered_providers())


class ResolveProviderTest(TestCase):
    """Provider resolution from DB record and fallback heuristic."""

    def test_resolves_from_db(self):
        AIModel.objects.create(
            name="resolve-test", provider="anthropic", active=True,
        )
        self.assertEqual(resolve_provider("resolve-test"), "anthropic")

    def test_fallback_gpt_prefix(self):
        self.assertEqual(resolve_provider("gpt-4o-nonexistent"), "openai")

    def test_fallback_o1_prefix(self):
        self.assertEqual(resolve_provider("o1-preview-nonexistent"), "openai")

    def test_fallback_default_anthropic(self):
        self.assertEqual(resolve_provider("unknown-model-xyz"), "anthropic")

    def test_inactive_model_uses_fallback(self):
        AIModel.objects.create(
            name="inactive-test", provider="openai", active=False,
        )
        # Inactive model not found, falls back to heuristic
        self.assertEqual(resolve_provider("inactive-test"), "anthropic")


class CreateChatModelTest(TestCase):
    """create_chat_model instantiates the correct provider."""

    def test_unknown_provider_raises_valueerror(self):
        AIModel.objects.create(
            name="bad-provider-model", provider="nonexistent_provider", active=True,
        )
        with self.assertRaises(ValueError) as ctx:
            create_chat_model("bad-provider-model")
        self.assertIn("nonexistent_provider", str(ctx.exception))
        self.assertIn("register_provider", str(ctx.exception))


class GetApiKeyTest(TestCase):
    """API key resolution from settings."""

    def test_returns_string(self):
        key = get_api_key("anthropic")
        self.assertIsInstance(key, str)

    def test_unknown_provider_returns_empty(self):
        key = get_api_key("nonexistent_provider_xyz")
        self.assertEqual(key, "")


class TranscriptionProviderTest(TestCase):
    """Transcription provider registry."""

    def test_openai_registered(self):
        # Should not raise
        client = create_transcription_client("openai")
        self.assertIsNotNone(client)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            create_transcription_client("nonexistent_transcription")

    def test_register_custom(self):
        mock = MagicMock(return_value="fake_client")
        register_transcription_provider("deepgram", mock)
        result = create_transcription_client("deepgram")
        mock.assert_called_once()
        self.assertEqual(result, "fake_client")
