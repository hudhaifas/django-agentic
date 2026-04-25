"""Tests for django_agentic chat module (agent graph, history, helpers)."""
from decimal import Decimal
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.contrib.auth import get_user_model
from langchain_core.messages import AIMessage, HumanMessage

from django_agentic.models import AIModel, SiteAIConfig
from django_agentic.chat import (
    AgentResponse, _normalize_content, build_system_message,
    _extract_chat_history, _usage_from_callback, _count_interrupt_actions,
    get_chat_history, _build_history_key,
)

User = get_user_model()


class UsageFromCallbackTest(TestCase):
    """Extract usage from UsageMetadataCallbackHandler."""

    def test_empty_callback(self):
        cb = MagicMock()
        cb.usage_metadata = {}
        self.assertEqual(_usage_from_callback(cb), {})

    def test_extracts_tokens(self):
        cb = MagicMock()
        cb.usage_metadata = {
            "model1": {
                "input_tokens": 100,
                "output_tokens": 50,
                "input_token_details": {
                    "cache_read": 20,
                    "cache_creation": 10,
                },
            }
        }
        usage = _usage_from_callback(cb)
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["cache_read_input_tokens"], 20)
        self.assertEqual(usage["cache_creation_input_tokens"], 10)

    def test_aggregates_multiple_models(self):
        cb = MagicMock()
        cb.usage_metadata = {
            "m1": {"input_tokens": 100, "output_tokens": 50, "input_token_details": {}},
            "m2": {"input_tokens": 200, "output_tokens": 100, "input_token_details": {}},
        }
        usage = _usage_from_callback(cb)
        self.assertEqual(usage["input_tokens"], 300)
        self.assertEqual(usage["output_tokens"], 150)

    def test_handles_ephemeral_tokens(self):
        cb = MagicMock()
        cb.usage_metadata = {
            "m1": {
                "input_tokens": 100, "output_tokens": 50,
                "input_token_details": {
                    "ephemeral_5m_input_tokens": 30,
                    "ephemeral_1h_input_tokens": 20,
                },
            }
        }
        usage = _usage_from_callback(cb)
        self.assertEqual(usage["cache_creation_input_tokens"], 50)


class CountInterruptActionsTest(TestCase):
    """Count pending HITL actions from graph state."""

    def test_none_state(self):
        self.assertEqual(_count_interrupt_actions(None), 1)

    def test_no_tasks(self):
        state = MagicMock()
        state.tasks = []
        self.assertEqual(_count_interrupt_actions(state), 1)

    def test_with_actions(self):
        action_req = {"action_requests": [{"name": "a"}, {"name": "b"}]}
        interrupt = MagicMock()
        interrupt.value = action_req
        task = MagicMock()
        task.interrupts = [interrupt]
        state = MagicMock()
        state.tasks = [task]
        self.assertEqual(_count_interrupt_actions(state), 2)


class BuildHistoryKeyTest(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="hkuser", password="p")

    def test_entity_scoped(self):
        entity = MagicMock()
        entity.pk = 42
        entity._meta = MagicMock()
        entity._meta.label = "myapp.MyModel"
        key = _build_history_key(self.user, entity, "thread-1")
        self.assertIn(str(self.user.pk), key)
        self.assertIn("myapp.MyModel", key)
        self.assertIn("42", key)

    def test_fallback_to_thread_id(self):
        key = _build_history_key(self.user, None, "my-thread")
        self.assertEqual(key, "my-thread")


class GetChatHistoryTest(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="histuser", password="p")

    def test_empty_history(self):
        entity = MagicMock()
        entity.pk = 999
        entity._meta = MagicMock()
        entity._meta.label = "test.Model"
        history = get_chat_history(user=self.user, entity=entity)
        self.assertEqual(history, [])
