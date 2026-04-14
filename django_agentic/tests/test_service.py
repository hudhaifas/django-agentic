"""Tests for django_agentic service helpers and ai_context."""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from django_agentic.models import AIModel, SiteAIConfig
from django_agentic.context import current_ai_user, current_ai_model_name
from django_agentic.service import (
    ai_context, AgentResponse, _normalize_content,
    _build_system_message, _classify_error,
    _extract_chat_history,
)

User = get_user_model()


class AIContextTest(TestCase):

    def setUp(self):
        self.free = AIModel.objects.create(
            name="ctx-free", provider="openai", allowed_for_free=True,
            input_cost_per_1m=Decimal("0.10"),
            output_cost_per_1m=Decimal("0.40"),
        )
        self.paid = AIModel.objects.create(
            name="ctx-paid", provider="anthropic",
            allowed_for_free=False,
            input_cost_per_1m=Decimal("3.00"),
            output_cost_per_1m=Decimal("15.00"),
        )
        SiteAIConfig.objects.update_or_create(pk=1, defaults={
            "default_free_model": self.free,
            "default_paid_model": self.paid,
        })
        self.user = User.objects.create_user(
            username="ctxuser", password="p",
        )

    def test_sets_context_vars(self):
        with ai_context(self.user) as ctx:
            self.assertEqual(current_ai_user.get(), self.user)
            self.assertEqual(
                current_ai_model_name.get(), ctx.model_name,
            )
            self.assertTrue(len(ctx.model_name) > 0)

    def test_resets_on_exit(self):
        with ai_context(self.user):
            pass
        self.assertIsNone(current_ai_user.get())
        self.assertIsNone(current_ai_model_name.get())

    def test_resets_on_exception(self):
        try:
            with ai_context(self.user):
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertIsNone(current_ai_user.get())
        self.assertIsNone(current_ai_model_name.get())


class AgentResponseTest(TestCase):

    def test_to_dict_without_interrupt(self):
        r = AgentResponse(success=True, message="hello", usage={"input_tokens": 10})
        d = r.to_dict()
        self.assertEqual(d["success"], True)
        self.assertEqual(d["message"], "hello")
        self.assertNotIn("interrupt", d)

    def test_to_dict_with_interrupt(self):
        r = AgentResponse(
            success=True, message="Confirm",
            interrupt={"message": "Confirm", "actions": []},
        )
        d = r.to_dict()
        self.assertIn("interrupt", d)

    def test_has_interrupt(self):
        self.assertFalse(AgentResponse().has_interrupt)
        self.assertTrue(AgentResponse(interrupt={}).has_interrupt)


class NormalizeContentTest(TestCase):

    def test_string_passthrough(self):
        self.assertEqual(_normalize_content("hello"), "hello")

    def test_list_of_blocks(self):
        blocks = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]
        self.assertEqual(_normalize_content(blocks), "Hello world")

    def test_empty(self):
        self.assertEqual(_normalize_content(""), "")
        self.assertEqual(_normalize_content(None), "")
        self.assertEqual(_normalize_content([]), "")


class BuildSystemMessageTest(TestCase):

    def test_static_only(self):
        self.assertEqual(_build_system_message("hello"), "hello")

    def test_with_dynamic(self):
        result = _build_system_message("static", "dynamic")
        self.assertEqual(result, "static\n\ndynamic")


class ClassifyErrorTest(TestCase):

    def test_credit_error(self):
        self.assertEqual(_classify_error("Insufficient credits"), "credit_limit")

    def test_api_error(self):
        self.assertEqual(_classify_error("rate limit exceeded"), "api_error")

    def test_empty(self):
        self.assertEqual(_classify_error(""), "")

    def test_unknown(self):
        self.assertEqual(_classify_error("something weird"), "unknown")


class ExtractChatHistoryTest(TestCase):

    def test_filters_tool_messages(self):
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            ToolMessage(content="result", name="t", tool_call_id="1"),
            AIMessage(content="Here you go"),
        ]
        history = _extract_chat_history(messages)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Here you go")

    def test_skips_empty_ai_with_tool_calls(self):
        messages = [
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ]
        history = _extract_chat_history(messages)
        self.assertEqual(len(history), 0)
