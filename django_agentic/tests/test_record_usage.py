"""Tests for AIService._record_usage — the unified usage logging pipeline.

Covers token-based cost (LLM), explicit cost (Whisper), credit deduction,
entity resolution, and error handling.
"""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model

from django_agentic.models import AIModel, SiteAIConfig, AIUsageLog
from django_agentic.credits import get_or_create_profile
from django_agentic.service import ai_service

User = get_user_model()


class RecordUsageTestBase(TestCase):

    def setUp(self):
        self.model = AIModel.objects.create(
            name="usage-test-model", provider="anthropic",
            input_cost_per_1m=Decimal("3.00"),
            output_cost_per_1m=Decimal("15.00"),
            cache_write_cost_per_1m=Decimal("3.75"),
            cache_read_cost_per_1m=Decimal("0.30"),
            allowed_for_free=True,
        )
        self.whisper = AIModel.objects.create(
            name="whisper-test", provider="openai",
            model_type="transcription",
            input_cost_per_1m=Decimal("0.01"),
            output_cost_per_1m=Decimal("0"),
            allowed_for_free=False,
        )
        SiteAIConfig.objects.update_or_create(pk=1, defaults={
            "default_free_model": self.model,
            "default_paid_model": self.model,
            "monthly_free_credits": Decimal("5.00"),
        })
        self.user = User.objects.create_user(username="usageuser", password="p")
        self.staff = User.objects.create_user(
            username="usagestaff", password="p", is_staff=True)


class TokenBasedCostTest(RecordUsageTestBase):
    """LLM calls: cost calculated from tokens × model pricing."""

    def test_records_tokens_and_cost(self):
        ai_service._record_usage(
            user=self.user, model_name="usage-test-model",
            workflow="test", node="test_node",
            usage={"input_tokens": 1000, "output_tokens": 500},
            output_json={},
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.prompt_tokens, 1000)
        self.assertEqual(log.completion_tokens, 500)
        self.assertEqual(log.total_tokens, 1500)
        expected_cost = (1000 / 1e6) * 3.0 + (500 / 1e6) * 15.0
        self.assertAlmostEqual(float(log.cost_usd), expected_cost, places=5)

    def test_records_cache_tokens(self):
        ai_service._record_usage(
            user=self.user, model_name="usage-test-model",
            workflow="test", node="cache_test",
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 5000,
                "cache_creation_input_tokens": 2000,
            },
            output_json={},
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.cache_read_tokens, 5000)
        self.assertEqual(log.cache_creation_tokens, 2000)

    def test_snapshots_pricing(self):
        ai_service._record_usage(
            user=self.user, model_name="usage-test-model",
            workflow="test", node="snap",
            usage={"input_tokens": 100, "output_tokens": 50},
            output_json={},
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.input_cost_per_1m_at_time, Decimal("3.00"))
        self.assertEqual(log.output_cost_per_1m_at_time, Decimal("15.00"))


class ExplicitCostTest(RecordUsageTestBase):
    """Non-token models (Whisper): cost passed explicitly."""

    def test_explicit_cost_stored(self):
        ai_service._record_usage(
            user=self.user, model_name="whisper-test",
            workflow="transcribe", node="whisper",
            usage={}, output_json={"audio_seconds": 120},
            explicit_cost_usd=Decimal("0.012000"),
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.cost_usd, Decimal("0.012000"))
        self.assertIsNone(log.prompt_tokens)
        self.assertIsNone(log.completion_tokens)

    def test_explicit_cost_overrides_token_calculation(self):
        ai_service._record_usage(
            user=self.user, model_name="whisper-test",
            workflow="transcribe", node="whisper",
            usage={"input_tokens": 999},  # should be ignored
            output_json={},
            explicit_cost_usd=Decimal("0.050000"),
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.cost_usd, Decimal("0.050000"))


class CreditDeductionTest(RecordUsageTestBase):
    """Credits deducted correctly during usage recording."""

    def test_staff_no_deduction(self):
        ai_service._record_usage(
            user=self.staff, model_name="usage-test-model",
            workflow="test", node="staff",
            usage={"input_tokens": 1000, "output_tokens": 500},
            output_json={},
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.used_free_credits, Decimal("0"))
        self.assertEqual(log.used_paid_credits, Decimal("0"))

    def test_non_staff_deducts_free_credits(self):
        profile = get_or_create_profile(self.user)
        profile.free_monthly_credits = Decimal("5.00")
        profile.purchased_credits = Decimal("0")
        profile.save()

        ai_service._record_usage(
            user=self.user, model_name="usage-test-model",
            workflow="test", node="deduct",
            usage={"input_tokens": 1_000_000, "output_tokens": 100_000},
            output_json={},
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertGreater(log.used_free_credits, Decimal("0"))
        profile.refresh_from_db()
        self.assertLess(profile.free_monthly_credits, Decimal("5.00"))

    def test_explicit_cost_deducts_credits(self):
        profile = get_or_create_profile(self.user)
        profile.purchased_credits = Decimal("1.00")
        profile.free_monthly_credits = Decimal("0")
        profile.save()

        ai_service._record_usage(
            user=self.user, model_name="whisper-test",
            workflow="transcribe", node="whisper",
            usage={}, output_json={},
            explicit_cost_usd=Decimal("0.100000"),
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.used_paid_credits, Decimal("0.100000"))
        profile.refresh_from_db()
        self.assertEqual(profile.purchased_credits, Decimal("0.900000"))

    def test_no_deduction_on_error(self):
        profile = get_or_create_profile(self.user)
        profile.free_monthly_credits = Decimal("5.00")
        profile.save()

        ai_service._record_usage(
            user=self.user, model_name="usage-test-model",
            workflow="test", node="error",
            usage={"input_tokens": 1000, "output_tokens": 500},
            output_json={}, error_text="API error",
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.used_free_credits, Decimal("0"))
        self.assertFalse(log.success)

    def test_no_tokens_no_cost_no_deduction(self):
        ai_service._record_usage(
            user=self.user, model_name="usage-test-model",
            workflow="test", node="empty",
            usage={}, output_json={},
        )
        log = AIUsageLog.objects.latest("created_at")
        self.assertEqual(log.used_free_credits, Decimal("0"))
        self.assertEqual(log.used_paid_credits, Decimal("0"))
