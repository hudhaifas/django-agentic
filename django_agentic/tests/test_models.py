"""Tests for django_agentic models."""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model

from django_agentic.models import AIModel, SiteAIConfig, UserAIProfile, AIUsageLog

User = get_user_model()


class AIModelTest(TestCase):

    def setUp(self):
        self.model = AIModel.objects.create(
            name="test-model", display_name="Test Model",
            provider="anthropic", model_type="chat",
            input_cost_per_1m=Decimal("3.00"),
            output_cost_per_1m=Decimal("15.00"),
            cache_write_cost_per_1m=Decimal("3.75"),
            cache_read_cost_per_1m=Decimal("0.30"),
        )

    def test_str(self):
        self.assertEqual(str(self.model), "Test Model")

    def test_str_fallback_to_name(self):
        self.model.display_name = ""
        self.model.save()
        self.assertEqual(str(self.model), "test-model")

    def test_calculate_cost(self):
        cost = self.model.calculate_cost({
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        })
        self.assertAlmostEqual(cost, 18.0, places=4)

    def test_calculate_cost_with_cache(self):
        cost = self.model.calculate_cost({
            "input_tokens": 0, "output_tokens": 100_000,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 500_000,
        })
        expected = (100_000 / 1e6) * 15 + (1e6 / 1e6) * 3.75 + (500_000 / 1e6) * 0.30
        self.assertAlmostEqual(cost, round(expected, 6), places=4)

    def test_estimate_cost(self):
        cost = self.model.estimate_cost(1_000_000, 1_000_000)
        self.assertEqual(cost, Decimal("18"))

    def test_table_name(self):
        self.assertTrue(AIModel._meta.db_table.startswith("django_agentic_"))


class SiteAIConfigTest(TestCase):

    def test_singleton(self):
        c1 = SiteAIConfig.load()
        c2 = SiteAIConfig.load()
        self.assertEqual(c1.pk, c2.pk)
        self.assertEqual(c1.pk, 1)

    def test_save_forces_pk_1(self):
        c = SiteAIConfig()
        c.save()
        self.assertEqual(c.pk, 1)

    def test_table_name(self):
        self.assertTrue(SiteAIConfig._meta.db_table.startswith("django_agentic_"))


class UserAIProfileTest(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="pass")

    def test_total_credits(self):
        profile = UserAIProfile.objects.create(
            user=self.user,
            free_monthly_credits=Decimal("2.00"),
            purchased_credits=Decimal("5.00"),
        )
        self.assertEqual(profile.total_credits, Decimal("7.00"))

    def test_clean_rejects_negative(self):
        profile = UserAIProfile(user=self.user, purchased_credits=Decimal("-1"))
        with self.assertRaises(Exception):
            profile.clean()

    def test_refill_free_credits(self):
        SiteAIConfig.objects.update_or_create(pk=1, defaults={"monthly_free_credits": Decimal("3.00")})
        profile = UserAIProfile.objects.create(
            user=self.user, free_monthly_credits=Decimal("0.50"),
        )
        profile.refill_free_credits()
        profile.refresh_from_db()
        self.assertEqual(profile.free_monthly_credits, Decimal("3.00"))
        self.assertIsNotNone(profile.credits_reset_at)


class AIUsageLogTest(TestCase):

    def setUp(self):
        self.model = AIModel.objects.create(
            name="log-test-model", provider="openai",
            input_cost_per_1m=Decimal("2.00"),
            output_cost_per_1m=Decimal("8.00"),
        )

    def test_auto_resolves_ai_model(self):
        log = AIUsageLog(
            model_name="log-test-model", workflow="test", node="test",
            prompt_tokens=1000, completion_tokens=500,
        )
        log.save()
        self.assertEqual(log.ai_model, self.model)

    def test_auto_calculates_cost(self):
        log = AIUsageLog(
            model_name="log-test-model", workflow="test", node="test",
            prompt_tokens=1_000_000, completion_tokens=1_000_000,
        )
        log.save()
        self.assertAlmostEqual(float(log.cost_usd), 10.0, places=4)

    def test_snapshots_pricing(self):
        log = AIUsageLog(
            model_name="log-test-model", workflow="test", node="test",
            prompt_tokens=100, completion_tokens=100,
        )
        log.save()
        self.assertEqual(log.input_cost_per_1m_at_time, Decimal("2.00"))
        self.assertEqual(log.output_cost_per_1m_at_time, Decimal("8.00"))


    def test_explicit_zero_cost_not_overwritten(self):
        """Regression: Decimal('0') is falsy — explicit cost_usd=0 must not be recalculated."""
        log = AIUsageLog(
            model_name="log-test-model", workflow="test", node="test",
            prompt_tokens=1_000_000, completion_tokens=1_000_000,
            cost_usd=Decimal("0"),  # Explicitly set to zero (e.g. staff user)
        )
        log.save()
        # Should NOT be overwritten by calculate_cost (which would give 10.0)
        self.assertEqual(log.cost_usd, Decimal("0"))
