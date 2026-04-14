"""Tests for django_agentic credit system."""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model

from django_agentic.models import AIModel, SiteAIConfig, UserAIProfile
from django_agentic.credits import (
    resolve_model_for_user, deduct_credits, get_or_create_profile,
    CreditLimitExceeded, AIServiceUnavailable,
)

User = get_user_model()


class CreditTestBase(TestCase):

    def setUp(self):
        self.free_model = AIModel.objects.create(
            name="free-model", display_name="Free",
            provider="openai", allowed_for_free=True,
            input_cost_per_1m=Decimal("0.15"),
            output_cost_per_1m=Decimal("0.60"),
        )
        self.paid_model = AIModel.objects.create(
            name="paid-model", display_name="Paid",
            provider="anthropic", allowed_for_free=False,
            input_cost_per_1m=Decimal("3.00"),
            output_cost_per_1m=Decimal("15.00"),
        )
        SiteAIConfig.objects.update_or_create(pk=1, defaults={
            "default_free_model": self.free_model,
            "default_paid_model": self.paid_model,
            "monthly_free_credits": Decimal("2.00"),
        })
        self.user = User.objects.create_user(
            username="credituser", password="pass",
        )
        self.staff = User.objects.create_user(
            username="staffuser", password="pass", is_staff=True,
        )


class ResolveModelTest(CreditTestBase):

    def test_staff_gets_paid_model(self):
        sel = resolve_model_for_user(self.staff)
        self.assertEqual(sel.model, self.paid_model)
        self.assertFalse(sel.is_free_tier)

    def test_user_with_purchased_credits_gets_paid(self):
        profile = get_or_create_profile(self.user)
        profile.purchased_credits = Decimal("10.00")
        profile.save()
        sel = resolve_model_for_user(self.user)
        self.assertEqual(sel.model, self.paid_model)
        self.assertFalse(sel.is_free_tier)

    def test_user_with_only_free_credits_gets_free(self):
        sel = resolve_model_for_user(self.user)
        self.assertEqual(sel.model, self.free_model)
        self.assertTrue(sel.is_free_tier)

    def test_user_with_no_credits_raises(self):
        profile = get_or_create_profile(self.user)
        profile.free_monthly_credits = Decimal("0")
        profile.purchased_credits = Decimal("0")
        profile.save()
        with self.assertRaises(CreditLimitExceeded):
            resolve_model_for_user(self.user)

    def test_missing_config_raises(self):
        SiteAIConfig.objects.update(default_paid_model=None)
        with self.assertRaises(AIServiceUnavailable):
            resolve_model_for_user(self.user)


class DeductCreditsTest(CreditTestBase):

    def test_staff_no_deduction(self):
        result = deduct_credits(self.staff, Decimal("1.00"), self.paid_model)
        self.assertEqual(result["purchased_deduct"], Decimal("0"))
        self.assertEqual(result["free_deduct"], Decimal("0"))

    def test_deducts_from_purchased_first(self):
        profile = get_or_create_profile(self.user)
        profile.purchased_credits = Decimal("5.00")
        profile.free_monthly_credits = Decimal("2.00")
        profile.save()
        result = deduct_credits(self.user, Decimal("1.00"), self.free_model)
        self.assertEqual(result["purchased_deduct"], Decimal("1.00"))
        self.assertEqual(result["free_deduct"], Decimal("0"))
        profile.refresh_from_db()
        self.assertEqual(profile.purchased_credits, Decimal("4.00"))

    def test_falls_back_to_free(self):
        profile = get_or_create_profile(self.user)
        profile.purchased_credits = Decimal("0.30")
        profile.free_monthly_credits = Decimal("2.00")
        profile.save()
        result = deduct_credits(self.user, Decimal("1.00"), self.free_model)
        self.assertEqual(result["purchased_deduct"], Decimal("0.30"))
        self.assertEqual(result["free_deduct"], Decimal("0.70"))

    def test_idempotency(self):
        from django_agentic.models import AIUsageLog
        AIUsageLog.objects.create(
            idempotency_key="test-key", model_name="free-model",
            workflow="test", node="test",
        )
        result = deduct_credits(self.user, Decimal("1.00"), self.free_model, idempotency_key="test-key")
        self.assertEqual(result["purchased_deduct"], Decimal("0"))


class GetOrCreateProfileTest(CreditTestBase):

    def test_creates_profile(self):
        self.assertFalse(UserAIProfile.objects.filter(user=self.user).exists())
        profile = get_or_create_profile(self.user)
        self.assertEqual(profile.free_monthly_credits, Decimal("2.00"))

    def test_returns_existing(self):
        p1 = get_or_create_profile(self.user)
        p2 = get_or_create_profile(self.user)
        self.assertEqual(p1.pk, p2.pk)
