"""Tests for django_agentic API views.

Uses DRF's APIClient with token-based auth so tests don't depend on
django.contrib.sessions being in INSTALLED_APPS.
"""
import json
from decimal import Decimal
from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from django_agentic.models import AIModel, SiteAIConfig, AIUsageLog

User = get_user_model()


@override_settings(ROOT_URLCONF="django_agentic.tests.urls")
class ViewTestBase(TestCase):

    def setUp(self):
        self.free = AIModel.objects.create(
            name="v-free", provider="openai", allowed_for_free=True,
            input_cost_per_1m=Decimal("0.10"),
            output_cost_per_1m=Decimal("0.40"),
        )
        self.paid = AIModel.objects.create(
            name="v-paid", provider="anthropic",
            allowed_for_free=False,
            input_cost_per_1m=Decimal("3.00"),
            output_cost_per_1m=Decimal("15.00"),
        )
        SiteAIConfig.objects.update_or_create(pk=1, defaults={
            "default_free_model": self.free,
            "default_paid_model": self.paid,
            "monthly_free_credits": Decimal("2.00"),
        })
        self.user = User.objects.create_user(
            username="viewuser", password="pass",
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)


class CreditStatusViewTest(ViewTestBase):

    def test_returns_200(self):
        r = self.client.get("/api/agentic/credits/")
        self.assertEqual(r.status_code, 200)

    def test_contains_expected_fields(self):
        r = self.client.get("/api/agentic/credits/")
        data = r.json()
        for key in ["free_monthly_credits", "purchased_credits", "total_credits",
                     "monthly_allowance", "is_unlimited", "current_model",
                     "is_free_tier", "available_models"]:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_unauthenticated_returns_401_or_403(self):
        c = APIClient()
        r = c.get("/api/agentic/credits/")
        self.assertIn(r.status_code, [401, 403])


class ModelOverrideViewTest(ViewTestBase):

    def test_set_override(self):
        r = self.client.patch(
            "/api/agentic/credits/model-override/",
            data=json.dumps({"model_id": str(self.paid.pk)}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["model_override_id"], str(self.paid.pk))

    def test_clear_override(self):
        r = self.client.patch(
            "/api/agentic/credits/model-override/",
            data=json.dumps({"model_id": None}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["model_override_id"])

    def test_invalid_model_returns_404(self):
        r = self.client.patch(
            "/api/agentic/credits/model-override/",
            data=json.dumps({"model_id": "00000000-0000-0000-0000-000000000000"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 404)


class UsageStatsViewTest(ViewTestBase):

    def test_returns_200(self):
        r = self.client.get("/api/agentic/credits/usage/")
        self.assertEqual(r.status_code, 200)

    def test_contains_structure(self):
        data = self.client.get("/api/agentic/credits/usage/").json()
        self.assertIn("daily", data)
        self.assertIn("by_model", data)
        self.assertIn("totals", data)
        self.assertEqual(data["scope"], "user")


class AgentChatViewTest(ViewTestBase):

    def test_missing_message_returns_400(self):
        r = self.client.post(
            "/api/agentic/agent/chat",
            data=json.dumps({"context": {"entity_class": "auth.User"}}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_missing_entity_class_returns_400(self):
        r = self.client.post(
            "/api/agentic/agent/chat",
            data=json.dumps({"message": "hi"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_invalid_entity_class_returns_400(self):
        r = self.client.post(
            "/api/agentic/agent/chat",
            data=json.dumps({
                "message": "hi",
                "context": {"entity_class": "nonexistent.Model"},
            }),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)


class AgentHistoryViewTest(ViewTestBase):

    def test_returns_empty_history(self):
        r = self.client.get("/api/agentic/agent/history", {
            "entity_class": "auth.User",
            "entity_id": str(self.user.pk),
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["history"], [])

    def test_missing_entity_class_returns_400(self):
        r = self.client.get("/api/agentic/agent/history")
        self.assertEqual(r.status_code, 400)
