import uuid
from decimal import Decimal
from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models


class AIModel(models.Model):
    """An LLM model configuration with pricing and capability metadata.

    Tracks provider, cost per million tokens (input/output/cache), and
    access control flags (allowed_for_free, allowed_for_paid).
    Seeded by migration 0004 with default models from OpenAI and Anthropic.
    """
    MODEL_TYPES = [
        ("chat", "Chat / Generation"),
        ("embedding", "Embedding"),
        ("transcription", "Transcription"),
        ("image", "Image Generation"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    display_name = models.CharField(max_length=200, blank=True, default="")
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES, default="chat", db_index=True)
    provider = models.CharField(max_length=50, choices=[
        ("anthropic", "Anthropic"), ("openai", "OpenAI"),
        ("google", "Google"), ("bedrock", "AWS Bedrock"),
    ], default="anthropic")
    input_cost_per_1m = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    output_cost_per_1m = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    cache_write_cost_per_1m = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    cache_read_cost_per_1m = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    active = models.BooleanField(default=True)
    allowed_for_free = models.BooleanField(default=True)
    allowed_for_paid = models.BooleanField(default=True)
    context_window = models.IntegerField(default=200000)
    max_output_tokens = models.IntegerField(default=8192)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["input_cost_per_1m"]

    def __str__(self):
        return self.display_name or self.name

    def calculate_cost(self, usage: dict) -> float:
        cost = 0.0
        cost += (usage.get("input_tokens", 0) / 1_000_000) * float(self.input_cost_per_1m)
        cost += (usage.get("output_tokens", 0) / 1_000_000) * float(self.output_cost_per_1m)
        cost += (usage.get("cache_creation_input_tokens", 0) / 1_000_000) * float(self.cache_write_cost_per_1m)
        cost += (usage.get("cache_read_input_tokens", 0) / 1_000_000) * float(self.cache_read_cost_per_1m)
        return round(cost, 6)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        cost = (Decimal(input_tokens) / 1_000_000) * self.input_cost_per_1m
        cost += (Decimal(output_tokens) / 1_000_000) * self.output_cost_per_1m
        return round(cost, 6)


class SiteAIConfig(models.Model):
    """Singleton site-wide AI configuration.

    Stores default free/paid models, monthly credit allowance, and
    per-step model overrides. Always uses pk=1 (singleton pattern).
    Access via ``SiteAIConfig.load()``.
    """
    default_free_model = models.ForeignKey(AIModel, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    default_paid_model = models.ForeignKey(AIModel, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    monthly_free_credits = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("2.00"))
    step_model_config = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name = "AI Configuration"
        verbose_name_plural = "AI Configuration"

    def __str__(self):
        return "Site AI Configuration"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class UserAIProfile(models.Model):
    """Per-user AI credit balance and model preferences.

    Created automatically via ``get_or_create_profile(user)``.
    Tracks free monthly credits, purchased credits, and optional
    per-user model override.
    """
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="ai_profile")
    free_monthly_credits = models.DecimalField(max_digits=10, decimal_places=6, default=Decimal("2.000000"))
    purchased_credits = models.DecimalField(max_digits=10, decimal_places=6, default=Decimal("0.000000"))
    model_override = models.ForeignKey(AIModel, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    credits_reset_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User AI Profile"

    def __str__(self):
        return f"AI Profile: {self.user}"

    @property
    def total_credits(self) -> Decimal:
        return self.free_monthly_credits + self.purchased_credits

    def clean(self):
        from django.core.exceptions import ValidationError
        if self.purchased_credits < 0:
            raise ValidationError({"purchased_credits": "Cannot be negative."})
        if self.free_monthly_credits < 0:
            raise ValidationError({"free_monthly_credits": "Cannot be negative."})

    def refill_free_credits(self):
        from django.utils import timezone
        config = SiteAIConfig.load()
        self.free_monthly_credits = config.monthly_free_credits
        self.credits_reset_at = timezone.now()
        self.save(update_fields=["free_monthly_credits", "credits_reset_at", "updated_at"])


class AIUsageLog(models.Model):
    """Immutable log entry for every LLM API call.

    Records token counts, cost, cache metrics, credit deductions,
    timing, and optional generic-FK link to the entity that triggered
    the call. Auto-resolves AIModel and calculates cost on save.
    """
    ERROR_TYPES = [
        ("", "None"), ("credit_limit", "Credit Limit"), ("api_error", "API Error"),
        ("validation", "Validation"), ("context_overflow", "Context Overflow"), ("unknown", "Unknown"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    idempotency_key = models.CharField(max_length=255, unique=True, null=True, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name="ai_usage_logs")
    ai_model = models.ForeignKey(AIModel, on_delete=models.SET_NULL, null=True, blank=True, related_name="usage_logs")
    model_name = models.CharField(max_length=100, db_index=True)
    workflow = models.CharField(max_length=100, db_index=True)
    node = models.CharField(max_length=100)
    # Generic entity link (replaces app-specific FKs)
    entity_type = models.ForeignKey(ContentType, on_delete=models.SET_NULL, null=True, blank=True)
    entity_id = models.CharField(max_length=255, blank=True, default="")
    entity = GenericForeignKey("entity_type", "entity_id")
    input_summary = models.TextField(blank=True, default="")
    output_json = models.JSONField(default=dict)
    prompt_tokens = models.IntegerField(null=True, blank=True)
    completion_tokens = models.IntegerField(null=True, blank=True)
    total_tokens = models.IntegerField(null=True, blank=True)
    cache_read_tokens = models.IntegerField(null=True, blank=True)
    cache_creation_tokens = models.IntegerField(null=True, blank=True)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)
    input_cost_per_1m_at_time = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    output_cost_per_1m_at_time = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    cache_write_cost_per_1m_at_time = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)
    cache_read_cost_per_1m_at_time = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)
    used_free_credits = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    used_paid_credits = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    provider_request_id = models.CharField(max_length=255, blank=True, default="")
    error_type = models.CharField(max_length=20, choices=ERROR_TYPES, blank=True, default="")
    request_time = models.DateTimeField(null=True, blank=True)
    response_time = models.DateTimeField(null=True, blank=True)
    duration_ms = models.IntegerField(null=True, blank=True)
    success = models.BooleanField(default=True)
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["model_name", "created_at"]),
            models.Index(fields=["workflow", "created_at"]),
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self):
        return f"{self.workflow}/{self.node} — {self.created_at}"

    def save(self, *args, **kwargs):
        if not self.ai_model_id and self.model_name:
            self.ai_model = AIModel.objects.filter(name=self.model_name, active=True).first()
        if self.ai_model:
            if not self.cost_usd:
                self.cost_usd = self.ai_model.calculate_cost({
                    "input_tokens": self.prompt_tokens or 0,
                    "output_tokens": self.completion_tokens or 0,
                    "cache_creation_input_tokens": self.cache_creation_tokens or 0,
                    "cache_read_input_tokens": self.cache_read_tokens or 0,
                })
            if self.input_cost_per_1m_at_time is None:
                self.input_cost_per_1m_at_time = self.ai_model.input_cost_per_1m
                self.output_cost_per_1m_at_time = self.ai_model.output_cost_per_1m
                self.cache_write_cost_per_1m_at_time = self.ai_model.cache_write_cost_per_1m
                self.cache_read_cost_per_1m_at_time = self.ai_model.cache_read_cost_per_1m
        super().save(*args, **kwargs)
