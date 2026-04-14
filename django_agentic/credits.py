"""Credit system — model selection and atomic deduction."""
import logging
from dataclasses import dataclass
from decimal import Decimal
from django.db import transaction
from .models import AIModel, SiteAIConfig, UserAIProfile

logger = logging.getLogger(__name__)


class CreditLimitExceeded(Exception):
    def __init__(self, available: Decimal, required: Decimal):
        self.available = available
        self.required = required
        super().__init__(f"Insufficient credits. Required: ${required}, Available: ${available}.")


class AIServiceUnavailable(Exception):
    pass


@dataclass
class ModelSelection:
    model: AIModel
    model_name: str
    estimated_cost: Decimal
    purchased_deduct: Decimal
    free_deduct: Decimal
    is_free_tier: bool


def get_or_create_profile(user) -> UserAIProfile:
    profile, _ = UserAIProfile.objects.get_or_create(
        user=user, defaults={"free_monthly_credits": SiteAIConfig.load().monthly_free_credits},
    )
    return profile


def resolve_model_for_user(user, input_tokens: int = 5000, output_tokens: int = 2000) -> ModelSelection:
    config = SiteAIConfig.load()
    profile = get_or_create_profile(user)
    paid_model = profile.model_override or config.default_paid_model
    free_model = config.default_free_model
    if not paid_model:
        raise AIServiceUnavailable("Paid AI model not configured. Contact admin.")
    if not free_model:
        raise AIServiceUnavailable("Free AI model not configured. Contact admin.")
    if not free_model.allowed_for_free:
        raise AIServiceUnavailable(f"Default free model '{free_model.display_name}' not allowed for free credits.")

    paid_cost = paid_model.estimate_cost(input_tokens, output_tokens)
    free_cost = free_model.estimate_cost(input_tokens, output_tokens)

    if user.is_staff:
        return ModelSelection(paid_model, paid_model.name, paid_cost, Decimal(0), Decimal(0), False)

    if profile.purchased_credits >= paid_cost:
        return ModelSelection(paid_model, paid_model.name, paid_cost, paid_cost, Decimal(0), False)

    total = profile.purchased_credits + profile.free_monthly_credits
    if total >= free_cost:
        used_p = min(profile.purchased_credits, free_cost)
        return ModelSelection(free_model, free_model.name, free_cost, used_p, free_cost - used_p, True)

    raise CreditLimitExceeded(available=total, required=free_cost)


def deduct_credits(user, cost: Decimal, model: AIModel, idempotency_key: str | None = None) -> dict:
    if user.is_staff:
        return {"purchased_deduct": Decimal(0), "free_deduct": Decimal(0)}
    if idempotency_key:
        from .models import AIUsageLog
        if AIUsageLog.objects.filter(idempotency_key=idempotency_key).exists():
            return {"purchased_deduct": Decimal(0), "free_deduct": Decimal(0)}

    profile = get_or_create_profile(user)
    with transaction.atomic():
        profile = UserAIProfile.objects.select_for_update().get(pk=profile.pk)
        can_use_free = getattr(model, "allowed_for_free", True)
        if can_use_free:
            used_p = min(profile.purchased_credits, cost)
            used_f = min(profile.free_monthly_credits, cost - used_p)
        else:
            used_p = min(profile.purchased_credits, cost)
            used_f = Decimal(0)
        profile.purchased_credits = max(Decimal(0), profile.purchased_credits - used_p)
        profile.free_monthly_credits = max(Decimal(0), profile.free_monthly_credits - used_f)
        profile.save(update_fields=["purchased_credits", "free_monthly_credits", "updated_at"])
    return {"purchased_deduct": used_p, "free_deduct": used_f}
