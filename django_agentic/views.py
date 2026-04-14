"""Credit status, model override, and usage stats API views."""
import datetime
import logging
from django.db.models import Sum, Count
from django.db.models.functions import TruncDate
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .credits import get_or_create_profile, resolve_model_for_user, CreditLimitExceeded, AIServiceUnavailable
from .models import AIModel, SiteAIConfig, AIUsageLog
from .serializers import AIModelChoiceSerializer, CreditStatusSerializer, ModelOverrideSerializer

logger = logging.getLogger(__name__)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def credit_status(request):
    profile = get_or_create_profile(request.user)
    config = SiteAIConfig.load()
    current_model, current_provider, is_free_tier = "unavailable", "", True
    try:
        sel = resolve_model_for_user(request.user)
        current_model = sel.model.display_name or sel.model_name
        current_provider = sel.model.provider
        is_free_tier = sel.is_free_tier
    except (CreditLimitExceeded, AIServiceUnavailable):
        pass
    data = {
        "free_monthly_credits": profile.free_monthly_credits,
        "purchased_credits": profile.purchased_credits,
        "total_credits": profile.total_credits,
        "monthly_allowance": config.monthly_free_credits,
        "credits_reset_at": profile.credits_reset_at,
        "is_unlimited": request.user.is_staff,
        "current_model": current_model,
        "current_model_provider": current_provider,
        "is_free_tier": is_free_tier,
        "model_override_id": profile.model_override_id,
        "available_models": AIModelChoiceSerializer(AIModel.objects.filter(active=True, model_type="chat"), many=True).data,
    }
    return Response(CreditStatusSerializer(data).data)


@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def model_override(request):
    ser = ModelOverrideSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    model_id = ser.validated_data.get("model_id")
    profile = get_or_create_profile(request.user)
    if model_id:
        try:
            model = AIModel.objects.get(pk=model_id, active=True)
        except AIModel.DoesNotExist:
            return Response({"detail": "Model not found."}, status=status.HTTP_404_NOT_FOUND)
        profile.model_override = model
    else:
        profile.model_override = None
    profile.save(update_fields=["model_override", "updated_at"])
    return Response({"model_override_id": profile.model_override_id})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def usage_stats(request):
    try:
        days = min(int(request.query_params.get("days", 30)), 90)
    except (ValueError, TypeError):
        days = 30
    cutoff = timezone.now() - datetime.timedelta(days=days)
    qs = AIUsageLog.objects.filter(created_at__gte=cutoff)
    scope = request.query_params.get("scope", "user")
    if not (scope == "all" and request.user.is_staff):
        qs = qs.filter(user=request.user)
        scope = "user"

    daily = (qs.annotate(date=TruncDate("created_at")).values("date").annotate(
        requests=Count("id"), cost=Sum("cost_usd"),
        input_tokens=Sum("prompt_tokens"), output_tokens=Sum("completion_tokens"),
        cache_read_tokens=Sum("cache_read_tokens"), cache_write_tokens=Sum("cache_creation_tokens"),
    ).order_by("date"))

    daily_map = {row["date"]: row for row in daily}
    today = timezone.now().date()
    filled = []
    for i in range(days - 1, -1, -1):
        d = today - datetime.timedelta(days=i)
        row = daily_map.get(d)
        filled.append({
            "date": d.isoformat(),
            "requests": row["requests"] if row else 0,
            "cost": float(row["cost"] or 0) if row else 0,
            "input_tokens": (row["input_tokens"] or 0) if row else 0,
            "output_tokens": (row["output_tokens"] or 0) if row else 0,
            "cache_read_tokens": (row["cache_read_tokens"] or 0) if row else 0,
            "cache_write_tokens": (row["cache_write_tokens"] or 0) if row else 0,
        })

    by_model = qs.values("model_name").annotate(
        requests=Count("id"), cost=Sum("cost_usd"), total_tokens=Sum("total_tokens")).order_by("-cost")
    by_workflow = qs.values("workflow").annotate(requests=Count("id"), cost=Sum("cost_usd")).order_by("-cost")
    totals = qs.aggregate(
        total_requests=Count("id"), total_cost=Sum("cost_usd"),
        total_input=Sum("prompt_tokens"), total_output=Sum("completion_tokens"),
        total_cache_read=Sum("cache_read_tokens"), total_cache_write=Sum("cache_creation_tokens"),
    )

    return Response({
        "scope": scope, "days": days, "daily": filled,
        "by_model": [{"model": r["model_name"], "requests": r["requests"],
                       "cost": float(r["cost"] or 0), "tokens": r["total_tokens"] or 0} for r in by_model],
        "by_workflow": [{"workflow": r["workflow"], "requests": r["requests"],
                          "cost": float(r["cost"] or 0)} for r in by_workflow],
        "totals": {
            "requests": totals["total_requests"] or 0, "cost": float(totals["total_cost"] or 0),
            "input_tokens": totals["total_input"] or 0, "output_tokens": totals["total_output"] or 0,
            "cache_read_tokens": totals["total_cache_read"] or 0, "cache_write_tokens": totals["total_cache_write"] or 0,
        },
    })


def _resolve_entity(request_data):
    """Resolve entity from POST request body context."""
    ctx = request_data.get("context", {})
    return _do_resolve_entity(ctx.get("entity_class"), ctx.get("entity_id"),
                              request_data.get("thread_id", ""))


def _resolve_entity_from_params(query_params):
    """Resolve entity from GET query params."""
    return _do_resolve_entity(query_params.get("entity_class"),
                              query_params.get("entity_id"),
                              query_params.get("thread_id", ""))


def _do_resolve_entity(entity_class, entity_id, thread_id):
    """Shared entity resolution logic."""
    from django.apps import apps

    if not entity_class:
        return None, None, None, "entity_class required"

    try:
        app_label, model_name = entity_class.split(".")
        model_cls = apps.get_model(app_label, model_name)
    except (ValueError, LookupError):
        return None, None, None, "Invalid entity class"

    if entity_id:
        try:
            entity = model_cls.objects.get(pk=entity_id)
        except model_cls.DoesNotExist:
            return None, None, None, "Entity not found"
    else:
        entity = model_cls()

    return entity, thread_id, entity_id, None


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def agent_history(request):
    """Retrieve existing conversation history for an entity.

    GET /api/agentic/agent/history?entity_class=myapp.MyModel&entity_id=UUID
    Returns the stored chat messages so the frontend can restore previous conversations.
    """
    from .service import ai_service

    entity, thread_id, entity_id, error = _resolve_entity_from_params(request.query_params)
    if error:
        code = 404 if error == "Entity not found" else 400
        return Response({"error": error}, status=code)

    history = ai_service.get_history(user=request.user, entity=entity, thread_id=thread_id)
    return Response({"history": history})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_chat(request):
    """Agent chat — server manages history via LangGraph checkpointer."""
    from .service import ai_service

    message = (request.data.get("message") or "").strip()
    if not message:
        return Response({"error": "message is required"}, status=400)

    entity, thread_id, entity_id, error = _resolve_entity(request.data)
    if error:
        code = 404 if error == "Entity not found" else 400
        return Response({"error": error}, status=code)

    try:
        result = ai_service.chat(message=message, user=request.user, entity=entity, thread_id=thread_id)
        return Response(result.to_dict())
    except CreditLimitExceeded as e:
        return Response({"error": str(e)}, status=402)
    except AIServiceUnavailable as e:
        return Response({"error": str(e)}, status=503)
    except Exception as e:
        logger.error("[agent_chat] %s", e)
        return Response({"error": "An unexpected error occurred."}, status=500)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_resume(request):
    """Resume an interrupted agent workflow after user approval/rejection.

    Mirrors silverstripe-ai's AgentController.resume().
    Expects: { approved: bool, context: { entity_class, entity_id }, thread_id }
    """
    from .service import ai_service

    approved = request.data.get("approved", True)

    entity, thread_id, entity_id, error = _resolve_entity(request.data)
    if error:
        code = 404 if error == "Entity not found" else 400
        return Response({"error": error}, status=code)

    try:
        result = ai_service.resume(
            user=request.user, entity=entity, thread_id=thread_id, approved=approved)
        return Response(result.to_dict())
    except CreditLimitExceeded as e:
        return Response({"error": str(e)}, status=402)
    except AIServiceUnavailable as e:
        return Response({"error": str(e)}, status=503)
    except Exception as e:
        logger.error("[agent_resume] %s", e)
        return Response({"error": "An unexpected error occurred."}, status=500)
