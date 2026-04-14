from rest_framework import serializers
from .models import AIModel


class AIModelChoiceSerializer(serializers.ModelSerializer):
    class Meta:
        model = AIModel
        fields = ["id", "name", "display_name", "provider",
                  "input_cost_per_1m", "output_cost_per_1m",
                  "allowed_for_free", "allowed_for_paid"]


class CreditStatusSerializer(serializers.Serializer):
    free_monthly_credits = serializers.DecimalField(max_digits=10, decimal_places=6)
    purchased_credits = serializers.DecimalField(max_digits=10, decimal_places=6)
    total_credits = serializers.DecimalField(max_digits=10, decimal_places=6)
    monthly_allowance = serializers.DecimalField(max_digits=10, decimal_places=2)
    credits_reset_at = serializers.DateTimeField(allow_null=True)
    is_unlimited = serializers.BooleanField()
    current_model = serializers.CharField()
    current_model_provider = serializers.CharField()
    is_free_tier = serializers.BooleanField()
    model_override_id = serializers.UUIDField(allow_null=True)
    available_models = AIModelChoiceSerializer(many=True)


class ModelOverrideSerializer(serializers.Serializer):
    model_id = serializers.UUIDField(required=False, allow_null=True)
