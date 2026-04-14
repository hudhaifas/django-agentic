"""Seed default AI model records and SiteAIConfig on first migrate."""

from decimal import Decimal
from django.db import migrations


MODELS = [
    # (name, display_name, provider, input_cost, output_cost, cache_write, cache_read, context_window, max_output, allowed_free)
    ("gpt-4.1-nano", "GPT-4.1 Nano", "openai", "0.10", "0.40", "0", "0", 1047576, 32768, True),
    ("gpt-4o-mini", "GPT-4o Mini", "openai", "0.15", "0.60", "0", "0", 128000, 4096, True),
    ("gpt-4.1-mini", "GPT-4.1 Mini", "openai", "0.40", "1.60", "0", "0", 1047576, 32768, True),
    ("gpt-4.1", "GPT-4.1", "openai", "2.00", "8.00", "0", "0", 1047576, 32768, False),
    ("gpt-4o", "GPT-4o", "openai", "2.50", "10.00", "0", "0", 128000, 4096, False),
    ("claude-haiku-3-5-20241022", "Claude Haiku 3.5", "anthropic", "0.80", "4.00", "1.000000", "0.080000", 200000, 8192, True),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "anthropic", "1.00", "5.00", "1.250000", "0.100000", 200000, 8192, True),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4", "anthropic", "3.00", "15.00", "3.750000", "0.300000", 200000, 8192, False),
    ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "anthropic", "3.00", "15.00", "3.750000", "0.300000", 200000, 8192, False),
]

FREE_MODEL = "claude-haiku-3-5-20241022"
PAID_MODEL = "claude-sonnet-4-20250514"


def seed(apps, schema_editor):
    AIModel = apps.get_model("django_ai", "AIModel")
    SiteAIConfig = apps.get_model("django_ai", "SiteAIConfig")

    for name, display, provider, inp, out, cw, cr, ctx, max_out, free in MODELS:
        AIModel.objects.update_or_create(name=name, defaults={
            "display_name": display, "provider": provider,
            "input_cost_per_1m": Decimal(inp), "output_cost_per_1m": Decimal(out),
            "cache_write_cost_per_1m": Decimal(cw), "cache_read_cost_per_1m": Decimal(cr),
            "context_window": ctx, "max_output_tokens": max_out,
            "allowed_for_free": free, "allowed_for_paid": True, "active": True,
        })

    free_model = AIModel.objects.get(name=FREE_MODEL)
    paid_model = AIModel.objects.get(name=PAID_MODEL)
    config, _ = SiteAIConfig.objects.get_or_create(pk=1, defaults={
        "monthly_free_credits": Decimal("2.00"),
    })
    config.default_free_model = free_model
    config.default_paid_model = paid_model
    config.save()


def reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai", "0003_remove_db_table_overrides"),
    ]
    operations = [
        migrations.RunPython(seed, reverse),
    ]
