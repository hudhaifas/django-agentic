"""Monthly free credit reset. Run via cron on the 1st of each month."""
from django.core.management.base import BaseCommand
from django.utils import timezone
from django_agentic.models import SiteAIConfig, UserAIProfile


class Command(BaseCommand):
    help = "Reset free monthly AI credits for all users"

    def handle(self, *args, **options):
        config = SiteAIConfig.load()
        amount = config.monthly_free_credits
        updated = UserAIProfile.objects.all().update(
            free_monthly_credits=amount, credits_reset_at=timezone.now(),
        )
        self.stdout.write(self.style.SUCCESS(f"Reset {updated} profiles to ${amount} free credits."))
