"""Rename all django_ai_* tables and indexes to django_agentic_*.

This migration accompanies the app label change from 'django_ai' to
'django_agentic' in apps.py. It renames the four tables and all their
associated indexes so the DB stays in sync with the new label.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('django_agentic', '0005_remove_provider_choices'),
    ]

    operations = [
        # ── Tables ──────────────────────────────────────────────────────
        migrations.AlterModelTable('AIModel', 'django_agentic_aimodel'),
        migrations.AlterModelTable('AIUsageLog', 'django_agentic_aiusagelog'),
        migrations.AlterModelTable('SiteAIConfig', 'django_agentic_siteaiconfig'),
        migrations.AlterModelTable('UserAIProfile', 'django_agentic_useraiprofile'),

        # ── Indexes on AIUsageLog ────────────────────────────────────────
        migrations.RenameIndex(
            model_name='aiusagelog',
            old_name='django_ai_a_model_n_053a01_idx',
            new_name='django_agentic_aiusagelog_model_name_idx',
        ),
        migrations.RenameIndex(
            model_name='aiusagelog',
            old_name='django_ai_a_workflo_9b3be0_idx',
            new_name='django_agentic_aiusagelog_workflow_idx',
        ),
        migrations.RenameIndex(
            model_name='aiusagelog',
            old_name='django_ai_a_user_id_6d10e0_idx',
            new_name='django_agentic_aiusagelog_user_id_idx',
        ),
    ]
