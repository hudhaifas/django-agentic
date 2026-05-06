"""Rename django_ai_* tables to django_agentic_* for existing installations.

This migration accompanies the app label change from 'django_ai' to
'django_agentic' in apps.py.

For fresh installations: migrations 0001-0005 already create tables with
the correct django_agentic_* names (after 0003 removes db_table overrides),
so this migration is a no-op.

For existing installations: tables were created as django_ai_* and need
renaming. The RENAME TABLE is conditional on the old table existing.
"""

from django.db import migrations


def rename_table_if_exists(old_name, new_name):
    sql = f"""
        SET @exists = (
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = '{old_name}'
        );
        SET @sql = IF(@exists > 0,
            'RENAME TABLE `{old_name}` TO `{new_name}`',
            'SELECT 1'
        );
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    """
    reverse_sql = f"""
        SET @exists = (
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = '{new_name}'
        );
        SET @sql = IF(@exists > 0,
            'RENAME TABLE `{new_name}` TO `{old_name}`',
            'SELECT 1'
        );
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    """
    return migrations.RunSQL(sql=sql, reverse_sql=reverse_sql)


class Migration(migrations.Migration):

    dependencies = [
        ('django_agentic', '0005_remove_provider_choices'),
    ]

    operations = [
        rename_table_if_exists('django_ai_aimodel', 'django_agentic_aimodel'),
        rename_table_if_exists('django_ai_aiusagelog', 'django_agentic_aiusagelog'),
        rename_table_if_exists('django_ai_siteaiconfig', 'django_agentic_siteaiconfig'),
        rename_table_if_exists('django_ai_useraiprofile', 'django_agentic_useraiprofile'),

        # Update Django's model state to use the new table names
        migrations.AlterModelTable('AIModel', 'django_agentic_aimodel'),
        migrations.AlterModelTable('AIUsageLog', 'django_agentic_aiusagelog'),
        migrations.AlterModelTable('SiteAIConfig', 'django_agentic_siteaiconfig'),
        migrations.AlterModelTable('UserAIProfile', 'django_agentic_useraiprofile'),
    ]
