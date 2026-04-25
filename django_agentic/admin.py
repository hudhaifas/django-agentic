import json
import datetime
from django.contrib import admin
from django.db.models import Sum, Count
from django.utils import timezone
from django.utils.html import escape
from django.utils.safestring import mark_safe
from .models import AIModel, AIUsageLog, SiteAIConfig, UserAIProfile


def _daily_usage_chart(qs, title="Usage (Last 30 Days)", days=30):
    """Generate Chart.js HTML for daily usage trends."""
    from django.db.models.functions import TruncDate
    cutoff = timezone.now() - datetime.timedelta(days=days)
    daily = (
        qs.filter(created_at__gte=cutoff)
        .annotate(date=TruncDate("created_at"))
        .values("date")
        .annotate(
            requests=Count("id"), cost=Sum("cost_usd"),
            input_tokens=Sum("prompt_tokens"), output_tokens=Sum("completion_tokens"),
            cache_read=Sum("cache_read_tokens"), cache_write=Sum("cache_creation_tokens"),
        )
        .order_by("date")
    )
    daily_map = {r["date"]: r for r in daily}
    today = timezone.now().date()

    dates, costs, inputs, outputs, cache_r, cache_w, reqs = [], [], [], [], [], [], []
    for i in range(days - 1, -1, -1):
        d = today - datetime.timedelta(days=i)
        r = daily_map.get(d)
        dates.append(d.strftime("%b %d"))
        costs.append(float(r["cost"] or 0) if r else 0)
        inputs.append(r["input_tokens"] or 0 if r else 0)
        outputs.append(r["output_tokens"] or 0 if r else 0)
        cache_r.append(r["cache_read"] or 0 if r else 0)
        cache_w.append(r["cache_write"] or 0 if r else 0)
        reqs.append(r["requests"] or 0 if r else 0)

    total_cost = f"${sum(costs):.4f}"
    total_reqs = sum(reqs)
    total_tokens = f"{sum(inputs)+sum(outputs)+sum(cache_r)+sum(cache_w):,}"
    cid = f"aichart_{id(qs) % 99999}"
    data = json.dumps({"d": dates, "c": costs, "i": inputs,
                        "o": outputs, "cr": cache_r, "cw": cache_w})

    safe_title = escape(title)
    js = (
        '(function(){'
        'var d=' + data + ';'
        'var ctx=document.getElementById("' + cid + '");'
        'if(ctx&&d.d.length>0){new Chart(ctx,{type:"line",data:{labels:d.d,datasets:['
        '{label:"Cost ($)",data:d.c,borderColor:"rgb(75,192,192)",backgroundColor:"rgba(75,192,192,0.1)",yAxisID:"y",tension:0.3,fill:true},'
        '{label:"Input",data:d.i,borderColor:"rgb(54,162,235)",backgroundColor:"rgba(54,162,235,0.1)",yAxisID:"y1",tension:0.3,fill:true},'
        '{label:"Output",data:d.o,borderColor:"rgb(255,159,64)",backgroundColor:"rgba(255,159,64,0.1)",yAxisID:"y1",tension:0.3,fill:true},'
        '{label:"Cache Read",data:d.cr,borderColor:"rgb(153,102,255)",backgroundColor:"rgba(153,102,255,0.1)",yAxisID:"y1",tension:0.3,fill:true},'
        '{label:"Cache Write",data:d.cw,borderColor:"rgb(255,99,132)",backgroundColor:"rgba(255,99,132,0.1)",yAxisID:"y1",tension:0.3,fill:true}'
        ']},options:{responsive:true,interaction:{mode:"index",intersect:false},'
        'scales:{y:{type:"linear",position:"left",title:{display:true,text:"Cost ($)"},ticks:{callback:function(v){return "$"+v.toFixed(3)}}},'
        'y1:{type:"linear",position:"right",grid:{drawOnChartArea:false},title:{display:true,text:"Tokens"},'
        'ticks:{callback:function(v){return v>=1e6?(v/1e6).toFixed(1)+"M":v>=1e3?(v/1e3).toFixed(0)+"K":v}}}'
        '}}})}})();'
    )
    return mark_safe(
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'
        '<div style="background:#fff;padding:20px;border:1px solid #ddd;border-radius:4px;margin:10px 0">'
        '<h3 style="margin-top:0">' + str(safe_title) + '</h3>'
        '<div style="display:flex;gap:40px;margin-bottom:15px">'
        '<div><strong>Requests:</strong> ' + str(total_reqs) + '</div>'
        '<div><strong>Cost:</strong> ' + str(total_cost) + '</div>'
        '<div><strong>Tokens:</strong> ' + str(total_tokens) + '</div></div>'
        '<canvas id="' + cid + '" style="max-height:300px"></canvas></div>'
        '<script>' + js + '</script>'
    )


@admin.register(AIModel)
class AIModelAdmin(admin.ModelAdmin):
    list_display = ("display_name", "name", "model_type", "provider", "input_cost_per_1m", "output_cost_per_1m",
                    "allowed_for_free", "allowed_for_paid", "active", "total_cost_30d")
    list_filter = ("model_type", "provider", "active", "allowed_for_free", "allowed_for_paid")

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        if db_field.name == "provider":
            from django_agentic.providers import get_registered_providers
            field = super().formfield_for_dbfield(db_field, request, **kwargs)
            field.help_text = f"Registered: {', '.join(get_registered_providers())}"
            return field
        return super().formfield_for_dbfield(db_field, request, **kwargs)

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        if db_field.name == "provider":
            from django_agentic.providers import get_registered_providers
            providers = get_registered_providers()
            kwargs["widget"] = admin.widgets.AdminTextInputWidget()
            field = super().formfield_for_dbfield(db_field, request, **kwargs)
            field.help_text = f"Registered providers: {', '.join(providers)}"
            return field
        return super().formfield_for_dbfield(db_field, request, **kwargs)
    list_editable = ("allowed_for_free", "allowed_for_paid", "active")
    readonly_fields = ("usage_charts",)

    def total_cost_30d(self, obj):
        cutoff = timezone.now() - datetime.timedelta(days=30)
        total = obj.usage_logs.filter(created_at__gte=cutoff).aggregate(total=Sum("cost_usd"))["total"]
        return f"${total:.4f}" if total else "$0"
    total_cost_30d.short_description = "Cost (30d)"

    def usage_charts(self, obj):
        if not obj.pk:
            return "Save the model first to see usage charts."
        return _daily_usage_chart(obj.usage_logs.all(), f"{obj.display_name or obj.name} — Usage")
    usage_charts.short_description = "Usage Statistics"

    def get_fieldsets(self, request, obj=None):
        base = [(None, {"fields": ("name", "display_name", "model_type", "provider",
                                    "input_cost_per_1m", "output_cost_per_1m",
                                    "cache_write_cost_per_1m", "cache_read_cost_per_1m",
                                    "active", "allowed_for_free", "allowed_for_paid",
                                    "context_window", "max_output_tokens")})]
        if obj and obj.pk:
            base.append(("Usage Statistics", {"fields": ("usage_charts",)}))
        return base


@admin.register(SiteAIConfig)
class SiteAIConfigAdmin(admin.ModelAdmin):
    list_display = ("__str__", "default_free_model", "default_paid_model", "monthly_free_credits")
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name in ("default_free_model", "default_paid_model"):
            kwargs["queryset"] = AIModel.objects.filter(active=True, model_type="chat")
        if db_field.name == "default_free_model":
            kwargs["queryset"] = kwargs["queryset"].filter(allowed_for_free=True)
        return super().formfield_for_foreignkey(db_field, request, **kwargs)
    def has_add_permission(self, request):
        return not SiteAIConfig.objects.exists()
    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(UserAIProfile)
class UserAIProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "free_monthly_credits", "purchased_credits", "total_credits",
                    "model_override", "credits_reset_at")
    list_filter = ("model_override",)
    search_fields = ("user__email", "user__username")
    readonly_fields = ("created_at", "updated_at", "usage_charts")

    def total_credits(self, obj):
        return f"${obj.total_credits:.6f}"
    total_credits.short_description = "Total"

    def usage_charts(self, obj):
        if not obj.pk:
            return "Save the profile first to see usage charts."
        return _daily_usage_chart(AIUsageLog.objects.filter(user=obj.user), f"{obj.user} — Usage")
    usage_charts.short_description = "Usage Statistics"

    def get_fieldsets(self, request, obj=None):
        base = [(None, {"fields": ("user", "free_monthly_credits", "purchased_credits",
                                    "model_override", "credits_reset_at", "created_at", "updated_at")})]
        if obj and obj.pk:
            base.append(("Usage Statistics", {"fields": ("usage_charts",)}))
        return base


@admin.register(AIUsageLog)
class AIUsageLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "workflow", "node", "model_name", "user",
                    "prompt_tokens", "completion_tokens",
                    "cache_read_tokens", "cache_creation_tokens",
                    "cost_usd", "used_free_credits", "used_paid_credits",
                    "error_type", "duration_ms", "success")
    list_filter = ("workflow", "model_name", "success", "error_type", "created_at")
    readonly_fields = ("id", "idempotency_key", "created_at", "request_time", "response_time",
                       "provider_request_id",
                       "input_cost_per_1m_at_time", "output_cost_per_1m_at_time",
                       "cache_write_cost_per_1m_at_time", "cache_read_cost_per_1m_at_time")
    search_fields = ("workflow", "node", "input_summary", "user__email", "provider_request_id")
    date_hierarchy = "created_at"
