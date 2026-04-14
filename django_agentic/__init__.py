"""django-agentic: AI framework for Django — LangGraph + LangChain.

Add 'django_agentic' to INSTALLED_APPS, run migrations, configure models in admin.

Checkpointer Configuration
--------------------------
Chat history is stored via LangGraph checkpointers. Configure in settings:

    DJANGO_AGENTIC = {
        "CHECKPOINTER": <checkpointer_instance>,
    }

Default: InMemorySaver (works out of the box, loses history on server restart).

For production with PostgreSQL:

    # pip install langgraph-checkpoint-postgres
    from langgraph.checkpoint.postgres import PostgresSaver
    DJANGO_AGENTIC = {
        "CHECKPOINTER": PostgresSaver.from_conn_string(
            "postgresql://user:pass@host:5432/dbname"
        ),
    }
    # Call DJANGO_AGENTIC["CHECKPOINTER"].setup() once (e.g. in a management command
    # or AppConfig.ready()) to create the checkpoint tables.

For production with Redis:

    # pip install langgraph-checkpoint-redis
    from langgraph.checkpoint.redis import RedisSaver
    DJANGO_AGENTIC = {
        "CHECKPOINTER": RedisSaver(redis_url="redis://localhost:6379"),
    }

For SQLite (local dev with persistence across restarts):

    # pip install langgraph-checkpoint-sqlite
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    DJANGO_AGENTIC = {
        "CHECKPOINTER": SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False)),
    }
"""
__version__ = "0.3.0"
