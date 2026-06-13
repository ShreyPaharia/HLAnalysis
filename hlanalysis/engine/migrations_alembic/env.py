"""Alembic environment for the engine's per-slot state.db.

Driven programmatically from ``StateDAL.run_migrations`` (no CLI step at
runtime). The DB url is injected via ``Config.set_main_option`` by the caller,
so this env builds an engine straight from that url. ``target_metadata`` is
None because every revision is hand-written (the baseline must reproduce the
exact pre-Alembic schema byte-for-byte — autogenerate from SQLModel metadata
would NOT match, e.g. the ALTER-appended fill.closed_pnl column).
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

config = context.config
target_metadata = None


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("alembic env: sqlalchemy.url not configured")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
