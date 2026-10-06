"""LCP owns conversation messages — the store comes back, this time as a first-class table

Revision ID: 024
Revises: 023

Context
-------
``routing_decisions.conversation_json`` used to hold a trimmed copy of the request
that produced each routing decision. simplify-lcp **M7 / alembic 022** dropped it —
it was 101 MB across 27,969 rows, 73 % of ``costs.db``. That was the right call at
the time (LCP had no use for a transcript, and nothing read it after the work layer
started reading Hermes' session store instead).

The requirement changed: LCP should own its conversation history outright rather
than reach into another application's schema. This revision adds:

* ``conv_messages`` — the message rows, one conversation to many messages.
* a handful of nullable rollup columns on ``conversations`` so an imported
  conversation can report the same facts the chat store does (window, message and
  tool-call counts, tokens). Nullable, so every existing writer is unaffected.

Content sizing
--------------
``content`` is capped by the *writer* at 64 KB per message; ``content_len`` always
records the true length and ``truncated`` flags the row, so a capped row is
self-describing and the audit view can say "1.2 MB, showing first 64 KB" rather
than silently lying. 578 of the 258 k rows in the current corpus exceed the cap.

Fidelity
--------
``active`` and ``compacted`` are carried across from the source store rather than
flattened: 68 k of the corpus is compacted context, and the UI filters on it. A
transcript that dropped those rows would misrepresent what the agent actually saw.
"""

import sqlalchemy as sa
from alembic import op

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


#: Nullable rollup columns added to the existing ``conversations`` table. The table
#: itself is created by ``work_conversations.ensure_schema()`` (it predates the
#: alembic line for this feature), so every add is guarded.
_CONVERSATION_COLUMNS = (
    ("source", "TEXT"),          # where the row came from: 'hermes' | 'live'
    ("agent_session_id", "TEXT"),  # the originating agent session id, when known
    ("started_at", "TEXT"),
    ("ended_at", "TEXT"),
    # ``message_count`` is the size of the stored transcript. Hermes' own
    # ``sessions.message_count`` means something different — it is the
    # post-compaction *live window* (verified 2026-10-06: for session
    # 20260918_165340_061a1c98 it was 1525, exactly the count of ``active=1``
    # conversation rows, while the transcript held 7241). Both numbers are useful
    # to an audit view, so they are kept apart rather than conflated.
    ("message_count", "INTEGER"),
    ("active_message_count", "INTEGER"),
    ("tool_call_count", "INTEGER"),
    ("input_tokens", "INTEGER"),
    ("output_tokens", "INTEGER"),
    ("imported_at", "TEXT"),
)


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _has_column(table: str, column: str) -> bool:
    insp = sa.inspect(op.get_bind())
    if table not in insp.get_table_names():
        return False
    return any(c["name"] == column for c in insp.get_columns(table))


def upgrade() -> None:
    # ``conversations`` is created by ``work_conversations.ensure_schema()`` at
    # runtime, not by this migration line — so on a database that has never been
    # served (a fresh install) the table is simply absent, and ALTERing it would
    # abort the whole upgrade. Guard on existence rather than assume.
    if _has_table("conversations"):
        for column, ddl in _CONVERSATION_COLUMNS:
            if not _has_column("conversations", column):
                op.execute("ALTER TABLE conversations ADD COLUMN %s %s" % (column, ddl))

    if not _has_table("conv_messages"):
        op.create_table(
            "conv_messages",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("conversation_id", sa.String(), nullable=False),
            # Position within the conversation. Stable under append: the
            # migration numbers messages by the source's own autoincrement id.
            sa.Column("seq", sa.Integer(), nullable=False),
            sa.Column("role", sa.String()),
            sa.Column("content", sa.Text()),
            # True length before the writer's cap, so a truncated row stays honest.
            sa.Column("content_len", sa.Integer()),
            sa.Column("truncated", sa.Integer(), server_default="0"),
            sa.Column("tool_name", sa.String()),
            sa.Column("tool_call_id", sa.String()),
            sa.Column("tool_calls", sa.Text()),
            sa.Column("reasoning", sa.Text()),
            sa.Column("ts", sa.String()),
            sa.Column("active", sa.Integer(), server_default="1"),
            sa.Column("compacted", sa.Integer(), server_default="0"),
            sa.Column("msg_hash", sa.String()),
            sa.Column("origin", sa.String()),  # 'migration' | 'capture'
            sa.UniqueConstraint("conversation_id", "seq", name="uq_conv_messages_conv_seq"),
        )
        op.create_index("idx_conv_messages_conv", "conv_messages", ["conversation_id", "seq"])
        op.create_index(
            "idx_conv_messages_hash", "conv_messages", ["conversation_id", "msg_hash"]
        )

    # The History table pages newest-first.
    if _has_column("conversations", "last_seen"):
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_last_seen "
            "ON conversations (last_seen DESC)"
        )


def downgrade() -> None:
    """Drop only what this revision added.

    ``conversations`` itself is left in place — it predates this revision and
    dropping it would destroy rollups that other code writes.
    """
    op.execute("DROP INDEX IF EXISTS idx_conversations_last_seen")
    op.execute("DROP INDEX IF EXISTS idx_conv_messages_hash")
    op.execute("DROP INDEX IF EXISTS idx_conv_messages_conv")
    if _has_table("conv_messages"):
        op.drop_table("conv_messages")
    for column, _ddl in reversed(_CONVERSATION_COLUMNS):
        if _has_column("conversations", column):
            op.execute("ALTER TABLE conversations DROP COLUMN %s" % column)
