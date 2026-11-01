"""history-only sync migration

Revision ID: 7db8aefff00f
Revises: bdb1127dca4f
Create Date: 2026-10-28 23:24:52.281142
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '7db8aefff00f'
down_revision = 'bdb1127dca4f'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # History-only migration: database already has the intended schema.
    pass

def downgrade() -> None:
    pass
