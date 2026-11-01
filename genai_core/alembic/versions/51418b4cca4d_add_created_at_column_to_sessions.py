"""Add created_at column to sessions

Revision ID: 51418b4cca4d
Revises: 98b19f60077e
Create Date: 2026-10-28 23:13:31.718328

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '51418b4cca4d'
down_revision: Union[str, Sequence[str], None] = '98b19f60077e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column(
        'sessions',
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )

def downgrade():
    op.drop_column('sessions', 'created_at')
