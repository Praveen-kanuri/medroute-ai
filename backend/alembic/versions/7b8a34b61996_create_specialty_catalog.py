"""create specialty catalog

Revision ID: 7b8a34b61996
Revises: 8199fd7b743c
Create Date: 2026-07-29 22:45:21.588165

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7b8a34b61996"
down_revision: str | Sequence[str] | None = "8199fd7b743c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "specialties",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_specialties")),
        sa.UniqueConstraint("slug", name=op.f("uq_specialties_slug")),
    )

    op.create_table(
        "specialty_taxonomy_mappings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("specialty_id", sa.UUID(), nullable=False),
        sa.Column("taxonomy_code", sa.String(), nullable=False),
        sa.Column("taxonomy_description", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["specialty_id"],
            ["specialties.id"],
            name=op.f("fk_specialty_taxonomy_mappings_specialty_id_specialties"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_specialty_taxonomy_mappings")),
        sa.UniqueConstraint(
            "taxonomy_code", name=op.f("uq_specialty_taxonomy_mappings_taxonomy_code")
        ),
    )
    op.create_index(
        "ix_specialty_taxonomy_mappings_specialty_id",
        "specialty_taxonomy_mappings",
        ["specialty_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_specialty_taxonomy_mappings_specialty_id", table_name="specialty_taxonomy_mappings"
    )
    op.drop_table("specialty_taxonomy_mappings")
    op.drop_table("specialties")
