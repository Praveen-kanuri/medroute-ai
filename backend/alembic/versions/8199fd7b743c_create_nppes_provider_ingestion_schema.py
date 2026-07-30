"""create nppes provider ingestion schema

Revision ID: 8199fd7b743c
Revises: 55c268c36fcf
Create Date: 2026-07-29 21:25:56.695044

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8199fd7b743c"
down_revision: str | Sequence[str] | None = "55c268c36fcf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_name", sa.String(), nullable=False),
        sa.Column("source_filename", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rows_read", sa.Integer(), nullable=False),
        sa.Column("rows_inserted", sa.Integer(), nullable=False),
        sa.Column("rows_updated", sa.Integer(), nullable=False),
        sa.Column("rows_rejected", sa.Integer(), nullable=False),
        sa.Column("error_summary", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'partially_completed')",
            name=op.f("ck_ingestion_runs_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_runs")),
    )
    op.create_index("ix_ingestion_runs_started_at", "ingestion_runs", ["started_at"], unique=False)
    op.create_index("ix_ingestion_runs_status", "ingestion_runs", ["status"], unique=False)

    op.create_table(
        "providers",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("npi", sa.String(), nullable=False),
        sa.Column("entity_type_code", sa.String(), nullable=False),
        sa.Column("organization_name", sa.String(), nullable=True),
        sa.Column("first_name", sa.String(), nullable=True),
        sa.Column("last_name", sa.String(), nullable=True),
        sa.Column("middle_name", sa.String(), nullable=True),
        sa.Column("name_prefix", sa.String(), nullable=True),
        sa.Column("name_suffix", sa.String(), nullable=True),
        sa.Column("credential", sa.String(), nullable=True),
        sa.Column("gender_code", sa.String(), nullable=True),
        sa.Column("enumeration_date", sa.Date(), nullable=True),
        sa.Column("last_update_date", sa.Date(), nullable=True),
        sa.Column("deactivation_date", sa.Date(), nullable=True),
        sa.Column("reactivation_date", sa.Date(), nullable=True),
        sa.Column("replacement_npi", sa.String(), nullable=True),
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
        sa.CheckConstraint(
            "entity_type_code IN ('1', '2')", name=op.f("ck_providers_entity_type_code")
        ),
        sa.CheckConstraint("npi ~ '^[0-9]{10}$'", name=op.f("ck_providers_npi_format")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_providers")),
        sa.UniqueConstraint("npi", name=op.f("uq_providers_npi")),
    )

    op.create_table(
        "provider_locations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("provider_id", sa.UUID(), nullable=False),
        sa.Column("address_purpose", sa.String(), nullable=False),
        sa.Column("address_line_1", sa.String(), nullable=True),
        sa.Column("address_line_2", sa.String(), nullable=True),
        sa.Column("city", sa.String(), nullable=True),
        sa.Column("state", sa.String(), nullable=True),
        sa.Column("postal_code", sa.String(), nullable=True),
        sa.Column("country_code", sa.String(), nullable=True),
        sa.Column("telephone_number", sa.String(), nullable=True),
        sa.Column("fax_number", sa.String(), nullable=True),
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
        sa.CheckConstraint(
            "address_purpose IN ('mailing', 'practice')",
            name=op.f("ck_provider_locations_address_purpose"),
        ),
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["providers.id"],
            name=op.f("fk_provider_locations_provider_id_providers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_provider_locations")),
        sa.UniqueConstraint(
            "provider_id", "address_purpose", name="uq_provider_locations_provider_id"
        ),
    )
    op.create_index(
        "ix_provider_locations_provider_id", "provider_locations", ["provider_id"], unique=False
    )

    op.create_table(
        "provider_taxonomies",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("provider_id", sa.UUID(), nullable=False),
        sa.Column("taxonomy_code", sa.String(), nullable=False),
        sa.Column("license_number", sa.String(), nullable=True),
        sa.Column("license_state", sa.String(), nullable=True),
        sa.Column("is_primary", sa.Boolean(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["providers.id"],
            name=op.f("fk_provider_taxonomies_provider_id_providers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_provider_taxonomies")),
        sa.UniqueConstraint(
            "provider_id", "taxonomy_code", name="uq_provider_taxonomies_provider_id"
        ),
    )
    op.create_index(
        "ix_provider_taxonomies_provider_id", "provider_taxonomies", ["provider_id"], unique=False
    )
    op.create_index(
        "ix_provider_taxonomies_taxonomy_code",
        "provider_taxonomies",
        ["taxonomy_code"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_provider_taxonomies_taxonomy_code", table_name="provider_taxonomies")
    op.drop_index("ix_provider_taxonomies_provider_id", table_name="provider_taxonomies")
    op.drop_table("provider_taxonomies")

    op.drop_index("ix_provider_locations_provider_id", table_name="provider_locations")
    op.drop_table("provider_locations")

    op.drop_table("providers")

    op.drop_index("ix_ingestion_runs_status", table_name="ingestion_runs")
    op.drop_index("ix_ingestion_runs_started_at", table_name="ingestion_runs")
    op.drop_table("ingestion_runs")
