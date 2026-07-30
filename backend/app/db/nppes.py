import uuid
from datetime import date, datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class IngestionStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class Provider(Base):
    """A healthcare provider identity sourced from the NPPES registry.

    NPPES is a public government provider directory (real doctor/organization
    identities, not patient data) — see docs/safety-design.md. Supports both
    NPPES entity types: 1 (individual) and 2 (organization).
    """

    __tablename__ = "providers"
    __table_args__ = (
        CheckConstraint("npi ~ '^[0-9]{10}$'", name="npi_format"),
        CheckConstraint("entity_type_code IN ('1', '2')", name="entity_type_code"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    npi: Mapped[str] = mapped_column(unique=True, nullable=False)
    entity_type_code: Mapped[str] = mapped_column(nullable=False)

    organization_name: Mapped[str | None] = mapped_column(default=None)
    first_name: Mapped[str | None] = mapped_column(default=None)
    last_name: Mapped[str | None] = mapped_column(default=None)
    middle_name: Mapped[str | None] = mapped_column(default=None)
    name_prefix: Mapped[str | None] = mapped_column(default=None)
    name_suffix: Mapped[str | None] = mapped_column(default=None)
    credential: Mapped[str | None] = mapped_column(default=None)
    gender_code: Mapped[str | None] = mapped_column(default=None)

    enumeration_date: Mapped[date | None] = mapped_column(Date, default=None)
    last_update_date: Mapped[date | None] = mapped_column(Date, default=None)
    deactivation_date: Mapped[date | None] = mapped_column(Date, default=None)
    reactivation_date: Mapped[date | None] = mapped_column(Date, default=None)
    replacement_npi: Mapped[str | None] = mapped_column(default=None)

    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ProviderLocation(Base):
    """A provider's mailing or practice-location address.

    Natural key (provider_id, address_purpose): the NPPES main file supplies
    exactly one mailing and one practice address per provider row, so this is
    sufficient to make re-ingestion idempotent without inventing a synthetic
    address identity NPPES doesn't provide.
    """

    __tablename__ = "provider_locations"
    __table_args__ = (
        UniqueConstraint(
            "provider_id", "address_purpose", name="uq_provider_locations_provider_id"
        ),
        CheckConstraint(
            "address_purpose IN ('mailing', 'practice')",
            name="address_purpose",
        ),
        Index("ix_provider_locations_provider_id", "provider_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("providers.id", ondelete="CASCADE"), nullable=False
    )
    address_purpose: Mapped[str] = mapped_column(nullable=False)

    address_line_1: Mapped[str | None] = mapped_column(default=None)
    address_line_2: Mapped[str | None] = mapped_column(default=None)
    city: Mapped[str | None] = mapped_column(default=None)
    state: Mapped[str | None] = mapped_column(default=None)
    postal_code: Mapped[str | None] = mapped_column(default=None)
    country_code: Mapped[str | None] = mapped_column(default="US")
    telephone_number: Mapped[str | None] = mapped_column(default=None)
    fax_number: Mapped[str | None] = mapped_column(default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ProviderTaxonomy(Base):
    """A provider's taxonomy (specialty) assignment, with license attributes.

    Natural key (provider_id, taxonomy_code): NPPES ties one license per
    taxonomy slot for a provider, so this pair identifies the assignment for
    idempotent re-ingestion.
    """

    __tablename__ = "provider_taxonomies"
    __table_args__ = (
        UniqueConstraint("provider_id", "taxonomy_code", name="uq_provider_taxonomies_provider_id"),
        Index("ix_provider_taxonomies_provider_id", "provider_id"),
        Index("ix_provider_taxonomies_taxonomy_code", "taxonomy_code"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("providers.id", ondelete="CASCADE"), nullable=False
    )
    taxonomy_code: Mapped[str] = mapped_column(nullable=False)
    license_number: Mapped[str | None] = mapped_column(default=None)
    license_state: Mapped[str | None] = mapped_column(default=None)
    is_primary: Mapped[bool] = mapped_column(nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class IngestionRun(Base):
    """Tracks one execution of a batch ingestion (e.g. one NPPES CSV import)."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'partially_completed')",
            name="status",
        ),
        Index("ix_ingestion_runs_status", "status"),
        Index("ix_ingestion_runs_started_at", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_name: Mapped[str] = mapped_column(nullable=False)
    source_filename: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(nullable=False, default=IngestionStatus.RUNNING)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    rows_read: Mapped[int] = mapped_column(nullable=False, default=0)
    rows_inserted: Mapped[int] = mapped_column(nullable=False, default=0)
    rows_updated: Mapped[int] = mapped_column(nullable=False, default=0)
    rows_rejected: Mapped[int] = mapped_column(nullable=False, default=0)

    # Bounded/sanitized summary only — never a full row dump or a private path.
    error_summary: Mapped[str | None] = mapped_column(default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
