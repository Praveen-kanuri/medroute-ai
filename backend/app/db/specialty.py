import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Specialty(Base):
    """A MedRoute AI specialty in the small, transparent supported catalog.

    See app/catalog/nucc_specialties.py for the source data and its
    NUCC taxonomy-code-set version/access date.
    """

    __tablename__ = "specialties"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str | None] = mapped_column(default=None)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SpecialtyTaxonomyMapping(Base):
    """Maps one authoritative NUCC taxonomy code to one MedRoute specialty.

    taxonomy_code is unique: each taxonomy code identifies exactly one
    specialty in this catalog (a specialty may still have more than one
    mapped code in the future, but a code never maps to more than one
    specialty).
    """

    __tablename__ = "specialty_taxonomy_mappings"
    __table_args__ = (Index("ix_specialty_taxonomy_mappings_specialty_id", "specialty_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    specialty_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("specialties.id", ondelete="CASCADE"), nullable=False
    )
    taxonomy_code: Mapped[str] = mapped_column(unique=True, nullable=False)
    taxonomy_description: Mapped[str | None] = mapped_column(default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
