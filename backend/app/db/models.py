import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class HealthCheckRecord(Base):
    """Infrastructure-only row used to prove the persistence stack works end to end.

    Not a business-domain model. Phase 1 introduces doctor/specialty/appointment
    schemas once that domain data is defined.
    """

    __tablename__ = "health_check_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
