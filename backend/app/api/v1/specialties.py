from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.db.specialty import Specialty
from app.schemas.provider_search import SpecialtyOut

router = APIRouter()


@router.get("/specialties")
async def list_specialties(
    session: AsyncSession = Depends(get_db_session),
) -> list[SpecialtyOut]:
    """List active specialties in the supported catalog, sorted by display name."""
    result = await session.scalars(
        select(Specialty).where(Specialty.is_active.is_(True)).order_by(Specialty.display_name)
    )
    return [
        SpecialtyOut(slug=s.slug, display_name=s.display_name, description=s.description)
        for s in result
    ]
