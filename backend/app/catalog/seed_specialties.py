"""Idempotently seed the specialty catalog from nucc_specialties.py.

Run from backend/:
    uv run python -m app.catalog.seed_specialties

Safe to run repeatedly: specialties are upserted by slug, mappings by
taxonomy_code, so re-running never creates duplicates. This is the only
supported way to populate specialties/specialty_taxonomy_mappings — do not
insert rows manually.
"""

import asyncio
import sys
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.nucc_specialties import SPECIALTY_SEEDS, SpecialtySeed
from app.db.session import get_sessionmaker
from app.db.specialty import Specialty, SpecialtyTaxonomyMapping


@dataclass
class SeedResult:
    specialties_inserted: int = 0
    specialties_updated: int = 0
    mappings_inserted: int = 0
    mappings_updated: int = 0


async def _upsert_specialty(session: AsyncSession, seed: SpecialtySeed) -> tuple[object, bool]:
    """Upsert one specialty by slug. Returns (specialty_id, was_inserted)."""
    existing_id = await session.scalar(select(Specialty.id).where(Specialty.slug == seed.slug))

    stmt = (
        insert(Specialty)
        .values(
            slug=seed.slug,
            display_name=seed.display_name,
            description=seed.description,
            is_active=True,
        )
        .on_conflict_do_update(
            index_elements=[Specialty.slug],
            set_={
                "display_name": seed.display_name,
                "description": seed.description,
                "is_active": True,
                "updated_at": func.now(),
            },
        )
        .returning(Specialty.id)
    )
    specialty_id = await session.scalar(stmt)
    return specialty_id, existing_id is None


async def _upsert_mapping(session: AsyncSession, seed: SpecialtySeed, specialty_id: object) -> bool:
    """Upsert one taxonomy mapping by taxonomy_code. Returns was_inserted."""
    existing = await session.scalar(
        select(SpecialtyTaxonomyMapping.id).where(
            SpecialtyTaxonomyMapping.taxonomy_code == seed.taxonomy_code
        )
    )

    stmt = insert(SpecialtyTaxonomyMapping).values(
        specialty_id=specialty_id,
        taxonomy_code=seed.taxonomy_code,
        taxonomy_description=seed.taxonomy_description,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[SpecialtyTaxonomyMapping.taxonomy_code],
        set_={
            "specialty_id": stmt.excluded.specialty_id,
            "taxonomy_description": stmt.excluded.taxonomy_description,
        },
    )
    await session.execute(stmt)
    return existing is None


async def seed_specialties() -> SeedResult:
    """Idempotently upsert every seed in SPECIALTY_SEEDS. Safe to call repeatedly."""
    session_factory = get_sessionmaker()
    result = SeedResult()

    async with session_factory() as session:
        for seed in SPECIALTY_SEEDS:
            specialty_id, specialty_inserted = await _upsert_specialty(session, seed)
            if specialty_inserted:
                result.specialties_inserted += 1
            else:
                result.specialties_updated += 1

            mapping_inserted = await _upsert_mapping(session, seed, specialty_id)
            if mapping_inserted:
                result.mappings_inserted += 1
            else:
                result.mappings_updated += 1

        await session.commit()

    return result


def main() -> int:
    result = asyncio.run(seed_specialties())
    print(f"specialties_inserted: {result.specialties_inserted}")
    print(f"specialties_updated: {result.specialties_updated}")
    print(f"mappings_inserted: {result.mappings_inserted}")
    print(f"mappings_updated: {result.mappings_updated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
