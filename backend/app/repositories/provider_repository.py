"""Async repository for deterministic provider search.

Joins providers -> provider_taxonomies -> specialty_taxonomy_mappings ->
specialties -> provider_locations in a single query (no N+1). A provider
must have at least one taxonomy row to be searchable — NPPES's taxonomy
columns are how "what does this provider do" is expressed, so a provider
with zero taxonomies isn't meaningfully discoverable by specialty.

Only one taxonomy row per provider is used (window-ranked: primary taxonomy
first, then alphabetically by code — deterministic) and only one location
row per provider is used (practice address preferred over mailing), so
results never contain duplicate providers caused by the joins themselves.

NPPES inclusion is not proof of licensing, credentialing, quality, or
appointment availability — see the API-level disclaimer.
"""

from dataclasses import dataclass
from datetime import date

from sqlalchemy import ColumnElement, and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.nppes import Provider, ProviderLocation, ProviderTaxonomy
from app.db.specialty import Specialty, SpecialtyTaxonomyMapping

# Safety bound on how many matching rows are ever fetched before ranking and
# pagination happen in the service layer. Well beyond fixture/pilot scale;
# a national-scale deployment would push ordering and LIMIT/OFFSET into SQL
# instead of ranking in Python.
MAX_FETCH_ROWS = 2000


@dataclass(frozen=True)
class ProviderSearchRow:
    npi: str
    entity_type_code: str
    organization_name: str | None
    first_name: str | None
    last_name: str | None
    middle_name: str | None
    name_prefix: str | None
    name_suffix: str | None
    credential: str | None
    is_active: bool
    last_update_date: date | None
    taxonomy_code: str
    is_primary_taxonomy: bool
    specialty_slug: str | None
    specialty_display_name: str | None
    specialty_description: str | None
    location_address_line_1: str | None
    location_address_line_2: str | None
    location_city: str | None
    location_state: str | None
    location_postal_code: str | None
    location_telephone_number: str | None
    location_address_purpose: str | None


async def search_providers(
    session: AsyncSession,
    *,
    specialty_slug: str | None = None,
    taxonomy_code: str | None = None,
    state: str | None = None,
    city: str | None = None,
    postal_code: str | None = None,
    entity_type: str | None = None,
    name: str | None = None,
    active_only: bool = True,
    fetch_limit: int = MAX_FETCH_ROWS,
) -> list[ProviderSearchRow]:
    """Return matching provider rows (pre-ranking, pre-pagination).

    Filters are ANDed together. Returns an empty list, not an error, when
    nothing matches.
    """
    taxonomy_rank = (
        func.row_number()
        .over(
            partition_by=ProviderTaxonomy.provider_id,
            order_by=(ProviderTaxonomy.is_primary.desc(), ProviderTaxonomy.taxonomy_code.asc()),
        )
        .label("rn")
    )
    taxonomy_query = (
        select(
            ProviderTaxonomy.provider_id.label("provider_id"),
            ProviderTaxonomy.taxonomy_code.label("taxonomy_code"),
            ProviderTaxonomy.is_primary.label("is_primary"),
            Specialty.slug.label("specialty_slug"),
            Specialty.display_name.label("specialty_display_name"),
            Specialty.description.label("specialty_description"),
            taxonomy_rank,
        )
        .select_from(ProviderTaxonomy)
        .outerjoin(
            SpecialtyTaxonomyMapping,
            SpecialtyTaxonomyMapping.taxonomy_code == ProviderTaxonomy.taxonomy_code,
        )
        .outerjoin(Specialty, Specialty.id == SpecialtyTaxonomyMapping.specialty_id)
    )
    if taxonomy_code is not None:
        taxonomy_query = taxonomy_query.where(ProviderTaxonomy.taxonomy_code == taxonomy_code)
    if specialty_slug is not None:
        taxonomy_query = taxonomy_query.where(Specialty.slug == specialty_slug)
    taxonomy_subquery = taxonomy_query.subquery("ranked_taxonomy")

    location_rank = (
        func.row_number()
        .over(
            partition_by=ProviderLocation.provider_id,
            order_by=(
                case((ProviderLocation.address_purpose == "practice", 0), else_=1),
                ProviderLocation.address_purpose.asc(),
            ),
        )
        .label("rn")
    )
    location_subquery = select(
        ProviderLocation.provider_id.label("provider_id"),
        ProviderLocation.address_purpose.label("address_purpose"),
        ProviderLocation.address_line_1.label("address_line_1"),
        ProviderLocation.address_line_2.label("address_line_2"),
        ProviderLocation.city.label("city"),
        ProviderLocation.state.label("state"),
        ProviderLocation.postal_code.label("postal_code"),
        ProviderLocation.telephone_number.label("telephone_number"),
        location_rank,
    ).subquery("ranked_location")

    query = (
        select(
            Provider.npi,
            Provider.entity_type_code,
            Provider.organization_name,
            Provider.first_name,
            Provider.last_name,
            Provider.middle_name,
            Provider.name_prefix,
            Provider.name_suffix,
            Provider.credential,
            Provider.is_active,
            Provider.last_update_date,
            taxonomy_subquery.c.taxonomy_code,
            taxonomy_subquery.c.is_primary,
            taxonomy_subquery.c.specialty_slug,
            taxonomy_subquery.c.specialty_display_name,
            taxonomy_subquery.c.specialty_description,
            location_subquery.c.address_line_1,
            location_subquery.c.address_line_2,
            location_subquery.c.city,
            location_subquery.c.state,
            location_subquery.c.postal_code,
            location_subquery.c.telephone_number,
            location_subquery.c.address_purpose,
        )
        .select_from(Provider)
        .join(
            taxonomy_subquery,
            and_(taxonomy_subquery.c.provider_id == Provider.id, taxonomy_subquery.c.rn == 1),
        )
        .outerjoin(
            location_subquery,
            and_(location_subquery.c.provider_id == Provider.id, location_subquery.c.rn == 1),
        )
    )

    conditions: list[ColumnElement[bool]] = []
    if active_only:
        conditions.append(Provider.is_active.is_(True))
    if entity_type is not None:
        conditions.append(Provider.entity_type_code == entity_type)
    if name is not None:
        pattern = f"%{name}%"
        conditions.append(
            (Provider.last_name.ilike(pattern)) | (Provider.organization_name.ilike(pattern))
        )
    if state is not None:
        conditions.append(location_subquery.c.state == state)
    if city is not None:
        conditions.append(location_subquery.c.city.ilike(city))
    if postal_code is not None:
        conditions.append(location_subquery.c.postal_code == postal_code)

    if conditions:
        query = query.where(and_(*conditions))

    query = query.limit(fetch_limit)

    result = await session.execute(query)
    return [
        ProviderSearchRow(
            npi=row.npi,
            entity_type_code=row.entity_type_code,
            organization_name=row.organization_name,
            first_name=row.first_name,
            last_name=row.last_name,
            middle_name=row.middle_name,
            name_prefix=row.name_prefix,
            name_suffix=row.name_suffix,
            credential=row.credential,
            is_active=row.is_active,
            last_update_date=row.last_update_date,
            taxonomy_code=row.taxonomy_code,
            is_primary_taxonomy=row.is_primary,
            specialty_slug=row.specialty_slug,
            specialty_display_name=row.specialty_display_name,
            specialty_description=row.specialty_description,
            location_address_line_1=row.address_line_1,
            location_address_line_2=row.address_line_2,
            location_city=row.city,
            location_state=row.state,
            location_postal_code=row.postal_code,
            location_telephone_number=row.telephone_number,
            location_address_purpose=row.address_purpose,
        )
        for row in result
    ]
