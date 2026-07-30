"""Chunked, streaming NPPES CSV ingestion with idempotent PostgreSQL upserts.

The file is streamed with csv.DictReader (never loaded fully into memory) and
processed in bounded chunks. Each chunk is written in its own AsyncSession/
transaction, so a failure in one chunk cannot corrupt rows already committed
by earlier chunks, and no single transaction spans the whole file.
"""

import csv
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.nppes import IngestionRun, IngestionStatus, Provider, ProviderLocation, ProviderTaxonomy
from app.db.session import get_sessionmaker
from app.ingestion.nppes_transform import (
    InvalidNPPESRowError,
    NormalizedAddress,
    NormalizedProvider,
    NormalizedTaxonomy,
    transform_row,
)

DEFAULT_CHUNK_SIZE = 500
_MAX_ERROR_REASONS = 20
_MAX_ERROR_SUMMARY_LENGTH = 2000

_PROVIDER_UPDATABLE_COLUMNS = (
    "entity_type_code",
    "organization_name",
    "first_name",
    "last_name",
    "middle_name",
    "name_prefix",
    "name_suffix",
    "credential",
    "gender_code",
    "enumeration_date",
    "last_update_date",
    "deactivation_date",
    "reactivation_date",
    "replacement_npi",
)
_LOCATION_UPDATABLE_COLUMNS = (
    "address_line_1",
    "address_line_2",
    "city",
    "state",
    "postal_code",
    "country_code",
    "telephone_number",
    "fax_number",
)
_TAXONOMY_UPDATABLE_COLUMNS = ("license_number", "license_state", "is_primary")


@dataclass
class IngestionResult:
    run_id: uuid.UUID
    status: IngestionStatus
    rows_read: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_rejected: int = 0
    error_reasons: list[str] = field(default_factory=list)


def _chunked(
    rows: Iterator[Mapping[str, str]], chunk_size: int
) -> Iterator[list[Mapping[str, str]]]:
    chunk: list[Mapping[str, str]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _provider_values(provider: NormalizedProvider) -> dict[str, object]:
    return {
        "npi": provider.npi,
        "entity_type_code": provider.entity_type_code,
        "organization_name": provider.organization_name,
        "first_name": provider.first_name,
        "last_name": provider.last_name,
        "middle_name": provider.middle_name,
        "name_prefix": provider.name_prefix,
        "name_suffix": provider.name_suffix,
        "credential": provider.credential,
        "gender_code": provider.gender_code,
        "enumeration_date": provider.enumeration_date,
        "last_update_date": provider.last_update_date,
        "deactivation_date": provider.deactivation_date,
        "reactivation_date": provider.reactivation_date,
        "replacement_npi": provider.replacement_npi,
    }


def _address_values(address: NormalizedAddress) -> dict[str, object]:
    return {
        "address_purpose": address.address_purpose,
        "address_line_1": address.address_line_1,
        "address_line_2": address.address_line_2,
        "city": address.city,
        "state": address.state,
        "postal_code": address.postal_code,
        "country_code": address.country_code,
        "telephone_number": address.telephone_number,
        "fax_number": address.fax_number,
    }


def _taxonomy_values(taxonomy: NormalizedTaxonomy) -> dict[str, object]:
    return {
        "taxonomy_code": taxonomy.taxonomy_code,
        "license_number": taxonomy.license_number,
        "license_state": taxonomy.license_state,
        "is_primary": taxonomy.is_primary,
    }


async def _upsert_providers(
    session: AsyncSession, providers: list[NormalizedProvider]
) -> tuple[dict[str, uuid.UUID], int, int]:
    """Upsert providers by NPI. Returns (npi -> id map, inserted count, updated count).

    Counts insert vs. update per *original* row (before de-duplication), so
    that an intra-chunk duplicate NPI (e.g. the same provider appearing twice
    in one file) is still reflected in the totals: the first occurrence
    counts as an insert, and any later occurrence of the same NPI in this
    chunk counts as an update — matching what actually happens to that row's
    data. This keeps rows_inserted + rows_updated == valid rows processed.
    """
    npis_in_order = [p.npi for p in providers]
    unique_npis = list(dict.fromkeys(npis_in_order))

    existing_npis: set[str] = set(
        await session.scalars(select(Provider.npi).where(Provider.npi.in_(unique_npis)))
    )

    seen_this_chunk: set[str] = set()
    inserted = 0
    updated = 0
    for npi in npis_in_order:
        if npi in existing_npis or npi in seen_this_chunk:
            updated += 1
        else:
            inserted += 1
        seen_this_chunk.add(npi)

    # De-duplicate by NPI (keep the last occurrence) for the actual write —
    # a single INSERT..ON CONFLICT statement cannot affect the same
    # conflict target twice.
    by_npi = {p.npi: p for p in providers}
    values = [_provider_values(p) for p in by_npi.values()]
    insert_stmt = insert(Provider).values(values)
    set_ = {col: getattr(insert_stmt.excluded, col) for col in _PROVIDER_UPDATABLE_COLUMNS}
    set_["updated_at"] = func.now()
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[Provider.npi], set_=set_
    ).returning(Provider.id, Provider.npi)

    result = await session.execute(upsert_stmt)
    npi_to_id = {row.npi: row.id for row in result}

    return npi_to_id, inserted, updated


async def _upsert_locations(session: AsyncSession, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    stmt = insert(ProviderLocation).values(rows)
    set_ = {col: getattr(stmt.excluded, col) for col in _LOCATION_UPDATABLE_COLUMNS}
    set_["updated_at"] = func.now()
    stmt = stmt.on_conflict_do_update(
        index_elements=[ProviderLocation.provider_id, ProviderLocation.address_purpose],
        set_=set_,
    )
    await session.execute(stmt)


async def _upsert_taxonomies(session: AsyncSession, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    stmt = insert(ProviderTaxonomy).values(rows)
    set_ = {col: getattr(stmt.excluded, col) for col in _TAXONOMY_UPDATABLE_COLUMNS}
    set_["updated_at"] = func.now()
    stmt = stmt.on_conflict_do_update(
        index_elements=[ProviderTaxonomy.provider_id, ProviderTaxonomy.taxonomy_code],
        set_=set_,
    )
    await session.execute(stmt)


async def _update_run_progress(
    session: AsyncSession, run_id: uuid.UUID, result: IngestionResult
) -> None:
    await session.execute(
        update(IngestionRun)
        .where(IngestionRun.id == run_id)
        .values(
            rows_read=result.rows_read,
            rows_inserted=result.rows_inserted,
            rows_updated=result.rows_updated,
            rows_rejected=result.rows_rejected,
        )
    )


async def ingest_nppes_file(
    file_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    source_name: str = "nppes",
) -> IngestionResult:
    """Stream, validate, and idempotently upsert an NPPES CSV file.

    Never logs or persists the file's absolute path — only its basename.
    """
    session_factory = get_sessionmaker()
    source_filename = file_path.name

    async with session_factory() as session:
        run = IngestionRun(
            source_name=source_name, source_filename=source_filename, status=IngestionStatus.RUNNING
        )
        session.add(run)
        await session.flush()
        run_id = run.id
        await session.commit()

    result = IngestionResult(run_id=run_id, status=IngestionStatus.RUNNING)
    had_chunk_failure = False

    try:
        with file_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for chunk in _chunked(reader, chunk_size):
                valid_providers: list[NormalizedProvider] = []
                for raw_row in chunk:
                    result.rows_read += 1
                    try:
                        valid_providers.append(transform_row(raw_row))
                    except InvalidNPPESRowError as exc:
                        result.rows_rejected += 1
                        if len(result.error_reasons) < _MAX_ERROR_REASONS:
                            result.error_reasons.append(str(exc)[:200])

                if not valid_providers:
                    continue

                try:
                    async with session_factory() as session:
                        npi_to_id, inserted, updated = await _upsert_providers(
                            session, valid_providers
                        )

                        by_npi = {p.npi: p for p in valid_providers}
                        location_rows: list[dict[str, object]] = []
                        taxonomy_rows: list[dict[str, object]] = []
                        for npi, provider_id in npi_to_id.items():
                            provider = by_npi[npi]
                            for address in provider.addresses:
                                location_rows.append(
                                    {"provider_id": provider_id, **_address_values(address)}
                                )
                            for taxonomy in provider.taxonomies:
                                taxonomy_rows.append(
                                    {"provider_id": provider_id, **_taxonomy_values(taxonomy)}
                                )

                        await _upsert_locations(session, location_rows)
                        await _upsert_taxonomies(session, taxonomy_rows)

                        result.rows_inserted += inserted
                        result.rows_updated += updated
                        await _update_run_progress(session, run_id, result)

                        await session.commit()
                except Exception as exc:
                    had_chunk_failure = True
                    result.rows_rejected += len(valid_providers)
                    if len(result.error_reasons) < _MAX_ERROR_REASONS:
                        result.error_reasons.append(f"chunk failed: {type(exc).__name__}")

                    async with session_factory() as session:
                        await _update_run_progress(session, run_id, result)
                        await session.commit()
    except FileNotFoundError:
        had_chunk_failure = True
        result.error_reasons.append("input file not found")

    if had_chunk_failure and (result.rows_inserted + result.rows_updated) == 0:
        result.status = IngestionStatus.FAILED
    elif had_chunk_failure:
        result.status = IngestionStatus.PARTIALLY_COMPLETED
    else:
        result.status = IngestionStatus.COMPLETED

    error_summary = "; ".join(result.error_reasons)[:_MAX_ERROR_SUMMARY_LENGTH] or None
    async with session_factory() as session:
        await session.execute(
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(
                status=result.status,
                completed_at=func.now(),
                rows_read=result.rows_read,
                rows_inserted=result.rows_inserted,
                rows_updated=result.rows_updated,
                rows_rejected=result.rows_rejected,
                error_summary=error_summary,
            )
        )
        await session.commit()

    return result
