"""Pure transformation and validation of a single NPPES CSV row.

No I/O and no database access here — this module only turns a raw CSV row
(mapping of column name to string) into normalized, typed data, or raises
InvalidNPPESRowError for structurally invalid rows. Kept separate from
nppes_service.py so it is trivially unit-testable.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

from app.ingestion import nppes_mapping as col

_NPPES_DATE_FORMAT = "%m/%d/%Y"


class InvalidNPPESRowError(ValueError):
    """Raised when a CSV row is structurally invalid and must be rejected."""


@dataclass(frozen=True)
class NormalizedAddress:
    address_purpose: str  # "mailing" or "practice"
    address_line_1: str | None
    address_line_2: str | None
    city: str | None
    state: str | None
    postal_code: str | None
    country_code: str | None
    telephone_number: str | None
    fax_number: str | None


@dataclass(frozen=True)
class NormalizedTaxonomy:
    taxonomy_code: str
    license_number: str | None
    license_state: str | None
    is_primary: bool


@dataclass(frozen=True)
class NormalizedProvider:
    npi: str
    entity_type_code: str
    organization_name: str | None
    first_name: str | None
    last_name: str | None
    middle_name: str | None
    name_prefix: str | None
    name_suffix: str | None
    credential: str | None
    gender_code: str | None
    enumeration_date: date | None
    last_update_date: date | None
    deactivation_date: date | None
    reactivation_date: date | None
    replacement_npi: str | None
    addresses: tuple[NormalizedAddress, ...]
    taxonomies: tuple[NormalizedTaxonomy, ...]


def blank_to_none(value: str | None) -> str | None:
    """Trim whitespace; treat blank CSV values consistently as None."""
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def validate_required_columns(row: Mapping[str, str]) -> None:
    missing = [name for name in col.REQUIRED_COLUMNS if name not in row]
    if missing:
        raise InvalidNPPESRowError(f"missing required column(s): {', '.join(missing)}")


def validate_npi(value: str | None) -> str:
    """Validate NPI as exactly 10 numeric digits. Leading zeros are preserved."""
    npi = blank_to_none(value)
    if npi is None or len(npi) != 10 or not npi.isdigit():
        raise InvalidNPPESRowError("NPI must be exactly 10 numeric digits")
    return npi


def parse_nppes_date(value: str | None) -> date | None:
    """Parse an NPPES MM/DD/YYYY date column. Blank values are None."""
    text = blank_to_none(value)
    if text is None:
        return None
    try:
        return datetime.strptime(text, _NPPES_DATE_FORMAT).date()
    except ValueError as exc:
        raise InvalidNPPESRowError(f"invalid date value: {text!r}") from exc


def normalize_state(value: str | None) -> str | None:
    text = blank_to_none(value)
    return text.upper() if text is not None else None


def normalize_country_code(value: str | None, *, default: str = "US") -> str | None:
    text = blank_to_none(value)
    return text.upper() if text is not None else default


def normalize_postal_code(value: str | None) -> str | None:
    """Trim whitespace only. Never coerce to a number — leading zeros matter."""
    return blank_to_none(value)


def _extract_address(
    row: Mapping[str, str], *, purpose: str, prefix_columns: dict[str, str]
) -> NormalizedAddress | None:
    address_line_1 = blank_to_none(row.get(prefix_columns["address_1"]))
    address_line_2 = blank_to_none(row.get(prefix_columns["address_2"]))
    city = blank_to_none(row.get(prefix_columns["city"]))
    state = normalize_state(row.get(prefix_columns["state"]))
    postal_code = normalize_postal_code(row.get(prefix_columns["postal_code"]))
    telephone_number = blank_to_none(row.get(prefix_columns["telephone"]))
    fax_number = blank_to_none(row.get(prefix_columns["fax"]))

    if not any((address_line_1, city, state, postal_code)):
        return None

    country_code = normalize_country_code(row.get(prefix_columns["country_code"]))
    return NormalizedAddress(
        address_purpose=purpose,
        address_line_1=address_line_1,
        address_line_2=address_line_2,
        city=city,
        state=state,
        postal_code=postal_code,
        country_code=country_code,
        telephone_number=telephone_number,
        fax_number=fax_number,
    )


def extract_addresses(row: Mapping[str, str]) -> tuple[NormalizedAddress, ...]:
    """Extract mailing and/or practice addresses. Skips a purpose entirely if blank."""
    mailing = _extract_address(
        row,
        purpose="mailing",
        prefix_columns={
            "address_1": col.COL_MAILING_ADDRESS_1,
            "address_2": col.COL_MAILING_ADDRESS_2,
            "city": col.COL_MAILING_CITY,
            "state": col.COL_MAILING_STATE,
            "postal_code": col.COL_MAILING_POSTAL_CODE,
            "country_code": col.COL_MAILING_COUNTRY_CODE,
            "telephone": col.COL_MAILING_TELEPHONE,
            "fax": col.COL_MAILING_FAX,
        },
    )
    practice = _extract_address(
        row,
        purpose="practice",
        prefix_columns={
            "address_1": col.COL_PRACTICE_ADDRESS_1,
            "address_2": col.COL_PRACTICE_ADDRESS_2,
            "city": col.COL_PRACTICE_CITY,
            "state": col.COL_PRACTICE_STATE,
            "postal_code": col.COL_PRACTICE_POSTAL_CODE,
            "country_code": col.COL_PRACTICE_COUNTRY_CODE,
            "telephone": col.COL_PRACTICE_TELEPHONE,
            "fax": col.COL_PRACTICE_FAX,
        },
    )
    return tuple(address for address in (mailing, practice) if address is not None)


def extract_taxonomies(
    row: Mapping[str, str], *, max_slots: int = col.MAX_TAXONOMY_SLOTS
) -> tuple[NormalizedTaxonomy, ...]:
    """Extract every populated taxonomy slot, not just the first.

    De-duplicates by taxonomy code (keeping the last slot) in case the same
    code appears in more than one slot for a provider — the database's
    (provider_id, taxonomy_code) natural key allows only one row per code.
    """
    by_code: dict[str, NormalizedTaxonomy] = {}
    for slot in range(1, max_slots + 1):
        code = blank_to_none(row.get(col.taxonomy_code_column(slot)))
        if code is None:
            continue
        license_number = blank_to_none(row.get(col.taxonomy_license_number_column(slot)))
        license_state = normalize_state(row.get(col.taxonomy_license_state_column(slot)))
        primary_switch = blank_to_none(row.get(col.taxonomy_primary_switch_column(slot)))
        by_code[code] = NormalizedTaxonomy(
            taxonomy_code=code,
            license_number=license_number,
            license_state=license_state,
            is_primary=(primary_switch or "").upper() == "Y",
        )
    return tuple(by_code.values())


def transform_row(row: Mapping[str, str]) -> NormalizedProvider:
    """Transform and validate one raw CSV row. Raises InvalidNPPESRowError if invalid."""
    validate_required_columns(row)

    entity_type_code = blank_to_none(row.get(col.COL_ENTITY_TYPE_CODE))
    if entity_type_code not in (col.ENTITY_TYPE_INDIVIDUAL, col.ENTITY_TYPE_ORGANIZATION):
        raise InvalidNPPESRowError(f"unsupported entity type code: {entity_type_code!r}")

    npi = validate_npi(row.get(col.COL_NPI))

    return NormalizedProvider(
        npi=npi,
        entity_type_code=entity_type_code,
        organization_name=blank_to_none(row.get(col.COL_ORGANIZATION_NAME)),
        first_name=blank_to_none(row.get(col.COL_FIRST_NAME)),
        last_name=blank_to_none(row.get(col.COL_LAST_NAME)),
        middle_name=blank_to_none(row.get(col.COL_MIDDLE_NAME)),
        name_prefix=blank_to_none(row.get(col.COL_NAME_PREFIX)),
        name_suffix=blank_to_none(row.get(col.COL_NAME_SUFFIX)),
        credential=blank_to_none(row.get(col.COL_CREDENTIAL)),
        gender_code=blank_to_none(row.get(col.COL_GENDER_CODE)),
        enumeration_date=parse_nppes_date(row.get(col.COL_ENUMERATION_DATE)),
        last_update_date=parse_nppes_date(row.get(col.COL_LAST_UPDATE_DATE)),
        deactivation_date=parse_nppes_date(row.get(col.COL_DEACTIVATION_DATE)),
        reactivation_date=parse_nppes_date(row.get(col.COL_REACTIVATION_DATE)),
        replacement_npi=blank_to_none(row.get(col.COL_REPLACEMENT_NPI)),
        addresses=extract_addresses(row),
        taxonomies=extract_taxonomies(row),
    )
