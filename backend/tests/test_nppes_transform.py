import pytest

from app.ingestion import nppes_mapping as col
from app.ingestion.nppes_transform import (
    InvalidNPPESRowError,
    blank_to_none,
    extract_addresses,
    extract_taxonomies,
    normalize_country_code,
    normalize_postal_code,
    normalize_state,
    parse_nppes_date,
    transform_row,
    validate_npi,
    validate_required_columns,
)


def _base_row(**overrides: str) -> dict[str, str]:
    row = {
        col.COL_NPI: "1234567890",
        col.COL_ENTITY_TYPE_CODE: "1",
        col.COL_LAST_NAME: "Smith",
        col.COL_FIRST_NAME: "Jane",
    }
    row.update(overrides)
    return row


def test_blank_to_none_treats_blank_and_whitespace_as_none() -> None:
    assert blank_to_none("") is None
    assert blank_to_none("   ") is None
    assert blank_to_none(None) is None
    assert blank_to_none(" hello ") == "hello"


def test_validate_required_columns_raises_on_missing_column() -> None:
    with pytest.raises(InvalidNPPESRowError):
        validate_required_columns({col.COL_NPI: "1234567890"})


def test_validate_required_columns_passes_when_present() -> None:
    validate_required_columns({col.COL_NPI: "1234567890", col.COL_ENTITY_TYPE_CODE: "1"})


def test_validate_npi_accepts_exactly_ten_digits() -> None:
    assert validate_npi("1234567890") == "1234567890"


def test_validate_npi_preserves_leading_zeros() -> None:
    assert validate_npi("0123456789") == "0123456789"


@pytest.mark.parametrize("value", ["12345", "12345678901", "123456789A", "", None, "  "])
def test_validate_npi_rejects_invalid_values(value: str | None) -> None:
    with pytest.raises(InvalidNPPESRowError):
        validate_npi(value)


def test_parse_nppes_date_parses_valid_date() -> None:
    from datetime import date

    assert parse_nppes_date("01/15/2010") == date(2010, 1, 15)


def test_parse_nppes_date_returns_none_for_blank() -> None:
    assert parse_nppes_date("") is None
    assert parse_nppes_date(None) is None


def test_parse_nppes_date_rejects_malformed_date() -> None:
    with pytest.raises(InvalidNPPESRowError):
        parse_nppes_date("2010-01-15")


def test_normalize_state_uppercases_and_trims() -> None:
    assert normalize_state(" ca ") == "CA"
    assert normalize_state("") is None


def test_normalize_country_code_defaults_to_us_when_blank() -> None:
    assert normalize_country_code("") == "US"
    assert normalize_country_code(None) == "US"
    assert normalize_country_code(" gb ") == "GB"


def test_normalize_postal_code_preserves_leading_zero() -> None:
    assert normalize_postal_code("00501") == "00501"
    assert normalize_postal_code(" 90001 ") == "90001"
    assert normalize_postal_code("") is None


def test_transform_row_individual_provider() -> None:
    row = _base_row(**{col.COL_CREDENTIAL: "MD", col.COL_GENDER_CODE: "F"})
    provider = transform_row(row)
    assert provider.npi == "1234567890"
    assert provider.entity_type_code == col.ENTITY_TYPE_INDIVIDUAL
    assert provider.first_name == "Jane"
    assert provider.last_name == "Smith"
    assert provider.organization_name is None
    assert provider.credential == "MD"


def test_transform_row_organization_provider() -> None:
    row = {
        col.COL_NPI: "9876543210",
        col.COL_ENTITY_TYPE_CODE: "2",
        col.COL_ORGANIZATION_NAME: "Springfield Clinic LLC",
    }
    provider = transform_row(row)
    assert provider.entity_type_code == col.ENTITY_TYPE_ORGANIZATION
    assert provider.organization_name == "Springfield Clinic LLC"
    assert provider.first_name is None
    assert provider.last_name is None


def test_transform_row_rejects_invalid_npi() -> None:
    with pytest.raises(InvalidNPPESRowError):
        transform_row(_base_row(**{col.COL_NPI: "123"}))


def test_transform_row_rejects_unsupported_entity_type() -> None:
    with pytest.raises(InvalidNPPESRowError):
        transform_row(_base_row(**{col.COL_ENTITY_TYPE_CODE: "3"}))


def test_transform_row_error_message_is_bounded_and_has_no_row_dump() -> None:
    row = _base_row(**{col.COL_NPI: "not-a-valid-npi-value-at-all"})
    with pytest.raises(InvalidNPPESRowError) as exc_info:
        transform_row(row)
    message = str(exc_info.value)
    assert len(message) < 200
    assert "Smith" not in message
    assert "Jane" not in message


def test_extract_addresses_returns_both_when_present() -> None:
    row = {
        col.COL_MAILING_ADDRESS_1: "123 Main St",
        col.COL_MAILING_CITY: "Springfield",
        col.COL_MAILING_STATE: "CA",
        col.COL_MAILING_POSTAL_CODE: "90001",
        col.COL_PRACTICE_ADDRESS_1: "456 Oak Ave",
        col.COL_PRACTICE_CITY: "Springfield",
        col.COL_PRACTICE_STATE: "CA",
        col.COL_PRACTICE_POSTAL_CODE: "90002",
    }
    addresses = extract_addresses(row)
    purposes = {a.address_purpose for a in addresses}
    assert purposes == {"mailing", "practice"}


def test_extract_addresses_skips_entirely_blank_purpose() -> None:
    row = {
        col.COL_PRACTICE_ADDRESS_1: "456 Oak Ave",
        col.COL_PRACTICE_CITY: "Springfield",
        col.COL_PRACTICE_STATE: "CA",
        col.COL_PRACTICE_POSTAL_CODE: "90002",
    }
    addresses = extract_addresses(row)
    assert len(addresses) == 1
    assert addresses[0].address_purpose == "practice"


def test_extract_addresses_preserves_leading_zero_postal_code() -> None:
    row = {
        col.COL_PRACTICE_ADDRESS_1: "10 Holtsville Rd",
        col.COL_PRACTICE_CITY: "Holtsville",
        col.COL_PRACTICE_STATE: "NY",
        col.COL_PRACTICE_POSTAL_CODE: "00501",
    }
    addresses = extract_addresses(row)
    assert addresses[0].postal_code == "00501"


def test_extract_taxonomies_supports_multiple_slots() -> None:
    row = {
        col.taxonomy_code_column(1): "207Q00000X",
        col.taxonomy_primary_switch_column(1): "Y",
        col.taxonomy_code_column(2): "208D00000X",
        col.taxonomy_primary_switch_column(2): "N",
    }
    taxonomies = extract_taxonomies(row)
    codes = {t.taxonomy_code for t in taxonomies}
    assert codes == {"207Q00000X", "208D00000X"}
    primary_codes = {t.taxonomy_code for t in taxonomies if t.is_primary}
    assert primary_codes == {"207Q00000X"}


def test_extract_taxonomies_deduplicates_repeated_code_across_slots() -> None:
    row = {
        col.taxonomy_code_column(1): "207Q00000X",
        col.taxonomy_code_column(2): "207Q00000X",
    }
    taxonomies = extract_taxonomies(row)
    assert len(taxonomies) == 1


def test_extract_taxonomies_empty_when_no_slots_populated() -> None:
    assert extract_taxonomies({}) == ()
