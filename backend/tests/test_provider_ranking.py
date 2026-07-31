"""Unit tests for deterministic provider-search ranking.

Pure Python: constructs ProviderSearchRow dataclasses directly, no
database, no FastAPI, no model API of any kind.
"""

from datetime import date

from app.repositories.provider_repository import ProviderSearchRow
from app.services.provider_ranking import provider_display_name, rank_provider_results


def _row(**overrides: object) -> ProviderSearchRow:
    defaults: dict[str, object] = {
        "npi": "1000000001",
        "entity_type_code": "1",
        "organization_name": None,
        "first_name": "Jane",
        "last_name": "Smith",
        "middle_name": None,
        "name_prefix": None,
        "name_suffix": None,
        "credential": None,
        "is_active": True,
        "last_update_date": date(2020, 1, 1),
        "taxonomy_code": "207Q00000X",
        "is_primary_taxonomy": True,
        "specialty_slug": "family-medicine",
        "specialty_display_name": "Family Medicine",
        "specialty_description": "Primary care across all ages.",
        "location_address_line_1": "456 Oak Ave",
        "location_address_line_2": None,
        "location_city": "Springfield",
        "location_state": "CA",
        "location_postal_code": "90002",
        "location_telephone_number": None,
        "location_address_purpose": "practice",
    }
    defaults.update(overrides)
    return ProviderSearchRow(**defaults)  # type: ignore[arg-type]


def test_provider_display_name_uses_organization_name_for_organizations() -> None:
    row = _row(organization_name="Springfield Clinic LLC", first_name=None, last_name=None)
    assert provider_display_name(row) == "Springfield Clinic LLC"


def test_provider_display_name_formats_individual_with_credential() -> None:
    row = _row(first_name="Jane", middle_name="Q", last_name="Smith", credential="MD")
    assert provider_display_name(row) == "Jane Q Smith, MD"


def test_rank_primary_taxonomy_before_non_primary() -> None:
    primary = _row(npi="1", is_primary_taxonomy=True)
    non_primary = _row(npi="2", is_primary_taxonomy=False)
    ranked = rank_provider_results([non_primary, primary])
    assert [r.npi for r in ranked] == ["1", "2"]


def test_rank_active_before_inactive() -> None:
    active = _row(npi="1", is_active=True)
    inactive = _row(npi="2", is_active=False)
    ranked = rank_provider_results([inactive, active])
    assert [r.npi for r in ranked] == ["1", "2"]


def test_rank_exact_specialty_match_first() -> None:
    matching = _row(npi="1", specialty_slug="cardiology")
    other = _row(npi="2", specialty_slug="dermatology")
    ranked = rank_provider_results([other, matching], specialty_slug="cardiology")
    assert [r.npi for r in ranked] == ["1", "2"]


def test_rank_exact_postal_code_match_first() -> None:
    matching = _row(npi="1", location_postal_code="90001")
    other = _row(npi="2", location_postal_code="10001")
    ranked = rank_provider_results([other, matching], postal_code="90001")
    assert [r.npi for r in ranked] == ["1", "2"]


def test_rank_exact_city_state_match_first() -> None:
    matching = _row(npi="1", location_city="Dallas", location_state="TX")
    other = _row(npi="2", location_city="Austin", location_state="TX")
    ranked = rank_provider_results([other, matching], city="Dallas", state="TX")
    assert [r.npi for r in ranked] == ["1", "2"]


def test_rank_city_match_is_case_insensitive() -> None:
    matching = _row(npi="1", location_city="dallas")
    ranked = rank_provider_results([matching], city="Dallas")
    assert [r.npi for r in ranked] == ["1"]


def test_rank_stable_tiebreak_by_name_then_npi() -> None:
    row_b = _row(npi="2", last_name="Brown", first_name="Amy")
    row_a = _row(npi="1", last_name="Adams", first_name="Amy")
    ranked = rank_provider_results([row_b, row_a])
    assert [r.npi for r in ranked] == ["1", "2"]


def test_rank_tiebreak_by_npi_when_names_equal() -> None:
    row_high = _row(npi="9999999999", last_name="Smith", first_name="Jane")
    row_low = _row(npi="1000000000", last_name="Smith", first_name="Jane")
    ranked = rank_provider_results([row_high, row_low])
    assert [r.npi for r in ranked] == ["1000000000", "9999999999"]


def test_rank_is_deterministic_across_multiple_calls() -> None:
    rows = [
        _row(npi="3", is_primary_taxonomy=False),
        _row(npi="1", is_primary_taxonomy=True),
        _row(npi="2", is_active=False),
    ]
    first = [r.npi for r in rank_provider_results(rows)]
    second = [r.npi for r in rank_provider_results(rows)]
    assert first == second


def test_rank_does_not_mutate_input_list() -> None:
    rows = [_row(npi="2"), _row(npi="1")]
    original_order = [r.npi for r in rows]
    rank_provider_results(rows)
    assert [r.npi for r in rows] == original_order


def test_rank_neutral_when_no_filters_given() -> None:
    row_b = _row(npi="2", specialty_slug="dermatology", location_postal_code="10001")
    row_a = _row(npi="1", specialty_slug="cardiology", location_postal_code="90001")
    # With no specialty/postal/city/state filters, only active + name/NPI tie-break apply.
    ranked = rank_provider_results([row_b, row_a])
    assert [r.npi for r in ranked] == [r.npi for r in [row_a, row_b]]
