"""Mapping from NPPES Data Dissemination File column names to internal fields.

This intentionally covers only the columns Phase 1A needs. The real national
NPPES file has additional columns (e.g. "Is Sole Proprietor", "Authorized
Official" fields, "Other Provider Identifier" blocks) that are not mapped
here. Phase 1A does not claim full-file coverage — only the identity,
address, and taxonomy fields required by the `providers` /
`provider_locations` / `provider_taxonomies` schema.

Column names below match the NPPES Data Dissemination File header exactly.
"""

# Identity columns.
COL_NPI = "NPI"
COL_ENTITY_TYPE_CODE = "Entity Type Code"
COL_ORGANIZATION_NAME = "Provider Organization Name (Legal Business Name)"
COL_LAST_NAME = "Provider Last Name (Legal Name)"
COL_FIRST_NAME = "Provider First Name"
COL_MIDDLE_NAME = "Provider Middle Name"
COL_NAME_PREFIX = "Provider Name Prefix Text"
COL_NAME_SUFFIX = "Provider Name Suffix Text"
COL_CREDENTIAL = "Provider Credential Text"
COL_GENDER_CODE = "Provider Gender Code"
COL_ENUMERATION_DATE = "Provider Enumeration Date"
COL_LAST_UPDATE_DATE = "Last Update Date"
COL_DEACTIVATION_DATE = "NPI Deactivation Date"
COL_REACTIVATION_DATE = "NPI Reactivation Date"
COL_REPLACEMENT_NPI = "Replacement NPI"

# Required columns: rows missing any of these are structurally invalid.
REQUIRED_COLUMNS = (COL_NPI, COL_ENTITY_TYPE_CODE)

ENTITY_TYPE_INDIVIDUAL = "1"
ENTITY_TYPE_ORGANIZATION = "2"

# Mailing address columns.
COL_MAILING_ADDRESS_1 = "Provider First Line Business Mailing Address"
COL_MAILING_ADDRESS_2 = "Provider Second Line Business Mailing Address"
COL_MAILING_CITY = "Provider Business Mailing Address City Name"
COL_MAILING_STATE = "Provider Business Mailing Address State Name"
COL_MAILING_POSTAL_CODE = "Provider Business Mailing Address Postal Code"
COL_MAILING_COUNTRY_CODE = "Provider Business Mailing Address Country Code (If outside U.S.)"
COL_MAILING_TELEPHONE = "Provider Business Mailing Address Telephone Number"
COL_MAILING_FAX = "Provider Business Mailing Address Fax Number"

# Practice-location address columns.
COL_PRACTICE_ADDRESS_1 = "Provider First Line Business Practice Location Address"
COL_PRACTICE_ADDRESS_2 = "Provider Second Line Business Practice Location Address"
COL_PRACTICE_CITY = "Provider Business Practice Location Address City Name"
COL_PRACTICE_STATE = "Provider Business Practice Location Address State Name"
COL_PRACTICE_POSTAL_CODE = "Provider Business Practice Location Address Postal Code"
COL_PRACTICE_COUNTRY_CODE = (
    "Provider Business Practice Location Address Country Code (If outside U.S.)"
)
COL_PRACTICE_TELEPHONE = "Provider Business Practice Location Address Telephone Number"
COL_PRACTICE_FAX = "Provider Business Practice Location Address Fax Number"

# Taxonomy slots. The real national file has 15 slots (_1.._15); Phase 1A
# reads a smaller, documented, easily-extendable number of slots, validated
# against the fixture rather than the full file.
MAX_TAXONOMY_SLOTS = 3


def taxonomy_code_column(slot: int) -> str:
    return f"Healthcare Provider Taxonomy Code_{slot}"


def taxonomy_license_number_column(slot: int) -> str:
    return f"Provider License Number_{slot}"


def taxonomy_license_state_column(slot: int) -> str:
    return f"Provider License Number State Code_{slot}"


def taxonomy_primary_switch_column(slot: int) -> str:
    return f"Healthcare Provider Primary Taxonomy Switch_{slot}"
