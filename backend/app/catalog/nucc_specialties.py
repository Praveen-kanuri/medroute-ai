"""MedRoute AI's small, transparent supported-specialty catalog.

Source: National Uniform Claim Committee (NUCC) Health Care Provider
Taxonomy Code Set, version 26.1 (effective 2026-07-01).
Reference: https://www.nucc.org/index.php/code-sets-mainmenu-41/provider-taxonomy-mainmenu-40
Cross-checked against: https://www.findacode.com/tools/taxonomy-codes.html,
https://npiprofile.com/taxonomy/code/208600000X,
https://npiprofile.com/taxonomy/code/2085R0202X
Accessed: 2026-07-29.

This is a deliberately small, curated starting set — not the full ~880-code
NUCC taxonomy. A taxonomy code with no entry here is simply unmapped (no
specialty shown for it); it is not an error.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialtySeed:
    slug: str
    display_name: str
    description: str
    taxonomy_code: str
    taxonomy_description: str
    # Plain-language keywords used only by the deterministic (non-LLM)
    # Phase 1D routing fallback — never persisted to the database, never a
    # medical claim, just a transparent keyword-overlap signal.
    keywords: tuple[str, ...] = ()


SPECIALTY_SEEDS: tuple[SpecialtySeed, ...] = (
    SpecialtySeed(
        slug="family-medicine",
        display_name="Family Medicine",
        description="Primary care across all ages.",
        taxonomy_code="207Q00000X",
        taxonomy_description="Allopathic & Osteopathic Physicians / Family Medicine",
        keywords=("checkup", "primary", "general", "family", "flu", "cold", "fever"),
    ),
    SpecialtySeed(
        slug="internal-medicine",
        display_name="Internal Medicine",
        description="Adult primary and consultative medicine.",
        taxonomy_code="207R00000X",
        taxonomy_description="Allopathic & Osteopathic Physicians / Internal Medicine",
        keywords=("fatigue", "diabetes", "adult", "chronic", "internal"),
    ),
    SpecialtySeed(
        slug="cardiology",
        display_name="Cardiology",
        description="Diagnosis and treatment of heart and vascular conditions.",
        taxonomy_code="207RC0000X",
        taxonomy_description=(
            "Allopathic & Osteopathic Physicians / Internal Medicine, Cardiovascular Disease"
        ),
        keywords=(
            "heart",
            "chest",
            "palpitations",
            "cardiac",
            "cardiovascular",
            "blood",
            "pressure",
        ),
    ),
    SpecialtySeed(
        slug="pediatrics",
        display_name="Pediatrics",
        description="Medical care for infants, children, and adolescents.",
        taxonomy_code="208000000X",
        taxonomy_description="Allopathic & Osteopathic Physicians / Pediatrics",
        keywords=("child", "infant", "baby", "kid", "adolescent", "pediatric"),
    ),
    SpecialtySeed(
        slug="psychiatry",
        display_name="Psychiatry",
        description="Diagnosis and treatment of mental health conditions.",
        taxonomy_code="2084P0800X",
        taxonomy_description=(
            "Allopathic & Osteopathic Physicians / Psychiatry & Neurology, Psychiatry"
        ),
        keywords=("anxiety", "depression", "mood", "mental", "stress", "sleep", "psychiatric"),
    ),
    SpecialtySeed(
        slug="dermatology",
        display_name="Dermatology",
        description="Diagnosis and treatment of skin, hair, and nail conditions.",
        taxonomy_code="207N00000X",
        taxonomy_description="Allopathic & Osteopathic Physicians / Dermatology",
        keywords=("skin", "rash", "acne", "mole", "itch", "hair", "nail", "dermatology"),
    ),
    SpecialtySeed(
        slug="orthopaedic-surgery",
        display_name="Orthopaedic Surgery",
        description="Surgical and non-surgical care of the musculoskeletal system.",
        taxonomy_code="207X00000X",
        taxonomy_description="Allopathic & Osteopathic Physicians / Orthopaedic Surgery",
        keywords=(
            "knee",
            "joint",
            "bone",
            "fracture",
            "back",
            "shoulder",
            "sprain",
            "orthopaedic",
            "orthopedic",
        ),
    ),
    SpecialtySeed(
        slug="obstetrics-gynecology",
        display_name="Obstetrics & Gynecology",
        description="Women's reproductive health, pregnancy, and childbirth.",
        taxonomy_code="207V00000X",
        taxonomy_description="Allopathic & Osteopathic Physicians / Obstetrics & Gynecology",
        keywords=("pregnancy", "pregnant", "gynecology", "obstetrics", "menstrual"),
    ),
    SpecialtySeed(
        slug="general-surgery",
        display_name="General Surgery",
        description="Surgical treatment of a broad range of conditions.",
        taxonomy_code="208600000X",
        taxonomy_description="Allopathic & Osteopathic Physicians / Surgery",
        keywords=("surgery", "surgical", "hernia", "appendix", "gallbladder"),
    ),
    SpecialtySeed(
        slug="diagnostic-radiology",
        display_name="Diagnostic Radiology",
        description="Medical imaging for diagnosis.",
        taxonomy_code="2085R0202X",
        taxonomy_description=(
            "Allopathic & Osteopathic Physicians / Radiology, Diagnostic Radiology"
        ),
        keywords=("xray", "x-ray", "scan", "imaging", "mri", "ct", "ultrasound", "radiology"),
    ),
)
