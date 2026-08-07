"""Phase 3A: extensible, non-diagnostic clinical-context contract.

This module defines a generic structured representation of what a user has
explicitly said about a specific concern (onset, duration, location,
laterality, progression, severity, associated features, risk factors, and
explicit negative findings), plus a bounded, reviewable summary produced
once a complaint-specific protocol (see app.services.clinical_intake_service)
has collected what it needs.

Deliberately protocol-agnostic: nothing here names "leg swelling" or any
other specific complaint — a new protocol only needs a
ClinicalProtocol definition (app.services.clinical_intake_service), never a
change to this schema or to app/graph/. Every fact carries its own
confidence and source so provenance is always inspectable; nothing here is
ever invented — a field the user did not address stays None/absent and is
listed in `unknown_fields` instead of being guessed at.

Never a diagnosis, treatment recommendation, or urgency/emergency
classification: ClinicalNavigationSummary.diagnosis and
.treatment_recommendation are typed as always-None so that can never change
without a schema edit, and possible_explanations are broad, non-diagnostic
categories only (see app.services.clinical_intake_service's protocol
definitions for the actual reviewable category text).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

_MAX_FACT_VALUE_LENGTH = 300
_MAX_CATEGORY_LENGTH = 120


class FactConfidence(StrEnum):
    """How a ClinicalFact's value was obtained. Never a medical-certainty
    claim — purely about extraction provenance."""

    EXPLICIT = "explicit"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class FactSource(StrEnum):
    """Which channel produced a ClinicalFact — mirrors
    ConversationState's input_modality plus a clarification-specific value
    for a fact obtained by answering a follow-up question."""

    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    CLARIFICATION_ANSWER = "clarification_answer"


class MentionStatus(StrEnum):
    """Whether a specific symptom/risk-factor phrase was affirmed, negated,
    or never clearly addressed in a piece of user-supplied text. "unknown"
    covers both "not mentioned at all" and "mentioned with genuine
    uncertainty" — either way, never treated as a positive finding. See
    app.services.mention_classification_service.classify_mention."""

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNKNOWN = "unknown"


class ClinicalFact(BaseModel):
    """One user-supplied data point with its own provenance. Never
    fabricated: every ClinicalFact must trace back to something the user
    actually said (their own words, or a narrow deterministic derivation
    such as "for 3 days" -> a structured Duration, see
    app.services.duration_extraction)."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(max_length=_MAX_FACT_VALUE_LENGTH)
    confidence: FactConfidence
    source: FactSource


class ClinicalContext(BaseModel):
    """Generic, extensible clinical context for one active complaint
    protocol. Every field below is optional/empty by default — a protocol
    only ever populates what the user actually addressed, and
    `unknown_fields`/`answered_protocol_fields` make clear which of the
    protocol's own questions have and have not yet been resolved."""

    model_config = ConfigDict(extra="forbid")

    protocol: str

    primary_concern: ClinicalFact | None = None
    onset: ClinicalFact | None = None
    duration: ClinicalFact | None = None
    location: ClinicalFact | None = None
    laterality: ClinicalFact | None = None
    progression: ClinicalFact | None = None
    severity: ClinicalFact | None = None

    associated_features: list[ClinicalFact] = Field(default_factory=list)
    risk_factors: list[ClinicalFact] = Field(default_factory=list)
    negative_findings: list[ClinicalFact] = Field(default_factory=list)

    # Keyed by red-flag category name (see app.safety.red_flag_rules) ->
    # MentionStatus value. Empty until the protocol's red-flag question has
    # been asked and answered (from a clarification answer) or found in the
    # original message (see app.services.clinical_intake_service's initial
    # extraction pass).
    red_flags: dict[str, str] = Field(default_factory=dict)

    # Whether app.graph.nodes.clinical_red_flag_gate_node has actually run
    # for this context yet -- set True only by that node itself, once,
    # regardless of its outcome. This is the ONLY signal
    # clinical_intake_agent_node trusts to decide whether it is safe to
    # build a ClinicalNavigationSummary or continue to specialty/provider
    # routing: it is deliberately independent of *which* protocol question
    # was most recently answered, so red-flag information supplied in the
    # original message (before any question was ever asked) cannot bypass
    # the gate the way checking "was red_flags the last field answered"
    # alone would.
    red_flag_gate_passed: bool = False

    # Protocol question field names (see
    # app.services.clinical_intake_service.ProtocolQuestion.field) already
    # resolved -- either from the original message or a clarification
    # answer -- so the same question is never asked twice.
    answered_protocol_fields: list[str] = Field(default_factory=list)

    # Free-form notes on what remains genuinely unknown (e.g.
    # "local_symptoms:redness") -- surfaced verbatim in
    # ClinicalNavigationSummary.important_unknowns.
    unknown_fields: list[str] = Field(default_factory=list)


class CareLevel(StrEnum):
    """A non-diagnostic urgency-adjacent recommendation about *how soon* to
    seek routine care — never an emergency classification (that stays the
    dedicated red-flag gate's job, see app.safety.red_flag_rules) and never
    a diagnosis."""

    ROUTINE = "routine"
    PROMPT = "prompt"


class PossibleExplanationCategory(BaseModel):
    """One broad, non-diagnostic category of possible causes, plus which of
    the user's own answers (if any) support it. `description` text must
    never name a specific disease or condition as *the* cause — only a
    general category (see app.services.clinical_intake_service's protocol
    definitions, which are the single reviewable source of this text)."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(max_length=_MAX_CATEGORY_LENGTH)
    description: str
    supporting_evidence: list[str] = Field(default_factory=list)


class ClinicalNavigationSummary(BaseModel):
    """A bounded, reviewable result produced once a protocol has collected
    what it needs and no red flag was affirmed (see
    app.safety.red_flag_rules). `diagnosis` and `treatment_recommendation`
    are always None -- typed that way so this can never silently change to
    carry either without an explicit schema edit."""

    model_config = ConfigDict(extra="forbid")

    protocol: str
    summary_of_reported_information: str
    possible_explanations: list[PossibleExplanationCategory]
    important_unknowns: list[str] = Field(default_factory=list)
    recommended_care_level: CareLevel
    specialty_candidates: list[str]
    diagnosis: None = None
    treatment_recommendation: None = None
    disclaimer: str
