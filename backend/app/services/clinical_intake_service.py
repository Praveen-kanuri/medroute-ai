"""Phase 3A: extensible, per-complaint clinical-context intake.

A `ClinicalProtocol` bundles everything one complaint needs: how to detect
it from free text, which follow-up questions to ask (in order, one at a
time), and how to turn collected answers into a bounded, non-diagnostic
`ClinicalNavigationSummary`. Adding a new protocol (a new complaint) never
requires changing app/graph/ — it only ever needs a new `ClinicalProtocol`
entry in `CLINICAL_PROTOCOLS` below (see
app.graph.nodes.clinical_intake_agent_node, which is entirely
protocol-agnostic).

Currently only one protocol is defined: unilateral/one-sided leg swelling.
Every extraction here is deterministic keyword/negation matching (see
app.services.mention_classification_service) — never a model call, never a
guess. A field the user never addressed stays unset and is recorded in
`ClinicalContext.unknown_fields` instead.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from app.safety.red_flag_rules import assess_red_flags
from app.schemas.clinical_context import (
    CareLevel,
    ClinicalContext,
    ClinicalFact,
    ClinicalNavigationSummary,
    FactConfidence,
    FactSource,
    MentionStatus,
    PossibleExplanationCategory,
)
from app.services.mention_classification_service import classify_mention

_TOKEN_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ProtocolQuestion:
    """One follow-up question a protocol asks, one at a time, only when
    `field` is not already in ClinicalContext.answered_protocol_fields."""

    field: str
    prompt: str


@dataclass(frozen=True)
class ClinicalProtocol:
    """A single complaint's detection rule, ordered questions, and
    candidate specialties. `candidate_specialty_slugs` must only ever
    contain slugs already present in app.catalog.nucc_specialties —
    see test_clinical_intake_service.py's catalog-validation test."""

    name: str
    display_name: str
    primary_concern_label: str
    location_label: str | None
    candidate_specialty_slugs: tuple[str, ...]
    questions: tuple[ProtocolQuestion, ...]
    detect: Callable[[str], bool]


# --- leg-swelling protocol detection and extraction -------------------------

_LEG_SWELLING_LOCATION_TOKENS = frozenset({"leg", "legs", "calf", "calves", "ankle", "ankles"})
_LEG_SWELLING_SYMPTOM_TOKENS = frozenset(
    {"swelling", "swollen", "swell", "swells", "edema", "puffy", "puffiness"}
)


def _detect_leg_swelling(text: str) -> bool:
    """This protocol is specifically for *unilateral* (one-sided) leg
    swelling — explicit bilateral swelling ("both legs", "bilateral") is a
    different clinical picture with different possible explanations and
    must never be processed through this protocol's unilateral-framed
    questions/summary. Unspecified laterality ("my leg is swollen", no
    left/right/bilateral/one-sided stated) also does not enter this
    protocol — it is left to the existing non-protocol specialty-routing
    path (a smaller, safer change than adding a blocking laterality
    question to every other existing test/flow)."""
    tokens = set(_tokenize(text))
    has_location = bool(tokens & _LEG_SWELLING_LOCATION_TOKENS)
    has_symptom = bool(tokens & _LEG_SWELLING_SYMPTOM_TOKENS)
    if not (has_location and has_symptom):
        return False
    return _extract_laterality(text) in ("left", "right", "one-sided")


_LATERALITY_ANCHOR_TOKENS = _LEG_SWELLING_LOCATION_TOKENS | {"side", "sided"}


def _mentioned_as_laterality(tokens: list[str], word: str) -> bool:
    """Whether `word` ("left"/"right") appears immediately next to an
    anatomical anchor token (leg/calf/ankle/side/...), e.g. "left leg" or
    "swelling on the right side". Plain proximity anywhere in the sentence
    is deliberately NOT enough: free/voice-dictated speech is full of
    "right"/"left" used as filler or tag-question ("Right?", "left it at
    that") with nothing to do with anatomy, and the whole point of this
    protocol is unilateral (one-sided) leg swelling specifically — a
    misread laterality would fabricate a clinical detail the user never
    reported (see this module's tests for the transcript that motivated
    this)."""
    for i, token in enumerate(tokens):
        if token != word:
            continue
        neighbors = tokens[max(0, i - 1) : i] + tokens[i + 1 : i + 2]
        if any(neighbor in _LATERALITY_ANCHOR_TOKENS for neighbor in neighbors):
            return True
    return False


def _extract_laterality(text: str) -> str | None:
    tokens = _tokenize(text)
    token_set = set(tokens)
    if "bilateral" in token_set or "both" in token_set:
        return "bilateral"
    has_left = _mentioned_as_laterality(tokens, "left")
    has_right = _mentioned_as_laterality(tokens, "right")
    if has_left and has_right:
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    if "unilateral" in token_set or ("one" in token_set and "sided" in token_set):
        return "one-sided"
    return None


_LOCAL_SYMPTOM_PHRASES: dict[str, list[str]] = {
    "pain": ["pain", "aching", "ache", "hurts", "hurting"],
    "redness": ["redness", "red", "reddish"],
    "warmth": ["warm", "warmth"],
    "tenderness": ["tender", "tenderness", "sore"],
}

_RISK_FACTOR_PHRASES: dict[str, list[str]] = {
    "recent_injury": ["injury", "injured", "trauma"],
    "recent_surgery": ["surgery", "surgical", "operation"],
    "prolonged_travel": ["travel", "flight", "long drive", "road trip"],
    "immobility": [
        "immobile",
        "immobility",
        "bed rest",
        "cast",
        "sitting for a long time",
        "sitting for hours",
    ],
    "pregnancy_or_postpartum": ["pregnant", "pregnancy", "postpartum"],
    "previous_clot_history": [
        "blood clot",
        "dvt",
        "clot before",
        "previous clot",
        "history of clots",
    ],
}


_PROGRESSION_PHRASES: dict[str, list[str]] = {
    "worsening": ["worse", "worsening", "worsened"],
    "improving": ["better", "improving", "improved"],
    "stable": ["stable", "unchanged"],
}


def _handle_onset(
    context: ClinicalContext, text: str, source: FactSource, is_direct_answer: bool
) -> tuple[ClinicalContext, bool]:
    lowered = text.lower()
    found = False
    if "sudden" in lowered:
        context.onset = ClinicalFact(
            value="sudden", confidence=FactConfidence.EXPLICIT, source=source
        )
        found = True
    elif "gradual" in lowered:
        context.onset = ClinicalFact(
            value="gradual", confidence=FactConfidence.EXPLICIT, source=source
        )
        found = True
    elif is_direct_answer and text.strip():
        context.onset = ClinicalFact(
            value=text.strip()[:300], confidence=FactConfidence.EXPLICIT, source=source
        )
        found = True

    # Negation-aware, unlike a plain substring check: "not worse"/"hasn't
    # worsened" must never be recorded as progression=worsening (a real bug
    # this replaced -- a naive "wors" in lowered matched even "not worsted",
    # asserting the opposite of what the user said in the final summary).
    for label, phrases in _PROGRESSION_PHRASES.items():
        status = classify_mention(text, phrases, allow_blanket_negation=False)
        if status == MentionStatus.AFFIRMED:
            context.progression = ClinicalFact(
                value=label, confidence=FactConfidence.DERIVED, source=source
            )
            found = True
            break

    return context, found


def _handle_local_symptoms(
    context: ClinicalContext, text: str, source: FactSource, is_direct_answer: bool
) -> tuple[ClinicalContext, bool]:
    found_any = False
    for label, phrases in _LOCAL_SYMPTOM_PHRASES.items():
        status = classify_mention(text, phrases)
        if status == MentionStatus.AFFIRMED:
            context.associated_features.append(
                ClinicalFact(value=label, confidence=FactConfidence.EXPLICIT, source=source)
            )
            found_any = True
        elif status == MentionStatus.NEGATED:
            context.negative_findings.append(
                ClinicalFact(value=label, confidence=FactConfidence.EXPLICIT, source=source)
            )
            found_any = True
    if not found_any:
        context.unknown_fields.append("local_symptoms")
    return context, found_any


_RED_FLAG_CATEGORIES = ("chest_pain", "difficulty_breathing", "fainting", "coughing_blood")

_RED_FLAG_CATEGORY_LABELS: dict[str, str] = {
    "chest_pain": "chest pain or discomfort",
    "difficulty_breathing": "difficulty breathing or shortness of breath",
    "fainting": "fainting",
    "coughing_blood": "coughing up blood",
}

_RED_FLAG_STATUS_PRECEDENCE = {
    MentionStatus.UNKNOWN.value: 0,
    MentionStatus.NEGATED.value: 1,
    MentionStatus.AFFIRMED.value: 2,
}


def _merge_red_flags_dict(existing: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    """Merge a fresh per-category red-flag assessment into the existing
    one, category by category, by precedence (affirmed > negated >
    unknown) — never wholesale overwrite. An existing AFFIRMED is never
    downgraded, and a new UNKNOWN never overwrites a prior explicit
    NEGATED/AFFIRMED for that same category: previously collected
    information is never discarded."""
    merged = dict(existing)
    for category, new_status in new.items():
        current_status = merged.get(category, MentionStatus.UNKNOWN.value)
        if _RED_FLAG_STATUS_PRECEDENCE[new_status] > _RED_FLAG_STATUS_PRECEDENCE[current_status]:
            merged[category] = new_status
    return merged


def _red_flags_resolved(flags: dict[str, str]) -> bool:
    """Whether the compound red_flags field can be considered fully
    addressed and safe to mark answered: either at least one of the four
    categories is explicitly AFFIRMED (which always demands immediate
    gate evaluation on its own, regardless of the other three), or every
    one of the four has been explicitly resolved (affirmed or negated).
    A *partial* negative result alone (e.g. only chest pain addressed) is
    never enough — the remaining unknown categories must still be asked
    about, via a dedicated question narrowed to just those (see
    red_flags_question_text)."""
    statuses = [
        flags.get(category, MentionStatus.UNKNOWN.value) for category in _RED_FLAG_CATEGORIES
    ]
    if any(status == MentionStatus.AFFIRMED.value for status in statuses):
        return True
    return all(status != MentionStatus.UNKNOWN.value for status in statuses)


def _remaining_red_flag_categories(flags: dict[str, str]) -> list[str]:
    return [
        category
        for category in _RED_FLAG_CATEGORIES
        if flags.get(category, MentionStatus.UNKNOWN.value) == MentionStatus.UNKNOWN.value
    ]


def red_flags_question_text(context: ClinicalContext) -> str:
    """The red-flags question to ask next. Narrowed to name only the
    categories still unknown once some (but not all) have already been
    resolved from an earlier partial statement (the initial message, or
    an answer to a different question) — the user is never re-asked
    about a category they already explicitly addressed. Falls back to
    the full four-category question when nothing is known yet."""
    remaining = _remaining_red_flag_categories(context.red_flags)
    if not remaining or len(remaining) == len(_RED_FLAG_CATEGORIES):
        return (
            "Have you had any chest pain or discomfort, difficulty breathing or "
            "shortness of breath, fainting, or coughing up blood?"
        )
    labels = [_RED_FLAG_CATEGORY_LABELS[category] for category in remaining]
    if len(labels) == 1:
        joined = labels[0]
    elif len(labels) == 2:
        joined = f"{labels[0]} or {labels[1]}"
    else:
        joined = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return f"Have you had any {joined}?"


def _merge_red_flags_into_context(
    context: ClinicalContext, new_flags: dict[str, str], source: FactSource
) -> ClinicalContext:
    """Merges `new_flags` into context.red_flags (see _merge_red_flags_dict)
    and rebuilds negative_findings/unknown_fields for the four red-flag
    categories fresh from the merged, authoritative state each time —
    never incrementally appended — so a category resolved across
    multiple partial statements is never recorded twice, and one that
    later flips from unknown to negated is never left stranded in
    unknown_fields."""
    context.red_flags = _merge_red_flags_dict(context.red_flags, new_flags)
    context.negative_findings = [
        fact for fact in context.negative_findings if fact.value not in _RED_FLAG_CATEGORIES
    ]
    context.unknown_fields = [
        note for note in context.unknown_fields if not note.startswith("red_flags:")
    ]
    for category in _RED_FLAG_CATEGORIES:
        status = context.red_flags.get(category, MentionStatus.UNKNOWN.value)
        if status == MentionStatus.NEGATED.value:
            context.negative_findings.append(
                ClinicalFact(value=category, confidence=FactConfidence.EXPLICIT, source=source)
            )
        elif status == MentionStatus.UNKNOWN.value:
            context.unknown_fields.append(f"red_flags:{category}")
    return context


def _handle_red_flags(
    context: ClinicalContext, text: str, source: FactSource, is_direct_answer: bool
) -> tuple[ClinicalContext, bool]:
    assessment = assess_red_flags(text)
    context = _merge_red_flags_into_context(context, assessment.as_dict(), source)
    return context, _red_flags_resolved(context.red_flags)


def _handle_risk_factors(
    context: ClinicalContext, text: str, source: FactSource, is_direct_answer: bool
) -> tuple[ClinicalContext, bool]:
    found_any = False
    for label, phrases in _RISK_FACTOR_PHRASES.items():
        status = classify_mention(text, phrases)
        if status == MentionStatus.AFFIRMED:
            context.risk_factors.append(
                ClinicalFact(value=label, confidence=FactConfidence.EXPLICIT, source=source)
            )
            found_any = True
        elif status == MentionStatus.NEGATED:
            context.negative_findings.append(
                ClinicalFact(value=label, confidence=FactConfidence.EXPLICIT, source=source)
            )
            found_any = True
    if not found_any:
        context.unknown_fields.append("risk_factors")
    return context, found_any


_FIELD_HANDLERS: dict[
    str, Callable[[ClinicalContext, str, FactSource, bool], tuple[ClinicalContext, bool]]
] = {
    "onset": _handle_onset,
    "local_symptoms": _handle_local_symptoms,
    "red_flags": _handle_red_flags,
    "risk_factors": _handle_risk_factors,
}


LEG_SWELLING_PROTOCOL = ClinicalProtocol(
    name="unilateral_leg_swelling",
    display_name="Leg swelling",
    primary_concern_label="leg swelling",
    location_label="leg",
    candidate_specialty_slugs=("internal-medicine", "family-medicine"),
    questions=(
        ProtocolQuestion(
            field="onset",
            prompt=(
                "Did the swelling come on suddenly or gradually, and has it been getting worse?"
            ),
        ),
        ProtocolQuestion(
            field="local_symptoms",
            prompt=(
                "Have you noticed any pain, redness, warmth, or tenderness in the swollen area?"
            ),
        ),
        ProtocolQuestion(
            field="red_flags",
            prompt=(
                "Have you had any chest pain or discomfort, difficulty breathing or "
                "shortness of breath, fainting, or coughing up blood?"
            ),
        ),
        ProtocolQuestion(
            field="risk_factors",
            prompt=(
                "Have you had any recent injury, surgery, prolonged travel, "
                "immobility, pregnancy or being postpartum, or a previous blood clot?"
            ),
        ),
    ),
    detect=_detect_leg_swelling,
)

CLINICAL_PROTOCOLS: tuple[ClinicalProtocol, ...] = (LEG_SWELLING_PROTOCOL,)


def match_protocol(concern_text: str | None) -> ClinicalProtocol | None:
    """The first protocol (in registration order) whose detector matches
    `concern_text`, or None when no protocol applies — the graph's
    clinical_intake_agent_node is then a complete no-op, preserving every
    other concern's existing behavior unchanged."""
    if not concern_text:
        return None
    for protocol in CLINICAL_PROTOCOLS:
        if protocol.detect(concern_text):
            return protocol
    return None


def get_protocol(name: str | None) -> ClinicalProtocol | None:
    if not name:
        return None
    for protocol in CLINICAL_PROTOCOLS:
        if protocol.name == name:
            return protocol
    return None


def build_initial_context(
    protocol: ClinicalProtocol,
    concern_text: str,
    *,
    duration_fact: ClinicalFact | None,
    source: FactSource,
) -> ClinicalContext:
    """The starting ClinicalContext for a brand-new turn: primary_concern/
    location are the protocol's own fixed labels (never invented from raw
    text), laterality/duration/every question field are extracted from
    `concern_text` only when a clear, explicit signal is present — anything
    not addressed stays unset and is recorded in unknown_fields, never
    guessed at."""
    context = ClinicalContext(protocol=protocol.name)
    context.primary_concern = ClinicalFact(
        value=protocol.primary_concern_label, confidence=FactConfidence.DERIVED, source=source
    )
    if protocol.location_label:
        context.location = ClinicalFact(
            value=protocol.location_label, confidence=FactConfidence.DERIVED, source=source
        )

    laterality = _extract_laterality(concern_text)
    if laterality is not None:
        context.laterality = ClinicalFact(
            value=laterality, confidence=FactConfidence.EXPLICIT, source=source
        )
    else:
        context.unknown_fields.append("laterality")

    if duration_fact is not None:
        context.duration = duration_fact
    else:
        context.unknown_fields.append("duration")

    for question in protocol.questions:
        context, found = _FIELD_HANDLERS[question.field](context, concern_text, source, False)
        if found and question.field not in context.answered_protocol_fields:
            context.answered_protocol_fields.append(question.field)

    return context


def next_missing_field(protocol: ClinicalProtocol, context: ClinicalContext) -> str | None:
    """The next protocol question still needing an answer, in the
    protocol's own fixed order — None once every question has been
    addressed (from the original message, a clarification answer, or
    both)."""
    for question in protocol.questions:
        if question.field not in context.answered_protocol_fields:
            return question.field
    return None


def question_for(protocol: ClinicalProtocol, field: str) -> str:
    for question in protocol.questions:
        if question.field == field:
            return question.prompt
    raise ValueError(f"{field!r} is not a question field of protocol {protocol.name!r}")


def apply_field_answer(
    context: ClinicalContext,
    protocol: ClinicalProtocol,
    field: str,
    text: str,
    *,
    source: FactSource,
) -> ClinicalContext:
    """Merge a direct clarification answer for `field` into `context`. Marks
    `field` answered even when the text matches none of this field's known
    phrases (the user *was* directly asked and *did* respond — never
    re-asking the identical question is more important than a perfect
    keyword match), so this always progresses the protocol forward."""
    handler = _FIELD_HANDLERS[field]
    updated_context, _ = handler(context, text, source, True)
    if field not in updated_context.answered_protocol_fields:
        updated_context.answered_protocol_fields.append(field)
    return updated_context


def scan_incidental_red_flags(
    context: ClinicalContext, text: str, source: FactSource
) -> ClinicalContext:
    """Safety net: scans ANY answer text for red-flag phrases, regardless
    of which protocol question was actually being asked. A user may
    mention a red-flag symptom while answering an unrelated question
    (e.g. "it came on suddenly, and I've also had some chest pain" while
    answering the *onset* question) — this must be caught immediately,
    not only at the initial message or the dedicated red-flags question.

    Uses the same precedence merge as _handle_red_flags (see
    _merge_red_flags_into_context) — previously collected categories are
    never discarded or downgraded. "red_flags" is only marked answered
    once _red_flags_resolved is true: either an AFFIRMED category was
    just found (which always demands immediate gate evaluation, so this
    node's safety-gate invariant check picks it up on its very next
    entry), or all four categories are now explicitly resolved. A
    *partial* incidental negative alone (e.g. only "no chest pain" while
    answering an unrelated question) must never suppress the dedicated
    red-flags question for the categories still unknown — see the
    reported completeness gap this function was corrected for.

    A bare blanket reply ("No.", "None") is deliberately NOT read as
    negating these categories here (allow_blanket_negation=False) — the
    user was answering a *different* question, so a blanket "no" to that
    question must not be misread as also covering these four unrelated
    categories they were never actually asked about."""
    assessment = assess_red_flags(text, allow_blanket_negation=False)
    new_flags = assessment.as_dict()
    if _merge_red_flags_dict(context.red_flags, new_flags) == context.red_flags:
        return context  # nothing new learned -- avoid a needless rebuild

    context = _merge_red_flags_into_context(context, new_flags, source)
    if (
        _red_flags_resolved(context.red_flags)
        and "red_flags" not in context.answered_protocol_fields
    ):
        context.answered_protocol_fields.append("red_flags")
    return context


# --- clinical-navigation summary --------------------------------------------

_PROMPT_CARE_RISK_LABELS = frozenset(
    {
        "immobility",
        "prolonged_travel",
        "recent_surgery",
        "previous_clot_history",
        "pregnancy_or_postpartum",
    }
)


def _summary_text(context: ClinicalContext) -> str:
    location_phrase = "leg swelling"
    if context.laterality is not None:
        if context.laterality.value == "bilateral":
            location_phrase = "swelling in both legs"
        elif context.laterality.value in ("left", "right"):
            location_phrase = f"{context.laterality.value}-sided leg swelling"
        elif context.laterality.value == "one-sided":
            location_phrase = "one-sided leg swelling"

    sentence = f"You reported {location_phrase}"
    if context.duration is not None:
        sentence += f" for {context.duration.value}"
    # Only "sudden"/"gradual" are short, grammatical onset descriptors fit
    # to splice into this clause. _handle_onset's fallback also records the
    # user's raw direct-answer text verbatim (e.g. rambling voice-dictated
    # speech with no explicit "sudden"/"gradual") when neither keyword is
    # present -- that raw text is never spliced mid-sentence here (it would
    # read as broken grammar); it is instead appended as its own quoted
    # sentence below, once, alongside progression if either is known.
    raw_onset_note: str | None = None
    if context.onset is not None:
        if context.onset.value in ("sudden", "gradual"):
            sentence += f", with {context.onset.value} onset"
        else:
            raw_onset_note = context.onset.value
    if context.progression is not None:
        sentence += f" that has been {context.progression.value}"
    sentence += "."
    if raw_onset_note is not None:
        sentence += f' You described the onset as: "{raw_onset_note}"'

    feature_values = [fact.value.replace("_", " ") for fact in context.associated_features]
    if feature_values:
        sentence += f" You also reported {', '.join(feature_values)}."

    risk_values = [fact.value.replace("_", " ") for fact in context.risk_factors]
    if risk_values:
        sentence += f" Relevant history includes {', '.join(risk_values)}."

    negated_values = [fact.value.replace("_", " ") for fact in context.negative_findings]
    if negated_values:
        sentence += f" You reported no {', '.join(negated_values)}."

    return sentence


def _possible_explanations(context: ClinicalContext) -> list[PossibleExplanationCategory]:
    risk_labels = {fact.value for fact in context.risk_factors}
    feature_labels = {fact.value for fact in context.associated_features}

    categories = [
        PossibleExplanationCategory(
            category="Prolonged standing, sitting, or minor strain",
            description=(
                "One-sided leg swelling may be associated with prolonged standing, "
                "sitting, or minor strain, which can let fluid build up in one leg. This "
                "is one possible category among several and is not a diagnosis — an "
                "in-person evaluation is still the appropriate next step."
            ),
            supporting_evidence=(
                ["prolonged travel or immobility reported"]
                if risk_labels & {"immobility", "prolonged_travel"}
                else []
            ),
        ),
        PossibleExplanationCategory(
            category="Local inflammation or minor injury",
            description=(
                "Pain, redness, warmth, or tenderness in the swollen area can point to "
                "local inflammation or a minor injury."
            ),
            supporting_evidence=[f"{label} reported" for label in sorted(feature_labels)]
            + (["recent injury reported"] if "recent_injury" in risk_labels else []),
        ),
        PossibleExplanationCategory(
            category="Fluid buildup or circulation-related causes",
            description=(
                "Several fluid- or circulation-related issues can also cause one-sided "
                "leg swelling, especially alongside pregnancy, recent surgery, or reduced "
                "mobility."
            ),
            supporting_evidence=[
                label.replace("_", " ")
                for label in sorted(risk_labels)
                if label in {"pregnancy_or_postpartum", "recent_surgery", "immobility"}
            ],
        ),
    ]

    if risk_labels & _PROMPT_CARE_RISK_LABELS:
        categories.append(
            PossibleExplanationCategory(
                category="Circulation concern warranting prompt attention",
                description=(
                    "Given the risk factors you reported, a circulation-related cause "
                    "that benefits from prompt medical attention (rather than routine "
                    "scheduling) is also possible."
                ),
                supporting_evidence=[label.replace("_", " ") for label in sorted(risk_labels)],
            )
        )

    return categories


def _care_level(context: ClinicalContext) -> CareLevel:
    """Unilateral leg swelling always recommends prompt in-person clinical
    evaluation, never "routine" -- this is a deliberate, cautious Phase 3A
    POC decision independent of which specific risk factors were reported:
    one-sided leg swelling is a recognized possible early sign of a
    circulation-related concern even with no other risk factors present,
    so this protocol never tells the user waiting for a routine
    appointment is appropriate. (A future, genuinely lower-concern
    protocol could still use CareLevel.ROUTINE -- that value is kept in
    the schema for that reason -- but leg swelling is not that case.)"""
    return CareLevel.PROMPT


def build_navigation_summary(
    protocol: ClinicalProtocol, context: ClinicalContext
) -> ClinicalNavigationSummary:
    """A bounded, reviewable summary — only ever called once every protocol
    question has been addressed and no red flag was affirmed (see
    app.graph.nodes.clinical_intake_agent_node)."""
    return ClinicalNavigationSummary(
        protocol=protocol.name,
        summary_of_reported_information=_summary_text(context),
        possible_explanations=_possible_explanations(context),
        important_unknowns=list(context.unknown_fields),
        recommended_care_level=_care_level(context),
        specialty_candidates=list(protocol.candidate_specialty_slugs),
        disclaimer=(
            "These are general possible categories based on common causes, not a "
            "diagnosis. Only a licensed clinician can diagnose or treat a medical "
            "condition."
        ),
    )
