"""Phase 3A: dedicated, deterministic red-flag safety gate.

A single, centralized, versionable rule set for a small number of
explicitly affirmed symptoms that must short-circuit the ordinary
clinical-navigation flow (see app.graph.nodes.clinical_red_flag_gate_node).
No model call is involved anywhere in this module — every decision is
plain, reviewable keyword/negation matching (see
app.services.mention_classification_service.classify_mention). This module
never identifies a disease and never makes the final emergency decision on
its own initiative beyond "was one of these specific phrases explicitly
affirmed" — the graph node that calls this is what actually short-circuits
the turn.

Extending this to a new complaint protocol never requires touching this
module: RED_FLAG_PHRASES is a fixed, small set shared by every protocol
that opts into a red-flag question (currently only unilateral leg
swelling — see app.services.clinical_intake_service).
"""

from pydantic import BaseModel, ConfigDict

from app.schemas.clinical_context import MentionStatus
from app.services.mention_classification_service import classify_mention

# Centralized, reviewable phrase catalog for each red-flag category. Keep
# this list short and specific — a false affirmation here short-circuits a
# user's entire turn into the emergency path, so every phrase must be an
# unambiguous, explicit statement of that symptom, never a vague synonym.
RED_FLAG_PHRASES: dict[str, list[str]] = {
    "chest_pain": [
        "chest pain",
        "chest discomfort",
        "chest pressure",
        "chest tightness",
    ],
    "difficulty_breathing": [
        "difficulty breathing",
        "trouble breathing",
        "shortness of breath",
        "short of breath",
        "hard to breathe",
        "cant breathe",
        "cannot breathe",
    ],
    "fainting": [
        "fainting",
        "fainted",
        "passed out",
        "passing out",
        "blacked out",
        "loss of consciousness",
    ],
    "coughing_blood": [
        "coughing blood",
        "coughing up blood",
        "cough blood",
        "coughed up blood",
        "blood when i cough",
    ],
}


class RedFlagAssessment(BaseModel):
    """Deterministic classification of the four leg-swelling red flags
    (Phase 3A). `any_affirmed` is the only signal
    app.graph.nodes.clinical_red_flag_gate_node acts on — an emergency
    short-circuit requires at least one explicit AFFIRMED, never a
    NEGATED or UNKNOWN result."""

    model_config = ConfigDict(extra="forbid")

    chest_pain: MentionStatus
    difficulty_breathing: MentionStatus
    fainting: MentionStatus
    coughing_blood: MentionStatus

    @property
    def any_affirmed(self) -> bool:
        return MentionStatus.AFFIRMED in (
            self.chest_pain,
            self.difficulty_breathing,
            self.fainting,
            self.coughing_blood,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "chest_pain": self.chest_pain.value,
            "difficulty_breathing": self.difficulty_breathing.value,
            "fainting": self.fainting.value,
            "coughing_blood": self.coughing_blood.value,
        }


def assess_red_flags(text: str, *, allow_blanket_negation: bool = True) -> RedFlagAssessment:
    """Classify `text` against each category in RED_FLAG_PHRASES
    independently — a reply can affirm one flag while negating another
    (e.g. "I have chest pain but no trouble breathing").

    allow_blanket_negation=False (see
    app.services.mention_classification_service.classify_mention) is used
    when `text` is an answer to a *different* question being scanned only
    for incidentally-mentioned red-flag information — a bare "No."/"None"
    there must not be read as negating these four categories, since the
    user was never actually asked about them in that turn."""
    return RedFlagAssessment(
        chest_pain=classify_mention(
            text, RED_FLAG_PHRASES["chest_pain"], allow_blanket_negation=allow_blanket_negation
        ),
        difficulty_breathing=classify_mention(
            text,
            RED_FLAG_PHRASES["difficulty_breathing"],
            allow_blanket_negation=allow_blanket_negation,
        ),
        fainting=classify_mention(
            text, RED_FLAG_PHRASES["fainting"], allow_blanket_negation=allow_blanket_negation
        ),
        coughing_blood=classify_mention(
            text, RED_FLAG_PHRASES["coughing_blood"], allow_blanket_negation=allow_blanket_negation
        ),
    )
