# Emergency escalation (Phase 1C): user-DECLARED signals only. MedRoute AI
# never infers an emergency from symptom text, a voice transcript, or media —
# see docs/safety-design.md.

# Words a safe navigation response must never contain. Applied (with the
# required "not a diagnosis" disclaimer phrase scrubbed out first) to any
# optional model-generated text before it is trusted: Groq response
# rephrasing (app/services/response_composition_service.py) and Groq vision
# observations (app/schemas/vision.py) alike.
FORBIDDEN_TERMS = (
    "diagnos",
    "treatment",
    "treat you",
    "urgent",
    "urgency",
    "prescri",
    "confidence",
    "cure",
)

# Additional terms rejected only in vision-observation text (Phase 2C) — a
# visual observation must never name a disease/condition or classify
# severity, even in cases (e.g. "infection") that aren't already covered by
# FORBIDDEN_TERMS above.
VISION_FORBIDDEN_TERMS = FORBIDDEN_TERMS + (
    "emergency",
    "disease",
    "cancer",
    "tumor",
    "malignant",
    "benign",
    "infection",
    "fracture",
)

NON_DIAGNOSTIC_DISCLAIMER = (
    "MedRoute AI does not diagnose conditions, prescribe medication, "
    "or replace professional medical advice."
)

INTAKE_DISCLAIMER = (
    "MedRoute AI provides navigation assistance and does not diagnose "
    "conditions, recommend treatment, verify provider credentials, or "
    "provide emergency services."
)

EMERGENCY_SAFETY_MESSAGE = (
    "If you are experiencing a medical emergency, call 911 (in the United "
    "States) or seek immediate emergency care. MedRoute AI cannot provide "
    "emergency assistance."
)
