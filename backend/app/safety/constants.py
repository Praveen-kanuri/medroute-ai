# Emergency escalation (Phase 1C): user-DECLARED signals only. MedRoute AI
# never infers an emergency from symptom text, a voice transcript, or media —
# see docs/safety-design.md.

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
