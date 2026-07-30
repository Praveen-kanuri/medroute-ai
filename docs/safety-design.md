# Safety Design

## What MedRoute AI is not

MedRoute AI is not a diagnostic or treatment system. It must never:

- Diagnose a medical condition.
- Prescribe or recommend medication or dosages.
- Present itself as a substitute for a licensed medical professional.

## What MedRoute AI does

- Collects self-reported symptoms and preferences (`SymptomIntake`).
- Produces an advisory specialty routing suggestion (`RoutingDecision`),
  always labeled as LLM-derived and non-diagnostic.
- Searches deterministic doctor/availability data.
- Simulates (does not perform) appointment booking.

## Emergency escalation

Any routing logic must check for emergency indicators before producing a
routine specialty suggestion. If detected, the system must prioritize
directing the user to emergency services over any specialty routing or
booking flow.

**Phase 1C (current):** `POST /api/v1/intake/validate` supports
*user-declared* emergency input only — `emergency_concern` (boolean) and
`emergency_signals` (a closed, non-exhaustive enum the user selects from,
e.g. difficulty breathing, chest pain/pressure). MedRoute AI never infers
an emergency from symptom text, a voice transcript, or image/video — only
an explicit user declaration triggers the `emergency` state, which takes
precedence over every other state and directs the user to call 911 (U.S.)
rather than continuing to media processing, specialty routing, or
provider search. See `app/safety/constants.py` for the exact disclaimer
and safety-message text, and `app/services/multimodal_intake_service.py`
for the precedence logic.

**Not yet implemented:** autonomous/inferred emergency detection from
free text or media (e.g., an LLM or vision model recognizing an emergency
the user didn't explicitly declare). `RoutingDecision.is_emergency`
remains the schema hook for a future LLM-derived routing signal
(Phase 2) — any such signal must still never *replace* the user-declared
path above; it would be additive, and still non-diagnostic.

## Separation of generated vs. deterministic content

- LLM-generated text (routing rationale) lives only in `RoutingDecision`.
- Doctor and appointment data (`Doctor`, `AppointmentSlot`) come solely
  from deterministic sources and are never produced or altered by an LLM.
- Booking confirmations (`BookingConfirmation`) are deterministic/simulated,
  never LLM-generated.

## Data

All development and test data is synthetic. Real patient data must never
be used in this repository.
