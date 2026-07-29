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

Any future routing logic must check for emergency indicators before
producing a routine specialty suggestion. If detected, the system must
prioritize directing the user to emergency services over any specialty
routing or booking flow. `RoutingDecision.is_emergency` exists as the
schema hook for this; actual detection logic is a future milestone
(Phase 2 in the roadmap) — Phase 0 ships no detection logic.

## Separation of generated vs. deterministic content

- LLM-generated text (routing rationale) lives only in `RoutingDecision`.
- Doctor and appointment data (`Doctor`, `AppointmentSlot`) come solely
  from deterministic sources and are never produced or altered by an LLM.
- Booking confirmations (`BookingConfirmation`) are deterministic/simulated,
  never LLM-generated.

## Data

All development and test data is synthetic. Real patient data must never
be used in this repository.
