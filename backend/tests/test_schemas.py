from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.schemas.booking import BookingConfirmation, BookingRequest, BookingStatus
from app.schemas.doctor import AppointmentSlot, Doctor
from app.schemas.intake import Severity, SymptomIntake
from app.schemas.routing import RoutingDecision


def test_symptom_intake_valid() -> None:
    intake = SymptomIntake(symptoms=["headache"], duration_days=2, severity=Severity.MILD)
    assert intake.preferred_language == "en"


def test_symptom_intake_requires_at_least_one_symptom() -> None:
    with pytest.raises(ValidationError):
        SymptomIntake(symptoms=[], duration_days=2, severity=Severity.MILD)


def test_symptom_intake_rejects_invalid_severity() -> None:
    with pytest.raises(ValidationError):
        SymptomIntake(symptoms=["headache"], duration_days=2, severity="critical")


def test_routing_decision_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(suggested_specialty="cardiology", confidence=1.5, rationale="test")
    with pytest.raises(ValidationError):
        RoutingDecision(suggested_specialty="cardiology", confidence=-0.1, rationale="test")


def test_doctor_rating_bounds() -> None:
    with pytest.raises(ValidationError):
        Doctor(id="d1", name="Dr. Test", specialty="cardiology", location="Test City", rating=6.0)
    with pytest.raises(ValidationError):
        Doctor(id="d1", name="Dr. Test", specialty="cardiology", location="Test City", rating=-1.0)


def test_appointment_slot_valid() -> None:
    start = datetime.now(UTC)
    end = start + timedelta(minutes=30)
    slot = AppointmentSlot(slot_id="s1", doctor_id="d1", start_time=start, end_time=end)
    assert slot.is_available is True


def test_appointment_slot_rejects_naive_datetimes() -> None:
    start = datetime.now()
    end = start + timedelta(minutes=30)
    with pytest.raises(ValidationError):
        AppointmentSlot(slot_id="s1", doctor_id="d1", start_time=start, end_time=end)


def test_appointment_slot_rejects_end_before_start() -> None:
    start = datetime.now(UTC)
    end = start - timedelta(minutes=30)
    with pytest.raises(ValidationError):
        AppointmentSlot(slot_id="s1", doctor_id="d1", start_time=start, end_time=end)


def test_booking_request_valid() -> None:
    request = BookingRequest(patient_name="Jane Doe", slot_id="s1")
    assert request.contact_email is None


def test_booking_confirmation_valid() -> None:
    confirmation = BookingConfirmation(
        booking_id="b1", status=BookingStatus.CONFIRMED, slot_id="s1"
    )
    assert confirmation.status == BookingStatus.CONFIRMED
