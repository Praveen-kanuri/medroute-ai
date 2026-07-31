from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """LLM-derived specialty suggestion. Advisory only, never a diagnosis."""

    suggested_specialty: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    is_emergency: bool = False
