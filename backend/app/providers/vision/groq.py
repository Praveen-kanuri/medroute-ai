"""Real vision provider using Groq's hosted, multimodal-input chat model.

All Groq SDK usage is isolated to this module. The vision analysis service
and graph node (app/services/vision_analysis_service.py,
app/graph/nodes.py) only ever depend on the abstract VisionProvider
interface. Groq's vision model (qwen/qwen3.6-27b, confirmed live against
Groq's own /docs/vision and /docs/structured-outputs pages — Groq's
Llama-4 Scout/Maverick vision models are not currently reachable via this
project's API access, and no Groq vision model currently supports the
strict json_schema response format) does not support strict JSON-schema
response formatting, so structured output is requested via json_object
mode plus an explicit shape description in the system prompt — exactly
the same pattern already used for this project's optional Groq routing/
response-rephrasing calls. This module's own output is always treated as
untrusted candidate data and is independently re-validated by the caller
against app.schemas.vision.VisionObservation — never trusted directly.
"""

import base64
import json
from typing import Any

from groq import AsyncGroq

from app.providers.vision.base import VisionProvider

_MAX_OBSERVATIONS_PER_CALL = 3

_SYSTEM_PROMPT = (
    "You describe only visibly observable attributes in a single medical-navigation "
    "demo image or video frame. You must never diagnose, never name or suggest a "
    "disease/condition, never suggest treatment, never assign an urgency or emergency "
    "level, and never claim medical certainty. Respond with only a JSON object of the "
    'exact form {"observations": [{"observation_type": "skin_appearance|'
    "swelling_or_lesion|wound_or_injury|posture_or_movement|general_appearance|"
    'other_visible_finding", "body_area": "<short area name>"|null, '
    '"visual_description": "<short, literal, non-diagnostic description>", '
    '"visible_attributes": ["<short attribute>", ...], "confidence": "low|medium|high", '
    '"limitations": "<short, plain-language limitation of a visual-only review>"}]}, '
    f"with at most {_MAX_OBSERVATIONS_PER_CALL} items, and nothing else — no chain-of-thought, "
    "no extra fields, no text outside the JSON object."
)


class GroqVisionAnalysisError(Exception):
    """Raised when the Groq vision API call fails or returns a malformed response."""


class GroqVisionProvider(VisionProvider):
    """Production vision provider using a Groq-hosted multimodal chat model."""

    def __init__(
        self, *, api_key: str, model: str, timeout_seconds: float, max_output_tokens: int
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens

    async def analyze_image(
        self, image_bytes: bytes, *, content_type: str = "image/jpeg"
    ) -> list[dict[str, Any]]:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{content_type};base64,{encoded}"

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe only the visibly observable attributes.",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=self._max_output_tokens,
                timeout=self._timeout_seconds,
            )
            content = response.choices[0].message.content
            if not content:
                raise GroqVisionAnalysisError("Vision provider returned no content.")
            parsed = json.loads(content)
            observations = parsed.get("observations")
        except GroqVisionAnalysisError:
            raise
        except Exception as exc:
            raise GroqVisionAnalysisError("Vision provider request failed.") from exc

        if not isinstance(observations, list):
            raise GroqVisionAnalysisError("Vision provider returned a malformed response.")
        return observations
