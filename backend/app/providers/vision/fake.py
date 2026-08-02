from typing import Any

from app.providers.vision.base import VisionProvider


class FakeVisionProvider(VisionProvider):
    """Deterministic fake vision provider for local development and tests.

    call_count lets tests assert on how many times analyze_image was
    actually invoked — e.g. proving a resumed clarification turn or a
    declared-emergency turn never triggers an avoidable extra call.
    """

    def __init__(
        self,
        *,
        observations: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._observations = observations if observations is not None else []
        self._error = error
        self.call_count = 0

    async def analyze_image(
        self, image_bytes: bytes, *, content_type: str = "image/jpeg"
    ) -> list[dict[str, Any]]:
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return self._observations
