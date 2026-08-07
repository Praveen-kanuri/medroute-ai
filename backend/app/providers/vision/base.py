from abc import ABC, abstractmethod
from typing import Any


class VisionProvider(ABC):
    """Abstract interface for a vision-capable model provider."""

    @abstractmethod
    async def analyze_image(
        self, image_bytes: bytes, *, content_type: str = "image/jpeg"
    ) -> list[dict[str, Any]]:
        """Analyze a single still image (or one video frame) and return a
        list of candidate observation dicts.

        Args:
            image_bytes: Raw, already-normalized image data (JPEG).
            content_type: The image's MIME type.

        Returns:
            Raw, UNVALIDATED candidate observation dicts. Callers must
            validate each one against app.schemas.vision.VisionObservation
            (adding source_type/frame_timestamp_seconds) before trusting
            it — this method makes no safety guarantees about its own
            output on its own.

        Raises:
            Implementations may raise provider-specific exceptions on
            network, authentication, timeout, or malformed-output
            failures. Callers must treat any exception as a failed
            analysis — never fabricate an observation when this raises.
        """
        raise NotImplementedError
