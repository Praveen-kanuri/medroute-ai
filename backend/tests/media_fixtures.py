"""Synthetic, non-identifying media fixtures for Phase 2C tests.

Every image/video below is generated in-process from solid colors — no
real photograph or recording is ever used or committed.
"""

import io

import cv2
import numpy as np
from PIL import Image


def make_jpeg_bytes(*, width: int = 64, height: int = 48) -> bytes:
    image = Image.new("RGB", (width, height), color=(120, 90, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def make_png_bytes(*, width: int = 64, height: int = 48) -> bytes:
    image = Image.new("RGB", (width, height), color=(90, 120, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_webp_bytes(*, width: int = 64, height: int = 48) -> bytes:
    image = Image.new("RGB", (width, height), color=(90, 90, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP")
    return buffer.getvalue()


def make_mp4_bytes(
    *, width: int = 64, height: int = 48, fps: float = 10.0, frame_count: int = 20
) -> bytes:
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for i in range(frame_count):
            frame = np.full((height, width, 3), i * 5 % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        with open(path, "rb") as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# A syntactically-valid-looking but undecodable "video" — enough header
# bytes to pass signature sniffing (an ISO-BMFF ftyp box), followed by
# garbage no real decoder can parse.
MALFORMED_MP4_BYTES = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100
