import pytest

from app.providers.llm.fake import FakeTextLLMProvider
from app.providers.speech_to_text.fake import FakeSpeechToTextProvider
from app.providers.text_to_speech.fake import FakeTextToSpeechProvider


@pytest.mark.asyncio
async def test_fake_text_llm_provider_echoes_prompt() -> None:
    provider = FakeTextLLMProvider()
    result = await provider.generate("hello")
    assert "hello" in result


@pytest.mark.asyncio
async def test_fake_speech_to_text_provider_reports_byte_count() -> None:
    provider = FakeSpeechToTextProvider()
    result = await provider.transcribe(b"1234567890")
    assert "10 bytes" in result


@pytest.mark.asyncio
async def test_fake_text_to_speech_provider_returns_bytes() -> None:
    provider = FakeTextToSpeechProvider()
    result = await provider.synthesize("hello")
    assert isinstance(result, bytes)
    assert b"hello" in result
