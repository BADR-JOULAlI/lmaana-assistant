from lmaana_assistant.asr.base import TranscriptResult
from lmaana_assistant.asr.lmaana import LmaanaASR
from lmaana_assistant.config import Settings


def test_lmaana_reports_missing_model_path():
    provider = LmaanaASR(Settings(_env_file=None, asr_backend="lmaana", asr_model_dir=None))

    assert provider.ready() is False
    assert provider.reason == "LMAANA_ASR_MODEL_DIR is not configured."


def test_lmaana_reports_missing_checkpoint(tmp_path):
    provider = LmaanaASR(Settings(_env_file=None, asr_backend="lmaana", asr_model_dir=tmp_path))

    assert provider.ready() is False
    assert provider.reason == "The Lmaana checkpoint file is missing."


def test_transcript_result_keeps_uncalibrated_confidence_nullable():
    result = TranscriptResult(
        text="شنو الوثائق؟",
        model="Lmaana/lmaana-2.4",
        revision="test-revision",
        audio_duration_ms=1200,
    )

    assert result.confidence is None
    assert result.audio_duration_ms == 1200
