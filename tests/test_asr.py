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


def test_valid_worker_segments_preserve_original_transcript_at_duration_boundary():
    text = "  شنو الوثائق؟\n"
    result = TranscriptResult(
        text=text,
        model="Lmaana/lmaana-2.4",
        revision="test-revision",
        audio_duration_ms=1200,
        segments=[{"text": text, "start_ms": 0, "end_ms": 1200}],
        timings_ms={"load": 0, "inference": 12.5},
    )
    assert result.text == result.segments[0].text == text
    assert result.segments[0].end_ms == result.audio_duration_ms
