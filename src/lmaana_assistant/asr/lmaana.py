"""Lazy adapter for the native Lmaana 2.4 fairseq2/OmniASR runtime.

The heavy runtime is intentionally optional. Importing the application must not
load a 3B checkpoint, and readiness must remain false when the model or runtime
is missing. The adapter follows the local model's greedy CTC inference path.
"""

import tempfile
from importlib.util import find_spec
from pathlib import Path
from time import monotonic

from lmaana_assistant.asr.base import TranscriptResult
from lmaana_assistant.audio import AudioSample
from lmaana_assistant.config import Settings
from lmaana_assistant.errors import DependencyUnavailable, InferenceTimeout


class LmaanaASR:
    name = "lmaana-2.4-native-omnilingual-asr"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.asr_model
        self.revision = settings.asr_revision
        self.model_dir = settings.asr_model_dir
        self._pipeline = None
        self._torch = None

    @property
    def checkpoint(self) -> Path | None:
        return self.model_dir / "model" / "consolidated.pt" if self.model_dir else None

    @property
    def tokenizer(self) -> Path | None:
        return (
            self.model_dir / "tokenizer" / "omniASR_tokenizer_written_v2.model"
            if self.model_dir
            else None
        )

    def _runtime_available(self) -> bool:
        return all(find_spec(name) is not None for name in ("torch", "fairseq2", "omnilingual_asr"))

    def _cuda_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    @property
    def reason(self) -> str | None:
        if self.model_dir is None:
            return "LMAANA_ASR_MODEL_DIR is not configured."
        if self.checkpoint is None or not self.checkpoint.is_file():
            return "The Lmaana checkpoint file is missing."
        if self.tokenizer is None or not self.tokenizer.is_file():
            return "The Lmaana tokenizer file is missing."
        if not self._runtime_available():
            return "Install torch, fairseq2 and omnilingual_asr in the selected runtime."
        if not self._cuda_available():
            return "CUDA is unavailable for the Lmaana 2.4 3B profile."
        return None

    def ready(self) -> bool:
        return self.reason is None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline, self._torch
        if not self.ready():
            raise DependencyUnavailable(
                "Lmaana 2.4 needs the local checkpoint, tokenizer and native fairseq2 runtime."
            )
        try:
            import torch
            from fairseq2.assets import AssetCard
            from fairseq2.data.tokenizers import load_tokenizer
            from fairseq2.models.hub import load_model
            from fairseq2.models.wav2vec2.asr import get_wav2vec2_asr_model_hub
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

            if not torch.cuda.is_available():
                raise DependencyUnavailable("Lmaana 2.4 requires a CUDA runtime for this profile.")
            model_card = AssetCard(
                "lmaana24_local",
                {
                    "model_family": "wav2vec2_asr",
                    "model_arch": "3b_v2",
                    "checkpoint": self.checkpoint.as_uri(),
                },
            )
            tokenizer_card = AssetCard(
                "lmaana24_tokenizer",
                {
                    "tokenizer_family": "char_tokenizer",
                    "tokenizer": self.tokenizer.as_uri(),
                },
            )
            get_wav2vec2_asr_model_hub().get_model_config(model_card)
            tokenizer = load_tokenizer(tokenizer_card)
            model = load_model(
                model_card, device=torch.device("cuda"), dtype=torch.bfloat16, mmap=True
            )
            model.eval()
            self._pipeline = ASRInferencePipeline(
                model_card=None,
                model=model,
                tokenizer=tokenizer,
                device="cuda",
                dtype=torch.bfloat16,
            )
            self._torch = torch
            return self._pipeline, self._torch
        except DependencyUnavailable:
            raise
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise DependencyUnavailable("The native Lmaana runtime could not be loaded.") from exc

    def transcribe(self, audio: AudioSample) -> TranscriptResult:
        started = monotonic()
        pipeline, _ = self._load()
        temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = Path(temporary.name)
        try:
            temporary.write(audio.data)
            temporary.close()
            try:
                results = pipeline.transcribe([str(path)], batch_size=1)
            except (OSError, RuntimeError, ValueError) as exc:
                raise DependencyUnavailable(
                    "Lmaana inference failed for the uploaded audio."
                ) from exc
        finally:
            path.unlink(missing_ok=True)
        if len(results) != 1 or not isinstance(results[0], str) or not results[0].strip():
            raise DependencyUnavailable("Lmaana returned an empty or invalid transcript.")
        elapsed = (monotonic() - started) * 1000
        if elapsed > self.settings.timeout_seconds * 1000:
            raise InferenceTimeout("Lmaana transcription exceeded the configured timeout.")
        return TranscriptResult(
            text=results[0].strip(),
            model=self.model,
            revision=self.revision,
            audio_duration_ms=round(audio.duration_seconds * 1000),
            confidence=None,
            warnings=["Greedy CTC decoding; confidence is not calibrated."],
            timings_ms={"total": round(elapsed, 2)},
        )

    def close(self) -> None:
        self._pipeline = None
        self._torch = None
