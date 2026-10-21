"""ASR provider selection with no silent model substitution."""

from lmaana_assistant.asr.unavailable import UnavailableASR
from lmaana_assistant.config import Settings


def make_asr(settings: Settings):
    if settings.asr_backend == "lmaana-http":
        from lmaana_assistant.asr.http import LocalLmaanaASR

        return LocalLmaanaASR(settings)
    if settings.asr_backend == "lmaana":
        from lmaana_assistant.asr.lmaana import LmaanaASR

        return LmaanaASR(settings)
    return UnavailableASR(settings)
