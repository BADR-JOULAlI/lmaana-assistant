"""ASR provider selection with no silent model substitution."""

from lmaana_assistant.asr.unavailable import UnavailableASR
from lmaana_assistant.config import Settings


def make_asr(settings: Settings):
    # The native fairseq2/OmniASR adapter is intentionally not emulated here.
    # Until it is implemented, both profiles expose an explicit unavailable state.
    return UnavailableASR(settings)
