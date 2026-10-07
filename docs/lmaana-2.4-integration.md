# Lmaana 2.4 integration target

The project's primary ASR model is **`Lmaana/lmaana-2.4`**. Whisper-large-v3-turbo is the comparison baseline. The current prototype implements text retrieval; speech inference is the next milestone.

The model card identifies a native OmniASR CTC 3B v2 checkpoint using fairseq2, with greedy CTC decoding and a written tokenizer. It is not a Transformers `from_pretrained()` package. It was adapted using Dataset13 and corrected MoulSot; evaluation must respect its held-out splits. Access to weights is gated on Hugging Face. [Official model card](https://huggingface.co/Lmaana/lmaana-2.4).

The public model metadata was inspected on 3 October 2026:

| Item | Selected value |
| --- | --- |
| Repository | `Lmaana/lmaana-2.4` |
| Revision | `537c5e5a0e2b1015b5ad10798be54f66bc3a2d7a` |
| Checkpoint file | `model/consolidated.pt` |
| Tokenizer file | `tokenizer/omniASR_tokenizer_written_v2.model` |
| Published checksum file | `SHA256SUMS` |

The API metadata response is the source for the revision and paths. Keep the revision pinned until a deliberate model update; do not follow `main` silently.

## Adapter work

The first voice slice is now implemented at the application boundary:

- `audio.validate_wav` accepts bounded, non-empty PCM WAV input only (10 MiB and
  30 seconds maximum), and rejects malformed or silent uploads.
- `asr.base` defines `ASRProvider`, `TranscriptResult` and timestamped segments.
- `POST /v1/transcriptions` validates audio before checking the provider and returns
  `415` for invalid audio or `503` when the native runtime is not installed.
- The default provider is explicit `unavailable`; it never substitutes Whisper or
  invents a transcript. The text answering route remains independent.

On the development machine, the verified checkpoint is available at
`D:/darija_asr_data/models/lmaana-2.4` and may be selected with the ignored local
setting `LMAANA_ASR_MODEL_DIR`. This path alone is insufficient: the current
Ubuntu-24.04 WSL environment does not yet contain `torch`, `fairseq2`,
`omnilingual_asr` or `torchaudio`, so the API correctly remains unavailable.

The remaining adapter work is deliberately runtime-specific:

1. Confirm existing Hugging Face access. The model page currently requires agreeing to share contact information before files can be accessed. If access has not been granted, the user must accept the model's terms in their own account. Never put a token in Git, logs, or the chat.
2. Download only the selected revision's required files and verify the published checkpoint checksum.
3. Establish a compatible fairseq2/OmniASR environment from the published recipe, then smoke-test one short clip. Native Windows compatibility is not yet verified; an isolated Linux/WSL runtime may be needed.
4. Implement the shared transcription contract with raw transcript, model/revision, timing, warnings, and nullable confidence. Preserve the release's tokenizer and preprocessing.
5. Profile the ASR model alone on the 8 GB RTX 5070 Laptop GPU. A 3B model changes the memory budget: begin with Qwen generation on CPU while measuring ASR, and enable coexistence only after profiling.
6. Expose the transcription endpoint and editable transcript in Streamlit. Add Whisper as a separately selected benchmark adapter, with no automatic substitution for Lmaana.

No weights have been downloaded or executed by this implementation. Configuration and `lmaana doctor` record the selected ASR target without claiming that transcription is available.
