# Local Lmaana 2.4 runtime

The repository keeps the Lmaana checkpoint outside Git. On the development
machine it is stored at `D:/darija_asr_data/models/lmaana-2.4` and the checkpoint
checksum has already been verified.

## Runtime prerequisites

Install the native OmniASR/fairseq2 runtime in the WSL distribution that will
serve the model. The application checks for `torch`, `fairseq2` and
`omnilingual_asr` before loading the checkpoint. It also requires CUDA for this
3B profile.

Check the runtime without loading the model:

```bash
python3 -c "import torch, fairseq2, omnilingual_asr; print(torch.cuda.is_available())"
```

## Configuration

Set the ignored local environment values:

```bash
export LMAANA_ASR_BACKEND=lmaana
export LMAANA_ASR_MODEL_DIR=/mnt/d/darija_asr_data/models/lmaana-2.4
export LMAANA_ASR_RUNTIME_SOURCE=/mnt/d/darija_asr_data/work/omnilingual-asr-v2/src
```

Start the API and inspect readiness:

```bash
lmaana serve
curl http://127.0.0.1:8000/health/ready
```

The response exposes `asr_ready` and `asr_reason`. A missing model, tokenizer or
runtime leaves text answering available while `POST /v1/transcriptions` returns
`503`. This is intentional: the service never silently substitutes Whisper.

## First smoke test

Use a short, non-silent PCM WAV file:

```bash
curl -X POST http://127.0.0.1:8000/v1/transcriptions \
  -F 'file=@sample.wav;type=audio/wav'
```

The first request loads the 3B checkpoint and is expected to be slower. Later
requests reuse the in-process pipeline. The returned confidence is `null` because
greedy CTC decoding has not been calibrated as a probability.
