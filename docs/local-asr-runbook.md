# Local Lmaana 2.4 runtime

The repository keeps the Lmaana checkpoint outside Git. On the development
machine it is stored at `D:/darija_asr_data/models/lmaana-2.4`. The existing
Ubuntu-24.04 runtime is
`/home/sail_ano/projects/mistral_darija_stt_benchmark/.venv-omni/bin/python`.

## Runtime prerequisites

Use the native OmniASR/fairseq2 runtime in the WSL distribution that will
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

## Windows interface with a preloaded WSL worker

This profile keeps the text API and Ollama on Windows and loads native Lmaana in
WSL. Both services listen on loopback. The native worker has no corpus and does
not call a generator. It retains the model across requests and reports `503`
while preloading, so the interface enables voice input only after loading ends.
Health checks remain responsive during loading and inference.

Inside Ubuntu, from the repository root, use the existing native environment:

```bash
export PYTHONPATH="$PWD/src"
export LMAANA_ASR_MODEL_DIR=/mnt/d/darija_asr_data/models/lmaana-2.4
export LMAANA_ASR_RUNTIME_SOURCE=/mnt/d/darija_asr_data/work/omnilingual-asr-v2/src
/home/sail_ano/projects/mistral_darija_stt_benchmark/.venv-omni/bin/python \
  -m uvicorn lmaana_assistant.asr.worker:create_worker --factory \
  --host 127.0.0.1 --port 8001 --workers 1
```

If needed, install `fastapi`, `uvicorn` and `python-multipart` into that existing
environment. Do not replace its torch/fairseq2/OmniASR installation.

In the ignored Windows `.env`, set:

```dotenv
LMAANA_ASR_BACKEND=lmaana-http
LMAANA_ASR_WORKER_URL=http://127.0.0.1:8001
LMAANA_OLLAMA_NUM_GPU=0
```

`LMAANA_OLLAMA_NUM_GPU=0` explicitly keeps Qwen on CPU while Lmaana occupies the
8 GB GPU. Start Ollama and then, in two Windows terminals at the repository root:

```powershell
.\.venv\Scripts\python.exe -m lmaana_assistant.cli serve
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

Inspect `http://127.0.0.1:8001/health/ready` for worker readiness and
`http://127.0.0.1:8000/health/ready` for `asr_ready`. Once loaded, use the
microphone or upload a WAV in Streamlit. Transcription fills the editable question;
answering requires a separate submit. A transcript outside the indexed sources
may correctly produce an abstention rather than a generated answer.
