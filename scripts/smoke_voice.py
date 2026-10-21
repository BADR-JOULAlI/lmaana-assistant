"""Exercise the running local voice -> transcript -> RAG HTTP path with an existing WAV."""

import argparse
import json
import sys
from pathlib import Path

import httpx


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.audio.resolve():
        parser.error("The report must not overwrite the audio input.")
    with httpx.Client(base_url=args.api_url, trust_env=False, timeout=300) as client:
        health = client.get("/health/ready")
        health.raise_for_status()
        if not health.json().get("asr_ready"):
            raise RuntimeError(health.json().get("asr_reason", "ASR is not ready."))
        with args.audio.open("rb") as audio:
            response = client.post(
                "/v1/transcriptions", files={"file": (args.audio.name, audio, "audio/wav")}
            )
        response.raise_for_status()
        transcript = response.json()
        response = client.post(
            "/v1/answers", json={"question": transcript["text"], "response_language": "auto"}
        )
        response.raise_for_status()
        report = {"transcription": transcript, "answer": response.json()}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
