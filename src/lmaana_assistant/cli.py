"""Command-line entry points for ingestion, serving, and local queries."""

import argparse
import json
import logging
import sys
from pathlib import Path

import httpx

from lmaana_assistant.config import Settings
from lmaana_assistant.ingestion.pipeline import ingest
from lmaana_assistant.retrieval.embeddings import make_embedder


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="lmaana")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="Build and activate a reviewed corpus.")
    ingest_parser.add_argument("manifest", type=Path)
    serve = commands.add_parser("serve", help="Start the local API (one worker).")
    serve.add_argument("--port", type=int, default=8000)
    ask = commands.add_parser("ask", help="Ask the running local API.")
    ask.add_argument("question")
    ask.add_argument("--api-url", default="http://127.0.0.1:8000")
    evaluation = commands.add_parser("evaluate", help="Evaluate annotated outputs offline.")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("--output", type=Path, help="Save the JSON report instead of stdout.")
    commands.add_parser("doctor", help="Print profile and corpus status without loading models.")
    args = parser.parse_args()
    try:
        settings = Settings()
        if args.command == "ingest":
            result = ingest(args.manifest, settings, make_embedder(settings))
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in ("release", "source_count", "chunk_count", "embedding")
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "serve":
            import uvicorn

            request_logger = logging.getLogger("lmaana.requests")
            if not request_logger.handlers:
                request_logger.addHandler(logging.StreamHandler())
            request_logger.setLevel(logging.INFO)
            request_logger.propagate = False
            uvicorn.run(
                "lmaana_assistant.api.app:create_app",
                factory=True,
                host="127.0.0.1",
                port=args.port,
                workers=1,
            )
        elif args.command == "ask":
            response = httpx.post(
                f"{args.api_url.rstrip('/')}/v1/answers",
                json={"question": args.question},
                timeout=settings.timeout_seconds * 2 + 10,
                trust_env=False,
            )
            response.raise_for_status()
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            from lmaana_assistant.evaluation.dataset import EvaluationDataset
            from lmaana_assistant.evaluation.runner import evaluate

            if args.output and (
                args.output.resolve() == args.dataset.resolve()
                or (args.output.exists() and args.output.samefile(args.dataset))
            ):
                raise ValueError("The report path must not overwrite the input dataset.")
            dataset = EvaluationDataset.model_validate_json(
                args.dataset.read_text(encoding="utf-8")
            )
            report = json.dumps(evaluate(dataset), ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report + "\n", encoding="utf-8")
                print(f"Evaluation saved to {args.output.resolve()}")
            else:
                print(report)
        else:
            print(
                json.dumps(
                    {
                        "python": sys.version.split()[0],
                        "planned_asr_model": settings.asr_model,
                        "planned_asr_revision": settings.asr_revision,
                        "transcription_available": False,
                        "embedding_backend": settings.embedding_backend,
                        "generator_backend": settings.generator_backend,
                        "active_corpus_exists": settings.active_path.exists(),
                        "data_dir": str(settings.data_dir.resolve()),
                    },
                    indent=2,
                )
            )
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
