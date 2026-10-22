"""Command-line entry points for ingestion, serving, and local queries."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from time import perf_counter

import httpx

from lmaana_assistant.config import BenchmarkSettings, Settings
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
    retrieval = commands.add_parser(
        "benchmark-retrieval", help="Rank supplied passages without opening the active index."
    )
    retrieval.add_argument("dataset", type=Path)
    retrieval.add_argument("--output", type=Path)
    retrieval.add_argument("--backend", choices=["lexical", "qwen"], default="lexical")
    retrieval.add_argument("--revision", default="", help="Full Qwen checkpoint commit hash.")
    retrieval.add_argument("--device", default="cpu")
    retrieval.add_argument("--min-score", type=float, default=0.18)
    retrieval.add_argument("--query-mode", choices=["raw", "expanded"], default="expanded")
    retrieval.add_argument(
        "--allow-download", action="store_true", help="Allow downloading the pinned Qwen model."
    )
    commands.add_parser("doctor", help="Print profile and corpus status without loading models.")
    args = parser.parse_args()
    try:
        # Offline tools must not depend on a machine's serving/ASR configuration.
        settings = None if args.command in {"evaluate", "benchmark-retrieval"} else Settings()
        if args.command in {"evaluate", "benchmark-retrieval"} and args.output:
            if args.output.resolve() == args.dataset.resolve() or (
                args.output.exists() and args.output.samefile(args.dataset)
            ):
                raise ValueError("The report path must not overwrite the input dataset.")
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
        elif args.command in {"evaluate", "benchmark-retrieval"}:
            from lmaana_assistant.evaluation.dataset import EvaluationDataset
            from lmaana_assistant.evaluation.runner import evaluate

            if args.command == "evaluate":
                dataset = EvaluationDataset.model_validate_json(
                    args.dataset.read_text(encoding="utf-8")
                )
                result = evaluate(dataset)
            else:
                from lmaana_assistant.evaluation.retrieval import (
                    RetrievalBenchmark,
                    benchmark_retrieval,
                )
                from lmaana_assistant.evaluation.runtime import runtime_metadata
                from lmaana_assistant.retrieval.embeddings import LexicalEmbedder

                dataset = RetrievalBenchmark.model_validate_json(
                    args.dataset.read_text(encoding="utf-8")
                )
                if args.backend == "lexical" and (args.revision or args.allow_download):
                    raise ValueError("Revision/download options apply only to the Qwen backend.")
                if not math.isfinite(args.min_score) or not 0 <= args.min_score <= 1:
                    raise ValueError(
                        "The cosine threshold must be finite and between zero and one."
                    )
                started = perf_counter()
                embedder = (
                    LexicalEmbedder()
                    if args.backend == "lexical"
                    else make_embedder(
                        BenchmarkSettings(
                            _env_file=None,
                            embedding_backend="qwen",
                            embedding_revision=args.revision,
                            embedding_device=args.device,
                            embedding_local_files_only=not args.allow_download,
                            asr_backend="unavailable",
                            generator_backend="excerpt",
                        )
                    )
                )
                model_load_ms = (perf_counter() - started) * 1000
                result = benchmark_retrieval(
                    dataset, embedder, min_score=args.min_score, query_mode=args.query_mode
                )
                result["model_load_ms"] = model_load_ms
                result["model_download_allowed"] = args.allow_download
                result["embedding_backend"] = args.backend
                result["requested_device"] = args.device if args.backend == "qwen" else "cpu"
                result["python"] = sys.version.split()[0]
                result["runtime"] = runtime_metadata(embedder)
            report = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report + "\n", encoding="utf-8")
                print(f"Evaluation saved to {args.output.resolve()}")
            else:
                print(report)
        else:
            from lmaana_assistant.asr.factory import make_asr

            asr = make_asr(settings)
            print(
                json.dumps(
                    {
                        "python": sys.version.split()[0],
                        "planned_asr_model": settings.asr_model,
                        "planned_asr_revision": settings.asr_revision,
                        "asr_backend": settings.asr_backend,
                        "asr_model_dir": (
                            str(settings.asr_model_dir.resolve())
                            if settings.asr_model_dir
                            else None
                        ),
                        "transcription_available": asr.ready(),
                        "asr_reason": getattr(asr, "reason", None),
                        "embedding_backend": settings.embedding_backend,
                        "generator_backend": settings.generator_backend,
                        "active_corpus_exists": settings.active_path.exists(),
                        "data_dir": str(settings.data_dir.resolve()),
                    },
                    indent=2,
                )
            )
            asr.close()
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
