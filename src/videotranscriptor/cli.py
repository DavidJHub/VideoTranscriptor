"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__, audio as audio_utils, backends as backend_registry, formats, pipeline
from .backends.canary import DEFAULT_CHUNK_SECONDS, DEFAULT_MODEL as CANARY_MODEL
from .backends.deepgram import DEFAULT_MODEL as DEEPGRAM_MODEL, ENV_KEY as DEEPGRAM_ENV_KEY
from .config import load_env
from .models import TranscriptionError

log = logging.getLogger("videotranscriptor")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PARTIAL = 2

BACKEND_CHOICES = ["deepgram", "canary", "both", "auto"]


#: subcommands recognised as the first argument; anything else is a file to
#: transcribe, which keeps the common case at `videotranscriptor talk.mp4`.
SUBCOMMANDS = ("transcribe", "backends", "doctor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videotranscriptor",
        description=(
            "Extract audio from video and transcribe it two ways: Deepgram's "
            "cloud API and NVIDIA Canary running locally."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            "  transcribe   transcribe media files (default, may be omitted)\n"
            "  backends     show which approaches are usable right now\n"
            "  doctor       check ffmpeg, credentials and local model deps\n"
            "\n"
            "examples:\n"
            "  videotranscriptor talk.mp4                     # both backends + comparison\n"
            "  videotranscriptor talk.mp4 -b deepgram         # cloud only\n"
            "  videotranscriptor talk.mp4 -b canary --device cuda\n"
            "  videotranscriptor *.mp4 -f srt -o subtitles/\n"
            "  videotranscriptor backends\n"
        ),
    )
    parser.add_argument("--version", action="version", version="videotranscriptor {}".format(__version__))
    _add_transcribe_arguments(parser)
    return parser


def build_simple_parser(command: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videotranscriptor {}".format(command), description=description
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print debug logging")
    return parser


def _add_transcribe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="*", type=Path, help="video or audio files")
    parser.add_argument(
        "-b", "--backend", default="both", choices=BACKEND_CHOICES,
        help=(
            "which approach to run: 'deepgram' (cloud), 'canary' (local), "
            "'both' (default, adds a comparison report), or 'auto' (first "
            "available)"
        ),
    )
    parser.add_argument("-o", "--output-dir", type=Path, help="where to write transcripts (default: <input>_transcripts/)")
    parser.add_argument(
        "-f", "--format", dest="output_formats", action="append",
        choices=list(formats.FORMATS),
        help="output format, repeatable (default: txt srt json)",
    )
    parser.add_argument(
        "-l", "--language", default="auto",
        help="spoken language as an ISO code, or 'auto' (default). Canary cannot auto-detect and falls back to en.",
    )
    parser.add_argument("--keep-audio", action="store_true", help="keep the extracted 16 kHz WAV")
    parser.add_argument("--keep-raw", action="store_true", help="include the raw backend response in the JSON output")
    parser.add_argument("--no-compare", action="store_true", help="skip the comparison report when running both backends")

    cloud = parser.add_argument_group("deepgram")
    cloud.add_argument("--deepgram-model", default=DEEPGRAM_MODEL, help="Deepgram model (default: %(default)s)")
    cloud.add_argument("--deepgram-key", help="API key; defaults to ${}".format(DEEPGRAM_ENV_KEY))
    cloud.add_argument("--diarize", action="store_true", help="label speakers (Deepgram only)")

    local = parser.add_argument_group("canary")
    local.add_argument("--canary-model", default=CANARY_MODEL, help="NeMo model id (default: %(default)s)")
    local.add_argument("--device", choices=["cuda", "cpu"], help="inference device (default: cuda when available)")
    local.add_argument(
        "--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
        help="max audio window per Canary pass (default: %(default)s)",
    )
    local.add_argument("--batch-size", type=int, default=4, help="chunks per Canary batch (default: %(default)s)")

    output = parser.add_argument_group("output")
    output.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    output.add_argument("-v", "--verbose", action="store_true", help="print debug logging")


def _selected_backend_names(choice: str) -> List[str]:
    if choice == "both":
        return list(backend_registry.DEFAULT_ORDER)
    if choice == "auto":
        for name in backend_registry.DEFAULT_ORDER:
            if backend_registry.build(name).check().ok:
                return [name]
        return []
    return [choice]


def _instantiate(names: Sequence[str], args: argparse.Namespace):
    options = {
        "deepgram": {
            "api_key": args.deepgram_key,
            "model": args.deepgram_model,
            "diarize": args.diarize,
        },
        "canary": {
            "model": args.canary_model,
            "device": args.device,
            "chunk_seconds": args.chunk_seconds,
            "batch_size": args.batch_size,
        },
    }
    return [backend_registry.build(name, **options.get(name, {})) for name in names]


def cmd_backends(_args: argparse.Namespace) -> int:
    print("Available transcription approaches:\n")
    any_ok = False
    for name in backend_registry.DEFAULT_ORDER:
        backend = backend_registry.build(name)
        status = backend.check()
        any_ok = any_ok or status.ok
        print("  {} {:<10} {}".format("OK " if status.ok else "-- ", name, backend.description))
        print("     {}".format(status.reason))
    print()
    if not any_ok:
        print("No backend is usable yet. Run `videotranscriptor doctor` for setup steps.")
        return EXIT_ERROR
    return EXIT_OK


def cmd_doctor(_args: argparse.Namespace) -> int:
    problems = 0

    print("ffmpeg")
    try:
        audio_utils.ensure_ffmpeg()
        print("  OK  ffmpeg and ffprobe found")
    except audio_utils.FFmpegError as exc:
        problems += 1
        print("  --  {}".format(exc))

    print("\ndeepgram (cloud approach)")
    status = backend_registry.build("deepgram").check()
    if status.ok:
        print("  OK  {}".format(status.reason))
    else:
        problems += 1
        print("  --  {}".format(status.reason))
        print("      export {}=...  (or put it in a .env file)".format(DEEPGRAM_ENV_KEY))

    print("\ncanary (local approach)")
    canary = backend_registry.build("canary")
    status = canary.check()
    if status.ok:
        print("  OK  {}".format(status.reason))
        if canary.resolve_device() == "cpu":
            print("      no CUDA device: expect roughly realtime-or-slower decoding")
    else:
        problems += 1
        print("  --  {}".format(status.reason))
        print("      pip install 'videotranscriptor[local]'")

    print()
    if problems:
        print("{} issue(s) found.".format(problems))
        return EXIT_ERROR
    print("Everything checks out.")
    return EXIT_OK


def cmd_transcribe(args: argparse.Namespace) -> int:
    if not args.inputs:
        print("error: no input files given (see --help)", file=sys.stderr)
        return EXIT_ERROR

    names = _selected_backend_names(args.backend)
    if not names:
        print(
            "error: no backend is usable. Run `videotranscriptor doctor` for setup steps.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    output_formats = args.output_formats or ["txt", "srt", "json"]
    backends = _instantiate(names, args)
    language = None if args.language == "auto" else args.language

    def progress(message: str) -> None:
        if not args.quiet:
            print("  {}".format(message), flush=True)

    exit_code = EXIT_OK
    for source in args.inputs:
        if not source.exists():
            print("error: no such file: {}".format(source), file=sys.stderr)
            exit_code = EXIT_ERROR
            continue

        if not args.quiet:
            print("\n=== {} ===".format(source))
        try:
            result = pipeline.run(
                source,
                backends=backends,
                output_dir=args.output_dir,
                output_formats=output_formats,
                language=language,
                keep_audio=args.keep_audio,
                include_raw=args.keep_raw,
                write_comparison=not args.no_compare,
                progress=progress,
            )
        except (audio_utils.FFmpegError, TranscriptionError, FileNotFoundError, OSError) as exc:
            print("error: {}: {}".format(source.name, exc), file=sys.stderr)
            exit_code = EXIT_ERROR
            continue

        exit_code = max(exit_code, _report(result, quiet=args.quiet))
    return exit_code


def _report(result: pipeline.PipelineResult, quiet: bool) -> int:
    if quiet:
        for run in result.successful:
            for path in run.outputs.values():
                print(path)
        return EXIT_ERROR if not result.successful else (
            EXIT_PARTIAL if result.failed else EXIT_OK
        )

    print()
    for run in result.runs:
        if run.ok and run.transcript is not None:
            transcript = run.transcript
            speed = transcript.realtime_factor
            print(
                "  {:<9} {} words, {} segments{}".format(
                    run.backend,
                    transcript.word_count,
                    len(transcript.segments),
                    ", {:.2f}x realtime".format(speed) if speed else "",
                )
            )
            for fmt, path in sorted(run.outputs.items()):
                print("            {:<4} {}".format(fmt, path))
        else:
            print("  {:<9} FAILED: {}".format(run.backend, run.error))

    if result.comparison is not None:
        print()
        for line in result.comparison.to_text().splitlines():
            print("  {}".format(line))
        if result.comparison_path:
            print("  report         : {}".format(result.comparison_path))

    if not result.successful:
        return EXIT_ERROR
    return EXIT_PARTIAL if result.failed else EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = "transcribe"
    if arguments and arguments[0] in SUBCOMMANDS:
        command = arguments.pop(0)

    if command == "backends":
        args = build_simple_parser(command, "Show which approaches are usable.").parse_args(arguments)
        handler = cmd_backends
    elif command == "doctor":
        args = build_simple_parser(command, "Check ffmpeg, credentials and model deps.").parse_args(arguments)
        handler = cmd_doctor
    else:
        args = build_parser().parse_args(arguments)
        handler = cmd_transcribe

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    load_env()
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
