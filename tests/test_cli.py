import pytest

from videotranscriptor import backends as backend_registry
from videotranscriptor.backends import CanaryBackend, DeepgramBackend
from videotranscriptor.cli import _selected_backend_names, build_parser, main
from videotranscriptor.config import parse_env


def test_default_run_uses_both_backends():
    assert _selected_backend_names("both") == ["deepgram", "canary"]


def test_single_backend_selection():
    assert _selected_backend_names("canary") == ["canary"]


def test_build_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        backend_registry.build("whisper")


def test_build_ignores_options_the_backend_does_not_accept():
    backend = backend_registry.build("deepgram", model="nova-2", device="cuda")
    assert isinstance(backend, DeepgramBackend)
    assert backend.model == "nova-2"


def test_build_passes_through_canary_options():
    backend = backend_registry.build("canary", model="nvidia/canary-1b", chunk_seconds=15.0)
    assert isinstance(backend, CanaryBackend)
    assert backend.chunk_seconds == 15.0


def test_parser_defaults():
    args = build_parser().parse_args(["clip.mp4"])
    assert [str(p) for p in args.inputs] == ["clip.mp4"]
    assert args.backend == "both"
    assert args.language == "auto"
    assert args.output_formats is None  # falls back to txt/srt/json


def test_repeated_format_flags_accumulate():
    args = build_parser().parse_args(["clip.mp4", "-f", "srt", "-f", "vtt"])
    assert args.output_formats == ["srt", "vtt"]


def test_unknown_format_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["clip.mp4", "-f", "docx"])


def test_backends_subcommand_lists_both(capsys):
    exit_code = main(["backends"])
    output = capsys.readouterr().out
    assert "deepgram" in output and "canary" in output
    assert exit_code in {0, 1}  # 1 when neither is configured on this machine


def test_transcribe_without_inputs_is_an_error(capsys):
    assert main([]) == 1
    assert "no input files" in capsys.readouterr().err


def test_missing_input_file_is_reported(capsys, tmp_path):
    assert main([str(tmp_path / "nope.mp4"), "-b", "deepgram"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_env_file_parsing():
    values = parse_env(
        "\n".join(
            [
                "# a comment",
                "DEEPGRAM_API_KEY=abc123",
                'export QUOTED="with spaces"',
                "SINGLE='single'",
                "TRAILING=value # inline comment",
                "malformed-line",
                "",
            ]
        )
    )
    assert values == {
        "DEEPGRAM_API_KEY": "abc123",
        "QUOTED": "with spaces",
        "SINGLE": "single",
        "TRAILING": "value",
    }


def test_load_env_does_not_override_the_real_environment(tmp_path, monkeypatch):
    from videotranscriptor.config import load_env

    env_file = tmp_path / ".env"
    env_file.write_text("DEEPGRAM_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.setenv("DEEPGRAM_API_KEY", "from-shell")
    load_env(env_file)
    import os

    assert os.environ["DEEPGRAM_API_KEY"] == "from-shell"

    load_env(env_file, override=True)
    assert os.environ["DEEPGRAM_API_KEY"] == "from-file"
