"""Reviewer CLI entry point, custom input and effective prompt."""

from pathlib import Path

from jevloop.cli import main


def test_custom_cli_preserves_messages_and_shows_prompt(tmp_path: Path, capsys) -> None:
    transcript = tmp_path / "demo.jsonl"
    assert (
        main(
            [
                "--mode",
                "offline",
                "--generator-mode",
                "mock",
                "--message",
                "first custom request",
                "--message",
                "second custom request",
                "--transcript",
                str(transcript),
            ]
        )
        == 0
    )
    assert len(transcript.read_text().splitlines()) == 2
    output = capsys.readouterr().out
    assert "actual effective system prompt:" in output
    assert "first custom request" in output
    assert "second custom request" in output
    assert "Deterministic mock" in output
