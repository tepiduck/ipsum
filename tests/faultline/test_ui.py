from pathlib import Path

import pytest

from faultline.cli import main
from faultline.storage import atomic_json
from faultline.ui import campaigns, choose, launch


def answers(*values):
    iterator = iter(values)
    return lambda _: next(iterator)


def test_menu_reprompts_and_back_does_not_execute():
    messages = []
    assert choose("Select", ["one"], answers("bad", "9", "0"), messages.append) is None
    assert messages.count("Enter one of the listed numbers.") == 2


def test_terminal_ui_dispatches_real_cli_arguments(tmp_path):
    calls = []
    code = launch(
        tmp_path,
        execute=lambda args: calls.append(args) or 0,
        ask=answers("2", "3", "", "0"),
        say=lambda text: None,
    )
    assert code == 0
    command = calls[0]
    assert command[:5] == ["run", "--backend", "eager", "--device", "cpu"]
    assert command[-1] == "--corpus"
    assert Path(command[6]).parent == tmp_path


def test_terminal_ui_demo_creates_a_fresh_destination(tmp_path):
    calls = []
    launch(
        tmp_path,
        execute=lambda args: calls.append(args) or 0,
        ask=answers("1", "1", "0"),
        say=lambda text: None,
    )
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == "demo"
    assert calls[0][-1] != calls[1][-1]
    assert all(not Path(call[-1]).exists() for call in calls)


def test_terminal_ui_handles_closed_input(tmp_path):
    def closed(_):
        raise EOFError

    assert launch(tmp_path, ask=closed, say=lambda text: None) == 0


def test_campaign_browser_handles_incomplete_and_malformed_runs(tmp_path):
    atomic_json(tmp_path / "completed" / "summary.json", {"counts": {"pass": 2}})
    atomic_json(tmp_path / "broken" / "summary.json", [])
    (tmp_path / "empty").mkdir()
    found = campaigns(tmp_path)
    assert [(path.name, label) for path, label in found] == [
        ("broken", "unreadable summary"),
        ("completed", "2 pass"),
    ]


def test_version_and_ui_are_discoverable(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "Faultline 0.15.0" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert "demo" in text and "ui" in text and "replay" in text
