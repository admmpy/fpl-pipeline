"""
Unit tests for dbt command task behaviour.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.dbt_tasks import run_dbt_command


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def stub_prefect_logger(monkeypatch):
    monkeypatch.setattr("tasks.dbt_tasks.get_run_logger", lambda: DummyLogger())


def test_run_dbt_command_injects_absolute_project_dir(monkeypatch, tmp_path):
    project_dir = tmp_path / "dbt_project"
    project_dir.mkdir()

    called = {}

    def fake_run(cmd, cwd, capture_output, text, check):
        called["cmd"] = cmd
        called["cwd"] = cwd
        called["capture_output"] = capture_output
        called["text"] = text
        called["check"] = check
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("tasks.dbt_tasks.shutil.which", lambda _: "/usr/bin/dbt")
    monkeypatch.setattr("tasks.dbt_tasks.subprocess.run", fake_run)

    result = run_dbt_command.fn(
        command=["dbt", "build"],
        project_dir=str(project_dir),
        profiles_dir=None,
    )

    expected_project_dir = str(project_dir.resolve())

    assert result.is_success is True
    assert result.return_code == 0
    assert called["cwd"] == expected_project_dir
    assert called["capture_output"] is True
    assert called["text"] is True
    assert called["check"] is False

    assert called["cmd"].count("--project-dir") == 1
    project_flag_idx = called["cmd"].index("--project-dir")
    assert called["cmd"][project_flag_idx + 1] == expected_project_dir


def test_run_dbt_command_normalises_existing_project_dir_flag(monkeypatch, tmp_path):
    project_dir = tmp_path / "canonical_project"
    project_dir.mkdir()

    called = {}

    def fake_run(cmd, cwd, capture_output, text, check):
        called["cmd"] = cmd
        called["cwd"] = cwd
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("tasks.dbt_tasks.shutil.which", lambda _: "/usr/bin/dbt")
    monkeypatch.setattr("tasks.dbt_tasks.subprocess.run", fake_run)

    result = run_dbt_command.fn(
        command=[
            "dbt",
            "build",
            "--project-dir",
            "some/other/path",
            "--project-dir=another/path",
            "--select",
            "model_x",
        ],
        project_dir=str(project_dir),
        profiles_dir=None,
    )

    expected_project_dir = str(project_dir.resolve())

    assert result.is_success is True
    assert called["cwd"] == expected_project_dir
    assert called["cmd"].count("--project-dir") == 1
    assert "--project-dir=another/path" not in called["cmd"]

    project_flag_idx = called["cmd"].index("--project-dir")
    assert called["cmd"][project_flag_idx + 1] == expected_project_dir
    select_idx = called["cmd"].index("--select")
    assert called["cmd"][select_idx + 1] == "model_x"


def test_run_dbt_command_returns_error_for_missing_project_dir(monkeypatch, tmp_path):
    missing_dir = tmp_path / "missing_project"
    subprocess_called = False

    def fake_run(*args, **kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tasks.dbt_tasks.shutil.which", lambda _: "/usr/bin/dbt")
    monkeypatch.setattr("tasks.dbt_tasks.subprocess.run", fake_run)

    result = run_dbt_command.fn(
        command=["dbt", "build"],
        project_dir=str(missing_dir),
        profiles_dir=None,
    )

    assert result.is_success is False
    assert result.return_code == 2
    assert "Invalid project_dir" in result.stderr
    assert subprocess_called is False


def test_run_dbt_command_returns_127_when_dbt_not_found(monkeypatch, tmp_path):
    project_dir = tmp_path / "dbt_project"
    project_dir.mkdir()

    monkeypatch.setattr("tasks.dbt_tasks.shutil.which", lambda _: None)

    result = run_dbt_command.fn(
        command=["dbt", "build"],
        project_dir=str(project_dir),
        profiles_dir=None,
    )

    assert result.is_success is False
    assert result.return_code == 127
    assert result.stderr == "dbt not found on PATH"


def test_run_dbt_command_propagates_subprocess_failure(monkeypatch, tmp_path):
    project_dir = tmp_path / "dbt_project"
    project_dir.mkdir()

    monkeypatch.setattr("tasks.dbt_tasks.shutil.which", lambda _: "/usr/bin/dbt")
    monkeypatch.setattr(
        "tasks.dbt_tasks.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failure"),
    )

    result = run_dbt_command.fn(
        command=["dbt", "build"],
        project_dir=str(project_dir),
        profiles_dir=None,
    )

    assert result.is_success is False
    assert result.return_code == 1
    assert result.stderr == "failure"


def test_run_dbt_command_resolves_profiles_dir(monkeypatch, tmp_path):
    project_dir = tmp_path / "dbt_project"
    profiles_dir = tmp_path / "profiles"
    project_dir.mkdir()
    profiles_dir.mkdir()

    called = {}

    def fake_run(cmd, cwd, capture_output, text, check):
        called["cmd"] = cmd
        called["cwd"] = cwd
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("tasks.dbt_tasks.shutil.which", lambda _: "/usr/bin/dbt")
    monkeypatch.setattr("tasks.dbt_tasks.subprocess.run", fake_run)

    result = run_dbt_command.fn(
        command=["dbt", "build"],
        project_dir=str(project_dir),
        profiles_dir=str(profiles_dir),
    )

    assert result.is_success is True
    assert called["cwd"] == str(project_dir.resolve())

    assert called["cmd"].count("--profiles-dir") == 1
    profiles_idx = called["cmd"].index("--profiles-dir")
    assert called["cmd"][profiles_idx + 1] == str(profiles_dir.resolve())
