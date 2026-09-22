"""Tests for :mod:`osbad.config`."""
import json
import logging

import pytest

import osbad.config as bconf


class TestFindRepoRoot:
    """Tests for :func:`osbad.config.find_repo_root`."""

    def test_finds_marker_in_current_dir(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        monkeypatch.chdir(tmp_path)
        assert bconf.find_repo_root() == tmp_path

    def test_finds_marker_in_parent_dir(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        child = tmp_path / "a" / "b"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        assert bconf.find_repo_root() == tmp_path

    def test_custom_marker(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("KEY=value\n")
        monkeypatch.chdir(tmp_path)
        assert bconf.find_repo_root(".env") == tmp_path

    def test_missing_marker_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            bconf.find_repo_root("does_not_exist.marker")


class TestArtifactsOutputDir:
    """Tests for :func:`osbad.config.artifacts_output_dir`."""

    def test_creates_cell_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bconf, "PIPELINE_OUTPUT_DIR", tmp_path)
        result = bconf.artifacts_output_dir("cell_42")
        assert result == tmp_path / "cell_42"
        assert result.exists()

    def test_idempotent_when_dir_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bconf, "PIPELINE_OUTPUT_DIR", tmp_path)
        first = bconf.artifacts_output_dir("cell_42")
        second = bconf.artifacts_output_dir("cell_42")
        assert first == second
        assert second.exists()


class TestJsonHpConfig:
    """Round-trip tests for the JSON hyperparameter config helpers."""

    def test_create_writes_file(self, tmp_path):
        out = tmp_path / "hp.json"
        payload = {"contamination": {"low": 0.0, "high": 0.5}}
        bconf.create_json_hp_config(str(out), payload)
        assert out.exists()
        assert json.loads(out.read_text()) == payload

    def test_load_returns_dict(self, tmp_path):
        out = tmp_path / "hp.json"
        payload = {"n_estimators": {"low": 100, "high": 500}}
        bconf.create_json_hp_config(str(out), payload)
        assert bconf.load_json_hp_config(str(out)) == payload

    def test_round_trip_preserves_content(self, tmp_path):
        out = tmp_path / "hp.json"
        payload = {"a": {"low": 1, "high": 2}, "b": {"low": 3.5, "high": 4.5}}
        bconf.create_json_hp_config(str(out), payload)
        assert bconf.load_json_hp_config(str(out)) == payload


class TestCustomFormatter:
    """Tests for :class:`osbad.config.CustomFormatter`."""

    def _make_record(self, level, msg):
        return logging.LogRecord(
            name="osbad",
            level=level,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_info_format_contains_message(self):
        formatter = bconf.CustomFormatter()
        out = formatter.format(self._make_record(logging.INFO, "hello"))
        assert "hello" in out

    def test_debug_format_contains_message(self):
        formatter = bconf.CustomFormatter()
        out = formatter.format(self._make_record(logging.DEBUG, "debugmsg"))
        assert "debugmsg" in out
