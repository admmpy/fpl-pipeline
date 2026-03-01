"""Security tests for redaction in autonomous evidence/log output."""

from __future__ import annotations

import json
from pathlib import Path

from agents.autonomous_nodes import record_evidence, redact_sensitive
from agents.autonomous_state import create_initial_state


def test_redact_sensitive_redacts_nested_secret_like_keys():
    payload = {
        "API_TOKEN": "abc",
        "nested": {
            "private_key": "xyz",
            "safe": 1,
            "items": [{"SECRET": "value"}, {"normal": "ok"}],
        },
    }

    redacted = redact_sensitive(payload)
    assert redacted["API_TOKEN"] == "[REDACTED]"
    assert redacted["nested"]["private_key"] == "[REDACTED]"
    assert redacted["nested"]["items"][0]["SECRET"] == "[REDACTED]"
    assert redacted["nested"]["items"][1]["normal"] == "ok"


def test_record_evidence_does_not_persist_secret_fields(tmp_path, monkeypatch):
    from agents import autonomous_nodes

    log_dir = tmp_path / "autonomous"
    monkeypatch.setattr(autonomous_nodes, "AUTONOMOUS_LOG_DIR", log_dir)
    monkeypatch.setattr(autonomous_nodes, "AUTONOMOUS_EVENTS_PATH", log_dir / "events.jsonl")

    state = create_initial_state("run-sec")
    state["state"] = "REJECTED"
    state["promotion_decision"] = {"API_KEY": "should-not-appear", "decision": "reject"}
    state["candidate_metrics"] = {"TOKEN_VALUE": "nope", "mae": 4.2}

    update = record_evidence(state)
    evidence_path = Path(update["evidence_path"])
    content = json.loads(evidence_path.read_text(encoding="utf-8"))

    serialised = json.dumps(content)
    assert "should-not-appear" not in serialised
    assert "nope" not in serialised
    assert "[REDACTED]" in serialised
