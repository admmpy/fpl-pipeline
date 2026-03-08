"""Tests for domain rules schema validation."""

from pathlib import Path

import pytest
import yaml

from agents.autonomous_nodes import DomainRulesError, load_domain_rules


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "config" / "domain_rules.yaml"


def test_default_rules_file_is_schema_valid():
    rules = load_domain_rules(DEFAULT_RULES)
    assert rules["version"] == "1.2.0"
    assert "schema" in rules
    assert "drift" in rules
    assert "gameweek_quality" in rules


def test_invalid_rules_file_fails_fast(tmp_path):
    invalid_rules = {
        "version": "1.0.0",
        "schema": {
            "required_columns": ["player_id"],
            "dtypes": {"player_id": "integer"},
            "null_limits": {},
            # Missing duplicate_key_limits
        },
        "split": {"holdout_gameweeks": 5},
    }
    rules_path = tmp_path / "invalid_rules.yaml"
    rules_path.write_text(yaml.safe_dump(invalid_rules), encoding="utf-8")

    with pytest.raises(DomainRulesError):
        load_domain_rules(rules_path)
