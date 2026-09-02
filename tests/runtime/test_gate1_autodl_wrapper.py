from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts" / "run_gate1_autodl.sh"
EXPECTED_BASE = "3ef46650595e94e1a4d2ca48dc01e4ef1c424f1b"


def test_gate1_wrapper_has_frozen_order_and_development_only_commands() -> None:
    assert WRAPPER.is_file(), "missing AutoDL Gate 1 wrapper"
    source = WRAPPER.read_text(encoding="utf-8")

    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert f"EXPECTED_CODE_BASE='{EXPECTED_BASE}'" in source
    assert "git merge-base --is-ancestor" in source
    assert "--mode" not in source
    assert source.count("--seed 2026") == 5
    assert source.count("--max-epochs 100") == 4

    commands = (
        "python scripts/audit_hidfilter_gradients.py",
        "--variant no_edge_top_p",
        "--variant no_top_p",
        "python scripts/run_stid_performance_diagnosis.py",
        "--variant no_family_top_p",
    )
    positions = tuple(source.index(command) for command in commands)
    assert positions == tuple(sorted(positions))


def test_gate1_wrapper_preflights_every_report_before_running_python() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    reports = (
        "gradient_audit_seed_2026.json",
        "no_edge_top_p_seed_2026.json",
        "no_top_p_seed_2026.json",
        "stid_seed_2026.json",
        "no_family_top_p_seed_2026.json",
    )

    for report in reports:
        assert report in source
    assert source.index("for report_path in") < source.index(
        "python scripts/audit_hidfilter_gradients.py"
    )
    assert "if [[ -e \"${report_path}\" ]]" in source
    assert "GATE 1 AUTODL RUNS COMPLETE" in source
    assert "NO TEST WAS RUN." in source
