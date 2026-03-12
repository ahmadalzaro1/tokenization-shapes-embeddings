---
phase: 02
slug: tokenizer-baseline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x (already installed) |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/ -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 02-01-01 | 01 | 1 | TOK-01 | unit | `uv run pytest tests/test_tokenizer.py::test_vocab_size_flag -q` | ⬜ pending |
| 02-01-02 | 01 | 1 | TOK-04 | unit | `uv run pytest tests/test_tokenizer.py::test_fertility_report_written -q` | ⬜ pending |
| 02-02-01 | 02 | 2 | TOK-01,TOK-02 | integration | `uv run pytest tests/test_tokenizer.py::test_tokenizer_files_exist -q` | ⬜ pending |
| 02-02-02 | 02 | 2 | TOK-03 | integration | `uv run pytest tests/test_tokenizer.py::test_fertility_table_conditions -q` | ⬜ pending |
| 02-03-01 | 03 | 3 | BASE-01 | unit | `uv run pytest tests/test_baseline.py::test_baseline_json_written -q` | ⬜ pending |
| 02-03-02 | 03 | 3 | BASE-02,BASE-03 | integration | `uv run pytest tests/test_baseline.py::test_d2_lower_than_d1 -q` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_tokenizer.py` — stubs for TOK-01 through TOK-04
- [ ] `tests/test_baseline.py` — stubs for BASE-01 through BASE-03

*Existing `tests/conftest.py` and pytest infrastructure from Phase 1 covers shared fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Fertility table shows measurable D1/D2/D3 difference | TOK-03 | Values are empirical — thresholds unknown until run | Check `fertility_report.json`: D3 tokens/word should differ from D1 and D2 |
| D2 val_bpb lower than D1 val_bpb | BASE-02 | val_bpb values are empirical | Check `baseline_results.json`: `d2.val_bpb < d1.val_bpb` |
| Baseline training runs complete within 5-min budget | BASE-01 | Timing is hardware-dependent | Each `train.py` run exits cleanly within 300s |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
