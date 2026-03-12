# Deferred Items — Phase 01-data-pipeline

## Out-of-scope issues discovered during execution (not fixed per scope boundary rule)

### 1. test_load_dataset_with_progress_keyboard_interrupt flaky failure

- **Discovered during:** 01-03 plan final verification
- **Plan where issue originates:** 01-02 (load_dataset_with_progress implementation)
- **Failure:** `test_load_dataset_with_progress_keyboard_interrupt` fails when a local dataset directory (`arabic-tashkeel-dataset/`) exists on disk. The monkeypatch targets `datasets.load_dataset` but the local-path branch calls `_load("parquet", data_files=...)` instead, causing the fake to receive an unexpected `data_files` kwarg.
- **Root cause:** Test assumes HF download path is taken, but local cache directory presence triggers an early return via the local-path branch.
- **Fix needed:** Either mock `LOCAL_DATASET_DIR.exists()` to return `False` in the test, or add a condition-check in the test to skip when local data is present.
- **Files:** `tests/test_pipeline.py::test_load_dataset_with_progress_keyboard_interrupt`, `build_dataset.py:load_dataset_with_progress`
- **Impact:** 1 failing test out of 28; does not affect production behavior. All plan 01-03 validation tests pass (8/8).
