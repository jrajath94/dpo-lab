"""Regression tests for the CLI entry points.

These exist because a NameError (missing `utils` import) in data.main()
slipped past 110 unit tests and killed a GPU run: the pure functions were
all tested, but main() itself was never executed. Every module main()
gets executed here with heavy work mocked out.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CONFIG = str(REPO_ROOT / "configs" / "dpo_qwen25_15b_qlora.yaml")


def _run_main(module_name: str, argv: list[str]):
    """Import module and run its main() with patched sys.argv."""
    import importlib
    mod = importlib.import_module(module_name)
    with mock.patch.object(sys, "argv", argv):
        mod.main()


def test_data_main_runs_through_validation(tmp_path):
    """data.main() must resolve every name it uses up to the HF download.

    build_splits_from_hf is mocked (no network); the test fails if main()
    raises NameError/AttributeError before reaching it, e.g. the missing
    `utils` import that killed the first pod run.
    """
    from dpo_weekend import data

    out_dir = tmp_path / "splits"
    with mock.patch.object(
        data, "build_splits_from_hf", return_value={}
    ) as m:
        _run_main("dpo_weekend.data",
                  ["data", "--config", CONFIG, "--out-dir", str(out_dir)])
    m.assert_called_once()
    kwargs = m.call_args.kwargs
    assert kwargs["n_train"] == 10000
    assert kwargs["n_margin_eval"] == 1000
    assert kwargs["n_gen"] == 200
    assert kwargs["seed"] == 42


def test_data_main_rejects_bad_config(tmp_path):
    """validate_config must fire from inside data.main(), not just in
    isolation."""
    import yaml
    from dpo_weekend import data

    bad = tmp_path / "bad.yaml"
    cfg = {"model": {"name": "x"}}  # missing everything else
    bad.write_text(yaml.safe_dump(cfg))
    with mock.patch.object(data, "build_splits_from_hf"):
        with pytest.raises(ValueError):
            _run_main("dpo_weekend.data",
                      ["data", "--config", str(bad),
                       "--out-dir", str(tmp_path / "s")])
