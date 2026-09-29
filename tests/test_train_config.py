"""Regression tests for the trl 1.14 DPOConfig break (run 2, 2026-09-28).

The pod installed trl==1.14.0, whose DPOConfig removed max_prompt_length.
build_dpo_config crashed before any gradient step. These tests lock the
fix: a signature filter drops only non-critical kwargs, and the real YAML
still builds a valid DPOConfig on the installed trl.
"""

import pytest

from dpo_weekend.train import _filter_kwargs_for


class _FakeOldConfig:
    """Mimics the trl 0.9-era DPOConfig signature."""

    def __init__(self, beta, max_length, max_prompt_length, learning_rate):
        pass


class _FakeNewConfig:
    """Mimics trl 1.14: max_prompt_length removed, everything else kept."""

    def __init__(self, beta, max_length, learning_rate):
        pass


def test_filter_keeps_everything_on_old_api():
    kwargs = {"beta": 0.1, "max_length": 1024, "max_prompt_length": 512,
              "learning_rate": 1e-4}
    out = _filter_kwargs_for(_FakeOldConfig, kwargs, {"beta"})
    assert out == kwargs


def test_filter_drops_removed_kwarg_with_warning(capsys):
    kwargs = {"beta": 0.1, "max_length": 1024, "max_prompt_length": 512,
              "learning_rate": 1e-4}
    out = _filter_kwargs_for(_FakeNewConfig, kwargs, {"beta"})
    assert out == {"beta": 0.1, "max_length": 1024, "learning_rate": 1e-4}
    assert "max_prompt_length" in capsys.readouterr().out


def test_filter_raises_on_missing_critical_kwarg():
    kwargs = {"beta": 0.1, "max_length": 1024}
    with pytest.raises(TypeError, match="critical"):
        _filter_kwargs_for(_FakeNewConfig, kwargs, {"beta", "missing_kwarg"})


def test_filter_passes_through_kwargs_catchall():
    class _CatchAll:
        def __init__(self, beta, **kwargs):
            pass

    kwargs = {"beta": 0.1, "anything_goes": 1}
    out = _filter_kwargs_for(_CatchAll, kwargs, {"beta"})
    assert out == kwargs


def test_build_dpo_config_with_real_yaml(tmp_path):
    """Runs on the pod (trl installed); skipped on the VM.

    This is the smoke gate that would have caught the run-2 crash in
    seconds: build the config from the real YAML and check the values
    that define the experiment.
    """
    trl = pytest.importorskip("trl")
    from dpo_weekend.train import build_dpo_config, load_config

    config = load_config("configs/dpo_qwen25_15b_qlora.yaml")
    dpo_args = build_dpo_config(config, tmp_path)
    assert isinstance(dpo_args, trl.DPOConfig)
    assert dpo_args.beta == pytest.approx(0.1)
    assert dpo_args.max_length == 1024
    assert dpo_args.num_train_epochs == 1
    assert dpo_args.learning_rate == pytest.approx(1e-4)
    assert dpo_args.seed == 42
    assert dpo_args.bf16 is True
    # transformers 5.x removed warmup_ratio; the config's warmup_ratio
    # value must arrive as warmup_steps (float < 1 = ratio semantics).
    assert dpo_args.warmup_steps == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Hub revision pin (audit finding 2026-09-28): from_pretrained was called
# without revision=, so a hub weight update would silently change the base
# model. The config now pins the exact revision and every loader passes it
# through via utils.hf_model_kwargs.
# ---------------------------------------------------------------------------

def test_config_pins_model_revision():
    import yaml

    with open("configs/dpo_qwen25_15b_qlora.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    rev = config["model"].get("revision")
    assert isinstance(rev, str) and len(rev) >= 7, \
        "model.revision must be a pinned hub revision, not empty"


def test_hf_model_kwargs_passes_revision_through():
    from dpo_weekend.utils import hf_model_kwargs

    out = hf_model_kwargs({"name": "org/model",
                           "revision": "abc1234",
                           "trust_remote_code": True})
    assert out == {"name": "org/model", "revision": "abc1234",
                   "trust_remote_code": True}


def test_hf_model_kwargs_revision_none_means_hub_default():
    from dpo_weekend.utils import hf_model_kwargs

    out = hf_model_kwargs({"name": "org/model"})
    assert out["revision"] is None
    assert out["trust_remote_code"] is False


def test_config_with_revision_still_validates():
    import yaml

    from dpo_weekend.utils import validate_config

    with open("configs/dpo_qwen25_15b_qlora.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)  # extra keys must not break validation
