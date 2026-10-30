import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.utils.npu_profiler import NPUProfilerConfig, NPUProfilerContents
from slime.utils.profile_utils import get_profiled_rollout_step, is_step_in_profile_window

pytest.importorskip("torch")


def test_npu_profiler_contents_parses_expected_flags():
    parsed = NPUProfilerContents.from_contents(["npu", "cpu", "memory", "shapes", "module", "stack"])

    assert parsed.include_npu is True
    assert parsed.with_cpu is True
    assert parsed.with_memory is True
    assert parsed.record_shapes is True
    assert parsed.with_modules is True
    assert parsed.with_stack is True
    assert parsed.has("cpu") is True
    assert parsed.has("invalid") is False


def test_npu_profiler_contents_rejects_invalid_items():
    with pytest.raises(ValueError, match="Profiler contents only supports"):
        NPUProfilerContents.from_contents(["npu", "bad_flag"])


def test_npu_profiler_config_applies_contents_to_legacy_flags():
    config = NPUProfilerConfig(
        enabled=True,
        contents=["cpu", "memory", "shapes", "module"],
        ranks=[0],
        current_rank=0,
    )

    assert config.with_cpu is True
    assert config.with_memory is True
    assert config.record_shapes is True
    assert config.with_modules is True
    assert config.with_stack is False
    assert config.uses_content("cpu") is True
    assert config.uses_content("npu") is False


@pytest.mark.parametrize(
    ("step", "step_start", "step_end", "expected"),
    [
        (0, 0, -1, True),
        (3, 2, 5, True),
        (5, 2, 5, False),
        (1, 2, 5, False),
    ],
)
def test_is_step_in_profile_window(step, step_start, step_end, expected):
    assert is_step_in_profile_window(step, step_start, step_end) is expected


def test_get_profiled_rollout_step_uses_computed_global_step():
    args = SimpleNamespace(
        wandb_always_use_train_step=True,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=32,
    )

    step = get_profiled_rollout_step(args, rollout_id=3, step_start=10, step_end=15)

    assert step == 12


def test_get_profiled_rollout_step_returns_none_outside_window():
    args = SimpleNamespace(
        wandb_always_use_train_step=False,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=32,
    )

    assert get_profiled_rollout_step(args, rollout_id=6, step_start=2, step_end=6) is None
