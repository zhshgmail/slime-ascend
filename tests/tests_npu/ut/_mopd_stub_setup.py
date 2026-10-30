"""Dependency stubs for the MOPD CPU unit tests (test_mopd.py).

Heavy or unavailable third-party imports (megatron, sglang_router, wandb,
PIL/transformers via processing_utils) are stubbed in sys.modules *before*
the slime modules are imported. test_mopd_npu.py imports this module first so
the code under test runs for real on CPU.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# ``slime.backends.megatron_utils`` imports megatron_adaptor for its NPU-side
# patching side effects. The MOPD tests replace Megatron with the lightweight
# stub below, so importing the real adaptor in an NPU CI image would make it
# initialize against that incomplete stub during test collection.
sys.modules.setdefault('megatron_adaptor', types.ModuleType('megatron_adaptor'))

# The shared helper remains in the tests root. The Ascend UT gate invokes
# pytest on tests/tests_npu/ut directly, so add that parent explicitly.
_TESTS_ROOT = Path(__file__).resolve().parents[2]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

# _cp_dist_helpers stubs ``megatron.core.mpu`` (shared with the CP tests).
import _cp_dist_helpers  # noqa: E402, F401
from megatron.core import mpu as _mpu_stub  # noqa: E402

pytest.importorskip("torch")

# Extend the shared mpu stub with the attributes MOPD code paths touch.
for _name, _fn in {
    "get_tensor_model_parallel_group": lambda: None,
    "get_tensor_model_parallel_world_size": lambda: 1,
    "get_tensor_model_parallel_rank": lambda: 0,
    "is_pipeline_last_stage": lambda: True,
    "get_data_parallel_group": lambda *a, **k: None,
    "get_data_parallel_world_size": lambda *a, **k: 1,
    "get_data_parallel_rank": lambda *a, **k: 0,
    "get_virtual_pipeline_model_parallel_world_size": lambda: None,
}.items():
    if not hasattr(_mpu_stub, _name):
        setattr(_mpu_stub, _name, _fn)

# megatron.core.packed_seq_params (imported by megatron_utils/data.py).
_psp_stub = types.ModuleType("megatron.core.packed_seq_params")


class _PackedSeqParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


_psp_stub.PackedSeqParams = _PackedSeqParams
sys.modules.setdefault("megatron.core.packed_seq_params", _psp_stub)
sys.modules["megatron.core"].packed_seq_params = _psp_stub


def _install_module_stub(name: str, attrs: dict) -> types.ModuleType:
    """Install a stub module in sys.modules and as an attribute of its parent."""
    stub = types.ModuleType(name)
    stub.__path__ = []
    for key, value in attrs.items():
        setattr(stub, key, value)
    if name not in sys.modules:
        sys.modules[name] = stub
        parent_name, _, child = name.rpartition(".")
        if parent_name and parent_name in sys.modules:
            setattr(sys.modules[parent_name], child, sys.modules[name])
    return sys.modules[name]


# slime.utils.processing_utils pulls in PIL + transformers; the MOPD payload
# builder only needs encode_image_for_rollout_engine.
_PROCESSING_UTILS_STUB = _install_module_stub(
    "slime.utils.processing_utils",
    {"encode_image_for_rollout_engine": lambda image: f"encoded:{image}"},
)

# slime.utils.logging_utils pulls in wandb.
_LOGGING_UTILS_STUB = _install_module_stub(
    "slime.utils.logging_utils",
    {"configure_logger": lambda *a, **k: None, "log": lambda *a, **k: None},
)

# slime.utils.train_metric_utils pulls in the wandb chain (data.py imports it).
_TRAIN_METRIC_UTILS_STUB = _install_module_stub(
    "slime.utils.train_metric_utils",
    {"log_perf_data_raw": lambda *a, **k: None},
)

# sglang_router (imported by slime.utils.arguments).
_SGLANG_ROUTER_STUB = _install_module_stub("sglang_router", {})


class _RouterArgs:
    @staticmethod
    def add_cli_args(parser, **kwargs):
        return parser


_LAUNCH_ROUTER_STUB = _install_module_stub("sglang_router.launch_router", {"RouterArgs": _RouterArgs})

# slime.backends.sglang_utils.arguments pulls in sglang proper.
_SGLANG_UTILS_STUB = _install_module_stub("slime.backends.sglang_utils", {})
_SGLANG_ARGS_STUB = _install_module_stub(
    "slime.backends.sglang_utils.arguments",
    {
        "sglang_parse_args": lambda *a, **k: None,
        "validate_args": lambda args: args,
    },
)

# slime.rollout.rm_hub is imported lazily inside combined_reward_func; stub it
# so tests can monkeypatch async_rm / batched_async_rm without the real deps.
rm_hub_stub = _install_module_stub(
    "slime.rollout.rm_hub",
    {
        "async_rm": None,  # set by tests via monkeypatch
        "batched_async_rm": None,
    },
)
