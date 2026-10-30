import logging

import torch

from slime.utils.common import is_npu

if is_npu():
    import megatron_adaptor  # noqa: F401

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLTextRotaryEmbedding

    _original_forward = Qwen3VLTextRotaryEmbedding.forward

    def _patched_forward(self, *args, packed_seq_params=None, packed_seq=None, cp_group=None, **kwargs):
        return _original_forward(self, *args, **kwargs)

    Qwen3VLTextRotaryEmbedding.forward = _patched_forward
except ImportError:
    pass

try:
    from mbridge.models.qwen3_vl.model import Qwen3VLModel

    _original_forward2 = Qwen3VLModel.forward

    def _patched_forward2(self, *args, loss_mask=None, **kwargs):
        return _original_forward2(self, *args, **kwargs)

    Qwen3VLModel.forward = _patched_forward2
except ImportError:
    pass
try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel  # pylint: disable=ungrouped-imports

    _original_forward3 = Qwen3VLModel.forward

    def _patched_forward3(self, *args, loss_mask=None, **kwargs):
        return _original_forward3(self, *args, **kwargs)

    Qwen3VLModel.forward = _patched_forward3
except ImportError:
    pass
logging.getLogger("megatron").setLevel(logging.WARNING)

from . import megatron_patch  # noqa: F401, E402
