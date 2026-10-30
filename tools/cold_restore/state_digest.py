"""Read actual state at the original before-train-step hook; never changes it."""
import argparse
import enum
import hashlib
import io
import json
import math
import os
from pathlib import Path
import numpy as np
import torch


def digest(value):
    if isinstance(value, torch.Tensor):
        x=value.detach().to('cpu').contiguous()
        h=hashlib.sha256(); raw=x.reshape(-1).view(torch.uint8).numpy()
        for start in range(0,raw.size,8*1024**2): h.update(memoryview(raw[start:start+8*1024**2]))
        finite=bool(torch.isfinite(x).all()) if x.is_floating_point() or x.is_complex() else True
        if not finite: raise ValueError('nonfinite tensor in actual state')
        return {'type':'tensor','dtype':str(x.dtype),'shape':list(x.shape),'sha256':h.hexdigest(),'finite':finite}
    if isinstance(value,np.ndarray):
        x=np.ascontiguousarray(value)
        return {'type':'numpy','dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(memoryview(x).cast('B')).hexdigest()}
    if isinstance(value,io.BytesIO): return digest(value.getvalue())
    if isinstance(value,bytes): return {'type':'bytes','length':len(value),'sha256':hashlib.sha256(value).hexdigest()}
    if isinstance(value,enum.Enum): return {'type':type(value).__module__+'.'+type(value).__qualname__,'value':digest(value.value)}
    if isinstance(value,(torch.dtype,torch.device)): return {'type':type(value).__name__,'value':str(value)}
    if value is None or isinstance(value,(str,bool,int)): return value
    if isinstance(value,float):
        if not math.isfinite(value): raise ValueError('nonfinite scalar in state')
        return value
    if isinstance(value,dict): return {'type':'dict','items':[[digest(k),digest(v)] for k,v in sorted(value.items(),key=lambda kv:repr(kv[0]))]}
    if isinstance(value,(tuple,list)): return {'type':type(value).__name__,'items':[digest(x) for x in value]}
    if isinstance(value,argparse.Namespace): return {'type':'Namespace','value':digest(vars(value))}
    raise TypeError('Unqualified state type: '+type(value).__module__+'.'+type(value).__qualname__)


def before_train_step(args,rollout_id,step_id,model,optimizer,opt_param_scheduler):
    """Original hook ABI; both control and resume must use this same observer.

    This hashes complete local model state, all optimizer master/moment buffers
    through the selected original DP-zero getter, scheduler and original RNG API.
    Readback may synchronize work. It is not a performance measurement.
    """
    from megatron.core import mpu
    from megatron.training.checkpointing import get_rng_state
    assert args.tensor_model_parallel_size==4 and args.data_parallel_size==1
    outroot=Path(os.environ['SLIME_COLD_STATE_DIR'])
    outroot.mkdir(parents=True,exist_ok=True)
    rank=torch.distributed.get_rank()
    # Preserve the real iteration-zero selector before control saves iteration one.
    # Selected MCore uses a truthy ckpt_step test, so --ckpt-step 0 cannot override
    # a later tracker. Future resume mounts this byte and the SAME control iter0.
    if rank==0 and rollout_id==1 and os.environ.get('SLIME_COLD_PRESERVE_SPLIT')=='1':
        tracker=Path(args.save)/'latest_checkpointed_iteration.txt'
        raw=tracker.read_bytes()
        if raw.strip()!=b'0': raise RuntimeError('Control split tracker is not actual iteration zero')
        (outroot/'split_latest_checkpointed_iteration.txt').write_bytes(raw)
    out={'rank':rank,'rollout_id':rollout_id,'step_id':step_id,'model':[],
         'optimizer':[],'scheduler':digest(opt_param_scheduler.state_dict()),
         'RNG':digest(get_rng_state('torch',mpu.get_tensor_model_parallel_group(),mpu.get_pipeline_model_parallel_group()))}
    for part in model: out['model'].append(digest(part.state_dict()))
    children=getattr(optimizer,'chained_optimizers',[optimizer])
    for child in children:
        # The original state_dict alone intentionally excludes parameter/moment buffers.
        state=child.get_parameter_state_dp_zero(use_gloo_comm=True,empty_data=False,return_on_all_ranks=False)
        if state is None: raise RuntimeError('DP1 parameter state unavailable')
        out['optimizer'].append({'metadata':digest(child.state_dict()),'master_and_moments':digest(state)})
        del state
    path=outroot/f'rollout_{rollout_id}_step_{step_id}_rank_{rank}.json'
    if path.exists(): raise FileExistsError(str(path))
    path.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
