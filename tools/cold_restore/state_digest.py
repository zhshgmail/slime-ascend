"""Read actual state at the original before-train-step hook; never changes it."""
import argparse
import enum
import hashlib
import io
import json
import math
import os
import sys
import re
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


def metadata_geometry(meta):
    out={}
    for key,value in meta.state_dict_metadata.items():
        out[key]={'type':type(value).__name__}
        if hasattr(value,'size'):
            out[key].update(dtype=str(value.properties.dtype),shape=list(value.size),chunks=sorted([{'offsets':list(c.offsets),'sizes':list(c.sizes)} for c in value.chunks],key=lambda x:x['offsets']))
    return out


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b""): h.update(chunk)
    return h.hexdigest()


def authenticated_json(path, expected_sha256):
    raw=Path(path).read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "") or hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError("Unauthenticated contract/identity")
    return json.loads(raw)


def load_contract(path, expected_sha256, verify_files=False):
    c=authenticated_json(path,expected_sha256)
    if c.get('schema')!='slime-original-tp4-v1' or len(c.get('inventory',{}))!=522:
        raise ValueError('Expected authenticated original 522-entry inventory')
    binding=c.get('metadata',{})
    metadata_path=Path(binding.get('path',''))
    if not metadata_path.is_file() or sha256_file(metadata_path)!=binding.get('sha256'):
        raise ValueError('Canonical metadata authentication failed')
    from torch.distributed.checkpoint import FileSystemReader
    if metadata_geometry(FileSystemReader(metadata_path.parent).read_metadata())!=c['inventory']:
        raise ValueError('Inventory not derived from authenticated canonical metadata')
    g=c['inventory']
    if len([k for k in g if k.startswith('rng_state/')])!=4:
        raise ValueError('Expected four original RNG objects')
    for rank in range(4):
        for member in ('param','exp_avg','exp_avg_sq'):
            if not any(f'dp_group_idx_{rank}.' in k and k.endswith('.'+member) and 'shape' in v for k,v in g.items()):
                raise ValueError('Missing original optimizer inventory')
    if not any('shape' in v and not k.startswith('optimizer.') for k,v in g.items()):
        raise ValueError('Missing original model inventory')
    for section in ('source','input','runtime'):
        files=c.get('identity',{}).get(section,{}).get('files',{})
        if not files or any(not re.fullmatch(r'[0-9a-f]{64}',h or '') for h in files.values()):
            raise ValueError('Unknown '+section+' identity')
        if verify_files:
            for name,h in files.items():
                if sha256_file(name)!=h:
                    raise ValueError('Identity file mismatch: '+name)
    rt=c['identity']['runtime']
    if not rt.get('python') or not rt.get('torch'): raise ValueError('Unknown runtime version')
    actual_runtime = {'python': f'{sys.version_info.major}.{sys.version_info.minor}',
                      'torch': str(torch.__version__)}
    expected_runtime = {name: rt[name] for name in ('python', 'torch')}
    if verify_files and expected_runtime != actual_runtime:
        raise ValueError('Actual runtime differs from identity: ' + json.dumps(
            {'expected': expected_runtime, 'actual': actual_runtime}, sort_keys=True))
    return c


def dict_items(node):
    if not isinstance(node,dict) or node.get('type')!='dict' or not isinstance(node.get('items'),list):
        raise ValueError('Expected original dictionary digest')
    return dict((json.dumps(k,sort_keys=True),v) for k,v in node['items'])


def tensor_digests(node):
    if isinstance(node,dict):
        if node.get('type')=='tensor':
            if not node.get('finite') or 'shape' not in node or math.prod(node['shape'])<=0 or not re.fullmatch(r'[0-9a-f]{64}',node.get('sha256','')):
                raise ValueError('Empty/invalid tensor digest')
            yield node
        else:
            for v in node.values(): yield from tensor_digests(v)
    elif isinstance(node,list):
        for v in node: yield from tensor_digests(v)


def require_step(out,rank,c,contract_sha256):
    if (out.get('rank'),out.get('rollout_id'),out.get('step_id'))!=(rank,1,0): raise ValueError('Wrong original phase/rank')
    for section in ('model','optimizer','scheduler','RNG'):
        if section not in out: raise ValueError('Missing original section '+section)
    if out.get('contract_sha256')!=contract_sha256 or out.get('identity')!=c['identity']: raise ValueError('Source/input/runtime identity mismatch')
    if not out.get('model') or not all(list(tensor_digests(x)) for x in out['model']): raise ValueError('Empty original model')
    expected_model={k for k,v in c['inventory'].items() if 'shape' in v and not k.startswith('optimizer.')}
    actual_model=out.get('model_inventory',[])
    if {v['key'] for v in actual_model}!=expected_model: raise ValueError('Original model inventory incomplete')
    for v in actual_model:
        expected=c['inventory'][v['key']]
        if v['shape']!=expected['shape'] or v['dtype']!=expected['dtype'] or not all(x>0 for x in v['sizes']):
            raise ValueError('Original model geometry mismatch')
    if not dict_items(out.get('scheduler')): raise ValueError('Missing scheduler')
    require_optimizer(out.get('optimizer'),rank,c)
    require_rng(out.get('RNG',{}))
    return True


def require_optimizer(children,rank,c):
    if not children: raise ValueError('Missing optimizer')
    found={k:[] for k in ('param','exp_avg','exp_avg_sq')}
    def inspect(node):
        if isinstance(node,dict):
            if node.get('type')=='dict':
                entries=dict_items(node)
                present=[k for k in found if json.dumps(k) in entries]
                if present:
                    if len(present)!=3: raise ValueError('Incomplete master/moment bucket')
                    for k in present: found[k].extend(tensor_digests(entries[json.dumps(k)]))
            for v in node.values(): inspect(v)
        elif isinstance(node,list):
            for v in node: inspect(v)
    for child in children:
        if not dict_items(child.get('metadata')): raise ValueError('Missing optimizer metadata')
        inspect(child.get('master_and_moments'))
    for k,values in found.items():
        expected=sorted((v['dtype'],v['shape']) for key,v in c['inventory'].items() if f'dp_group_idx_{rank}.' in key and key.endswith('.'+k))
        if sorted((v['dtype'],v['shape']) for v in values)!=expected: raise ValueError('Original '+k+' inventory mismatch')


def require_rng(rng):
    if rng.get('type')!='list' or not rng.get('items'): raise ValueError('Missing original RNG list')
    for item in rng['items']:
        fields=dict_items(item)
        for name in ('random_rng_state','np_rng_state','torch_rng_state','cuda_rng_state','rng_tracker_states'):
            value=fields.get(json.dumps(name))
            if value is None or (isinstance(value,dict) and not value.get('items',value.get('shape',value.get('length')))):
                raise ValueError('Missing/empty original RNG '+name)
        py=fields[json.dumps('random_rng_state')];npstate=fields[json.dumps('np_rng_state')]
        if not isinstance(py,dict) or py.get('type')!='tuple' or len(py.get('items',[]))!=3 or not py['items'][1].get('items'):
            raise ValueError('Incomplete Python RNG state')
        if not isinstance(npstate,dict) or npstate.get('type')!='tuple' or len(npstate.get('items',[]))!=5 or npstate['items'][1].get('type')!='numpy' or not npstate['items'][1].get('shape'):
            raise ValueError('Incomplete NumPy RNG state')
        tracker=fields[json.dumps('rng_tracker_states')]
        if not dict_items(tracker) or not list(tensor_digests(tracker)): raise ValueError('Empty device RNG tracker')
        for name in ('torch_rng_state','cuda_rng_state'):
            if not list(tensor_digests(fields[json.dumps(name)])): raise ValueError('Missing RNG tensor')


def original_model_inventory(model):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
    rows=[]
    def visit(value):
        if isinstance(value,ShardedTensorFactory):
            visit(value.build())
        elif isinstance(value,ShardedTensor):
            if value.flattened_range is not None: raise ValueError("Unqualified flattened model shard")
            rows.append({"key":value.key,"shape":list(value.global_shape),"dtype":str(value.dtype),"offsets":list(value.global_offset),"sizes":[1]*value.prepend_axis_num+list(value.local_shape)})
        elif isinstance(value,dict):
            for child in value.values(): visit(child)
        elif isinstance(value,(list,tuple)):
            for child in value: visit(child)
    for part in model: visit(part.sharded_state_dict())
    if not rows: raise ValueError("Original sharded model inventory unavailable")
    return rows


def before_train_step(args,rollout_id,step_id,model,optimizer,opt_param_scheduler):
    """Original hook ABI; both control and resume must use this same observer.

    This hashes complete local model state, all optimizer master/moment buffers
    through the selected original DP-zero getter, scheduler and original RNG API.
    Readback may synchronize work. It is not a performance measurement.
    """
    # Only the paired post-update boundary is observed. Before the first update,
    # original distributed optimizer moments/master states may be lazy.
    if rollout_id!=1 or step_id!=0: return
    from megatron.core import mpu
    from megatron.training.checkpointing import get_rng_state
    assert args.tensor_model_parallel_size==4 and args.data_parallel_size==1
    contract_sha=os.environ['SLIME_COLD_CONTRACT_SHA256']
    contract=load_contract(os.environ['SLIME_COLD_CONTRACT'],contract_sha,verify_files=True)
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
    out={'contract_sha256':contract_sha,'identity':contract['identity'],'rank':rank,'rollout_id':rollout_id,'step_id':step_id,'model':[],
         'optimizer':[],'scheduler':digest(opt_param_scheduler.state_dict()),
         'RNG':digest(get_rng_state('torch',mpu.get_tensor_model_parallel_group(),mpu.get_pipeline_model_parallel_group()))}
    out['model_inventory']=original_model_inventory(model)
    for part in model: out['model'].append(digest(part.state_dict()))
    children=getattr(optimizer,'chained_optimizers',[optimizer])
    for child in children:
        # The original state_dict alone intentionally excludes parameter/moment buffers.
        state=child.get_parameter_state_dp_zero(use_gloo_comm=True,empty_data=False,return_on_all_ranks=False)
        if state is None: raise RuntimeError('DP1 parameter state unavailable')
        out['optimizer'].append({'metadata':digest(child.state_dict()),'master_and_moments':digest(state)})
        del state
    require_step(out,rank,contract,contract_sha)
    path=outroot/f'rollout_{rollout_id}_step_{step_id}_rank_{rank}.json'
    if path.exists(): raise FileExistsError(str(path))
    path.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
