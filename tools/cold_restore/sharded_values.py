"""Observe actual MCore save values; this is not a full-restore verdict.

Use the original save dictionary and factory preprocessing.  Geometry and bytes
come from the SAME ShardedTensor, rather than an unrelated state_dict/inventory.
No model, optimizer, process group or RNG state is synthesized here.
"""
import hashlib
import inspect
import json
import os
import re
from pathlib import Path

import torch

from tools.cold_restore.state_digest import digest, metadata_geometry, sha256_file


def _source(module):
    p = Path(inspect.getfile(module)).resolve()
    return {"path": str(p), "sha256": sha256_file(p)}


def read_expected(metadata_file, metadata_sha256):
    """Authenticate the real original checkpoint metadata before parsing it."""
    p = Path(metadata_file)
    if not p.is_file() or sha256_file(p) != metadata_sha256:
        raise ValueError("checkpoint metadata hash mismatch")
    from torch.distributed.checkpoint import FileSystemReader
    return metadata_geometry(FileSystemReader(p.parent).read_metadata())


def collect_save_values(original_save_dict):
    """Consume the actual result of original generate_state_dict.

    Original save_preprocess mutates its input containers while expanding
    factories.  Pass a fresh original save dictionary, never a live cached one.
    Its distributed integrity collective is deliberately not called by this
    local observer.  Global coverage is checked separately against real DCP
    metadata; no local record is a completeness certificate.
    """
    from megatron.core.dist_checkpointing import mapping, state_dict_utils
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedObject
    sharded, common = state_dict_utils.save_preprocess(
        original_save_dict, validate_access_integrity=False)
    records = []

    def visit(x):
        if isinstance(x, ShardedTensor):
            x.validate_metadata_integrity()
            if x.data is None or x.flattened_range is not None:
                raise ValueError("missing or unsupported flattened tensor data")
            actual = digest(x.data)
            sizes = [1] * x.prepend_axis_num + list(x.data.shape)
            # Actual tensor dtype/shape are checked again here and emitted in
            # the record that owns its value digest, never in a separate list.
            if actual['dtype'] != str(x.dtype) or list(x.data.shape) != list(x.local_shape):
                raise ValueError("shard data/geometry mismatch")
            records.append({"kind": "tensor", "key": x.key,
                "global_shape": list(x.global_shape),
                "offsets": list(x.global_offset), "sizes": sizes,
                "replica_id": x.replica_id, "value": actual})
        elif isinstance(x, ShardedObject):
            x.validate_metadata_integrity()
            # Includes original Python/NumPy/CPU/device/tracker RNG content,
            # tensor byte extents and object structure, if supplied by the
            # original RNG collector.  Does not invent absent device state.
            records.append({"kind": "object", "key": x.unique_key,
                "replica_id": x.replica_id, "value": digest(x.data)})
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        else:
            raise TypeError("unhandled original sharded value: " + str(type(x)))

    visit(sharded)
    return {"schema": "original-sharded-values-v1", "records": records,
        "common_values": digest(common),
        "sources": [_source(mapping), _source(state_dict_utils)],
        "all_complete_state_equal": False,
        "qualification": "actual supplied save values only; native completeness unqualified"}


def collect_original(args, model, optimizer, scheduler, rng_state, iteration=None,
                     expected_source_files=None):
    """Native integration adapter: use the SAME original save entry interfaces.

    rng_state must come from original get_rng_state('torch_dist', tp, pp), not a
    digest JSON or CPU substitute.  Invocation is untested on the native model.
    The optimizer's own sharded_state_dict supplies its semantic keys and actual
    parameter/moment values; arbitrary optimizer metadata is not substituted.
    """
    from megatron.training import checkpointing
    from megatron.core.dist_checkpointing import mapping, state_dict_utils
    # Source assertions are against external authenticated contract entries,
    # not against hashes supplied by a capture about itself.
    sources = [_source(m) for m in (checkpointing, mapping, state_dict_utils)]
    observer_path = Path(__file__).resolve()
    sources.append({'path':str(observer_path), 'sha256':sha256_file(observer_path)})
    if expected_source_files is None or any(
        expected_source_files.get(r['path']) != r['sha256'] for r in sources
    ):
        raise ValueError('original source files not authenticated by contract')
    if args.ckpt_format != "torch_dist" or args.no_save_optim or args.no_save_rng:
        raise ValueError("original full-state save contract required")
    if optimizer is None or optimizer.is_stub_optimizer or rng_state is None:
        raise ValueError("actual optimizer and original RNG required")
    md = checkpointing._build_sharded_state_dict_metadata(args)
    # Original save_checkpoint unwraps DDP/precision wrappers before generating
    # the save dictionary. Preserve that exact source operation here as well.
    model = checkpointing.unwrap_model(model)
    state = checkpointing.generate_state_dict(
        args, model, optimizer, scheduler, rng_state, iteration=iteration,
        optim_sd_kwargs={"metadata": md}, model_sd_kwargs={"metadata": md})
    result = collect_save_values(state)
    result['sources'].append(_source(checkpointing))
    return result


def verify_tensor_coverage(captures, expected, required_keys):
    """Compare supplied actual value records to authenticated checkpoint chunks.

    required_keys must come from read_expected, never from the capture itself.
    A caller may explicitly select a PARTIAL diagnostic subset.  Successful
    coverage says only those tensors have complete value-bearing records.
    Replica chunks must agree byte-for-byte.  Common/optimizer semantics and
    device RNG completeness are not certified by this routine.
    """
    required = set(required_keys)
    if not required or any(k not in expected or 'shape' not in expected[k] for k in required):
        raise ValueError("expected nonempty authenticated tensor key set")
    actual = {}
    for cap in captures:
        if cap.get('schema') != 'original-sharded-values-v1':
            raise ValueError('unexpected capture schema')
        for r in cap['records']:
            if r['kind'] != 'tensor' or r['key'] not in required:
                continue
            k, v, e = r['key'], r['value'], expected[r['key']]
            if v.get('type') != 'tensor' or v.get('finite') is not True:
                raise ValueError('invalid actual tensor value')
            if not re.fullmatch('[0-9a-f]{64}', v.get('sha256', '')):
                raise ValueError('invalid actual tensor digest')
            if r['global_shape'] != e['shape'] or v['dtype'] != e['dtype']:
                raise ValueError('actual tensor dtype/global shape mismatch: ' + k)
            if r['sizes'] != [1] * (len(r['sizes']) - len(v['shape'])) + v['shape']:
                raise ValueError('actual tensor shape/local extent mismatch: ' + k)
            chunk = (tuple(r['offsets']), tuple(r['sizes']))
            if {'offsets': r['offsets'], 'sizes': r['sizes']} not in e['chunks']:
                raise ValueError('actual tensor chunk not in checkpoint: ' + k)
            previous = actual.setdefault(k, {}).setdefault(chunk, v)
            if previous != v:
                raise ValueError('replica values disagree: ' + k)
    for k in required:
        want = {(tuple(c['offsets']), tuple(c['sizes'])) for c in expected[k]['chunks']}
        if set(actual.get(k, {})) != want:
            raise ValueError('missing actual tensor value chunks: ' + k)
    return {'covered_tensor_keys': sorted(required),
        'all_complete_state_equal': False,
        'scope': 'value-bearing tensor chunk coverage only, not complete-state equality'}


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Prepared original Slime hook ABI; NOT YET native-qualified.

    At the same original post-update boundary, preserve the actual split
    tracker and record original SAVE payload values.  No complete-state verdict
    is produced.  Raw original RNG object bytes are retained for independent
    decoding/API validation, rather than trusting a JSON RNG extent claim.
    """
    if (rollout_id, step_id) != (1, 0):
        return
    from megatron.core import mpu
    from megatron.training.checkpointing import get_rng_state
    from tools.cold_restore.state_digest import load_contract
    if args.tensor_model_parallel_size != 4 or args.data_parallel_size != 1:
        raise ValueError('original TP4/DP1 required')
    csha = os.environ['SLIME_COLD_CONTRACT_SHA256']
    c = load_contract(os.environ['SLIME_COLD_CONTRACT'], csha, verify_files=True)
    rank = torch.distributed.get_rank()
    out = Path(os.environ['SLIME_COLD_STATE_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0 and os.environ.get('SLIME_COLD_PRESERVE_SPLIT') == '1':
        tracker = (Path(args.save) / 'latest_checkpointed_iteration.txt').read_bytes()
        if tracker.strip() != b'0':
            raise ValueError('actual split tracker is not iteration zero')
        with (out / 'split_latest_checkpointed_iteration.txt').open('xb') as f:
            f.write(tracker)
    rng = get_rng_state('torch_dist', mpu.get_tensor_model_parallel_group(),
                        mpu.get_pipeline_model_parallel_group())
    from megatron.core.dist_checkpointing.mapping import ShardedObject
    if not isinstance(rng, ShardedObject) or rng.key != 'rng_state':
        raise TypeError('original torch_dist RNG object required')
    raw = out / f'rollout_1_step_0_rank_{rank}_rng.pt'
    with raw.open('xb') as f:
        # Same object-list representation as original MCore torch strategy.
        torch.save([rng.data], f)
    cap = collect_original(args, model, optimizer, opt_param_scheduler, rng,
        expected_source_files=c['identity']['source']['files'])
    cap.update(rank=rank, rollout_id=rollout_id, step_id=step_id,
               contract_sha256=csha, identity=c['identity'],
               metadata=c['metadata'],
               original_rng_file={'path':str(raw),'bytes':raw.stat().st_size,
                                  'sha256':sha256_file(raw)})
    with (out / f'rollout_1_step_0_rank_{rank}_sharded.json').open('x') as f:
        json.dump(cap, f, sort_keys=True, indent=2)
        f.write('\n')
