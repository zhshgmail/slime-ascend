"""Read the actual sharded-values producer format; never certify full restore.

Exit 1: missing/invalid/different observations. Exit 2: reported values equal,
but complete native model/optimizer/device-RNG semantics remain unqualified.
Diagnostic tensor subsets are explicit and never enter the original denominator.
"""
import argparse
import copy
import json
from pathlib import Path

from tools.cold_restore.sharded_values import verify_tensor_coverage
from tools.cold_restore.state_digest import load_contract, digest, sha256_file


def compare_common(left, right, rules):
    l = copy.deepcopy([v['common_values'] for v in left])
    r = copy.deepcopy([v['common_values'] for v in right])
    if len(l) != len(r):
        return False
    def field(node, key):
        if node.get('type') == 'Namespace':
            node = node['value']
        if node.get('type') != 'dict':
            raise ValueError('expected actual dictionary/Namespace digest')
        matches = [pair for pair in node['items'] if pair[0] == key]
        if len(matches) != 1:
            raise ValueError('missing/duplicate common field: ' + key)
        return matches[0]
    for row in rules:
        path = row['path']
        if len(path) != 2 or path[0] != 'args' or path[1] not in ('load', 'save', 'start_rollout_id', 'num_rollout'):
            raise ValueError('only exact declared original continuation startup differences allowed')
        for lc, rc in zip(l, r):
            lv = field(field(lc, 'args')[1], path[1])
            rv = field(field(rc, 'args')[1], path[1])
            if lv[1] != digest(row['left']) or rv[1] != digest(row['right']):
                raise ValueError('declared startup difference does not match actual observations')
            lv[1] = rv[1] = {'declared_original_startup_difference': row['reason']}
    return l == r


def _record_map(captures, expected, selected):
    values = {}
    for cap in captures:
        for row in cap['records']:
            k = row['key']
            if k not in selected:
                continue
            key = (k, tuple(row['offsets']), tuple(row['sizes'])) if row['kind'] == 'tensor' else (k,)
            if (row['kind'] == 'tensor') != ('shape' in expected[k]):
                raise ValueError('actual value kind differs from checkpoint: ' + k)
            if key in values and values[key] != row['value']:
                raise ValueError('replica value mismatch: ' + k)
            values[key] = row['value']
    objects = {k for k in selected if 'shape' not in expected[k]}
    if {k[0] for k in values if len(k) == 1} != objects:
        raise ValueError('missing actual original object values')
    return values


def read_side(root, contract, contract_sha, diagnostic):
    root = Path(root)
    if diagnostic:
        # One actual CPU capture, not four invented rank files.
        return [json.loads(root.read_text())]
    caps = []
    for rank in range(4):
        cap = json.loads((root / f'rollout_1_step_0_rank_{rank}_sharded.json').read_text())
        if (cap.get('rank'), cap.get('rollout_id'), cap.get('step_id')) != (rank, 1, 0):
            raise ValueError('wrong original capture rank/phase')
        if cap.get('contract_sha256') != contract_sha or cap.get('identity') != contract['identity']:
            raise ValueError('capture source/input/runtime contract mismatch')
        raw = root / f'rollout_1_step_0_rank_{rank}_rng.pt'
        binding = cap['original_rng_file']
        if not raw.is_file() or raw.stat().st_size != binding['bytes'] or sha256_file(raw) != binding['sha256']:
            raise ValueError('actual original raw RNG artifact missing or changed')
        if raw.stat().st_size > 2 * 1024**2:
            raise ValueError('raw RNG object exceeds this bounded reader scope')
        import torch
        state = torch.load(raw, map_location='cpu', weights_only=False)
        if not isinstance(state, list) or len(state) != 1:
            raise ValueError('not original object-list RNG serialization')
        rng = [r for r in cap['records'] if r['kind'] == 'object' and r['key'].startswith('rng_state/')]
        if len(rng) != 1 or digest(state[0]) != rng[0]['value']:
            raise ValueError('raw RNG bytes do not match captured original object value')
        caps.append(cap)
    return caps


def main():
    p = argparse.ArgumentParser()
    for name in ('control', 'resume', 'output', 'contract'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--contract-sha256', required=True)
    p.add_argument('--allow-common-differences', type=Path)
    p.add_argument('--diagnostic-tensor-keys', type=Path,
                   help='Explicit partial CPU capture files and tensor key list; never complete-state qualification')
    a = p.parse_args()
    out = {'all_complete_state_equal': False, 'validation_errors': [],
           'reported_values_equal': False,
           'scope': 'PARTIAL_TENSOR_DIAGNOSTIC' if a.diagnostic_tensor_keys else 'ORIGINAL_CAPTURE_COMPARISON_NATIVE_SEMANTICS_UNQUALIFIED',
           'native_model_optimizer_device_rng_semantics': 'NOT_QUALIFIED'}
    try:
        c = load_contract(a.contract, a.contract_sha256, verify_files=True)
        selected = set(json.loads(a.diagnostic_tensor_keys.read_text())) if a.diagnostic_tensor_keys else set(c['inventory'])
        if not selected or not selected <= c['inventory'].keys():
            raise ValueError('empty or unknown expected original key selection')
        tensors = {k for k in selected if 'shape' in c['inventory'][k]}
        if a.diagnostic_tensor_keys and selected != tensors:
            raise ValueError('diagnostic mode accepts only explicit tensor keys')
        left = read_side(a.control, c, a.contract_sha256, a.diagnostic_tensor_keys)
        right = read_side(a.resume, c, a.contract_sha256, a.diagnostic_tensor_keys)
        for side in (left, right):
            verify_tensor_coverage(side, c['inventory'], tensors)
        lm = _record_map(left, c['inventory'], selected)
        rm = _record_map(right, c['inventory'], selected)
        # Every allowed startup difference supplies an exact expected left/right
        # value; scheduler/optimizer/common content is never broadly skipped.
        rules = json.loads(a.allow_common_differences.read_text()) if a.allow_common_differences else []
        common_equal = compare_common(left, right, rules)
        out.update(compared_keys=sorted(selected), reported_values_equal=lm == rm,
                   common_values_equal=common_equal)
        if lm != rm:
            out['validation_errors'].append('actual value records differ')
        if not common_equal:
            out['validation_errors'].append('common values differ; no broad startup-field skip applied')
    except (ValueError, TypeError, KeyError, OSError) as e:
        out['validation_errors'].append(type(e).__name__ + ': ' + str(e))
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    raise SystemExit(1 if out['validation_errors'] else 2)


if __name__ == '__main__':
    main()
