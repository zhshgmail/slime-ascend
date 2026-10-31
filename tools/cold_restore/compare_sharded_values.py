"""Read the actual sharded-values producer format; never certify full restore.

Exit 1: missing/invalid/different observations. Exit 2: reported values equal,
but complete native model/optimizer/device-RNG semantics remain unqualified.
Diagnostic tensor subsets are explicit and never enter the original denominator.
"""
import argparse
import copy
import errno
import json
import math
import pickle
import re
from pathlib import Path

from tools.cold_restore.sharded_values import verify_tensor_coverage
from tools.cold_restore.state_digest import load_contract, digest, sha256_file


def validate_digest(node, path='common_values'):
    """Validate the original digest grammar, not merely truthiness/equality."""
    if node is None or isinstance(node, (str, bool, int)):
        return
    if isinstance(node, float):
        if not math.isfinite(node):
            raise ValueError(path + ': nonfinite scalar')
        return
    if not isinstance(node, dict) or not isinstance(node.get('type'), str):
        raise ValueError(path + ': expected original typed digest or scalar')
    kind = node['type']
    if kind == 'dict':
        if set(node) != {'type', 'items'} or not isinstance(node['items'], list):
            raise ValueError(path + ': invalid dictionary digest')
        seen = set()
        for pair in node['items']:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(path + ': dictionary entry must be a key/value pair')
            validate_digest(pair[0], path + '.key')
            validate_digest(pair[1], path + '.value')
            key = dictionary_key(pair[0])
            if key in seen:
                raise ValueError(path + ': duplicate dictionary key')
            seen.add(key)
    elif kind in ('list', 'tuple'):
        if set(node) != {'type', 'items'} or not isinstance(node['items'], list):
            raise ValueError(path + ': invalid sequence digest')
        for value in node['items']:
            validate_digest(value, path + '[]')
    elif kind == 'Namespace':
        if set(node) != {'type', 'value'} or not isinstance(node['value'], dict) or node['value'].get('type') != 'dict':
            raise ValueError(path + ': invalid original Namespace digest')
        validate_digest(node['value'], path + '.Namespace')
    elif kind in ('tensor', 'numpy'):
        fields = {'type', 'dtype', 'shape', 'sha256'} | ({'finite'} if kind == 'tensor' else set())
        if set(node) != fields or not isinstance(node['dtype'], str) or not isinstance(node['shape'], list):
            raise ValueError(path + ': invalid array digest')
        if any(type(n) is not int or n < 0 for n in node['shape']) or not re.fullmatch('[0-9a-f]{64}', node.get('sha256', '')):
            raise ValueError(path + ': invalid array shape/hash')
        if kind == 'tensor' and node['finite'] is not True:
            raise ValueError(path + ': nonfinite tensor')
    elif kind == 'class':
        if (set(node) != {'type', 'module', 'qualname', 'source_sha256'}
                or not isinstance(node['module'], str)
                or not isinstance(node['qualname'], str)
                or not all(part.isidentifier() for part in
                           (node['module'] + '.' + node['qualname']).split('.'))):
            raise ValueError(path + ': invalid class identity')
        source = node['source_sha256']
        if ((node['module'] == 'builtins' and source is not None)
                or (node['module'] != 'builtins' and
                    (not isinstance(source, str) or not re.fullmatch('[0-9a-f]{64}', source)))):
            raise ValueError(path + ': invalid class source identity')
    elif kind == 'bytes':
        if set(node) != {'type', 'length', 'sha256'} or type(node['length']) is not int or node['length'] < 0 or not re.fullmatch('[0-9a-f]{64}', node.get('sha256', '')):
            raise ValueError(path + ': invalid bytes digest')
    elif kind in ('dtype', 'device'):
        if set(node) != {'type', 'value'} or not isinstance(node['value'], str):
            raise ValueError(path + ': invalid torch descriptor')
    elif '.' in kind and set(node) == {'type', 'value'}:
        validate_digest(node['value'], path + '.enum')
    else:
        raise ValueError(path + ': unknown digest type ' + kind)


def dictionary_key(key):
    """Reject digest keys that cannot originate from a Python dictionary."""
    if isinstance(key, (bool, int, float)):
        # Python treats True, 1 and 1.0 as the same dictionary key.
        return ('number', key)
    if key is None or isinstance(key, str):
        return (type(key).__name__, key)
    kind = key['type']
    if kind in ('dict', 'list', 'Namespace', 'numpy'):
        raise ValueError('common_values: unhashable original dictionary key')
    if kind == 'tuple':
        return ('tuple', tuple(dictionary_key(x) for x in key['items']))
    return ('typed', json.dumps(key, sort_keys=True, allow_nan=False))


def validate_common(node, original=False):
    validate_digest(node)
    if not isinstance(node, dict) or node.get('type') != 'dict' or not node['items']:
        raise ValueError('common_values: nonempty original dictionary digest required')
    if original:
        fields = {json.dumps(k, sort_keys=True): v for k, v in node['items']}
        for key in ('args', 'optimizer', 'opt_param_scheduler'):
            value = fields.get(json.dumps(key))
            if key == 'args' and isinstance(value, dict) and value.get('type') == 'Namespace':
                value = value['value']
            if not isinstance(value, dict) or value.get('type') != 'dict' or not value['items']:
                raise ValueError('common_values: missing original structured ' + key)
        if fields.get(json.dumps('checkpoint_version')) != 3.0:
            raise ValueError('common_values: missing original checkpoint_version3.0')


def compare_common(left, right, rules):
    if not isinstance(rules, list):
        raise ValueError('startup difference rules must be a list')
    for cap in left + right:
        validate_common(cap['common_values'])
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
        if not isinstance(row, dict) or set(row) != {'path', 'left', 'right', 'reason'} or not isinstance(row['path'], list) or not isinstance(row['reason'], str) or not row['reason']:
            raise ValueError('invalid exact startup difference rule')
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
            validate_digest(row['value'], 'records.' + k)
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
        cap = json.loads(root.read_text())
        if not isinstance(cap, dict):
            raise ValueError('capture must be an object')
        validate_common(cap['common_values'])
        return [cap]
    caps = []
    for rank in range(4):
        cap = json.loads((root / f'rollout_1_step_0_rank_{rank}_sharded.json').read_text())
        if not isinstance(cap, dict):
            raise ValueError('capture must be an object')
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
        try:
            state = torch.load(raw, map_location='cpu', weights_only=False)
        except (pickle.UnpicklingError, EOFError, UnicodeError) as e:
            raise ValueError('cannot decode original RNG archive: ' + type(e).__name__) from e
        except torch.OutOfMemoryError:
            raise
        except RuntimeError as e:
            # Recognized malformed archive errors only; unrelated runtime or
            # resource failures propagate, as do KeyboardInterrupt/SystemExit.
            if any(msg in str(e).lower() for msg in ('magic number', 'pytorchstreamreader', 'not a zip archive')):
                raise ValueError('cannot decode original RNG archive: ' + str(e)) from e
            raise
        if not isinstance(state, list) or len(state) != 1:
            raise ValueError('not original object-list RNG serialization')
        rng = [r for r in cap['records'] if r['kind'] == 'object' and r['key'].startswith('rng_state/')]
        if len(rng) != 1 or digest(state[0]) != rng[0]['value']:
            raise ValueError('raw RNG bytes do not match captured original object value')
        caps.append(cap)
        validate_common(cap['common_values'], original=True)
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
    except OSError as e:
        if e.errno in (errno.ENOMEM, errno.ENOSPC, errno.EMFILE, errno.ENFILE):
            raise
        out['validation_errors'].append(type(e).__name__ + ': ' + str(e))
    except (ValueError, TypeError, KeyError, IndexError) as e:
        out['validation_errors'].append(type(e).__name__ + ': ' + str(e))
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    raise SystemExit(1 if out['validation_errors'] else 2)


if __name__ == '__main__':
    main()
