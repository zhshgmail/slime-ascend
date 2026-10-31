"""Bind supplied actual paths to the original capture contract; no model/device run.

Identity is an externally reviewed source/input/runtime file map. This utility
checks supplied bytes and real DCP metadata; it cannot authenticate provenance
by itself and never certifies native model/optimizer/RNG completeness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from tools.cold_restore.sharded_values import read_expected
from tools.cold_restore.state_digest import load_contract

REQUIRED_SOURCE_SUFFIXES = (
    '/megatron/training/checkpointing.py',
    '/megatron/core/dist_checkpointing/mapping.py',
    '/megatron/core/dist_checkpointing/state_dict_utils.py',
    '/megatron/core/optimizer/distrib_optimizer.py',
    '/megatron/core/optimizer/optimizer.py',
    '/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py',
    '/megatron/core/optimizer_param_scheduler.py',
    '/tools/cold_restore/sharded_values.py',
    '/tools/cold_restore/state_digest.py',
    '/slime/backends/megatron_utils/model.py',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--metadata-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    identity = json.loads(a.identity.read_text())
    files = identity.get('source', {}).get('files', {})
    if any(not any(path.endswith(suffix) for path in files)
           for suffix in REQUIRED_SOURCE_SUFFIXES):
        raise ValueError('Missing actual original caller/producer/optimizer/scheduler source pins')
    contract = {'schema': 'slime-original-tp4-v1',
        'inventory': read_expected(a.metadata, a.metadata_sha256),
        'metadata': {'path': str(a.metadata.resolve()), 'sha256': a.metadata_sha256},
        'identity': identity}
    # Authenticated canonical geometry and every supplied source/input/runtime
    # path are checked in the SAME namespace the hook will consume.
    raw = (json.dumps(contract, sort_keys=True, indent=2) + '\n').encode()
    sha = hashlib.sha256(raw).hexdigest()
    # A rejected identity must not leave a final contract for a later command.
    # Validate in the same directory, then publish without replacing an existing
    # contract, retaining the original exclusive-create behavior.
    with tempfile.NamedTemporaryFile(dir=a.output.parent,
                                     prefix='.capture-contract-',
                                     delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
            load_contract(temporary, sha, verify_files=True)
            os.link(temporary, a.output)
        finally:
            temporary.unlink()
    print(json.dumps({'contract': str(a.output.resolve()), 'sha256': sha,
        'inventory_entries': len(contract['inventory']),
        'scope': 'Source/input/runtime bytes and original metadata bound only',
        'native_full_capture': 'NOT_RUN', 'all_complete_state_equal': False}))


if __name__ == '__main__':
    main()
