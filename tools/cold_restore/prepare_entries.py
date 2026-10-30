"""Prepare two explicit original train.py argument sets; no execution/resources."""
from pathlib import Path
import argparse,json,hashlib

def replace(argv,name,value):
    at=argv.index(name);argv[at+1]=str(value)

def main():
    p=argparse.ArgumentParser();p.add_argument('--original-argv',type=Path,required=True);p.add_argument('--checkpoint-root',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--contract-sha256',required=True);a=p.parse_args()
    from tools.cold_restore.state_digest import load_contract
    load_contract(a.contract,a.contract_sha256)
    base=json.loads(a.original_argv.read_text());assert base[0].endswith('/train.py')
    assert base[base.index('--tensor-model-parallel-size')+1]=='4' and base[base.index('--global-batch-size')+1]=='16'
    assert '--load-debug-rollout-data' not in base and '--load-forge-rollout-data' not in base
    a.output.mkdir(parents=True,exist_ok=False)
    modes={}
    for mode in ['control','resume']:
        argv=base.copy();replace(argv,'--num-rollout',2);replace(argv,'--save',f'/run/{mode}-checkpoint-output')
        replace(argv,'--start-rollout-id',0 if mode=='control' else 1)
        if mode=='resume': replace(argv,'--load',a.checkpoint_root)
        argv+=['--save-debug-rollout-data',f'/run/{mode}-rollout-traces/{{rollout_id}}.pt','--save-debug-train-data',f'/run/{mode}-train-traces/{{rollout_id}}_{{rank}}.pt','--custom-megatron-before-train-step-hook-path','tools.cold_restore.state_digest.before_train_step']
        modes[mode]={'argv':argv,'env_add':{'SLIME_COLD_STATE_DIR':f'/run/{mode}-state-digests','SLIME_COLD_CONTRACT':str(a.contract),'SLIME_COLD_CONTRACT_SHA256':a.contract_sha256,'SLIME_COLD_PRESERVE_SPLIT':'1' if mode=='control' else '0'},'entry':'Normal original train.py; same source/model/assets/TP4/16/4096/33graphs/offload/rolebackup/math','environment_delivery':'REQUIRED: export every env_add key BEFORE original Ray-head start, or explicitly forward all keys in worker train_env_vars. Driver-only late export is insufficient.','native':'NOT_RUN'}
        (a.output/f'{mode}_argv.json').write_text(json.dumps(argv,indent=2)+'\n')
    (a.output/'ENTRIES.json').write_text(json.dumps({'original_argv_sha256':hashlib.sha256(a.original_argv.read_bytes()).hexdigest(),'profiles':modes,'comparison':'Compare control before rollout1 training to resume before rollout1 training by exact local model/master/moments/scheduler/RNG digests for everyrank. Compare actual saved rollout/train inputs before attributing next-step endpoint differences; stochastic SGLang generator state is not restored by these trainer checkpoints. No input replay or deterministic flag is silently enabled.','selected_checkpoint':'Root must bind intended split checkpoint0 from SAME control trajectory: iter_0000000 and rollout from its canonical save root plus actual split_latest_checkpointed_iteration.txt captured by the control hook at rollout1, read-only as latest_checkpointed_iteration.txt. Do not point resume at control root after tracker advances to1. This is a shallow mount view, no 44GB copy. Root must bind split checkpoint0 for a valid interrupted/uninterrupted pair; existing author and independent checkpoints are neither the same trajectory nor a completed pair.','resource_authority':'NONE: prepare-only. Separate finite Root admission and unique writer required.'},indent=2)+'\n')
if __name__=='__main__': main()
