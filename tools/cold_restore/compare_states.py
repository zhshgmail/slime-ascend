"""Compare all persisted logical DCP tensors/objects plus common state on CPU.

No tolerance, wildcard ignore, native initialization, model or optimizer fabrication.
Common startup differences require explicit path + exact expected left/right values.
"""
from pathlib import Path
import argparse,io,json,hashlib,math
import torch
from torch.distributed.checkpoint import FileSystemReader,DefaultLoadPlanner,load
from tools.cold_restore.state_digest import digest,load_contract,dict_items,tensor_digests,metadata_geometry,require_rng

class CPUObjectPlanner(DefaultLoadPlanner):
    def __init__(self): super().__init__(flatten_state_dict=False,flatten_sharded_tensors=False)
    def load_bytes(self,read_item,value):
        self.state_dict[read_item.dest_index.fqn]=torch.load(value,map_location='cpu',weights_only=False)

geometry=metadata_geometry

def read_key(root,key,meta,max_tensor_bytes):
    item=meta.state_dict_metadata[key]
    if hasattr(item,'size'):
        size=math.prod(item.size)*torch.empty((),dtype=item.properties.dtype).element_size()
        if size>max_tensor_bytes: raise MemoryError(f'{key}: {size} bytes exceeds explicit CPU cap')
        value=torch.empty(item.size,dtype=item.properties.dtype,device='cpu')
    else:
        lengths=[v.length for k,v in meta.storage_data.items() if k.fqn==key]
        if sum(lengths)>512*1024**2: raise MemoryError('Object exceeds 512MiB CPU cap: '+key)
        value=None
    state={key:value};load(state,storage_reader=FileSystemReader(root),planner=CPUObjectPlanner())
    return state[key]

def main():
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');p.add_argument('--max-tensor-bytes',type=int,default=4*1024**3);p.add_argument('--allow-common-differences',type=Path);p.add_argument('--complete-original',action='store_true');p.add_argument('--contract',type=Path);p.add_argument('--contract-sha256');a=p.parse_args()
    lm=FileSystemReader(a.left).read_metadata();rm=FileSystemReader(a.right).read_metadata();lg=geometry(lm);rg=geometry(rm);r={'torch':torch.__version__,'left':str(a.left),'right':str(a.right),'geometry_equal':lg==rg,'keys':len(lg),'tensors':sum('shape' in v for v in lg.values()),'objects':sum('shape' not in v for v in lg.values()),'RNG_keys':[k for k in lg if k.startswith('rng_state/')],'metadata_only':a.metadata_only,'compared':[],'differences':[]}
    c=None
    if a.complete_original:
        try:
            c=load_contract(a.contract,a.contract_sha256,verify_files=True)
            if lg!=c['inventory'] or rg!=c['inventory']: raise ValueError('Actual DCP inventory differs from authenticated original522')
        except (ValueError,TypeError,OSError) as e: r['differences'].append(str(e))
    if lg!=rg: r['differences'].append('logical metadata geometry/key mismatch')
    if not a.metadata_only and not r['differences']:
        for key in sorted(lg):
            left=read_key(a.left,key,lm,a.max_tensor_bytes);ld=digest(left);del left
            right=read_key(a.right,key,rm,a.max_tensor_bytes);rd=digest(right);del right
            if a.complete_original and key.startswith('rng_state/'):
                for rng in (ld,rd): require_rng(rng)
            r['compared'].append({'key':key,'equal':ld==rd,'left':ld,'right':rd})
            if ld!=rd:r['differences'].append(key)
        lc=torch.load(a.left/'common.pt',map_location='cpu',weights_only=False);rc=torch.load(a.right/'common.pt',map_location='cpu',weights_only=False)
        if a.complete_original:
            for common in (lc,rc):
                if not all(k in common and common[k] for k in ('args','optimizer','opt_param_scheduler')) or common.get('iteration')!=0:
                    raise ValueError('Missing original common sections or wrong split iteration')
        # No blanket args/scheduler skips. Explicit declared startup differences only.
        for row in json.loads(a.allow_common_differences.read_text()) if a.allow_common_differences else []:
            path=row['path']
            if a.complete_original and (len(path)!=2 or path[0]!='args' or path[1] not in ('load','save','start_rollout_id','num_rollout')):
                raise ValueError('Complete-original permits only declared continuation/path startup differences')
            lv=lc;rv=rc
            for part in path[:-1]:lv=vars(lv)[part] if isinstance(lv,argparse.Namespace) else lv[part];rv=vars(rv)[part] if isinstance(rv,argparse.Namespace) else rv[part]
            lk=vars(lv) if isinstance(lv,argparse.Namespace) else lv;rk=vars(rv) if isinstance(rv,argparse.Namespace) else rv
            assert lk[path[-1]]==row['left'] and rk[path[-1]]==row['right'],row
            lk[path[-1]]=rk[path[-1]]=('DECLARED_STARTUP_DIFFERENCE',row['reason'])
        r['common_equal']=digest(lc)==digest(rc)
        if not r['common_equal']:r['differences'].append('common.pt logicalstate')
    r['qualification']='METADATA_ONLY_NO_RESTORE_OR_VALUE_PASS' if a.metadata_only else 'PERSISTED_LOGICAL_STATE_COMPARISON_ONLY_NOT_PROCESS_RESTORE'
    r['complete_original_state_equal']=False
    if a.complete_original: r['complete_state_validation']='UNQUALIFIED_WITHDRAWN_RNG_CONTENT_AND_VALUE_COVERAGE_PENDING'
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    raise SystemExit(1 if r['differences'] else 2 if a.complete_original and not a.metadata_only else 0)
if __name__=='__main__': main()
