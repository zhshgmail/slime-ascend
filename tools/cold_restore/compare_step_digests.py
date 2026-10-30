"""Validate original TP4 pre-step completeness before comparing digests."""
from pathlib import Path
import argparse,json
from tools.cold_restore.state_digest import load_contract,require_step

def main():
 p=argparse.ArgumentParser()
 for name in ('control','resume','output','contract'): p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--contract-sha256',required=True);a=p.parse_args();rows=[];errors=[];observations=[]
 try:
  c=load_contract(a.contract,a.contract_sha256)
  for rank in range(4):
   name=f'rollout_1_step_0_rank_{rank}.json';left=json.loads((a.control/name).read_text());right=json.loads((a.resume/name).read_text())
   require_step(left,rank,c,a.contract_sha256);require_step(right,rank,c,a.contract_sha256)
   observations.extend([left,right])
   rows.append({'rank':rank,'equal':left==right,'different_sections':[k for k in set(left)|set(right) if left.get(k)!=right.get(k)]})
  for key,expected in c['inventory'].items():
   if 'shape' not in expected or key.startswith('optimizer.'): continue
   for side in (0,1):
    actual={(tuple(v['offsets']),tuple(v['sizes'])) for out in observations[side::2] for v in out['model_inventory'] if v['key']==key}
    required={(tuple(v['offsets']),tuple(v['sizes'])) for v in expected['chunks']}
    if actual!=required: raise ValueError('Incomplete original model shard coverage: '+key)
 except (ValueError,KeyError,TypeError,OSError) as e: errors.append(str(e))
 result={'ranks':rows,'validation_errors':errors,'reported_sections_equal':not errors and len(rows)==4 and all(x['equal'] for x in rows),'all_complete_state_equal':False,'complete_state_validation':'UNQUALIFIED_WITHDRAWN_MODEL_VALUES_NOT_BOUND_TO_INVENTORY','qualification':'Validated hook digests; native invocation provenance, actual generation/training inputs and next-step numerics/performance require separate evidence'}
 a.output.write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(1 if errors or not result['reported_sections_equal'] else 2)
if __name__=='__main__':main()
