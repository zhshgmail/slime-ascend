"""Require every TP4 rank's actual pre-step state; no input/effect qualification."""
from pathlib import Path
import argparse,json

def main():
 p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--resume',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 for rank in range(4):
  name=f'rollout_1_step_0_rank_{rank}.json';left=json.loads((a.control/name).read_text());right=json.loads((a.resume/name).read_text())
  assert left['rank']==right['rank']==rank and left['rollout_id']==right['rollout_id']==1
  rows.append({'rank':rank,'equal':left==right,'different_sections':[k for k in set(left)|set(right) if left.get(k)!=right.get(k)]})
 result={'ranks':rows,'all_complete_state_equal':all(x['equal'] for x in rows),'qualification':'Actual hook digests only; does not compare generation/training inputs or prove next-step numerics/performance'};a.output.write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(0 if result['all_complete_state_equal'] else 1)
if __name__=='__main__':main()
