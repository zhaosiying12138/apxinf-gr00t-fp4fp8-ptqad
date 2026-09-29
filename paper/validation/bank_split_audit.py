#!/usr/bin/env python3
"""CPU-only exact-vector audit of the official LIBERO-10 initial-state partitions.

This checks bank vectors before simulator restoration/settling. It neither runs
simulation nor substitutes for the recorded restored and postsettle hashes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'eval'))
from run_recovery_eval import TASKS


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder):
    import numpy as np
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('Run this audit without an initialized CUDA context')
    splits={'development':[0,1],'collection':[2,3],'heldout':list(range(10,20))}
    rows=[]
    for task in TASKS:
        path=Path(folder)/(task+'.pruned_init')
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('float64'))]):
            bank=torch.load(path,map_location='cpu',weights_only=True)
        hashes=[]
        for vector in bank:
            array=np.asarray(vector,dtype='<f8').copy(order='C')
            if not np.isfinite(array).all():raise ValueError('Nonfinite initial-state vector: '+task)
            array[array==0]=0.0
            hashes.append(hashlib.sha256(array.tobytes(order='C')).hexdigest())
        records={name:[{'index':i,'sha256':hashes[i]} for i in indices] for name,indices in splits.items()}
        sets={name:{row['sha256'] for row in group} for name,group in records.items()}
        intersections={a+'__'+b:sorted(sets[a]&sets[b]) for a,b in
                       (('development','collection'),('development','heldout'),('collection','heldout'))}
        if len(set.union(*sets.values()))!=14 or any(intersections.values()):
            raise ValueError('Repeated selected initial-state vector: '+task)
        rows.append({'task':task,'file':str(path.resolve()),'file_sha256':digest(path),
                     'bank_count':len(bank),'all_bank_unique_vectors':len(set(hashes)),
                     'state_shape':list(np.asarray(bank[0]).shape),'selected':records,'split_intersections':intersections})
    if torch.cuda.is_initialized():raise RuntimeError('Unexpected CUDA initialization')
    return {'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'gpu_execution':False,
            'cuda_initialized':False,'torch':torch.__version__,'numpy':np.__version__,
            'scope':'Within each task: exact official bank vector uniqueness before simulator restoration and settling; no claim of statistical independence or different postsettle states.',
            'vector_encoding':'C-contiguous little-endian float64 bytes, finite values, signed zero normalized to +0',
            'tasks':len(rows),'selected_vectors':14*len(rows),'selected_splits_disjoint':True,
            'script_sha256':digest(__file__),'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank-dir',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    output=Path(args.out)
    if output.exists():raise FileExistsError(output)
    result=audit(args.bank_dir)
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','tasks','selected_vectors','selected_splits_disjoint','cuda_initialized')}))


if __name__=='__main__':main()
