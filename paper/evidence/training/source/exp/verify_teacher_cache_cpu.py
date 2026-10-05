"""Verify serialized teacher-cache contents with GPU visibility disabled."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def verify_payload(payload, metadata, student_checkpoint, seed, expected_count):
    import torch
    if payload.get("version") != 3 or payload.get("metadata") != metadata:
        raise ValueError("Serialized cache version/metadata disagrees with sidecar")
    samples=payload.get("samples")
    if not isinstance(samples,list) or len(samples)!=expected_count:
        raise ValueError("Serialized cache sample count differs from frozen budget")
    sources=metadata["source_observation_files"]
    if len(sources)!=expected_count:
        raise ValueError("Serialized cache source count differs")
    for index,sample in enumerate(samples):
        if type(sample.get("seed")) is not int or sample["seed"]!=seed+index:
            raise ValueError(f"Cache sample {index} replay seed differs")
        provenance=sample.get("provenance",{})
        if (provenance.get("source_kind")!="student_rollout" or
                provenance.get("student_checkpoint")!=str(student_checkpoint)):
            raise ValueError(f"Cache sample {index} is not from winning QAD")
        source=torch.load(sources[index]["path"],map_location="cpu",weights_only=True)
        if provenance!={k:v for k,v in source.items() if k!="inputs"}:
            raise ValueError(f"Cache sample {index} source provenance differs")
        inputs=sample["inputs"]
        pred,action,mask=sample["pred"],inputs["action"],inputs["action_mask"]
        if (not all(torch.is_tensor(x) for x in (pred,action,mask)) or
                pred.device.type!="cpu" or pred.shape!=action.shape or mask.shape!=pred.shape or
                action.shape[0]!=1 or not torch.isfinite(pred).all() or not torch.isfinite(action).all() or
                not torch.isfinite(mask).all() or not ((mask==0)|(mask==1)).all() or mask.sum()<=0):
            raise ValueError(f"Cache sample {index} has invalid velocity/action/mask")
    return {"version":3,"samples_verified":len(samples),"device":"cpu"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",required=True)
    parser.add_argument("--metadata",required=True)
    parser.add_argument("--student-checkpoint",required=True)
    parser.add_argument("--seed",required=True,type=int)
    parser.add_argument("--expected-count",required=True,type=int)
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch
    payload=torch.load(args.cache,map_location="cpu",weights_only=True)
    metadata=json.loads(Path(args.metadata).read_text())
    print(json.dumps(verify_payload(payload,metadata,Path(args.student_checkpoint).resolve(),
                                    args.seed,args.expected_count)))


if __name__=="__main__":
    main()
