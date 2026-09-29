#!/usr/bin/env python3
"""Count the real OPD hook on a CPU fixture; never a formal-run measurement."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
import test_probe_distill as probe


def check():
    fixture=probe.ProbeTests();fixture.setUp()
    try:
        class Trainer:
            def __init__(self):
                self.args=SimpleNamespace(gradient_accumulation_steps=16)
                self.current_gradient_accumulation_steps=16
                self.state=SimpleNamespace(global_step=0)
                self.teacher_calls=0
                def backward(loss):
                    if not loss.requires_grad or not probe.torch.isfinite(loss):
                        raise RuntimeError('Invalid fixture teacher backward')
                    self.teacher_calls+=1;loss.backward()
                self.accelerator=SimpleNamespace(gradient_accumulation_steps=1,num_processes=1,backward=backward)
            def training_step(self,model,inputs):
                loss=model(inputs)['loss']/16;loss.backward();return loss.detach()

        trainer=Trainer();probe.install_sequential_probe(Trainer,fixture.path,1.,4)
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            for step in range(100):
                trainer.state.global_step=step;fixture.model.zero_grad(set_to_none=True)
                for _ in range(16):trainer.training_step(fixture.model,fixture.inputs)
        records=re.findall(r'\[opd\] step=(\d+) probe=(\d+) mse=(\S+) weight=(\S+) microbatch=1',stream.getvalue())
        updates=sorted(set(int(row[0]) for row in records))
        if (trainer.teacher_calls,len(records),updates)!=(400,80,[4,24,44,64,84]):
            raise RuntimeError('Teacher hook cadence or logger sparsity differs')
        if probe.torch.cuda.is_initialized():raise RuntimeError('CPU-only check initialized CUDA')
        return {
            'kind':'CPU fixture execution of actual teacher hook, not a formal training measurement',
            'optimizer_updates_simulated':100,'accumulation_microbatches_per_update':16,
            'teacher_backward_calls_in_fixture':trainer.teacher_calls,
            'logged_records_in_fixture':len(records),'logged_update_numbers':updates,
            'producer_sha256':hashlib.sha256((ROOT/'rl/probe_distill.py').read_bytes()).hexdigest(),
            'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cuda_initialized':False}
    finally:fixture.tearDown()


if __name__=='__main__':print(json.dumps(check(),indent=2))
