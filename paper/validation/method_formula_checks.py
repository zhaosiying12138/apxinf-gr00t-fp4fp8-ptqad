#!/usr/bin/env python3
"""CPU-only numerical checks for the method's cadence, mask and matrix formulas."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import contextlib

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rl'))
sys.path.insert(0,str(ROOT/'quant/ptq'))
import torch
from torch import nn
from probe_distill import (CACHE_VERSION,architecture,flow_signature,replay_context,
                           masked_velocity_mse,install_sequential_probe)
from quantizers import gptq_nvfp4


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A=nn.Parameter(torch.randn(2,16)*0.03)
        self.lora_B=nn.Parameter(torch.randn(132,2)*0.03)
        self.backbone=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=range(16))))
        self.action_head=SimpleNamespace(model=SimpleNamespace(transformer_blocks=range(32)),
                                        vl_self_attention=SimpleNamespace(transformer_blocks=range(4)))
    def forward(self,inputs):
        action=inputs['action'];noise=torch.randn_like(action)
        t=(1-torch.distributions.Beta(1.5,1.).sample((1,)))*.999
        x=((1-t)*noise+t*action)[...,:16]
        pred=(x@self.lora_A.T)@self.lora_B.T
        return {'pred_actions':pred,'loss':pred.square().mean()}


def check():
    torch.set_num_threads(2);torch.manual_seed(90929)
    model=Policy();inputs={'action':torch.randn(1,40,132)}
    mask=torch.zeros_like(inputs['action']);mask[:,:16,:7]=1;inputs['action_mask']=mask
    # An independent masked reduction: padding error is deliberately much larger.
    student=torch.zeros_like(mask,requires_grad=True);teacher=torch.full_like(mask,100.)
    teacher[mask.bool()]=2.
    loss=masked_velocity_mse(student,teacher,mask);loss.backward()
    measured_mask_loss=loss.item();padding_gradient=student.grad[~mask.bool()].abs().max().item()
    if measured_mask_loss!=4. or padding_gradient!=0.:
        raise RuntimeError('Mask formula or padded-output derivative differs')
    expected=torch.full((112,),-4/112)
    mask_gradient_error=(student.grad[mask.bool()]-expected).abs().max().item()
    if mask_gradient_error!=0.:raise RuntimeError('Masked MSE normalization differs')
    # Algebra uses explicitly different shapes, so an A/B transpose swap fails.
    x=torch.randn(5,16,dtype=torch.float64);a=model.lora_A.detach().double();b=model.lora_B.detach().double()
    lora_error=(((x@a.T)@b.T)-x@(b@a).T).abs().max().item()
    if lora_error>1e-14:raise RuntimeError('LoRA multiplication order differs')
    # Execute the actual upstream sampler and arithmetic assignments in
    # isolation, without constructing its model or importing any GPU backend.
    upstream=Path(os.environ.get('GR00T_REPO',str(Path.home()/'codebase/groot-fsdp2/Isaac-GR00T')))/'gr00t/model/gr00t_n1d7/gr00t_n1d7.py'
    tree=ast.parse(upstream.read_text())
    head=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Gr00tN1d7ActionHead')
    sample=next(node for node in head.body if isinstance(node,ast.FunctionDef) and node.name=='sample_time')
    scope={};exec(compile(ast.Module(body=[sample],type_ignores=[]),str(upstream),'exec'),scope)
    beta_values=torch.tensor([.0,.25,1.])
    owner=SimpleNamespace(beta_dist=SimpleNamespace(sample=lambda shape:beta_values.clone()),
                          config=SimpleNamespace(noise_s=.999),num_timestep_buckets=1000)
    times=scope['sample_time'](owner,3,'cpu',torch.float32)
    time_error=(times-.999*(1-beta_values)).abs().max().item()
    forward=next(node for node in head.body if isinstance(node,ast.FunctionDef) and node.name=='forward')
    assignments=[node for node in ast.walk(forward) if isinstance(node,ast.Assign) and
                 isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('noisy_trajectory','velocity','t_discretized')]
    actions=torch.randn(3,40,132);noise=torch.randn_like(actions)
    values={'actions':actions,'noise':noise,'t':times[:,None,None],'self':owner}
    exec(compile(ast.Module(body=assignments,type_ignores=[]),str(upstream),'exec'),values)
    interpolation_error=(values['noisy_trajectory']-((1-times[:,None,None])*noise+times[:,None,None]*actions)).abs().max().item()
    velocity_error=(values['velocity']-(actions-noise)).abs().max().item()
    if time_error or interpolation_error or velocity_error or not torch.equal(values['t_discretized'],torch.floor(times*1000).long()):
        raise RuntimeError('Actual upstream flow/time expressions differ')
    update=next(node for node in ast.walk(head) if isinstance(node,ast.Assign) and
                isinstance(node.targets[0],ast.Name) and node.targets[0].id=='actions' and
                isinstance(node.value,ast.BinOp) and 'pred_velocity' in ast.unparse(node.value))
    step=compile(ast.Module(body=[update],type_ignores=[]),str(upstream),'exec')
    values={'actions':noise.clone(),'dt':.25,'pred_velocity':actions-noise,'vel_strength':torch.ones_like(actions)}
    for _ in range(4):exec(step,values)
    integration_error=(values['actions']-actions).abs().max().item()
    if integration_error>2e-6:raise RuntimeError('Forward Euler sign/time direction differs')
    # Teacher hook is the real implementation. Demo backward happens first on
    # each microbatch; the independent counter belongs only to probe backward.
    with tempfile.TemporaryDirectory() as raw:
        path=Path(raw)/'temporary_probe.pt'
        with replay_context(model,47,'none'),torch.no_grad():target=model(inputs)['pred_actions']+.2
        torch.save({'version':CACHE_VERSION,'metadata':{'architecture':architecture(model),
            'flow_config':flow_signature(model),'model_dtype':'float32','autocast_dtype':'none'},
            'samples':[{'seed':47,'inputs':inputs,'pred':target}]},path)
        class Trainer:
            def __init__(self):
                self.args=SimpleNamespace(gradient_accumulation_steps=16)
                self.current_gradient_accumulation_steps=16;self.state=SimpleNamespace(global_step=0)
                self.probes=0
                def backward(value):
                    if not value.requires_grad or not torch.isfinite(value):raise RuntimeError('Invalid teacher backward')
                    self.probes+=1;value.backward()
                self.accelerator=SimpleNamespace(num_processes=1,gradient_accumulation_steps=1,backward=backward)
            def training_step(self,model,inputs):
                loss=model(inputs)['loss']/16;loss.backward();return loss.detach()
        trainer=Trainer();install_sequential_probe(Trainer,path,weight=1.,every=4)
        counts=[];stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            for update in range(4):
                before=trainer.probes;trainer.state.global_step=update
                model.zero_grad(set_to_none=True)
                for _ in range(16):trainer.training_step(model,inputs)
                counts.append(trainer.probes-before)
        if counts!=[0,0,0,16] or len(stream.getvalue().splitlines())!=16:
            raise RuntimeError('OPD cadence or first-period logging differs')
    # Check U orientation against a direct inverse and its own factorization.
    x=torch.randn(50,32);h=x.T@x;hd=h+.01*h.diag().mean()*torch.eye(32)
    inv=torch.linalg.inv(hd);u=torch.linalg.cholesky(inv,upper=True)
    inverse_factor_error=(u.T@u-inv).abs().max().item()
    w=torch.randn(7,32);output,scales=gptq_nvfp4(w,h,return_metadata=True)
    if not torch.isfinite(output).all() or scales['block_scales'].shape!=(7,2):
        raise RuntimeError('GPTQ output metadata differs')
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only audit initialized CUDA')
    return {'mask_valid_elements':int(mask.sum()),'masked_mse':measured_mask_loss,
            'padding_output_gradient_max_abs':padding_gradient,'valid_gradient_reference_max_abs':mask_gradient_error,
            'lora_matrix_order_max_abs':lora_error,'probe_backwards_per_optimizer_update':counts,
            'first_period_teacher_log_lines':16,'gptq_upper_factor_inverse_max_abs':inverse_factor_error,
            'gptq_scale_shape':list(scales['block_scales'].shape),'cuda_initialized':False,
            'flow_time_transform_max_abs':time_error,'flow_interpolation_max_abs':interpolation_error,
            'flow_velocity_target_max_abs':velocity_error,'four_step_euler_endpoint_max_abs':integration_error,
            'upstream_flow_source':str(upstream),'upstream_flow_source_sha256':hashlib.sha256(upstream.read_bytes()).hexdigest()}


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
