"""Exercise trace hooks with the real GR00T Euler sampler and tiny CPU layers.

The optional external source check uses GR00T_REPO (or the local checkout). It
does not load weights, import GR00T, or claim CUDA/model-quality validation.
"""
import ast
import copy
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
from torch import nn

from exp.action_chunk_diagnostics import compare, infer_chunk
from rl.gr00t_runtime import action_mask
from tests.test_action_chunk_diagnostics import SPEC, payload


class Encoder(nn.Module):
    def forward(self, actions, times, embodiment):
        return actions


class TinyDiT(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.125))

    def forward(self, hidden, embodiment):
        return hidden * self.scale + 0.25


class ProductionSampler(nn.Module):
    def __init__(self, method):
        super().__init__()
        head = self.action_head = nn.Module()
        head.config = SimpleNamespace(action_horizon=40, add_pos_embed=False,
                                      use_alternate_vl_dit=False)
        head.action_dim, head.action_horizon = 132, 40
        head.num_inference_timesteps, head.num_timestep_buckets = 4, 1000
        head.action_encoder, head.action_decoder = Encoder(), Decoder()
        head.model = TinyDiT()
        head.sample = MethodType(method, head)

    def get_action(self, inputs):
        if 'action' in inputs or 'action_mask' in inputs:
            raise AssertionError('Diagnostic endpoint must not enable RTC')
        return self.action_head.sample(torch.zeros(1, 3, 132), inputs['state'],
                                       torch.zeros(1, dtype=torch.long), {}, inputs)


def external_sampler():
    root = Path(os.environ.get('GR00T_REPO',
                '/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T'))
    source = root / 'gr00t/model/gr00t_n1d7/gr00t_n1d7.py'
    if not source.is_file():
        raise unittest.SkipTest('Set GR00T_REPO to exercise the real external sampler')
    tree = ast.parse(source.read_text())
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == 'get_action_with_features']
    if len(methods) != 1:
        raise AssertionError('Production sampler identity changed')
    method = methods[0]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    namespace = {'torch': torch, 'BatchFeature': lambda data: data}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return namespace['get_action_with_features']


def traced_payload(arm='bf16'):
    out = payload(arm)
    out['input_manifest']['format'] = 'gr00t_action_inputs_v2'
    record = out['records'][0]
    record['integration_trace'] = {
        'states': torch.zeros(4, 1, 40, 132, dtype=torch.bfloat16),
        'velocities': torch.zeros(4, 1, 40, 132, dtype=torch.bfloat16),
        'time_buckets': torch.tensor([[0], [250], [500], [750]], dtype=torch.int64),
        'scope': "Each policy's own integration states and velocities; not shared-state field error or robot rollout drift"}
    return out


class ProductionTraceTests(unittest.TestCase):
    def test_real_sampler_trace_preserves_endpoint_rng_and_modes(self):
        model = ProductionSampler(external_sampler()).train()
        model.action_head.action_encoder.eval()
        endpoint = torch.full((1, 40, 132), 999.)
        inputs = {'action': endpoint, 'action_mask': action_mask(endpoint, SPEC),
                  'state': torch.full((1, 2, 132), 123.)}
        before = torch.random.get_rng_state().clone()
        plain = infer_chunk(model, inputs, 40, SPEC)
        traced = infer_chunk(model, inputs, 40, SPEC, record_trace=True)
        self.assertTrue(torch.equal(plain['action_pred'], traced['action_pred']))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(model.training)
        self.assertFalse(model.action_head.action_encoder.training)
        trace = traced['integration_trace']
        self.assertEqual(trace['states'].shape, (4, 1, 40, 132))
        self.assertTrue(torch.equal(trace['states'][0], traced['initial_noise']))
        self.assertEqual(trace['time_buckets'].flatten().tolist(), [0, 250, 500, 750])
        # Decoder includes state-prefix rows; hooks must retain action rows only.
        self.assertTrue(torch.equal(trace['velocities'], trace['states'] * 0.125 + 0.25))
        self.assertTrue(torch.equal(trace['states'][1:],
                                   trace['states'][:-1] + trace['velocities'][:-1] / 4))
        self.assertTrue(torch.equal(traced['action_pred'],
                                   trace['states'][-1] + trace['velocities'][-1] / 4))
        self.assertFalse(model.action_head.action_encoder._forward_pre_hooks)
        self.assertFalse(model.action_head.action_decoder._forward_hooks)

    def test_hooks_and_modes_restored_when_production_inference_fails(self):
        model = ProductionSampler(external_sampler()).train()
        def fail(*args):
            raise RuntimeError('intentional decoder failure')
        model.action_head.action_decoder.forward = fail
        endpoint = torch.zeros(1, 40, 132)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            infer_chunk(model, {'action': endpoint, 'action_mask': action_mask(endpoint, SPEC),
                               'state': torch.zeros(1, 2, 132)}, 30, SPEC, record_trace=True)
        self.assertTrue(model.training)
        self.assertFalse(model.action_head.action_encoder._forward_pre_hooks)
        self.assertFalse(model.action_head.action_decoder._forward_hooks)


class IntegrationMetricTests(unittest.TestCase):
    def test_padding_excluded_and_last_valid_step_included(self):
        ref, cur = traced_payload(), traced_payload('qad_opd')
        trace = cur['records'][0]['integration_trace']
        trace['states'][1:, :, 16:] = 100
        trace['velocities'][:, :, :, 7:] = 100
        report = compare(ref, cur)['task_macro']
        self.assertEqual(report['integration_state_step_4_mse'], 0)
        self.assertEqual(report['integration_own_path_velocity_step_4_mse'], 0)
        trace['velocities'][3, 0, 15, 6] = 2
        report = compare(ref, cur)['task_macro']
        self.assertAlmostEqual(report['integration_own_path_velocity_step_4_mse'], 4 / 112)

    def test_independent_inputs_require_both_traces(self):
        for arms in ((0,), (0, 1)):
            pair = [traced_payload(), traced_payload('ptq')]
            for index in arms:
                del pair[index]['records'][0]['integration_trace']
            with self.assertRaisesRegex(ValueError, 'full integration traces'):
                compare(*pair)

    def test_wrong_time_or_initial_noise_or_scope_or_nonfinite_rejected(self):
        changes = [lambda t: t['time_buckets'].__setitem__((1, 0), 251),
                   lambda t: t.update(time_buckets=t['time_buckets'].float()),
                   lambda t: t['states'].__setitem__((0, 0, 0, 0), 1),
                   lambda t: t.update(scope='shared-state field error'),
                   lambda t: t['velocities'].__setitem__((0, 0, 39, 131), float('nan')),
                   lambda t: t.update(states=t['states'][:3])]
        for change in changes:
            ref, cur = traced_payload(), traced_payload('ptq')
            change(cur['records'][0]['integration_trace'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare(ref, cur)


if __name__ == '__main__':
    unittest.main()
