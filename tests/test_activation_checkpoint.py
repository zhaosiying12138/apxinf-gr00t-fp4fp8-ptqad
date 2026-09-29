"""CPU gradient/recomputation checks with frozen inputs and eval-mode LoRA."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from activation_checkpoint import checkpoint_block, install_activation_checkpointing


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8,8), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(2,8))
        self.lora_B = nn.Parameter(torch.randn(8,2))
        self.dropout = nn.Dropout(.2)
        self.calls = 0
    def forward(self, x, scale=1):
        self.calls += 1
        x = self.dropout(x)
        return (x @ self.weight.T + x @ self.lora_A.T @ self.lora_B.T).sin() * scale


class ActivationCheckpointTests(unittest.TestCase):
    def test_eval_and_training_outputs_gradients_rng_and_keys_match(self):
        for training in (False, True):
            torch.manual_seed(93)
            plain = Block().train(training)
            recompute = copy.deepcopy(plain)
            keys = list(recompute.state_dict())
            checkpoint_block(recompute)
            x = torch.randn(3,8)  # frozen input; non-reentrant keeps LoRA gradients
            rng = torch.random.get_rng_state()
            expected = plain(x, scale=2); expected.square().mean().backward()
            after_plain = torch.random.get_rng_state()
            torch.random.set_rng_state(rng)
            actual = recompute(x, scale=2); actual.square().mean().backward()
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            torch.testing.assert_close(plain.lora_B.grad,recompute.lora_B.grad,rtol=0,atol=0)
            torch.testing.assert_close(plain.lora_A.grad,recompute.lora_A.grad,rtol=0,atol=0)
            self.assertEqual(list(recompute.state_dict()),keys)
            self.assertEqual(recompute.calls,2)
            self.assertTrue(torch.equal(after_plain,torch.random.get_rng_state()))
            with torch.no_grad(): recompute(x)
            self.assertEqual(recompute.calls,3)

    def test_real_qwen_dit_vl_with_project_lora_when_gr00t_is_available(self):
        if os.environ.get("GR00T_REPO"):
            sys.path.insert(0, os.environ["GR00T_REPO"])
        try:
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
            from gr00t.model.modules.dit import DiT, SelfAttentionTransformer
        except ImportError:
            self.skipTest("Set GR00T_REPO and use its training Python for the real-module integration")
        import lora_qad
        torch.manual_seed(93)
        config = Qwen3VLTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=64,rope_scaling={"rope_type":"default",
            "mrope_section":[1,1,2],"mrope_interleaved":True},use_cache=False)
        config._attn_implementation = "sdpa"
        plain = nn.Module(); plain.backbone = nn.Module(); plain.backbone.model = nn.Module()
        plain.backbone.model.language_model = Qwen3VLTextModel(config)
        plain.action_head = nn.Module()
        plain.action_head.model = DiT(num_attention_heads=4,attention_head_dim=8,output_dim=32,
                                     num_layers=2,cross_attention_dim=32,interleave_self_attention=True)
        plain.action_head.vl_self_attention = SelfAttentionTransformer(
            num_attention_heads=4,attention_head_dim=8,num_layers=1)
        recompute = copy.deepcopy(plain)  # clone BEFORE installing forward closures
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"ptq_recipe.json").write_text("{}")
            with patch.dict(os.environ,{"GR00T_BASE_CKPT":directory}), patch.object(lora_qad,"SCOPE","head+lang_all"):
                lora_qad.install_lora(plain); lora_qad.install_lora(recompute)
        recompute.load_state_dict(plain.state_dict())
        plain.eval(); recompute.eval()
        self.assertEqual(install_activation_checkpointing(recompute)["blocks"],{"language":2,"dit":2,"vl":1})
        inputs = torch.arange(8).reshape(1,8); action = torch.randn(1,4,32); time = torch.tensor([7])
        def run(model):
            text = model.backbone.model.language_model(input_ids=inputs).last_hidden_state
            value = model.action_head.model(action,model.action_head.vl_self_attention(text),timestep=time)
            value.square().mean().backward()
            return value
        expected, actual = run(plain), run(recompute)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        reference = dict(plain.named_parameters())
        for name,parameter in recompute.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(reference[name].grad,parameter.grad,rtol=0,atol=0)
        for group in ("backbone", "action_head"):
            self.assertGreater(sum(p.grad.abs().sum().item() for n,p in recompute.named_parameters()
                                   if n.startswith(group) and n.endswith("lora_B") and p.grad is not None),0.)


if __name__ == "__main__":
    unittest.main()
