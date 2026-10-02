"""Resolve the maintained source symbols used by appendix A without importing models.

Run: python paper/validation/check_source_refs.py
This is a source-reference check, not a numerical or GPU test.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PYTHON = {
    "quant/torch_fp4.py": ["_round_grid", "_quant_e2m1", "_quant_e4m3", "_nvfp4_tensor_scale", "_nvfp4_dequant", "fake_quant_nvfp4_torch"],
    "quant/fp4_quant.py": ["encode_e2m1", "decode_e2m1", "encode_e4m3", "decode_e4m3", "nvfp4_quantize", "nvfp4_dequantize", "swizzle_scales"],
    # A.4/A.6: the executable activation QDQ behind the two-branch formula.
    "quant/native_activation.py": ["_torch_native_activation_qdq", "native_activation_qdq_torch", "encode_native_activation"],
    "quant/ptq/quantizers.py": ["nvfp4_dequant", "fp8_e4m3_dequant", "layer_mse_tr", "rtnc_best_clip", "gptq_nvfp4"],
    "quant/ptq/collector.py": ["accumulate", "main"],
    "quant/ptq/verify_calibration.py": ["verify", "check_matrix", "check_architecture"],
    "quant/ptq/bake.py": ["inventory", "alloc", "tied_aliases", "make_plan", "main"],
    # A.3.4: category-bank identification, K-axis padding, calibration and bake.
    "quant/ptq/category_fp4.py": ["category_inventory", "prepare_bank", "quantize_bank", "memory_budget", "load_calibration"],
    "quant/ptq/collector_category.py": ["accumulate", "main"],
    "quant/ptq/bake_category.py": ["main"],
    "quant/ptq/awq.py": ["search_site"],
    "quant/ptq/folds.py": ["build_sites", "consumer_col_to_producer_col"],
    "rl/lora_qad.py": ["install_lora", "load_initial_adapter", "install_gradient_audit", "install_trainer_hooks"],
    "rl/w4a4_lora.py": ["_dense_base", "install_w4a4_lora"],
    "rl/w4a4_deploy.py": ["install_saved_w4a4_adapter"],
    "rl/recovery_batch.py": ["resolve_batch"],
    "rl/activation_checkpoint.py": ["checkpoint_block", "install_activation_checkpointing"],
    "rl/gr00t_runtime.py": ["restore_checkpoint_model_config", "configure_libero_data", "verify_libero_statistics", "libero_action_spec", "action_mask"],
    "rl/capture_onpolicy.py": ["policy_input_precision", "install_capture"],
    "rl/opd_probe_cache.py": ["main"],
    "rl/probe_distill.py": ["masked_velocity_mse", "replay_context", "ProbeAnchor.validate_model", "ProbeAnchor.loss", "install_sequential_probe"],
    "rl/lora_merge_bake.py": ["main"],
    # A.3.4/A.7: recipe-gated ordinary/category activation-only installation.
    "rl/scoped_quant.py": ["mark_scope", "configure_quant_recipe", "_pad_qdq_slice", "install_activation_only", "install_scoped_activation", "mark_activation_scope"],
    "eval/serve_recovery.py": ["main"],
    "eval/run_gr00t_server_fp4vla.py": ["main"],
    "eval/rollout_seeded.py": ["install_bank_resets", "main"],
    "eval/run_recovery_eval.py": ["validate_resets", "parse_log", "main"],
    "eval/compare_recovery.py": ["compare_round", "paired_summary", "exact_mcnemar_p"],
}
NATIVE_ROOT = "third_party/apxinf-robo/apxinf/"
NATIVE = {
    "crates/apxinf-cuda/src/context.rs": ["fp4_workspace"],
    "crates/apxinf-cuda/src/kernels/fp4.rs": ["Fp4WeightView", "fp4_graph_buffer_bytes", "fp4_linear_bf16", "fp4_linear"],
    "crates/apxinf-cuda/adapters/cublaslt_fp4_adapter.cu": ["apxinf_nvfp4_quantize_activation", "apxinf_fp4_gemm_rowmajor_scaled_f16", "apxinf_fp4_prepare_rowmajor_f16", "apxinf_fp4_gemm_rowmajor_prepared_f16"],
    "crates/apxinf-cuda/adapters/fp4_plan_cache.cuh": ["scales", "prepare", "get", "execute"],
    "crates/apxinf-cuda/src/kernels/fp4_contract_tests.rs": ["fp4_contract_rowmajor_and_tensor_scale", "fp4_graph_replay_bf16_and_distinct_scales", "fp4_activation_padding_zero_after_capture"],
    "crates/apxinf-model/src/pi05/fp4_weights.rs": ["Fp4DeviceWeights", "from_artifact_dir"],
    "crates/apxinf-model/src/pi05/model/blocks/fp4.rs": ["Fp4Blocks", "gemm_maybe_fp4", "workspace_requirements"],
    "crates/apxinf-model/examples/pi05_fp4_graph_smoke.rs": ["main"],
    "crates/apxinf-py/src/lib.rs": ["execution_mode"],
}


def python_symbols(text):
    symbols = {}
    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                symbols[name] = node.lineno
                visit(node.body, name + ".")
    visit(ast.parse(text).body)
    return symbols


def check(path, expected, python):
    content = (ROOT / path).read_text(encoding="utf-8-sig")
    found = python_symbols(content) if python else {}
    refs = []
    for name in expected:
        if python:
            line = found.get(name)
        else:
            pattern = r"\b(?:pub\s+)?(?:fn|struct)\s+" + re.escape(name) + r"\b" if path.endswith(".rs") else r"\b" + re.escape(name) + r"\s*\("
            match = re.search(pattern, content)
            line = content.count("\n", 0, match.start()) + 1 if match else None
        if line is None:
            raise ValueError(f"Missing referenced symbol {path}::{name}")
        refs.append({"symbol": name, "line": line})
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), "symbols": refs}


def main():
    files = [check(path, names, True) for path, names in PYTHON.items()]
    files.extend(check(NATIVE_ROOT + path, names, False) for path, names in NATIVE.items())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "git_head": head,
              "scope": "Working-tree file hashes; line numbers are generated navigation, not stable citations.",
              "status": "passed", "files": files,
              "symbol_count": sum(len(item["symbols"]) for item in files)}
    output = Path(__file__).with_name("source_refs.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"source refs passed: {len(files)} files, {report['symbol_count']} symbols -> {output}")


if __name__ == "__main__":
    main()
