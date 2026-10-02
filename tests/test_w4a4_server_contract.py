"""CPU guards for the W4A4 server's activation imports; no model loading."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class W4A4ServerContractTest(unittest.TestCase):
    def test_activation_scope_is_not_reimported_inside_main(self):
        tree = ast.parse((ROOT / "eval" / "run_gr00t_server_fp4vla.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        nested_imports = [
            node for node in ast.walk(main)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "mark_activation_scope" for alias in node.names)
        ]
        self.assertEqual(nested_imports, [])

    def test_relocated_project_resolves_its_own_rl_and_quant_modules(self):
        # Execute only the real server preamble before its first GR00T import.
        # An isolated subprocess excludes this checkout from normal sys.path;
        # copied module identities also detect fallback to an old absolute path.
        check = """
import ast
from pathlib import Path
import sys
server = Path(sys.argv[1])
nodes = []
for node in ast.parse(server.read_text()).body:
    if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('gr00t.'):
        break
    nodes.append(node)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(server), 'exec'),
     {'__file__': str(server), '__name__': 'relocated_server_preamble'})
import scoped_quant, native_activation, torch_fp4, torch
root = server.parent.parent
for module, subdir in ((scoped_quant, 'rl'), (native_activation, 'quant'), (torch_fp4, 'quant')):
    assert Path(module.__file__).resolve().parent == root / subdir, module.__file__
assert not any(name == 'gr00t' or name.startswith('gr00t.') for name in sys.modules)
assert not torch.cuda.is_initialized()
"""
        with tempfile.TemporaryDirectory(prefix="relocated w4a4 ") as tmp:
            root = Path(tmp).resolve()
            for relative in ("eval/run_gr00t_server_fp4vla.py", "rl/scoped_quant.py",
                             "quant/native_activation.py", "quant/fp4_quant.py", "quant/torch_fp4.py"):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            for adapter in ("0", "1"):
                with self.subTest(adapter=adapter):
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "FP4VLA_QUANT": "0",
                           "FP4VLA_W4A4": "1", "FP4VLA_W4A4_ADAPTER": adapter}
                    result = subprocess.run(
                        [sys.executable, "-I", "-c", check,
                         str(root / "eval/run_gr00t_server_fp4vla.py")],
                        cwd=root, env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
