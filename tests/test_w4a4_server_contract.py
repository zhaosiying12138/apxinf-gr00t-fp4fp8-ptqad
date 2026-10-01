"""Static guard for the W4A4 server's activation branch imports."""
import ast
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
