"""Local manuscript links survive the section-to-HTML directory move."""
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
import build_html
import gptq_reference_publication as gptq


class PublicationLinks(unittest.TestCase):
    def render(self, text):
        return build_html.render_markdown(
            text, source_dir=ROOT / "paper/sections", output_dir=ROOT / "paper")

    def test_local_targets_move_without_changing_queries_or_external_links(self):
        text = (
            "[doc](../../docs/guide.md#part)\n\n"
            "[scope](../publication_scope_v12.json)\n\n"
            "[encoded](../evidence/my%20file.json?raw=1#entry)\n\n"
            "[directory](../evidence/)\n\n"
            "[remote](https://example.com/a?q=1#part) [anchor](#part) "
            "[mail](mailto:author@example.com) [root](/docs/guide) [cdn](//example.com/path)"
        )
        rendered = self.render(text)
        for target in ("../docs/guide.md#part", "publication_scope_v12.json",
                       "evidence/my%20file.json?raw=1#entry", "evidence/",
                       "https://example.com/a?q=1#part", "#part", "mailto:author@example.com",
                       "/docs/guide", "//example.com/path"):
            self.assertIn('href="' + target + '"', rendered)
        self.assertIn(">doc</a>", rendered)

    def test_code_and_protected_math_are_not_rewritten(self):
        tick = chr(96)
        code = "[literal](../../docs/code.md)"
        math = r"\text{[literal](../../docs/math.md)}"
        source = tick * 3 + "text\n" + code + "\n" + tick * 3 + "\n\n" + tick + code + tick
        source += "\n\n$" + math + "$\n\n[real](../../docs/real.md)"
        with mock.patch.object(build_html.subprocess, "run", return_value=SimpleNamespace(stdout="[]")) as renderer:
            protected, _, sources = build_html.math_and_code(source)
        self.assertEqual(sources, [{"tex": math, "display": False}])
        self.assertEqual(json.loads(renderer.call_args.kwargs["input"]), sources)
        rendered = self.render(protected)
        self.assertIn("<code>" + code + "</code>", rendered)
        self.assertIn(code + "\n</code></pre>", rendered)
        self.assertIn("FPVMATHTOKEN0END", rendered)
        self.assertIn('href="../docs/real.md"', rendered)

    def test_gptq_links_resolve_for_readme_sections_zhihu_and_html(self):
        report = {"status": "not_performed", "decision": {"status": "not_performed"}}
        root_text, source_text = gptq.render(report, detailed=True), gptq.render(report)
        self.assertEqual(root_text, gptq.NOT_PERFORMED_TEXT)
        # Visible prose/labels and all experimental claims are unchanged.
        strip_links = lambda text: re.sub(r"\]\([^)]+\)", "]", text)
        self.assertEqual(strip_links(root_text), strip_links(source_text))
        expected = [ROOT / "paper/publication_scope_v12.json",
                    ROOT / "paper/evidence/gptq_reference/not_performed.json"]
        for directory, text in ((ROOT, root_text), (ROOT / "paper/sections", source_text),
                                (ROOT / "paper/zhihu", source_text)):
            targets = re.findall(r"\]\(([^)]+)\)", text)
            self.assertEqual([(directory / target).resolve() for target in targets], expected)
        rendered = self.render(source_text)
        for target in ("publication_scope_v12.json", "evidence/gptq_reference/not_performed.json"):
            self.assertIn('href="' + target + '"', rendered)


if __name__ == "__main__":
    unittest.main()
