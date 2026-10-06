"""Review-build contracts use temporary synthetic assets, never experiment data."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'paper'))
import build_review as review


class ReviewContractTests(unittest.TestCase):
    def test_only_current_explicit_review_status_is_accepted(self):
        _, protocol = review.review_protocol({'status': 'v12 RTN W4A4 审阅稿'})
        self.assertEqual(protocol['id'], 'w4a4-recovery-v12-rtn')
        self.assertEqual(protocol['sha256'], review.PROTOCOL_SHA256)
        for status in ('v11 W4A4 审阅稿', 'v12 W4A4 审阅稿', 'v12 RTN 已核验'):
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                review.review_protocol({'status': status})

    def test_frozen_protocol_bytes_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp)/'paper'
            (paper.parent/'exp').mkdir()
            (paper.parent/'exp'/review.PROTOCOL_NAME).write_text('{}')
            with mock.patch.object(review, 'P', paper), self.assertRaises(RuntimeError):
                review.review_protocol({'status': 'v12 RTN 审阅稿'})

    def test_all_and_only_seven_pending_screenshots_are_required(self):
        registry = {name: {'refresh_pending': True} for name in review.EXPECTED_PENDING}
        self.assertEqual(review.pending_screenshots(registry), review.EXPECTED_PENDING)
        registry.pop('shot_collect')
        with self.assertRaises(RuntimeError): review.pending_screenshots(registry)
        registry['shot_collect'] = {'refresh_pending': True}
        registry['shot_unreviewed'] = {'refresh_pending': True}
        with self.assertRaises(RuntimeError): review.pending_screenshots(registry)

    def test_pending_charts_never_read_evidence(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(review.figures, 'FIGS', Path(tmp)), \
                mock.patch.object(review.figures, 'read', side_effect=AssertionError('No data reads')), \
                mock.patch.object(review.figures, 'budget_rows', side_effect=AssertionError('No budget reads')):
            review.pending_charts()
            for name in ('ladder', 'budget_ladder', 'ptq_frontier'):
                svg = (Path(tmp)/(name+'.svg')).read_text()
                self.assertIn('xxx', svg)
                self.assertIn(review.RED, svg)
                self.assertNotRegex(svg, r'>\d+(?:\.\d+)? GB<')

    def test_archive_binds_protocol_and_excludes_pending_and_old_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paper = root/'paper'
            for folder in ('sections', 'figs', 'zhihu/images', 'validation'):
                (paper/folder).mkdir(parents=True)
            (root/'exp').mkdir()
            protocol = (ROOT/'exp'/review.PROTOCOL_NAME).read_bytes()
            (root/'exp'/review.PROTOCOL_NAME).write_bytes(protocol)
            names = sorted(review.EXPECTED_PENDING) + [f'shot_unchanged_{i}' for i in range(10)]
            names += ['e1_latency', 'swizzle_layout', 'gptq_block', 'recovery_protocol',
                      'ladder', 'budget_ladder', 'ptq_frontier']
            registry = {name: {'title': '图 1　test', 'note': 'test',
                              'refresh_pending': name in review.EXPECTED_PENDING} for name in names}
            (paper/'figures.json').write_text(json.dumps(registry), encoding='utf-8')
            (paper/'meta.json').write_text(json.dumps({'status': 'v12 RTN 审阅稿'}), encoding='utf-8')
            (paper/'sections/01.md').write_text('#b42318 xxx\n' + '\n'.join(
                '{{fig:'+name+'}}' for name in names), encoding='utf-8')
            for name in names:
                (paper/'figs'/(name+'.png')).write_bytes(b'original synthetic screenshot')
                (paper/'zhihu/images'/(name+'.png')).write_bytes(b'generated synthetic image')
            for path in review.INDEPENDENT_ENGINE_INPUTS:
                (root/path).parent.mkdir(parents=True, exist_ok=True)
                (root/path).write_bytes(b'{}')
            for name in ('analysis_plan_w4a4.json', 'paired_uncertainty.py'):
                (paper/name).write_text('forbidden old source')
            (paper/'paper.html').write_text('synthetic rendering')
            (paper/'zhihu/article.md').write_text('synthetic rendering')

            def engine():
                review.figures._DATA_INPUTS.update({path: {'path': path,
                    'sha256': hashlib.sha256(b'{}').hexdigest(), 'bytes': 2}
                    for path in review.INDEPENDENT_ENGINE_INPUTS})

            with mock.patch.object(review, 'P', paper), \
                    mock.patch.object(review.figures, 'FIGS', paper/'figs'), \
                    mock.patch.object(review.figures, 'e1_chart', side_effect=engine), \
                    mock.patch.object(review.figures, 'swizzle_layout'), \
                    mock.patch.object(review.figures, 'gptq_block'), \
                    mock.patch.object(review.figures, 'recovery_protocol'), \
                    mock.patch.object(review.cairosvg, 'svg2png'), \
                    mock.patch.object(review.subprocess, 'run'):
                review.main()
            with zipfile.ZipFile(paper/'apxinf-gr00t-fp4fp8-ptqad-review.zip') as bundle:
                files = set(bundle.namelist())
                manifest = json.loads(bundle.read('review-manifest.json'))
                bound_protocol = manifest['experiment_protocol']
                self.assertEqual(hashlib.sha256(bundle.read(bound_protocol['path'])).hexdigest(),
                                 bound_protocol['sha256'])
                self.assertEqual(manifest['screenshots'], 10)
                self.assertEqual(manifest['screenshot_slots'], 17)
                self.assertNotIn('v11', json.dumps(manifest))
                for name in review.EXPECTED_PENDING:
                    self.assertNotIn('zhihu/images/'+name+'.png', files)
                self.assertNotIn('analysis_plan_w4a4.json', files)
                self.assertNotIn('paired_uncertainty.py', files)
                for record in manifest['files']:
                    self.assertEqual(hashlib.sha256(bundle.read(record['path'])).hexdigest(),
                                     record['sha256'])
            for name in names:
                if name.startswith('shot_'):
                    self.assertEqual((paper/'figs'/(name+'.png')).read_bytes(),
                                     b'original synthetic screenshot')


if __name__ == '__main__': unittest.main()
