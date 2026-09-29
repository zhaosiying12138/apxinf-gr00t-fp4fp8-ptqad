"""CPU layout tests with synthetic scores only; never read publication outcomes."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import random
import re
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

MODULE = Path(__file__).resolve().parents[1]/'make_figs.py'
spec = importlib.util.spec_from_file_location('frontier_figure_layout', MODULE)
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)
BOUNDS = (100, 124, 1040, 505)
# Synthetic byte budgets approximate the difficult close spacing, not evidence.
BUDGETS = (6288000000, 2909000000, 2732000000, 2673000000, 2790000000, 2790000000, 2790000000)
NAMES = ('bf16', 'fp8', 'head_ffn', 'ptq', 'qad', 'continued_qad', 'qad_opd')
CASES = {
    'all_zero': [0]*7,
    'all_full': [100]*7,
    'equal_middle': [50]*7,
    'nearby': [93, 91, 90, 89, 92, 91, 88],
    'opd_decline': [93, 90, 85, 83, 96, 94, 0],
}


def synthetic_points(scores):
    points=[]
    for i,(name,size,score) in enumerate(zip(NAMES,BUDGETS,scores)):
        residual=117000000 if i>=4 else 0
        cost={'source_bytes':BUDGETS[0], 'base_bytes':size-residual,
              'residual_bytes':residual, 'total_bytes':size, 'compression_x':BUDGETS[0]/size}
        points.append({'name':name, 'role':'bf16' if i==0 else 'recovery' if i>=4 else 'pure_ptq_reference',
                       'count':100, 'successes':score, 'macro_success_rate':score/100,
                       'encoding_budget':{'physical':dict(cost),'known_alias_deduplicated':dict(cost)}})
    return points


def markers_for(scores):
    groups={}
    for i,(size,score) in enumerate(zip(BUDGETS,scores)):
        groups.setdefault((size,score),[]).append(chr(65+i))
    return [{'x':100+940*size/(max(BUDGETS)*1.06), 'y':505-381*score/100,
             'label':','.join(letters)} for (size,score),letters in groups.items()]


class FrontierLayoutTests(unittest.TestCase):
    def check_layout(self, markers):
        original=copy.deepcopy(markers)
        labels=figs.frontier_label_layout(markers,BOUNDS)
        self.assertEqual(markers,original, 'Layout moved a measured point')
        self.assertEqual([v['label'] for v in labels],[v['label'] for v in markers])
        for marker,label in zip(markers,labels):
            x0,y0,x1,y1=label['box']
            self.assertGreaterEqual(x0,BOUNDS[0]+8)
            self.assertGreaterEqual(y0,BOUNDS[1]+8)
            self.assertLessEqual(x1,BOUNDS[2]-8)
            self.assertLessEqual(y1,BOUNDS[3]-8)
            self.assertEqual(label['leader'][0],(marker['x'],marker['y']))
            ex,ey=label['leader'][1]
            self.assertTrue(x0<=ex<=x1 and y0<=ey<=y1 and (ex in (x0,x1) or ey in (y0,y1)))
            for point in markers:
                self.assertTrue(x1 < point['x']-9 or x0 > point['x']+9 or
                                y1 < point['y']-9 or y0 > point['y']+9,
                                'Label covers a fixed marker')
        for i,left in enumerate(labels):
            a,b,c,d=left['box']
            for right in labels[i+1:]:
                e,f,g,h=right['box']
                self.assertTrue(c+5<e or g+5<a or d+5<f or h+5<b, 'Labels overlap')
        self.assertEqual(labels,figs.frontier_label_layout(markers,BOUNDS), 'Layout is nondeterministic')
        return labels

    def test_zero_and_full_boundaries(self):
        for name in ('all_zero','all_full'):
            with self.subTest(case=name):
                self.check_layout(markers_for(CASES[name]))

    def test_equal_and_close_success_rates(self):
        for name in ('equal_middle','nearby'):
            with self.subTest(case=name):
                self.check_layout(markers_for(CASES[name]))

    def test_opd_decline(self):
        self.check_layout(markers_for(CASES['opd_decline']))

    def test_deterministic_dense_stress(self):
        rng=random.Random(20260929)
        for case in range(50):
            with self.subTest(case=case):
                self.check_layout(markers_for([rng.randrange(85,101) for _ in range(7)]))

    def test_axes_corners_and_long_merged_label(self):
        self.check_layout([{'x':100,'y':124,'label':'A,B,C'},
                           {'x':1040,'y':124,'label':'D'},
                           {'x':100,'y':505,'label':'E'},
                           {'x':1040,'y':505,'label':'F,G'}])

    def test_svg_preserves_points_and_merged_letters(self):
        namespace={'s':'http://www.w3.org/2000/svg'}
        for name,scores in CASES.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix='synthetic-frontier-layout-') as tmp:
                with patch.object(figs,'FIGS',Path(tmp)), patch.object(figs,'frontier_rows',return_value=({},synthetic_points(scores))), patch.object(figs,'_GENERATED_SVGS',set()):
                    figs.ptq_frontier()
                root=ET.fromstring((Path(tmp)/'ptq_frontier.svg').read_text())
                labels=[node for node in root.findall('s:text',namespace) if 'textLength' in node.attrib]
                markers=markers_for(scores)
                self.assertEqual([node.text for node in labels],[m['label'] for m in markers])
                circles=root.findall('s:circle',namespace)
                self.assertEqual(len(circles),4)
                for i,node in enumerate(circles):
                    self.assertAlmostEqual(float(node.attrib['cx']),100+940*BUDGETS[i]/(max(BUDGETS)*1.06))
                    self.assertAlmostEqual(float(node.attrib['cy']),505-381*scores[i]/100)
                if len(set(scores[4:]))==1:
                    self.assertIn('E,F,G',[node.text for node in labels])
                diamonds=[node for node in root.findall('s:path',namespace) if node.attrib.get('fill')==figs.TEAL]
                recovery_coordinates=set((100+940*BUDGETS[i]/(max(BUDGETS)*1.06),505-381*scores[i]/100) for i in range(4,7))
                actual_centers=[]
                for node in diamonds:
                    coords=[float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',node.attrib['d'])]
                    self.assertEqual(len(coords),8)
                    actual_centers.append((sum(coords[::2])/4,sum(coords[1::2])/4))
                self.assertEqual(len(actual_centers),len(recovery_coordinates))
                for actual,expected in zip(sorted(actual_centers),sorted(recovery_coordinates)):
                    for value,wanted in zip(actual,expected):
                        self.assertAlmostEqual(value,wanted)

    def test_segment_rectangle_edge_cases(self):
        box=(0,0,10,10)
        self.assertTrue(figs.segment_hits_box((-1,5),(11,5),box))
        self.assertTrue(figs.segment_hits_box((0,-2),(0,12),box))
        self.assertTrue(figs.segment_hits_box((5,5),(5,5),box))
        self.assertFalse(figs.segment_hits_box((-2,-1),(12,-1),box))
        self.assertFalse(figs.segment_hits_box((-1,1),(1,-2),box))


if __name__=='__main__':
    unittest.main()
