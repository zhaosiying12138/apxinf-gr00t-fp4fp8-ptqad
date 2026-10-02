"""CPU-only legacy API regression and current v11 publication tests.

The legacy snapshot API retains its seven-arm/100-episode regression fixture.
Only the current five-arm/160-episode disk fixture reaches v11 chart readers.
All fixtures and rendered files are temporary, visibly synthetic test data.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P));sys.path.insert(0,str(P.parent/'tests'))
import test_ptq_frontier as fixtures
from frontier_evidence import f,validate_frontier,digest
from collect_frontier_evidence import collect
import make_figs
import build_final_frontier
write=fixtures.write


V11_ARMS=('bf16','ptq','qad','continued_qad','qad_opd')
V11_RECIPE='all_nvfp4_gptq_category'
# Synthetic byte budgets; these are not measured publication outcomes.
V11_SOURCE_BYTES=(6400000000,6288000000)
V11_BASE_BYTES=(1800000000,1770000000)
V11_RESIDUAL_BYTES=146000000


def write_v11_chart_fixture(root,scores=(148,80,144,142,147)):
    """Write real JSON inputs, then run the current frontier builder unchanged."""
    root=Path(root);paper=root/'paper';evidence=paper/'evidence'
    if len(scores)!=5 or any(type(score) is not int or not 0<=score<=160 for score in scores):
        raise ValueError('Synthetic scores must contain five integer counts in 0..160')
    arms={}
    for name,score in zip(V11_ARMS,scores):
        per_task={task:{'episodes':16,'successes':max(0,min(16,score-16*i))}
                  for i,task in enumerate(f.TASKS)}
        for value in per_task.values():value['success_rate']=value['successes']/16
        arms[name]={'count':160,'successes':score,'macro_success_rate':score/160,
                    'per_task':per_task}
    paired={'environment_pairing_verified':True,'protocol_consistency_verified':True,
            'source_accounting_verified':True,'arms':arms}
    for baseline in ('qad','continued_qad'):
        paired['opd_vs_'+baseline]={'paired_count':160,
            'success_rate_difference_opd_minus_baseline':arms['qad_opd']['successes']/160-arms[baseline]['successes']/160}
    pair_path=evidence/'paired_comparison.json';write(pair_path,paired)
    unique={'source_tensor_bytes':V11_SOURCE_BYTES[1],
            'target_full_bytes':V11_BASE_BYTES[1],
            'full_compression_x':V11_SOURCE_BYTES[1]/V11_BASE_BYTES[1],
            'eligible_tensor_elements':3100000000,
            'elements_by_format':{'nvfp4':3100000000,'fp8':0,'bf16':0},
            'fraction_of_eligible':{'nvfp4':1.0,'fp8':0.0,'bf16':0.0}}
    memory={'source_tensor_bytes':V11_SOURCE_BYTES[0],
            'target_full_checkpoint_bytes':V11_BASE_BYTES[0],
            'full_checkpoint_compression_x':V11_SOURCE_BYTES[0]/V11_BASE_BYTES[0],
            'eligible_tensor_count':479,'linear_params':3150000000,
            'nvfp4_params':3150000000,'fp8_params':0,'bf16_params':0,
            'fraction_of_eligible_params':{'nvfp4':1.0,'fp8':0.0,'bf16':0.0},
            'known_tied_alias_deduplicated':unique}
    recipe_path=evidence/'selected_recipe/category_ptq_recipe.json'
    recipe={'version':'synthetic-category-v11','recipe':'category_nvfp4_extension','memory':memory}
    write(recipe_path,recipe)
    write(evidence/'selected_recipe/category_memory.json',{
        'format':'selected_category_recipe_memory_v2','recipe':'category_nvfp4_extension',
        'source_recipe':'category_ptq_recipe.json','source_recipe_sha256':digest(recipe_path),'memory':memory})
    write(evidence/'selected_recipe/category_bake_manifest.json',{
        'status':'complete','version':recipe['version'],'memory':memory})
    inventory={'schema_version':'v11-selected-all-nvfp4-category','recipe':V11_RECIPE,
        'ladder':[V11_RECIPE],'recipes':{V11_RECIPE:memory},
        'selected_recipe_artifacts':{
            'memory':'paper/evidence/selected_recipe/category_memory.json',
            'ptq_recipe':'paper/evidence/selected_recipe/category_ptq_recipe.json',
            'bake_manifest':'paper/evidence/selected_recipe/category_bake_manifest.json'},
        'recovery_residual':{'scope':'all_ordinary_linear','rank':32,'alpha':64,'linear_modules':468,
            'dtype':'bfloat16','bytes_per_element':2,'tensor_elements':V11_RESIDUAL_BYTES//2,
            'target_bytes':V11_RESIDUAL_BYTES}}
    inventory_path=evidence/'recipe_inventory.json';write(inventory_path,inventory)
    exposed={name:{'successes':row['successes'],'episodes':row['count'],
                    'macro_success_rate':row['macro_success_rate'],'per_task':row['per_task']}
             for name,row in arms.items()}
    final={'format':'publication_final_results_v1','status':'complete','selected_recipe':V11_RECIPE,
           'source':{'heldout_comparison':build_final_frontier.identity(pair_path)},
           'public_arms':{name:row for name,row in exposed.items() if name!='continued_qad'},
           'control':{'continued_qad':exposed['continued_qad']},'fixture':'synthetic CPU test only'}
    final_path=paper/'final_results.json';write(final_path,final)
    output=evidence/'frontier_comparison.json'
    with mock.patch.object(build_final_frontier,'ROOT',root):
        return build_final_frontier.build(final_path,pair_path,inventory_path,output)


class LegacyPublishedFrontier(unittest.TestCase):
    """Retain the old snapshot API's raw-log and tamper regression contract."""
    def setUp(self):
        self.fixture=fixtures.FrontierTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        t=self.fixture;t.heldout()
        patch=mock.patch.object(f,'__file__',str(t.root/'eval/compare_ptq_frontier.py'));patch.start();self.addCleanup(patch.stop)
        self.result=f.compare(t.plan,t.round,t.out)
        write(t.out/'frontier_comparison.json',self.result)
        self.paper=t.root/'published-paper';(self.paper/'evidence').mkdir(parents=True)
        shutil.copyfile(t.budget,self.paper/'evidence/recipe_inventory.json')

    def snapshot(self):return collect(self.fixture.out,self.paper)

    def rewrite(self,path,value):
        write(path,value)
        file=self.paper/'evidence/frontier/evidence_manifest.json';mapping=f.load(file)
        for row in mapping['mapping']:
            if self.paper/row['published_path']==path:row.update(bytes=path.stat().st_size,sha256=digest(path))
        write(file,mapping)

    def test_snapshot_preserves_json_and_validates_without_original_weights_or_logs(self):
        report=self.snapshot();self.assertEqual(report['status'],'complete')
        self.assertEqual((self.paper/'evidence/frontier_comparison.json').read_bytes(),(self.fixture.out/'frontier_comparison.json').read_bytes())
        for folder in (self.fixture.root/'weights',self.fixture.round,self.fixture.out,self.fixture.dev):shutil.rmtree(folder)
        verified,files=validate_frontier(self.paper)
        self.assertEqual(verified,self.result);self.assertGreater(len(files),100)
        self.assertFalse(any(path.suffix=='.safetensors' for path in files))

    def test_missing_raw_log_fails(self):
        self.snapshot();(self.paper/'evidence/frontier/references/heldout_fp8'/(f.TASKS[0]+'.log')).unlink()
        with self.assertRaises((RuntimeError,FileNotFoundError)):validate_frontier(self.paper)

    def test_changed_raw_log_fails_even_if_copy_manifest_is_updated(self):
        self.snapshot();path=self.paper/'evidence/frontier/references/heldout_fp8'/(f.TASKS[0]+'.log')
        path.write_text(path.read_text()+'changed after evaluation\n')
        m=self.paper/'evidence/frontier/evidence_manifest.json';data=f.load(m)
        for row in data['mapping']:
            if self.paper/row['published_path']==path:row.update(bytes=path.stat().st_size,sha256=digest(path))
        write(m,data)
        with self.assertRaisesRegex(ValueError,'raw log'):validate_frontier(self.paper)

    def test_net_residual_omission_rejected(self):
        self.snapshot();path=self.paper/'evidence/frontier_comparison.json';data=f.load(path)
        data['points'][-1]['encoding_budget']['physical']['residual_bytes']=0
        self.rewrite(path,data)
        with self.assertRaisesRegex(RuntimeError,'points differ'):validate_frontier(self.paper)

    def test_reference_omission_rejected(self):
        self.snapshot();path=self.paper/'evidence/frontier_comparison.json';data=f.load(path)
        del data['references']['fp8'];self.rewrite(path,data)
        with self.assertRaisesRegex(RuntimeError,'Reference results'):validate_frontier(self.paper)

    def test_pairing_is_rechecked_from_copied_episodes(self):
        self.snapshot();original=f.read_heldout
        def altered(folder,protocol):
            manifest,arm=original(folder,protocol)
            if Path(folder).name=='heldout_fp8':arm['episodes'][1]['restored_state_sha256']='f'*64
            return manifest,arm
        with mock.patch.object(f,'read_heldout',side_effect=altered):
            with self.assertRaisesRegex(ValueError,'pairing'):validate_frontier(self.paper)

    def test_other_main_round_rejected(self):
        self.snapshot();main=f.load(self.paper/'evidence/frontier/main/paired_comparison.json');main['arms']['ptq']['successes']-=1
        with self.assertRaisesRegex(RuntimeError,'another five-arm'):validate_frontier(self.paper,main)

    def test_snapshot_refuses_overwrite(self):
        self.snapshot()
        with self.assertRaisesRegex(RuntimeError,'overwrite'):self.snapshot()


class CurrentV11PublishedFrontier(unittest.TestCase):
    """Exercise current builders/readers with five arms and real file identities."""
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='synthetic-v11-frontier-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.result=write_v11_chart_fixture(self.root)
        self.evidence=self.root/'paper/evidence'
        for patcher in (mock.patch.object(make_figs,'ROOT',self.root),
                        mock.patch.object(make_figs,'_DATA_INPUTS',{}),
                        mock.patch.object(make_figs,'_GENERATED_SVGS',set()),
                        mock.patch.object(build_final_frontier,'ROOT',self.root)):
            patcher.start();self.addCleanup(patcher.stop)

    def rebuild(self):
        return build_final_frontier.build(self.root/'paper/final_results.json',
            self.evidence/'paired_comparison.json',self.evidence/'recipe_inventory.json',
            self.evidence/'frontier_comparison.json')

    def test_chart_missing_data_and_malformed_budget_fail_closed(self):
        path=self.evidence/'frontier_comparison.json';payload=path.read_bytes();path.unlink()
        with self.assertRaises(FileNotFoundError):make_figs.frontier_rows()
        path.write_bytes(payload)
        self.assertEqual([row['name'] for row in make_figs.frontier_rows()[1]],list(V11_ARMS))
        changed=copy.deepcopy(self.result);changed['points'][-1]['encoding_budget']['physical']['total_bytes']-=8
        write(path,changed);make_figs._DATA_INPUTS.clear()
        with self.assertRaisesRegex(ValueError,'net encoding'):make_figs.frontier_rows()

    def test_chart_writes_only_temporary_fixture_output(self):
        folder=self.root/'test-render';folder.mkdir()
        with mock.patch.object(make_figs,'FIGS',folder):
            make_figs.ptq_frontier()
        import xml.etree.ElementTree as ET
        doc=ET.parse(folder/'ptq_frontier.svg')
        self.assertEqual(doc.getroot().attrib['width'],'1120')
        self.assertEqual(doc.getroot().attrib['height'],'922')

    def test_current_recipe_artifact_identity_is_read_from_disk(self):
        path=self.evidence/'selected_recipe/category_bake_manifest.json'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'source identity differs'):make_figs.frontier_rows()

    def test_selected_recipe_digest_is_rechecked(self):
        path=self.evidence/'selected_recipe/category_ptq_recipe.json'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'source recipe digest differs'):make_figs.frontier_rows()

    def test_current_frontier_rejects_missing_control_or_extra_reference(self):
        for mutation in ('missing_control','extra_reference'):
            changed=copy.deepcopy(self.result)
            if mutation=='missing_control':changed['points']=[p for p in changed['points'] if p['name']!='continued_qad']
            else:changed['references']={'fp8':{}};changed['reference_order']=['fp8']
            write(self.evidence/'frontier_comparison.json',changed);make_figs._DATA_INPUTS.clear()
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'five arms'):
                make_figs.frontier_rows()

    def test_builder_rechecks_final_pair_identity(self):
        (self.evidence/'frontier_comparison.json').unlink()
        path=self.evidence/'paired_comparison.json';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'comparison identity differs'):self.rebuild()

    def test_current_builder_refuses_overwrite(self):
        with self.assertRaisesRegex(ValueError,'existing output'):self.rebuild()


if __name__=='__main__':unittest.main()
