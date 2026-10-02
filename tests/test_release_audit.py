"""CPU tests for the read-only release evidence auditor."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import audit_recovery_release as audit


def selection_fixture(root, prefix=''):
    """Small producer-shaped v11 source records, never real experiment output."""
    checker=audit.Audit(root,payload=False)
    base=root/'checkpoint';base.mkdir()
    (base/'ptq_recipe.json').write_text('{"recipe":"ordinary"}')
    parent_sha=checker.digest(base/'ptq_recipe.json')['sha256']
    (base/'category_ptq_recipe.json').write_text(json.dumps({'parent_recipe_sha256':parent_sha}))
    category_sha=checker.digest(base/'category_ptq_recipe.json')['sha256']
    candidate='all_nvfp4_gptq_category';path=root/'selection/selection.json';path.parent.mkdir()
    folder=path.parent/prefix/candidate;folder.mkdir(parents=True)
    values={'eval_manifest.json':{'protocol_sha256':'a'*64,'purpose':'development','checkpoint':str(base)},
            'task_results.json':{},'summary.json':{}}
    for name,value in values.items():(folder/name).write_text(json.dumps(value))
    selection={'selected_recipe':candidate,'source_sha256':{
        prefix+candidate+'/'+name:checker.digest(folder/name)['sha256'] for name in values},
        'arms':{candidate:{'checkpoint':str(base),'model_identity':{'path':str(base),'metadata':{
            'ptq_recipe.json':parent_sha,'category_ptq_recipe.json':category_sha}}}}}
    protocol={'id':'w4a4-recovery-v11-category','version':11,'w4a4':True,
              'quantization_scope':{'category_w4a4_required':True}}
    run={'protocol_sha256':'a'*64,'selected_ptq_recipe_sha256':parent_sha}
    return checker,path,base,folder,selection,protocol,run


class ReleaseAuditTest(unittest.TestCase):
    def test_v11_sources_beside_selection_are_bound_without_development_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            checker,path,base,folder,selection,protocol,run=selection_fixture(Path(tmp))
            audit.selection_evidence_checks(checker,path,selection,protocol,run)
            audit.selected_recipe_checks(checker,base,run,selection,protocol)
            self.assertTrue(checker.checks)
            self.assertTrue(all(row['status']=='matched' for row in checker.checks),checker.checks)
            checked={row['id'] for row in checker.checks}
            self.assertIn('selection.ptq_recipe_binding',checked)
            self.assertIn('selection.category_recipe_binding',checked)

    def test_explicit_nested_source_keys_remain_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            checker,path,base,folder,selection,protocol,run=selection_fixture(Path(tmp),'development/')
            audit.selection_evidence_checks(checker,path,selection,protocol,run)
            self.assertTrue(all(row['status']=='matched' for row in checker.checks))

    def test_missing_declared_direct_source_never_falls_back_to_nested_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            checker,path,base,folder,selection,protocol,run=selection_fixture(Path(tmp))
            target=path.parent/'development'/folder.name;target.parent.mkdir();folder.rename(target)
            audit.selection_evidence_checks(checker,path,selection,protocol,run)
            self.assertTrue(any(row['status']=='unverifiable' for row in checker.checks))
            self.assertFalse(any(row['id'].startswith('selection.raw_protocol.') for row in checker.checks))

    def test_tampered_manifest_and_incomplete_source_map_are_not_accepted(self):
        for mutation in ('tampered','incomplete','ambiguous'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                checker,path,base,folder,selection,protocol,run=selection_fixture(Path(tmp))
                if mutation=='tampered':(folder/'eval_manifest.json').write_text('{}')
                elif mutation=='incomplete':selection['source_sha256'].pop(folder.name+'/summary.json')
                else:
                    selection['source_sha256'].update({'development/'+k:v for k,v in list(selection['source_sha256'].items())})
                audit.selection_evidence_checks(checker,path,selection,protocol,run)
                self.assertTrue(any(row['status']=='failed' for row in checker.checks))

    def test_v11_recipe_hashes_cannot_be_swapped_or_inferred_from_current_bytes(self):
        for mutation in ('swap','missing_category','tampered_category'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                checker,path,base,folder,selection,protocol,run=selection_fixture(Path(tmp))
                metadata=selection['arms'][folder.name]['model_identity']['metadata']
                if mutation=='swap':run['selected_ptq_recipe_sha256']=metadata['category_ptq_recipe.json']
                elif mutation=='missing_category':del metadata['category_ptq_recipe.json']
                else:(base/'category_ptq_recipe.json').write_text('{}')
                audit.selected_recipe_checks(checker,base,run,selection,protocol)
                self.assertTrue(any(row['status']!='matched' for row in checker.checks))

    def test_legacy_category_protocol_preserves_explicit_old_hash_semantics(self):
        for version,name in ((4,'category-fp4-stress-v4'),(5,'category-fp4-exploratory-v5')):
            with self.subTest(version=version),tempfile.TemporaryDirectory() as tmp:
                checker,path,base,folder,selection,protocol,run=selection_fixture(Path(tmp),'development/')
                protocol={'version':version,'id':name}
                run['selected_ptq_recipe_sha256']=checker.digest(base/'category_ptq_recipe.json')['sha256']
                selection.pop('source_sha256');selection['arms'][folder.name].pop('model_identity')
                audit.selected_recipe_checks(checker,base,run,selection,protocol)
                self.assertTrue(all(row['status']=='matched' for row in checker.checks))
                audit.selection_evidence_checks(checker,path,selection,protocol,run)
                self.assertFalse(any(row['status']=='failed' for row in checker.checks))
                self.assertTrue(any(row['status']=='unverifiable' for row in checker.checks))
                self.assertTrue(any(row['id'].startswith('selection.raw_protocol.') and row['status']=='matched'
                                    for row in checker.checks))

    def test_equal_and_missing_have_distinct_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "evidence.json"
            path.write_text("{}\n")
            checker = audit.Audit(root)
            ident = checker.digest(path)
            self.assertTrue(checker.file(path, ident, "present"))
            self.assertFalse(checker.file(root / "absent", ident, "absent"))
            self.assertEqual(checker.checks[-1]["status"], "unverifiable")

    def test_hash_contradiction_is_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "evidence.json"
            path.write_text("{}\n")
            checker = audit.Audit(root)
            self.assertFalse(checker.file(path, {"sha256": "0" * 64}, "changed"))
            self.assertEqual(checker.checks[-1]["status"], "failed")

    def test_protocol_prose_mismatch_is_failed(self):
        checker = audit.Audit(ROOT, payload=False)
        protocol = {
            "partitions": {
                "development": {"init_state_indices": [4]},
                "collection": {"init_state_indices": [5]},
                "heldout": {"init_state_indices": [6]},
            },
            "selection": {
                "pressure_rule": {"min_drop_from_bf16": 0.05},
                "rule": "at least 0.20 below BF16",
                "train_seed": 1, "qad_optimizer_steps": 2,
                "continuation_optimizer_steps": 1, "recovery_scope": "head",
                "rank": 1, "alpha": 2, "effective_demo_batch": 1,
                "opd_every": 1, "qad_learning_rates": [1e-4], "opd_weights": [1.0],
            },
        }
        run = {"train_seed": 1, "qad_steps": 2, "continuation_steps": 1,
               "lora_scope": "head", "rank": 1, "alpha": 2,
               "effective_demo_batch": 1, "opd_every": 1,
               "qad_learning_rates": [1e-4], "opd_weights": [1.0]}
        audit.protocol_checks(checker, protocol, run)
        pressure = next(x for x in checker.checks if x["id"] == "protocol.pressure_text")
        self.assertEqual(pressure["status"], "failed")

    def test_all_nvfp4_recipe_passes_w4a4_format_check(self):
        checker = audit.Audit(ROOT, payload=False)
        audit.format_check(checker, {"memory": {"fraction_of_eligible_params": {"nvfp4": 1.0, "fp8": 0.0}}})
        row = next(x for x in checker.checks if x["id"] == "quantization.fp8_coverage")
        self.assertEqual(row["status"], "matched")

    def test_missing_format_declaration_is_unverifiable(self):
        checker = audit.Audit(ROOT, payload=False)
        audit.format_check(checker, {"memory": {}})
        row = next(x for x in checker.checks if x["id"] == "quantization.fp8_coverage")
        self.assertEqual(row["status"], "unverifiable")

    def test_audit_run_does_not_create_or_mutate_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "run_manifest.json").write_text(json.dumps({"status": "initialized"}) + "\n")
            before = sorted(p.relative_to(root) for p in root.rglob("*"))
            report = audit.audit_run(run, root, payload=False)
            after = sorted(p.relative_to(root) for p in root.rglob("*"))
            self.assertEqual(before, after)
            self.assertEqual(report["status"], "unverifiable")
            self.assertTrue(report["read_only"])


if __name__ == "__main__":
    unittest.main()
