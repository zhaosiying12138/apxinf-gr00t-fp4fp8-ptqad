"""Small serialized-cache fixtures; all tensors are CPU tensors."""
import os
from pathlib import Path
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"]=""

from exp.verify_teacher_cache_cpu import verify_payload


class TeacherCachePayloadFixture(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        self.temp=tempfile.TemporaryDirectory(prefix="fp4vla-cache-cpu-")
        self.path=Path(self.temp.name)
        self.model=self.path/"qad"
        self.source=self.path/"sample_000000.pt"
        provenance={"source_kind":"student_rollout","student_checkpoint":str(self.model),
                    "student_statistics_sha256":"a"*64,"task_text":"fixture","capture_index":0}
        torch.save({**provenance,"inputs":{}},self.source)
        self.metadata={"source_observation_files":[{"path":str(self.source)}]}
        self.payload={"version":3,"metadata":self.metadata,"samples":[{
            "seed":100,"provenance":provenance,
            "inputs":{"action":torch.ones(1,2,3),"action_mask":torch.ones(1,2,3)},
            "pred":torch.ones(1,2,3),
        }]}

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        return verify_payload(self.payload,self.metadata,self.model,100,1)

    def test_valid_cpu_payload(self):
        self.assertEqual(self.verify()["samples_verified"],1)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_short_payload_rejected(self):
        self.payload["samples"]=[]
        with self.assertRaisesRegex(ValueError,"sample count"):
            self.verify()

    def test_sidecar_disagreement_rejected(self):
        self.payload["metadata"]={"changed":True}
        with self.assertRaisesRegex(ValueError,"metadata disagrees"):
            self.verify()

    def test_wrong_replay_seed_rejected(self):
        self.payload["samples"][0]["seed"]=101
        with self.assertRaisesRegex(ValueError,"seed differs"):
            self.verify()

    def test_wrong_student_rejected(self):
        self.payload["samples"][0]["provenance"]["student_checkpoint"]="other"
        with self.assertRaisesRegex(ValueError,"not from winning QAD"):
            self.verify()

    def test_nonfinite_teacher_prediction_rejected(self):
        self.payload["samples"][0]["pred"][0,0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"invalid velocity"):
            self.verify()

    def test_nonbinary_mask_rejected(self):
        self.payload["samples"][0]["inputs"]["action_mask"][0,0,0]=.5
        with self.assertRaisesRegex(ValueError,"invalid velocity"):
            self.verify()

    def test_source_provenance_mismatch_rejected(self):
        self.payload["samples"][0]["provenance"]["capture_index"]=7
        with self.assertRaisesRegex(ValueError,"source provenance"):
            self.verify()


if __name__=="__main__":
    unittest.main()
