"""CPU-only Git fixtures for the scoped demo LFS verifier; no LFS/network call."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'setup/06_install_recovery.sh'
BLOCK=SCRIPT.read_text().split('# BEGIN DEMO_LFS_VERIFY\n',1)[1].split('# END DEMO_LFS_VERIFY',1)[0]
CODE=BLOCK.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]


class DemoLfsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.repo=Path(self.temp.name);self.prefix='demo_data/libero_demo/'
        self.git('init','-q')
        self.payloads={self.prefix+'data/chunk-000/episode_000000.parquet':b'PAR1 TEST DATA PAR1',
                       self.prefix+'videos/chunk-000/image/episode_000000.mp4':b'FAKE VIDEO FIXTURE'}
        for name,data in self.payloads.items():
            p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('version https://git-lfs.github.com/spec/v1\n'
                         f'oid sha256:{hashlib.sha256(data).hexdigest()}\nsize {len(data)}\n')
        path=self.repo/self.prefix/'meta/info.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'total_episodes':1,'total_videos':1}))
        self.git('add','.');self.git('-c','user.name=CPU Fixture','-c','user.email=fixture@example.invalid',
                                    '-c','core.hooksPath=/dev/null','commit','-qm','temporary fixture')
        for name,data in self.payloads.items():(self.repo/name).write_bytes(data)
    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.repo),*args],stderr=subprocess.STDOUT)
    def run_verifier(self):
        return subprocess.run([sys.executable,'-I','-S','-c',CODE,str(self.repo),'HEAD'],capture_output=True,text=True)
    def test_hydrated_exact_payloads_pass(self):
        result=self.run_verifier();self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('2 payloads',result.stdout)
    def test_pointer_is_not_data(self):
        name=next(iter(self.payloads));(self.repo/name).write_bytes(self.git('show','HEAD:'+name))
        result=self.run_verifier();self.assertNotEqual(result.returncode,0)
        self.assertIn('Unresolved Git LFS pointer',result.stderr)
    def test_corruption_rejected_even_at_same_size(self):
        name=next(iter(self.payloads));data=self.payloads[name];(self.repo/name).write_bytes(b'X'+data[1:])
        result=self.run_verifier();self.assertNotEqual(result.returncode,0)
        self.assertIn('size/SHA',result.stderr)
    def test_missing_payload_rejected(self):
        (self.repo/next(iter(self.payloads))).unlink()
        result=self.run_verifier();self.assertNotEqual(result.returncode,0)
        self.assertIn('Missing regular demo payload',result.stderr)
    def test_setup_network_scope_is_only_the_demo(self):
        text=SCRIPT.read_text()
        pulls=[line for line in text.splitlines() if ' lfs pull ' in line]
        self.assertEqual(pulls,["git -C \"$GROOT\" lfs pull --include='demo_data/libero_demo/**' --exclude='' ".rstrip()])
        self.assertIn('command -v git-lfs',text)
        self.assertIn('GIT_LFS_SKIP_SMUDGE=1 git clone',text)
        self.assertIn('GIT_LFS_SKIP_SMUDGE=1 git -C "$GROOT" checkout',text)
        self.assertNotIn('lfs fetch --all',text)

if __name__=='__main__':unittest.main(verbosity=2)
