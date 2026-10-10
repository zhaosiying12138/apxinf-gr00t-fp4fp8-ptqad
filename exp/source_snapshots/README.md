# Initialization source snapshots

These files preserve exact Git blob bytes for evidence verification. They are
not execution entry points. The directory name is the file's SHA-256; the cost
collector verifies that digest against the run's recorded `implementation_sha256`.
The same relative path is retained under `source/` in portable evidence archives.

- `9d5888bf96dc81191142f9ae1a896047176a5791c5d9c48aa932186cd084f9e7/run_high_fp4_v3.py`
  is the unchanged `exp/run_high_fp4_v3.py` blob from Git commit `c44f65a`.

An initialization snapshot does not prove which source a later resumed process
imported. Invocation receipts retain that separate provenance.

## Evaluation readiness amendment

- `9615ab64fd4a9cba46defe31ab1d0d58a4dd344a9d1dac1cc40f03497af9f00a/run_recovery_eval.py`
  preserves the exact pre-amendment evaluation runner from commit
  `051165de0583f48bf91aa5000ffeed17168bfa7b` (28,351 bytes). It is a historical
  source snapshot, not an execution entry point or retrospective execution
  attestation. See `docs/server_readiness_timeout_2026-10-10.md` for the preserved
  startup failure and final publication identity requirements.
