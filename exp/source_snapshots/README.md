# Initialization source snapshots

These files preserve exact Git blob bytes for evidence verification. They are
not execution entry points. The directory name is the file's SHA-256; the cost
collector verifies that digest against the run's recorded `implementation_sha256`.
The same relative path is retained under `source/` in portable evidence archives.

- `9d5888bf96dc81191142f9ae1a896047176a5791c5d9c48aa932186cd084f9e7/run_high_fp4_v3.py`
  is the unchanged `exp/run_high_fp4_v3.py` blob from Git commit `c44f65a`.

An initialization snapshot does not prove which source a later resumed process
imported. Invocation receipts retain that separate provenance.
