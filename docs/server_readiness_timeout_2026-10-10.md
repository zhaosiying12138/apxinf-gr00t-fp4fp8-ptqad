# Evaluation server readiness amendment, 2026-10-10

This is an infrastructure-only amendment for the already unfinished v12
continued-QAD development evaluation. Training, new quantization, and
result-dependent reruns remain frozen. It does not authorize another experiment.

## Preserved failure

Run root: `results/reruns/rtn_w4a4_release_20261006_01/recovery_v12`.

- Failed output: `artifacts/failed_dev_continued_qad_startup_timeout_20261009`.
- Failed stage log: `logs/failed_dev_continued_qad_startup_timeout_20261009.log`.
- The log records `TimeoutError: Server did not become ready` and return code 1.
  The output contains only `eval_manifest.json` and the first task's server log;
  there is no rollout log, task result, or summary. Zero episodes were scored.
- A later resume attempt (PID 3785847) stopped at the existing-log guard, also
  without scoring episodes, as recorded by the controlling task.

The controlling task resumed the existing driver as PID 3858752 on 2026-10-10
at 03:17 Asia/Shanghai, with log `/tmp/fp4vla_v12_resume_20261010.log`.
This note records launch context, not continuing process status or completion.
No producer source is to be edited while that evaluation is active.

## Exact change and CPU validation

`eval/run_recovery_eval.py` accepts `PTQAD_SERVER_READY_TIMEOUT_S`, with its
previous default of 180 retained. The authorized unfinished-evaluation launch
uses 600. Noninteger, zero, and negative values fail argument validation before
checkpoint validation, output creation, or server launch. New manifests record
`server_ready_timeout_s`; preserved older manifests are not rewritten.

The value bounds readiness attempts, each with a one-second connection timeout
and a one-second sleep after failure. It is not a strict elapsed-time deadline.
`PTQAD_ZMQ_TIMEOUT_MS=120000` remains unchanged; the rollout subprocess timeout,
checkpoint, numerical switches, seeds, bank indices, and scoring are unchanged.

Six CPU-only tests in `tests/test_recovery_server_ready_timeout.py` passed
(0.016 seconds). They cover the default, positive overrides including 1, invalid
values, successful readiness with the separate rollout timeout preserved, and
server exit. Every socket, subprocess, and sleep is mocked; these tests launch
no inference or simulation and create only temporary synthetic manifests.

## Source identity and final publication

The exact pre-amendment Git blob from commit
`051165de0583f48bf91aa5000ffeed17168bfa7b` is preserved, byte for byte, at:

`exp/source_snapshots/9615ab64fd4a9cba46defe31ab1d0d58a4dd344a9d1dac1cc40f03497af9f00a/run_recovery_eval.py`

The snapshot has 28,351 bytes. Its directory name is its verified SHA-256.
The amended runner SHA-256 at review is
`c1677e4ec6745551d88925247e61fd647112305e97a353293f875b4e1eb8a659`.
Only `main` changes; AST comparison confirms `parse_log` and `validate_resets`
are identical. A historical Git snapshot does not independently attest which
bytes an earlier running process imported.

Publication nevertheless hashes the entire runner: raw-log materialization
records it as the parser, and `validate_publication.py` checks that full-file
identity and the runtime source inventory. Generate final collector/runtime
receipts against the final published source bytes after the authorized run is
complete. Include the current runner plus this prior-source snapshot in the
source bundle; `paper/capture_runtime.py --source-file` can explicitly add the
snapshot to the runtime inventory. Preserve earlier receipts and failed-run
files unchanged. The current synthetic test fixtures and archived v11 paper
artifacts are not the final v12 release evidence.
