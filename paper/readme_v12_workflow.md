## 从编译到执行

以下命令均来自仓库实际脚本；大模型、校准缓存、训练 checkpoint 和教师张量不进入 Git。先定义新运行目录，避免把旧实验混入证据：

```bash
export PROJECT="$PWD"
source setup/recovery-env.sh
export PY="$PTQAD_PYTHON" GR00T="$GR00T_REPO" LIBERO_PY="$LIBERO_PYTHON"
export BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export DATASET="$GR00T/demo_data/libero_demo"
export RUN="$PROJECT/results/reruns/rtn_w4a4_release_$(date -u +%Y%m%dT%H%M%SZ)"
export PROTOCOL_SRC="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"
export PROTOCOL="$PROTOCOL_SRC"
export RECOVERY="$RUN/recovery_v12"
export PTQAD_ZMQ_TIMEOUT_MS=120000
mkdir -p "$RUN"
```

### 0. CPU 契约和源码静态检查

```bash
"$PY" paper/run_cpu_checks.py --out "$PROJECT/paper/_build/cpu-checks-$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" -m py_compile quant/ptq/bake.py quant/ptq/bake_category.py \
  eval/run_recovery_eval.py exp/run_w4a4_recovery.py exp/make_w4a4_selection.py
```

### 1. W4A4 RTN PTQ 与 CategorySpecificLinear

`quant/ptq/bake.py` 的 `rtn` 配方不读取 Hessian，直接生成全 NVFP4 普通层；`bake_category.py --method rtn_all` 补齐七个 category bank。

```bash
export PURE_RTN="$RUN/pure_rtn" CATEGORY_OUT="$RUN/rtn_category"
"$PY" quant/ptq/bake.py --base "$BASE" --out "$PURE_RTN" --recipe rtn --calibration-mode none
"$PY" quant/ptq/bake_category.py --parent "$PURE_RTN" \
  --out "$CATEGORY_OUT" --method rtn_all --rtn-clip 1.0
```

### 2. development、教师轨迹和选择文件

冻结协议只允许 development 选择压力臂；教师轨迹和学生 collection 只用于训练，held-out 不得参与选择。

```bash
export DEV="$RUN/development" TEACHER="$RUN/teacher_supervision_v12_clean" COLLECTION="$RUN/collection" SELECTION="$RUN/selection_final"
mkdir -p "$DEV" "$COLLECTION" "$SELECTION"
# A new bake lives under RUN, so standalone reproduction uses a local protocol
# copy whose pressure checkpoint path and SHA are bound to this run. The
# checked-in PROTOCOL_SRC remains immutable and is the only input accepted by
# the formal publication writer.
cp "$PROTOCOL_SRC" "$RUN/recovery_protocol_v12_rtn_w4a4.local.json"
export PROTOCOL="$RUN/recovery_protocol_v12_rtn_w4a4.local.json"
"$PY" - "$PROTOCOL" "$CATEGORY_OUT" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["selection"]["pressure_candidate_checkpoints"]["rtn_w4a4_category"] = str(pathlib.Path(sys.argv[2]).resolve())
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
print("standalone protocol sha256:", hashlib.sha256(p.read_bytes()).hexdigest())
PY
FP4VLA_QUANT=0 FP4VLA_W4A4=0 "$PY" eval/run_recovery_eval.py --checkpoint "$BASE" --out "$DEV/bf16" --purpose development --seed 940000 --episodes 5 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6920 --protocol-file "$PROTOCOL"
FP4VLA_QUANT=0 FP4VLA_W4A4=1 "$PY" eval/run_recovery_eval.py --checkpoint "$CATEGORY_OUT" --out "$DEV/rtn_w4a4_category" --purpose development --seed 940000 --episodes 5 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6921 --protocol-file "$PROTOCOL"
python3 exp/make_w4a4_selection.py --protocol-file "$PROTOCOL" --bf16 "$DEV/bf16" --candidate rtn_w4a4_category --candidate-output "$DEV/rtn_w4a4_category" --out "$SELECTION"
FP4VLA_QUANT=0 FP4VLA_W4A4=0 "$PY" eval/run_recovery_eval.py --checkpoint "$BASE" --out "$TEACHER" --purpose teacher_supervision --seed 950000 --episodes 4 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6922 --protocol-file "$PROTOCOL"
"$PY" exp/verify_teacher_replay.py --root "$TEACHER" --protocol-file "$PROTOCOL" --teacher "$BASE"
```

### 3. QAD、continued-QAD、OPD 与五臂 held-out

本轮教师数据包含 37 条成功轨迹的 148 个窗口；每 4 次策略调用采一窗、每回合最多四窗，覆盖偏向回合前段。训练按固定路径顺序读取，micro batch=1、最多累积 16 个微批，每轮保留四窗尾批；完成 2,000 次更新时，按样本数、完成轮数及加载规则推导读取 29,600 窗。该数是含尾批的读取预算，并非独立轨迹数或逐样本计数。学生状态采集也采用相同上限，并保留失败回合的状态。

恢复驱动会依次执行两个 QAD 学习率、QAD 选择、学生状态 collection、教师缓存、continued-QAD、两个 OPD 权重和最终五臂评测。完整且身份匹配的阶段可复用；半途失败的评测必须保留故障记录并改用新输出目录。当前训练 checkpoint 不含优化器与调度器状态，不能把重启训练称为无损续跑；详见 [故障恢复规则](docs/reproduce-ptqad.md)。

```bash
export PTQAD_ZMQ_TIMEOUT_MS=120000
"$PY" exp/run_w4a4_recovery.py --run-dir "$RECOVERY" --protocol-file "$PROTOCOL" \
  --ptq-selection "$SELECTION/selection.json" --base "$BASE" --gr00t-repo "$GR00T" \
  --python "$PY" --rollout-python "$LIBERO_PY" --dataset "$DATASET" \
  --capture-dataset "$TEACHER" --port-base 6920 --validate-only
"$PY" exp/run_w4a4_recovery.py --run-dir "$RECOVERY" --protocol-file "$PROTOCOL" \
  --ptq-selection "$SELECTION/selection.json" --base "$BASE" --gr00t-repo "$GR00T" \
  --python "$PY" --rollout-python "$LIBERO_PY" --dataset "$DATASET" \
  --capture-dataset "$TEACHER" --port-base 6920 --until all --allow-opd-nonimprovement
```

运行期间，可在已设置同一组路径变量的另一个终端查看记录进度。该命令只读日志与阶段收据；未完成记录本身不能证明进程仍在运行，日志步数也不是最终成功率。

```bash
python3 exp/high_fp4_status.py --development-root "$SELECTION" --recovery-root "$RECOVERY"
```

完整动作块的固定观测诊断入口见 [动作诊断说明](docs/ACTION_CHUNK_DIAGNOSTICS.md)。它复用正式加载器，从最终清单解析五臂，验证实际初始噪声后比较有效动作区域；训练分区观测只用于拟合分析，不作为独立泛化证据。诊断须在训练和正式评测结束、GPU 空闲后串行执行，验证范围与运行状态以该文档为准。

### 4. 证据、图表、HTML 和知乎稿

只有冻结 v12 发布运行的 `final_manifest.json` complete 且五臂各 160 回合时才材料化；命令不会把超时或半成品写进正文。上一步独立复现使用了 `$RUN/recovery_protocol_v12_rtn_w4a4.local.json`，其 SHA 与仓库冻结协议不同；它可以用于验证方法，但不能直接喂给正式发布工具。正式发布必须切换到本次冻结 v12 运行目录，并令 `PROTOCOL="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"`。

```bash
export RECOVERY="$PROJECT/results/reruns/rtn_w4a4_release_20261006_01/recovery_v12"
export PROTOCOL="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"
export CATEGORY_OUT="$("$PY" - "$RECOVERY/final_manifest.json" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(pathlib.Path(m["selected_pressure_checkpoint"]).resolve())
PY
)"
python3 paper/extract_final_evidence.py --run-dir "$RECOVERY" --out /tmp/v12_final_results.json
export OPD_MERGE="$("$PY" - "$RECOVERY/final_manifest.json" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
p = m.get("selected_opd_model_identity", {}).get("path")
if not p:
    raise SystemExit("selected_opd_model_identity.path is missing")
print(pathlib.Path(p).resolve())
PY
)"
python3 paper/build_recipe_inventory_v12.py --checkpoint "$CATEGORY_OUT" --recovery-manifest "$OPD_MERGE/recovery_manifest.json" --recipe-name rtn_all --out /tmp/v12_recipe_inventory.json
python3 paper/materialize_final_evidence.py --final-manifest "$RECOVERY/final_manifest.json" --recipe-inventory /tmp/v12_recipe_inventory.json --out paper/_build/final_bundle_v12 --orchestrator-run "$RECOVERY"
python3 paper/install_final_evidence.py --bundle paper/_build/final_bundle_v12 --archive paper/_build/evidence_archive_v12_previous
python3 paper/install_final_evidence.py --verify paper/evidence
python3 paper/collect_search_costs.py --run-dir "$RECOVERY" --out paper/evidence/search_costs
python3 paper/collect_search_costs.py --verify paper/evidence/search_costs
python3 paper/build_final_frontier.py --final-results paper/evidence/final_results.json --paired-comparison paper/evidence/paired_comparison.json --inventory paper/evidence/recipe_inventory.json --out paper/evidence/frontier_comparison.json
python3 paper/install_final_evidence.py --register-supplements paper/evidence --supplement-root paper/evidence
python3 paper/install_final_evidence.py --verify paper/evidence
```

安装器只迁移本轮科学证据和截图登记；上面的命令重新归档完整搜索成本，并在 runtime、search_costs、action_diagnostics、gptq_reference 和 frontier_comparison 生成后原子登记这些补充证据。随后从最终五臂的评测清单读取 checkpoint，记录环境、软件包和源码哈希：

```bash
"$PY" - "$RECOVERY" "$GR00T" "$PY" "$LIBERO_PY" <<'PY'
import json, pathlib, subprocess, sys
run, groot, server_python, rollout_python = sys.argv[1:]
final = json.loads((pathlib.Path(run) / "final_manifest.json").read_text())
round_dir = pathlib.Path(final["heldout_round"])
command = [sys.executable, "paper/capture_runtime.py", "--run-dir", run,
           "--out", "paper/evidence/runtime", "--gr00t", groot,
           "--server-python", server_python, "--rollout-python", rollout_python]
for arm in ("bf16", "ptq", "qad", "continued_qad", "qad_opd"):
    record = json.loads((round_dir / f"heldout_{arm}" / "eval_manifest.json").read_text())
    command += ["--checkpoint", f"{arm}={record['checkpoint']}"]
subprocess.run(command, check=True)
PY
```

<!-- include: readme_v12_supplement_commands.md -->

### 5. 截图登记

最终五臂完成后先生成本轮受控脚本，再用仓库内的 Windows/WSL 采集工具逐张运行；训练和评测进程必须退出，GPU 必须空闲。

```bash
CAPTURE_ROOT="$(dirname "$RECOVERY")/captures/w4a4_final"
python3 paper/prepare_w4a4_captures.py --final-manifest "$RECOVERY/final_manifest.json" --out "$CAPTURE_ROOT"
```

<!-- include: readme_v12_capture_commands.md -->

截图登记完毕后，再从同一套证据生成文稿与发布包。先前的输出目录或临时文件若已存在，工具会拒绝覆盖；复跑时使用新的 staging 名称并保留来源记录。

```bash
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/update_v12_release.py
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/write_readme_v12.py
sudo apt-get install -y libcairo2 fontconfig fonts-noto-cjk
node --version  # Node.js >=18; install from https://nodejs.org/ if absent.
npm install --prefix paper/_build/renderer --save-exact playwright@1.58.2
node paper/_build/renderer/node_modules/playwright/cli.js install --with-deps chromium
python3 paper/make_figs.py && bash paper/figs/render_pngs.sh
uv run --with-requirements paper/requirements-build.txt python paper/build_html.py
python3 paper/export_zhihu.py
node paper/qa_browser.cjs
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/validate_publication.py
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/package_publication.py --check
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/package_publication.py
```

<!-- include: readme_v12_engine_commands.md -->
