# 原生 FP4 Graph 路径的 CPU 构建收据

此目录只证明源码可编译、纯 CPU arena 大小测试通过、独立算子 CPU
self-test 通过，以及维护补丁能在指定 APXInf 提交上应用并反向核对。
**这些收据不证明 GPU 数值、CUDA Graph 捕获/重放、闭环成功率或性能通过。**

- `build_manifest.json`：源码、wheel、原生二进制和日志的 SHA256。
- `native_fp4_cpu_build.log`：一个纯 CPU 单元测试通过；GPU 测试仅编译。
- `native_fp4_model_gate_build.log`：最小完整模型 RequireGraph 入口编译成功。
- `native_fp4_wheel_build.log`：CPython 3.12 原生扩展打包成功。
- `native_fp4_opbench_build.log`：共用 CUDA adapter 的 opbench 重编及 CPU self-test。
- `native_patch_check.json`：针对清洁基线的 apply 检查与当前源码的 reverse 检查。
- `evidence_manifest.json`：本目录已归档文件的字节数与 SHA256。

收据生成时未安装新 wheel、未执行 GPU 测试。后续应在独占 GPU 时运行
`PTQAD_GPU_EXCLUSIVE=1 bash exp/run_native_graph_gates.sh <新结果目录>`。
该入口逐一执行数值、脏 scale padding、Graph 重放和完整模型 RequireGraph
验收；仅在全部通过后安装 wheel。之后还须重新运行受影响的原生模型与
算子基准，记录实际 `execution_mode`，再形成性能结论。
