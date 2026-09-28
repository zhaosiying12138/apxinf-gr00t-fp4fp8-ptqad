# 附录 A. 核心源码走读与 APXInf 框架解析

本附录面向希望复现或扩展本工作的读者，分两部分：A.1 解释 APXInf 引擎的代码框架——我们的全部引擎侧工作都在它的既有模式内完成；A.2–A.5 逐段走读我们实现的核心源码（引擎侧 CUDA/Rust 与 PyTorch 侧校准/恢复/评测栈），每段代码都说明"为什么这样写"。全部代码位于 fp4vla 仓库（`quant/ptq/`、`rl/`）与引擎补丁（`patches/apxinf-fp4vla-engine.patch`，1387 行，可对引擎主干一键应用）。

# A.1 APXInf 引擎代码框架解析

## A.1.1 三层 crate 拓扑

APXInf 是 Rust 编写的推理引擎，其 crate 分层决定了"加一种新精度/新模型"的全部工作量边界：

- **apxinf-core**：与设备无关的类型层——`Tensor`（dtype+shape+storage）、`DType`（F16/BF16/…）、`Shape`、统一错误 `Error`。任何算子签名都以 core 类型书写，因此上层模型代码不感知 CUDA 细节。
- **apxinf-cuda**：设备层——`CudaContext`（stream 句柄）、`CudaBuffer`（显存分配，`copy_to_host`/`ptr()` 回读与借用）、`kernels/`（Rust 算子入口）与 `ffi/`（对 C 接口的声明绑定）。**关键约束：`ffi` 模块是私有的**（`lib.rs` 里 `mod ffi;`），上层 crate 不能直接 import——所以我们的跨 crate 小工具（如 bf16↔f16 cast）以 `pub fn xxx_ffi(...)` 薄包装的形式放在 `kernels/fp4.rs` 里转发，这是引擎内跨 crate 暴露能力的既定模式。
- **apxinf-model**：模型层——每个支持的组织（π0.5、GR00T N1.7、Qwen3-VL、WALL-OSS…）一个模块，内含 weights 加载、blocks（执行拓扑）、runtime/executor（推理循环）。

CUDA 调用不走 Rust FFI 直连，而是经过 **适配器层**：`adapters/*.cu` 中的 `extern "C"` C++ 函数由 nvcc 编译，经 `build.rs` 的 `kernel_files` 显式列表链接进 crate，Rust 侧在 `ffi/cublaslt.rs` 声明签名。新增一个 GEMM 算子的完整集成链条是：写 adapter `.cu` → `build.rs` 注册 → Rust `ffi` 声明 → `kernels/` 包装 → 模型层调用。我们的 fp4 路径严格按此链条落地。

## A.1.2 精度 monomorphization：一种精度 = 一套 Blocks

引擎对精度的组织方式是**编译期单态化**而非运行期分派：以 π0.5 为例，`model/blocks/` 下每种精度一个完整实现（`bf16.rs` 621 行 / `fp8_static.rs` 1014 行 / `int8_dynamic.rs` 575 行），`ModelVariant` 枚举在 `model.rs` 的 `infer()/preprocess_rgb()` 等 match 点静态分派：

```rust
pub(in crate::pi05) enum ModelVariant { Fp8Static(Pi05Model<Fp8StaticBlocks>), Bf16(...), ... }
```

这套模式直接决定了我们的 fp4 实现形态：新增 `Nvfp4Static` 变体需要覆盖 `ModelVariant` 的**全部** match 臂（漏一处是 E0004 非穷尽错误），并且要么克隆一份完整 Blocks（fp8 的做法，千行级），要么做**组合式**设计。我们选择了后者（见 A.2.4）——这既是对引擎模式的遵循，也是对它的一次改进实验。

## A.1.3 工件与校准约定

引擎的量化模型是**离线工件 + 在线执行**：权重以量化格式离线转换成工件目录，加载器只上传字节、从不重量化。π0.5 的 FP8 profile 约定放在 `<model-dir>/calibration.json`（`load.rs` 的 `options.calibration_path` 或模型根自动识别，fp8_weights.rs 甚至带校准 JSON 的身份校验）。我们的 NVFP4 工件沿用并显式化这一约定：

```
<dir>/<stem>.packed.u8   # rows x K/2 字节，E2M1 对（偶数 k 在低 nibble）
<dir>/<stem>.scale.u8    # cuBLASLt VEC16_UE4M3 物理 swizzle 布局
<dir>/manifest.json      # 每张量 {shape, block=16, tscale, rel_frob}
```

**部分工件是合法状态**：豁免表（哪些张量留在 BF16）由工件的存在性表达——加载器查不到就回退 BF16。这个"存在即量化"的约定让混合精度部署不需要改任何引擎代码。

## A.1.4 Python 侧：robo 层与 serving 拓扑

APXInf-robo（RLinf/APXinf-robo）是模型之上的机器人层：`Gr00tPolicy` 家族用 `precision=/calibration=/embodiment=` 构造（π0.5 走 `AutoPolicy`+`model_variant=`），LIBERO 评测与 OpenPI 兼容 server 都在这层。本文的闭环评测栈与该拓扑同构：eval server（策略常驻、批处理观测）+ ZMQ rollout 客户端（MuJoCo/EGL 在独立 venv）——**评测即部署形态**，量化写盘产出的 PTQ/LoRA 合并 checkpoint 换入模型路径即可，零代码改动。引擎侧需在同一 venv 安装 NVIDIA gr00t 包（`--no-deps`）+ transformers 4.57.3。

# A.2 引擎内的 NVFP4 执行路径：五步源码走读

## A.2.1 CUDA 适配器：block-scaled GEMM（cublaslt_fp4_adapter.cu）

适配器是整条 fp4 路径里唯一"跟硬件对话"的文件。核心是三件事：GEMM 描述符的 scale 模式设置、CUDA 13 列主序布局适配、以及自持句柄：

```c
static cublasLtHandle_t fp4_lt_handle() {
    static cublasLtHandle_t handle = nullptr;
    static std::once_flag once;
    std::call_once(once, [] { cublasLtCreate(&handle); });
    return handle;
}

extern "C" cublasStatus_t apxinf_fp4_gemm_f16(
    int m, int n, int k,
    const void* a_packed, const void* b_packed,     // E2M1 对，各 M*K/2 与 N*K/2 字节
    const void* a_scale_swz, const void* b_scale_swz, // 物理 swizzle 后的 E4M3 块缩放
    __half* c, void* workspace, size_t workspace_bytes, cudaStream_t stream)
{
    cublasLtMatmulDesc_t op; ...;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof(nn));

    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_swz, ...);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_swz, ...);

    // CUDA 13：只有列主序布局；A 存 KxM (ld=k, opT 得 MxK)，B 存 KxN，C 为 MxN F16
    cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, k, m, k);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, k, n, k);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F,     m, n, m);
    ...
    cublasLtMatmulAlgoGetHeuristic(fp4_lt_handle(), op, la, lb, lc, lc, pref, 1, &heur, &nres);
    cublasLtMatmul(fp4_lt_handle(), op, &one, a_packed, la, b_packed, lb, &zero,
                   c, lc, c, lc, &heur.algo, workspace, workspace_bytes, stream);
}
```

写这段代码时踩过的坑值得记录：(i) CUDA 13 移除了行主序布局常量，全部按列主序重新推导存储方向——权重 `[N,K]` 行主序与 `[K,N]` 列主序 ld=K 是同一段字节，这是让 HF 转置习惯与 cuBLASLt 共存的关键；(ii) scale 指针与 scale 模式是描述符属性而非布局属性，设错位置会静默落到逐 tensor 缩放；(iii) `CUBLAS_COMPUTE_32F + CUDA_R_32F` 的组合是该 scale 模式下唯一可用的高精度路径。

**物理 swizzle 布局**——cuBLASLt 要求块缩放字节按固定物理公式排布（§3.1 的探针测定过程），适配器与离线转换器共用同一公式：

```
PS = 512 * ceil(KB/4)
off(r, b) = PS*(r/128) + 512*(b/4) + 16*(r%32) + 4*((r/32)%4) + (b%4)
```

## A.2.2 在线激活量化核：与 CPU 参考逐位一致

适配器里的第二个导出函数把 F16 激活在线量化成 NVFP4（packed + swizzled scale）。这段代码的目标不是"快"而是**逐位确定**——它必须与离线转换器（`fp4_quant.py`）产出完全相同的字节，否则在线路径与离线工件的数值语义就分裂了：

```c
__device__ __forceinline__ uint8_t fp4_enc_e4m3(float v) {
    if (v >= 448.0f) return 0x7E;              // satfinite
    if (v <= 0.0f)   return 0;
    float m = v, e = 0;
    while (m >= 2.0f) { m *= 0.5f; e += 1.0f; } // 规格化到 [1,2)
    while (m <  1.0f) { m *= 2.0f; e -= 1.0f; }
    int exp = (int)e + 7;
    if (exp <= 0) { /* 次正规格上的 RNE（ties-to-even） */ ... }
    float gm = (m - 1.0f) * 8.0f; int fl = (int)gm; float frac = gm - fl;
    int man = frac > 0.5f ? fl + 1 : frac < 0.5f ? fl : (fl % 2 == 0 ? fl : fl + 1);
    if (man > 7) { man = 0; exp += 1; }
    return (uint8_t)((exp << 3) | man);
}

__global__ void nvfp4_quantize_activation_kernel(
    const __half* __restrict__ x, uint8_t* __restrict__ packed,
    uint8_t* __restrict__ scale_swz, int rows, int kb)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int r = tid / kb, b = tid - r * kb;        // 每线程负责一个 16 元素块
    float vals[16], amax = 0.0f;
    for (int j = 0; j < 16; ++j) { vals[j] = __half2float(blk[j]); amax = fmaxf(amax, fabsf(vals[j])); }
    uint8_t sb = fp4_enc_e4m3(amax * (1.0f / 6.0f));   // 块缩放：amax/6 编码为 E4M3
    float sd = fp4_dec_e4m3(sb);
    for (int j = 0; j < 16; j += 2) {
        float q0 = min(fp4_e2m1_mag(fabsf(vals[j]) / sd), 6.0f);  // 除法（非倒数乘）
        ...  // E2M1 阶梯码 + 符号位，两值装一字节（偶数 k 低 nibble）
    }
    size_t off = PS*(r/128) + 512ull*(b/4) + 16ull*(r%32) + 4ull*((r/32)%4) + (b%4);
    scale_swz[off] = sb;                       // 直接写入物理 swizzle 位置
}
```

两个刻意的设计决定：(i) **E4M3 编码用手写位级函数而非 `__nv_cvt_float_to_fp8`**——硬件内建函数的舍入模式与 CPU 参考不同，我们要的是 RNE-on-log-lattice 的精确复刻；(ii) **归一化除法而非乘倒数**——`x / sd` 与 `x * (1/sd)` 在 fp32 下给出不同舍入，注释里明确标注这是为了匹配 CPU 路径。E2M1 阶梯函数 `fp4_e2m1_mag` 用显式分支处理全部中点（0.25→0、0.75→1.0、1.75→2.0、3.5→4.0、5.0→4.0），保证 ties-to-even 的**码字**序而非数值序。这些细节共同保证了 kernel 与 Python 金标准逐字节一致（gold_check4 corr=1.000000）。

## A.2.3 Rust 算子与工件加载器

`kernels/fp4.rs` 的 `fp4_linear` 是模型层看到的唯一入口——借用工件缓冲的零拷贝视图、在线量化激活、调用 GEMM：

```rust
pub struct Fp4WeightView<'a> {
    pub packed: &'a CudaBuffer,   // E2M1 对
    pub scale:  &'a CudaBuffer,   // 物理 swizzle 字节
    pub rows: usize, pub k: usize,
}

pub fn fp4_linear(ctx: &CudaContext, x: &Tensor, w: &Fp4WeightView<'_>) -> Result<Tensor> {
    let (m, k) = (dims[0], dims[1]);
    let packed_bytes = CudaBuffer::alloc_on(ctx, m * k / 2)?;
    let kb = k / 16;
    // 物理布局的缓冲尺寸公式：512*ceil(kb/4) * ceil(m/128)
    let scale_bytes = 512 * ((kb + 3) / 4) * ((m + 127) / 128);
    let scale_buf = CudaBuffer::alloc_on(ctx, scale_bytes)?;
    unsafe { ffi::apxinf_nvfp4_quantize_activation(x_buf.ptr(), packed_bytes.ptr(),
                scale_buf.ptr(), m as i32, k as i32, stream) };
    unsafe { ffi::apxinf_fp4_gemm_f16(m, w.rows, k, packed_bytes.ptr(), w.packed.ptr(),
                scale_buf.ptr(), w.scale.ptr(), out.ptr(), ws.ptr(), ws.len(), stream) };
    Ok(out.into_tensor(Shape::new(vec![m, w.rows]), DType::F16))
}
```

`pi05/fp4_weights.rs` 加载器只做三件事：解析 manifest、按 stem 约定定位文件、`CudaBuffer` 上传——**注释里写明"工件不可变（QAD 冻结 scale），本加载器永不重量化"**，这是引擎"离线工件/在线执行"哲学的贯彻。它还带真实工件的设备往返回归测试（逐字节比对，环境变量门控），以及前述"部分工件合法"的豁免表语义。

## A.2.4 混合拓扑执行器：组合式 Fp4Blocks 与逐站路由

fp8_static.rs 的做法是克隆整个执行拓扑（千行级）。我们没有克隆，而是组合：`Fp4Blocks { inner: Bf16Blocks, fp4: Fp4TensorMap }`——视觉前处理、嵌入、KV cache 类型全部委托给既有 BF16 实现（`vision_layer_bf16` 等自由函数直接复用），只在 GEMM 站点插入路由。路由的心脏是这一个函数：

```rust
/// 混合拓扑的路由心脏：权重有工件张量则走 fp4 GEMM，否则 BF16 直通（豁免语义）。
pub fn gemm_maybe_fp4(
    ctx: &Context, x_bf16: &Tensor, weight_bf16: &Tensor,
    fp4: Option<&Fp4DeviceWeight>,
) -> Result<Tensor> {
    let Some(w4) = fp4 else {
        return kernels::gemm::bf16(ctx, x_bf16, weight_bf16);   // 豁免层：BF16 直通
    };
    let x_f16 = bf16_to_f16(ctx, x_bf16)?;   // fp4 站点付一次 dtype 转换
    let view = Fp4WeightView { packed: &w4.packed, scale: &w4.scale, rows: w4.rows, k: w4.k };
    kernels::fp4::fp4_linear(ctx, &x_f16, &view)  // 输出 F16（引擎 dtype 边界）
}
```

语言层与动作层的每个 Linear 站点都换成 `gemm_maybe_fp4(...)`。两个非平凡的工程点：其一，引擎的 Gemma 执行器把 qkv 与 gate_up 存为 **PACKED 权重**（k/q/v 行拼接、dual-geglu 交错），而 HF checkpoint 是分开的张量——所以工件转换器（`quant/nvfp4_convert_packed.py`）必须**先按引擎内存布局拼接、再量化**，工件名用引擎内存名；RMSNorm 增益按输入通道折叠进 q/k/v（`w *= (1 + ln.weight)`）、gate/up 则乘 `(1 + post_ln.weight)`——这正是 §4.5 讨论的"变换折叠"在工件产线里的既有先例。其二，fp4 站点的 dtype 边界：引擎中间张量是 BF16，而量化核要求 F16 输入，故每个 fp4 站点付一次 bf16→f16 cast（输出侧同理），这在批注里如实标注为"豁免语义的代价"。

## A.2.5 变体接线与可见性

`load.rs` 与 `model.rs` 的接线是纯机械但坑密布的部分：`ModelVariantChoice::Nvfp4Static`（config.rs 的 as_str/from_str/resolve 三处）+ `ModelVariant::Nvfp4Static` 第四变体（model.rs 的全部 match 点）+ load.rs 分支（`fp4/` 目录约定或 `LoadOptions.fp4_artifact` 显式路径）+ builder `build_nvfp4_static_model`。引擎的封装边界要求解锁六处可见性（`CudaBuffer::as_ptr`、`into_tensor`、`Bf16Blocks::ctx` 等）——这些是"向引擎添加精度"的固有摩擦，patches 文件里逐处可见。

# A.3 PyTorch 侧：校准 PTQ 套件（quant/ptq/）

## A.3.1 校准采集器：精确 Hessian 的一钩子采集（collector.py）

GPTQ 与逐层 MSE 评估都需要每层的二阶统计。我们没有存激活本身（189 层 × 43k 行会占数 GB），而是利用恒等式 $\sum_i\lVert x_i\Delta W^\top\rVert^2=\operatorname{tr}(\Delta W H \Delta W^\top)$（$H=\sum_i x_i x_i^\top$），只存 K×K 的 Hessian 与 K 维绝对均值——评估任意权重扰动对的输出影响是精确的而非采样的：

```python
def mk_hook(name):
    def hook(mod, args):
        x = args[0]
        K = mod.weight.shape[1]
        if K % 16 != 0: return
        xf = x.detach().reshape(-1, K).float()
        e = acc[name]
        e["H"] = xf.t() @ xf if e["H"] is None else e["H"] + xf.t() @ xf  # fp32 累积
        e["abs"] += xf.abs().sum(0)
        e["n"] += xf.shape[0]
    return hook
```

`forward_pre_hook` 保证拿到的是线性层的**输入**（对于 q/k/v 就是各自 norm 之后的张量）；Hessian 在 GPU 上按批累积（`e["H"] += xf.t() @ xf`，每批一次 `+=`）、结束时落盘 CPU fp32（5.49GB）。两个容易做错的细节：其一，外积与累加前先把 BF16 输入 `.float()` 升精度——每层全程约 43k 行的求和若留在 BF16 的 8 位尾数里，H 的旁对角元会被舍入噪声淹没；其二，H 保持**未归一化**的求和形式、不除以行数 n——恒等式 $\sum_i\lVert x_i\Delta W^\top\rVert^2=\operatorname{tr}(\Delta W H \Delta W^\top)$ 要求 $H$ 恰为 $\sum_i x_i x_i^\top$，取平均会让逐层 MSE 的绝对值差一个 $1/n$ 因子（GPTQ 的 H⁻¹ 方向不受标量因子影响，它在求逆中消掉，但逐层裁剪搜索比较的正是 MSE 绝对值，必须用未归一化形式）。挂载范围是除动作头外、输入维为 16 倍数的全部 backbone 线性层，共 189 个（动作头的量化在写盘时用 RTN 完成）。AWQ 的通道显著度（absmean）在同一钩子里免费获得。

## A.3.2 量化器与 GPTQ 的 NVFP4 块格式适配（quantizers.py）

`nvfp4_dequant` 与引擎侧参考（`torch_fp4.fake_quant_nvfp4_torch`）逐位一致（单测 maxdiff=0.0）——这是"离线校准的数字就是引擎 kernel 消费的数字"的保证。语义：每 16 列求 amax，除以 E2M1 满格值 6 并归一到 E4M3 域后编码为块缩放，值本体在 E2M1 中点阶梯上取最近格（`searchsorted` 于中点表，保证与参考实现的舍入方向一致）。

**GPTQ 部分的数学**。逐层目标是 $\min_{\Delta W}\operatorname{tr}(\Delta W H \Delta W^\top)$。按列贪心时，每一步是一个小型等式约束二次规划：把第 $i$ 列钉在格点 $q_i$ 上，对剩余列求最优整体平移。闭式解：补偿系数 = 剩余列 Hessian 子矩阵之逆 $H[i{:},\,i{:}]^{-1}$ 的首行对角归一化——这是经典 OBQ/OBS 的逐步精确解。逐列重新求逆是 $O(K^4)$；GPTQ 的实现技巧是一次分解代替全部求逆：$H$ 加阻尼（$\lambda=0.01\cdot\operatorname{mean}(\operatorname{diag}H)$，标准 GPTQ 设定；对角为 0 的死通道先置零剔除、对角补 1）后求逆，取**上三角 Cholesky 因子 U**。为什么 U 的行给出同样的系数：上三角因子的尾主子阵 $U[i{:},\,i{:}]$ 恰是 $(H^{-1})[i{:},\,i{:}]$ 的上三角 Cholesky；而 Cholesky 因子的首行满足 $R_{0j}=M_{0j}/R_{00}$（首行性质），故 $U[i,\,i{+}1:]/U[i,i]$ 既等于 $(H^{-1})[i{:},\,i{:}]$ 首行归一化，又由 SPD 分块恒等式等于 $H[i{:},\,i{:}]^{-1}$ 首行归一化——即逐步精确 OBS 解。我们在随机 SPD 矩阵上逐步数值核验了三者（$H^{-1}$ 原行 / $U$ 行 / 子矩阵逆行）至任意步：$U$ 行与子矩阵逆行逐位一致，而"$H^{-1}$ 第 $i$ 行"的通俗写法仅在第一列成立——这正是 GPTQ 用 Cholesky 因子而非 H⁻¹ 本身的原因。与 Frantar 原实现的差别仅在工程层面：原版按 128 列分块做惰性更新以省显存，我们的层 $K\le 8192$，直接全局单循环，可读性优先、GPU 代价可忽略。

**NVFP4 块适配**。GPTQ 的适配难点在于 NVFP4 的缩放是**每 16 列一块、随块内当前值变化**的，而经典 GPTQ 假设逐列固定的均匀格。我们的做法：列仍从左到右处理，但**每进入一个新 16 列块，就从当前（已被误差反馈修改过的）权重值重算该块的逐行 E4M3 缩放**，再在该固定缩放下逐列量化-反馈（原理图见 §3.2）：

```python
for i in range(K):
    if i % block == 0:                      # 块入口：从当前值重算块缩放
        blk = W[:, i:i+block]
        amax = blk.abs().amax(1) * clip     # clip<1 是学习裁剪旋钮（MSE 校准）
        ts = (amax.max() / 448.0).clamp_min(1e-30)      # per-tensor 二级缩放
        s_row = _quant_e4m3(amax / 6.0 / ts).clamp_min(1e-9) * ts
    x = W[:, i] / s_row
    idx = torch.clamp(torch.searchsorted(mids, x.abs()), 0, 6)
    q = grid[idx] * torch.sign(x) * s_row  # E2M1 中点阶梯
    d = Hi[i, i]                            # 上三角 Cholesky 因子的对角（见上文数学）
    err = (W[:, i] - q) / d
    W[:, i+1:] -= err.unsqueeze(1) * Hi[i, i+1:].unsqueeze(0)   # OBS 误差反馈
    W[:, i] = q
```

**逐层裁剪搜索**在量化器外面套一层：`rtnc_best_clip` 对 c ∈ {1.0, 0.95, …, 0.5} 逐个完整跑上述量化（RTN 路径）或把 c 传入 GPTQ 的块入口（GPTQ 路径），以 `layer_mse_tr`（A.3.1 的 Hessian 恒等式，无须跑网络）取输出 MSE 最小者——max 校准（c=1）与 MSE 校准（c<1）在同一循环里合流。这就是正文 §4.3 的结论的出处：即便逐层输出 MSE 被压到 RTN 的 76%，全 NVFP4 的闭环仍为 0%——误差的**放置**比总量重要。

## A.3.3 量化写盘：分配表即策略（bake.py）

五种配方是五个纯函数，把模块名映射到精度——NVIDIA 分配的复刻与我们的激进版只差几行：

```python
def alloc_mixed(name, W):            # NVIDIA mixed_nvfp4 复刻（2.02×）
    if name.startswith("backbone.model.model.visual."): return "fp8"   # ViT 全 FP8
    if name == "backbone.lm_head":                     return "fp8"
    if ".language_model.layers." in name:
        if name.endswith(("self_attn.o_proj", "mlp.down_proj")): return "fp8"  # 敏感投影保护
        return "nvfp4_gptq"                                  # LLM 主投影 NVFP4+GPTQ
    if name.startswith("action_head."):
        base = name.split("action_head.model.")[-1]
        if base.startswith("ff.net.0.proj") or base == "ff.net.2": return "nvfp4_gptq"
        if base.startswith("transformer_blocks.") or base == "proj_out_1": return "fp8"
        return "bf16"                       # timestep 编码器/动作解码器不量化
    return "fp8"

def alloc_aggr(name, W):             # 本文激进方案（2.88×）：动作头整体 NVFP4
    ...
    if name.startswith("action_head."):
        ...  # 全部 nvfp4；仅 o_proj/down_proj → fp8
    return "nvfp4_gptq"              # 视觉塔也 NVFP4（实测：闭环致死）
```

而**恢复基座（2.44×）**就是"backbone 全 FP8 + 动作头全 NVFP4"——与 mixed 的差别恰是把 NVIDIA 保留 FP8/FP16 的动作头四类投影推进到 NVFP4（贡献亮点 1）。写盘方式是**按原键、原形状把量化取值写回 safetensors**（张量仍是普通 BF16，数值已落在量化格点上，不新增键或结构），因此现有评测栈零改动即可闭环；压缩账目（nvfp4=0.5625 字节/参数含 E4M3 缩放、fp8=1 字节+行缩放）随 checkpoint 落盘成 `ptq_recipe.json`。

## A.3.4 AWQ 折叠映射：三个非平凡边界（folds.py）

折叠映射表覆盖了 GQA 注意力（o_proj 的通道缩放须折叠进 v_proj 输出列，且同一 kv 通道被 2 个 q 头共享——映射函数处理 head_dim 分解）、GeGLU（down_proj 缩放精确折叠进 up_proj 列，因为 y=gelu(g)·u 对 u 线性）、以及**不可折叠点**（视觉塔 SiLU-MLP 的 fc2：逐元素非线性阻挡精确折叠）。搜索判据用 A.3.1 的 Hessian 恒等式精确评估输出 MSE——结果是全部站点 α=0（§4.5 的负结果）。

# A.4 量化域 LoRA 恢复栈（rl/）

## A.4.1 加性 LoRA：训练前向与部署函数严格同一（lora_qad.py）

恢复训练的核心决定是把"量化"从训练循环里**请出去**：基座权重是量化写盘时定格的量化值（不变），LoRA 残差是精确梯度的 BF16 低秩分支：

```python
def make_fwd(m, s):   # s = alpha/r
    # v2 加性语义：量化基座（量化值已写进 m.weight，永不再变）+ 精确梯度低秩残差。
    # 部署 = lora_merge_bake（W_baked + (B@A)*s，BF16 加法，不重量化）。
    def fwd(x):
        y = torch.nn.functional.linear(x, m.weight, m.bias)
        z = torch.nn.functional.linear(x, m.lora_A)     # (…, r)
        z = torch.nn.functional.linear(z, m.lora_B)     # (…, N)
        return y + (z * s).to(y.dtype)
    return fwd
```

三个由此而来的性质：(i) **无 STE 近似**——量化基座不需要梯度（冻结），残差路径是普通线性函数，梯度精确；(ii) **训练=部署**——合并只是把这个加法算一次（W_baked + BA·s），闭环评测的函数与训练完全一致；(iii) **原生速度**——0.43 秒/步，而"逐前向把 W+BA 量化合并"的朴素写法是 75 秒/步（170×），后者曾在 24GB 卡上把 500 步推到 10 小时量级。训练器通过替换 `Gr00tTrainer.__init__` 注入（构造后遍历模块装 LoRA、冻结基座），`QAD_OPD_KL_W>0` 时再包一层 `compute_loss` 做 probe 缓存 teacher-KL（BF16 教师探针 38 秒离线缓存，训练循环内只跑学生前向——这是对早期 twin-forward 方案在 WSL 上确定性崩溃的修复）。

## A.4.2 LoRA 合并写盘的教训（lora_merge_bake.py）

合并逻辑本身三行（全局加载全部 shard → `W_baked + (B@A)·α/r` → 按原索引回写），但它踩中了两个值得记录的坑：HF 保存的 LoRA 键是 `X.lora_A` 而 base 是 `X.weight`（查表要补后缀）；lora_A 与其 base 权重可能落在**不同 shard**（必须全局加载后合并再回写，逐 shard 处理会 KeyError 崩在半路）。

## A.4.3 RWR 在线对照臂（lora_rwr.py）

PPO 家族对照的落地是两阶段离线环：eval server 以 `FP4VLA_LOG_DIR` 记录批量 rollout（观测 JPEG 压缩 + 动作块），训练侧按任务日志 mtime 分段、按 slot 拼接 64 步目标、经 `apply_action`（含相对坐标转换与归一化，z-range 校验=1.00）注入训练批。GR00T 的统一动作空间是 132 维，libero 占据前 7 槽——这个映射不是从文档查的，而是从演示数据的方差结构测出来的（var>0 恰在 0..6）：

```python
def inject_action(batch, target):
    a  = torch.zeros(1, 64, 132, dtype=torch.bfloat16)
    mk = torch.zeros(1, 64, 132, dtype=torch.float32)
    a[0, :, :7]  = torch.from_numpy(target).to(torch.bfloat16)
    mk[0, :, :7] = 1.0
    inner = dict(batch["inputs"]) if "inputs" in batch else dict(batch)
    inner["action"], inner["action_mask"] = a, mk
    return inner
```

# A.5 闭环评测链：quantize-once 与零改动部署

早期按前向 fake-quant 的评测慢 ~50 倍并触发 rollout 端 ZMQ 超时连环崩。修复是 `rl/scoped_quant.py` 的 **quantize-once**：加载时把 in-scope 线性层权重一次性替换为 NVFP4 反量化值，前向恢复原生 BF16 速度——数值与逐前向完全等价：

```python
def mark_scope(model):
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and inscope(name) and ...:
                mod.weight.data = fake_quant_nvfp4_torch(w).to(w.dtype)
    nn.Linear.forward = _ORIG   # 恢复原生前向；权重已携带量化噪声
```

量化写盘脚本（A.3.3）是它的持久化版本。eval server（`run_gr00t_server_fp4vla.py`）在此之上只加三个环境变量钩子：`FP4VLA_QUANT/SCOPE`（量化注入）与 `FP4VLA_LOG_DIR`（A.4.3 的 rollout 日志）。于是全部实验臂——五臂 PTQ、LoRA 合并产物、RWR 产物——都以同一形态进入闭环：**换 checkpoint 路径，不换代码**。

# A.6 复现索引

| 组件 | 位置 | 一条命令 |
|---|---|---|
| 校准采集 | quant/ptq/collector.py | `.venv/bin/python .../collector.py`（16 批×8 窗，189 层 Hessian 5.49GB） |
| 五臂量化写盘 | quant/ptq/bake.py | `bake.py --recipe mixed --out weights/ptq_bakes/gr00t_ptq_mixed` |
| QAD/OPD-LoRA | rl/lora_qad.py | `GR00T_BASE_CKPT=<fp8臂> QAD_STEPS=500 ... lora_qad.py`（OPD 加 `QAD_OPD_KL_W=1.0`） |
| LoRA 合并 | rl/lora_merge_bake.py | `--ckpt <checkpoint-500> --out <部署目录>` |
| RWR 臂 | rl/lora_rwr.py + rwr_chain.sh | 采集（`FP4VLA_LOG_DIR=...`）→ 训练 → 评测 |
| 闭环全量 | groot-fsdp2/run_libero_eval_fp4vla.sh | `bash run_libero_eval_fp4vla.sh <CKPT> <TAG> 10` |
| 引擎侧 | patches/apxinf-fp4vla-engine.patch | `git apply` 于引擎主干（1387 行，adapter/算子/加载器/拓扑/变体全链） |

引擎补丁的完整文件清单：`cublaslt_fp4_adapter.cu`（GEMM+量化核+cast 核）、`build.rs`/`ffi/cublaslt.rs`（注册与声明）、`kernels/fp4.rs`（fp4_linear+测试）、`pi05/fp4_weights.rs`（工件加载器+往返测试）、`pi05/model/blocks/fp4.rs`（Fp4Blocks 混合拓扑）、`pi05/config.rs`/`model.rs`/`load.rs`（Nvfp4Static 变体接线）、外加 fp8 在 sm_120 的输出 dtype 修复。每个文件的 spike/验证脚本在 fp4vla 仓库 `spike/` 与 `quant/` 下成对出现。
