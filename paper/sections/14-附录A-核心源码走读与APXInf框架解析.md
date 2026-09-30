# 附录 A. 一份量化权重如何成为可评测的策略

源码走读围绕一份模型的生命周期展开。首先定义量化数值，再用真实输入校准，随后在量化基座上训练低秩残差，最后导出并进行闭环评测。原生 APXInf 章节单独解释 packed 权重如何进入 CUDA 算子。

建议阅读时同时打开 `quant/ptq/collector.py`、`quant/ptq/bake.py`、`rl/lora_qad.py` 和 `eval/run_recovery_eval.py`。它们分别对应统计采集、权重生成、恢复训练和环境评测四个阶段。函数名与张量形状用于定位实现，源码摘要见 `paper/validation/source_refs.json`。

## A.1 先确定输入、产物与执行路径

理解实现首先要区分数值、存储和计算。设原线性层为 $y=xW^\top+b$，$W$ 的形状为 $[N,K]$，$x$ 展平后的形状为 $[M,K]$。

| 表示 | 磁盘中保存什么 | 执行方式 | 主要用途 |
|---|---|---|---|
| 低位宽打包权重 | E2M1 数据、E4M3 块缩放、FP32 张量缩放 | 原生量化激活和低精度 GEMM | APXInf 算子与引擎实验 |
| PTQ 反量化权重 | $Q(W)$ 转回源 dtype 后的普通张量 | PyTorch 浮点线性层 | 单独研究权重扰动与闭环行为 |
| 恢复后的稠密权重 | $W_{\mathrm{baked}}+(\alpha/r)BA$ 再转回源 dtype | PyTorch 浮点线性层 | 与既有 GR00T 策略服务器兼容的恢复评测 |

接下来先走完 GR00T 的 PyTorch 路径：源 checkpoint → 校准统计 → PTQ 基座 → LoRA 恢复 → 稠密导出 → LIBERO 评测。A.8 再展开 APXInf 的 packed 权重路径，其中原生算子还会在线量化激活。这样，同一个低精度数值格式在两种执行方式中的作用就有了明确位置。两条路径的测量范围统一列在 A.10。

## A.2 把浮点权重映射到低精度格点

这一阶段输入原始矩阵与裁剪系数，输出低精度编码能够表示的数值。先统一舍入和缩放约定，后面的校准、写盘与 CUDA 实现才能比较同一对象。

### A.2.1 E2M1、E4M3 与最近偶数舍入

`quant/torch_fp4.py::_round_grid` 是 PyTorch 数值路径的共同舍入入口。E2M1 的非负格点完整包含

$$
\{0,0.5,1,1.5,2,3,4,6\}.
$$

符号位独立处理。E4M3FN 格点包含零、次正规数和最大有限幅值 448；正幅值编码范围为 `0x00..0x7e`，`0x7f` 留给 NaN。这里的 FN 指有限数格式约定，不存在可用的无穷大编码；量化有限权重时，超范围幅值饱和到端点。

两个量化器都采用最近值、恰好中点时取偶数编码，即 round-to-nearest, ties-to-even。设相邻幅值码为 $j,j+1$，中点为 $m_j$。`searchsorted` 在中点先选择低侧；如果低侧码 $j$ 为奇数，再移动到高侧。如此得到 `0.25→0`、`0.75→1`、`1.75→2`、`3.5→4`、`5→4`。偶数指**编码的最低有效位**，不是结果数值是否为偶整数。

格点决策以 FP64 进行，随后转回调用者要求的 dtype。这用于明确边界上的比较，不表示模型采用 FP64 推理。NumPy 参考实现 `quant/fp4_quant.py` 独立完成编码与解码；E4M3 还用 PyTorch 原生 `float8_e4m3fn` 转换交叉检查。CPU 测试覆盖格点、中点、中点两侧的相邻浮点数以及正负号，使验证不仅限于随机样本。

### A.2.2 一个张量缩放与每 16 个数的块缩放

`_nvfp4_tensor_scale` 从整个原始矩阵确定一个 FP32 二级缩放 $\tau$；裁剪系数 $c\in(0,1]$ 同时参与其定标。本文实现采用

$$
\tau=\operatorname{FP32}\!\left(\max(c\max|W|/448,2^{-149})\right),
$$

全零矩阵则约定 $\tau=1$。对每行连续 16 个元素形成的块，先计算 $a_{r,b}=c\max_j|W_{r,16b+j}|$，再计算可编码的块缩放

$$
s_{r,b}=R_{\mathrm{E4M3}}\!\left(\frac{a_{r,b}}{6\tau}\right),\qquad
q_{r,16b+j}=R_{\mathrm{E2M1}}\!\left(\frac{W_{r,16b+j}}{s_{r,b}\tau}\right).
$$

最终反量化值为 $q_{r,16b+j}s_{r,b}\tau$。除法中的零分母以安全值代替，仅用于选择编码；真实块缩放仍保留零，因而全零块严格解码为零。极小矩阵的 $\tau$ 至少为一个可表示的 FP32 次正规数，不因下溢而变成零。按块或按行分批转换时，调用者必须传入同一个全张量 $\tau$，不能给每个分片重新定标。

`quant/ptq/quantizers.py::nvfp4_dequant` 返回上述数值的 FP32 矩阵。`fake_quant_nvfp4_torch` 则转回输入 dtype，并通过

```python
return dequant + (W - W.detach())
```

提供恒等直通梯度。括号内前向数值为零，反向对 $W$ 的导数为一；这是一种梯度估计约定；加性 LoRA 恢复冻结基座，直接沿残差分支求导。

混合配方中的 FP8 使用 `fp8_e4m3_dequant`：每个输出行独立取 FP32 缩放 $s_r=\max_k|W_{r,k}|/448$，将 $W_{r,k}/s_r$ 舍入为 E4M3，再乘回 $s_r$。零行和极小尺度同样显式处理。它是逐行缩放的 weight-only FP8 量化器，与每 16 个元素共享一个缩放的 NVFP4 格式不同。

## A.3 用真实输入选择量化参数并生成 PTQ 基座

单看权重误差会忽略输入通道的使用频率。校准器先收集模型实际前向的输入统计，量化器据此选择裁剪和误差补偿，写盘程序最后将各层结果组成一份可加载的完整 checkpoint。

### A.3.1 完整模型的输入统计

`quant/ptq/collector.py` 从源 checkpoint 读取模型配置，通过 `training.start_from_checkpoint` 加载权重，并检查 16 层语言栈、32 个 DiT 块和 4 层 VL 模块。`gr00t_runtime.py::restore_checkpoint_model_config` 保留 checkpoint 的结构参数，仅合并显式训练开关；`configure_libero_data` 使用源 LIBERO 模态定义构造数据覆盖，不引入其他机器人更长的动作窗口。归一化检查读取实际 `state_action_processor.statistics`，并保留完整源处理器元数据。校准针对完整模型中真正被执行的线性层，不凭文件键推断某层必然存在或执行。

对一层输入 $x$，pre-hook 将除最后一维外的轴展平，得到 $X\in\mathbb R^{R\times K}$。`accumulate` 计算

$$
H\mathrel{+}=X^\top X,\qquad
u_k\mathrel{+}=\sum_r|X_{r,k}|,\qquad
n\mathrel{+}=R.
$$

模型前向采用 BF16；每个外积显式升至 FP32、关闭 autocast 和 TF32，随后转移到 CPU 累加。这样，显存中不必同时保留所有层的 $K\times K$ 统计矩阵。`H` 是未归一化的经验二阶矩，$u/n$ 是输入通道绝对均值，`calls` 与 `n` 分别记录层执行次数和实际输入行数。

对固定的已采集输入，若权重误差为 $\Delta W$，则

$$
\|X\Delta W^\top\|_F^2
=\operatorname{tr}(\Delta W H\Delta W^\top).
$$

`layer_mse_tr` 直接实现右式。该恒等式使搜索不必保存全部激活，这里的 H 描述有限校准样本上的局部线性目标。将 $H$ 除以行数会改变误差的尺度而不改变同层裁剪候选的排序；阻尼也同比缩放时，GPTQ 的补偿比例保持一致。

`calib_meta.json` 记录完整结构、源 shard 摘要、配置和统计文件摘要、实际窗口数、各层行数、缓存摘要及非 Linear 目标。完成前检查所有已挂钩层均被执行；迭代数据集中的窗口可能重复，因此“消费了多少窗口”与“唯一训练样本数”分别表述。共享的输出词投影不独立采集 Hessian，其量化值由规范词嵌入决定。

### A.3.2 固定张量缩放下的 GPTQ

`rtnc_best_clip` 先在 `{1.0,0.95,0.9,0.85,0.8,0.7,0.6,0.5}` 中，以 RTN 候选和迹公式选择裁剪系数。随后 `gptq_nvfp4` 仅对选中的系数运行一次补偿；这与“为每个系数完整运行 GPTQ 再择优”的计算预算不同。

GPTQ 先根据原始权重确定全张量 $\tau$，再处理校准输入恒零的通道，并为 $H$ 加入 $0.01\operatorname{mean}(\operatorname{diag}H)I$ 阻尼。对逆矩阵做上三角 Cholesky 分解，得到 $H^{-1}=U^\top U$。进入一个新的 16 列块时，依据已经接受前面误差反馈的权重，重新选择该块每行的 E4M3 缩放；$\tau$ 始终不变。

核心更新可写成如下伪代码，变量名对应实现中的 `ts`、`scales`、`Hi`：

```python
tau = tensor_scale(original_W, clip)  # 整个矩阵唯一
for i in range(K):
    if i % 16 == 0:
        scale = block_scale(compensated_W[:, i:i+16], tau, clip)
    q = round_E2M1(compensated_W[:, i] / (scale * tau)) * scale * tau
    error = (compensated_W[:, i] - q) / U[i, i]
    compensated_W[:, i+1:] -= error[:, None] * U[i, i+1:][None, :]
    compensated_W[:, i] = q
```

`U[i,j]/U[i,i]` 的依据是逐步消元：固定一列后，用 Schur 补得到剩余子问题的逆矩阵；其结果可由对应的尾部 Cholesky 因子表达。下一步使用消元后的子问题，而非原逆矩阵未经处理的尾部主子阵。

固定 $\tau$ 的作用是确保所有输出块仍属于同一个可表示的 NVFP4 张量。`return_metadata=True` 可同时返回 $\tau$ 和每块解码后的 E4M3 缩放，用于重新编码和检查可表示性。GPTQ 默认返回反量化矩阵，供 BF16 checkpoint 写盘；packed 导出是后续独立转换步骤。全局逐列补偿也有计算开销，量化墙钟时间须按实际执行记录。

### A.3.3 AWQ 搜索的有效误差与可折叠条件

`quant/ptq/awq.py::search_site` 依据输入绝对均值 $a_k$ 搜索 $s_k\propto\max(a_k,10^{-12})^\alpha$，再将 $s$ 归一化为均值一。候选指数为 `0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0`。正的尺度保证补偿变换可逆；$\alpha=0$ 对应恒等变换。

若把输入改成 $xs$、权重改成 $W/s$，未量化网络保持不变。量化后的有效误差必须在原输入坐标中计算：

$$
\Delta W_{\mathrm{eff}}=Q(W/s)s-W.
$$

对于 producer–consumer 对，消费者使用同样的有效误差；生产者另计算 $Q(W_ps)-W_ps$ 的局部输出误差。两者之和是局部搜索代理，不是这两个量化算子串联后的精确端到端误差。CPU 测试将这些表达式与显式矩阵输出比较。

`folds.py` 决定缩放能放在哪些位置。RMSNorm 的增益、LayerNorm 的增益和偏置可吸收输入通道缩放；门控 MLP 的 up 分支是线性支路，故 down 输入缩放可折入 up 的输出通道。语言使用 gated SiLU，π0.5 则有 GeGLU，二者的非线性形式分别处理。普通视觉 MLP 的非线性不允许任意缩放直接穿过激活；共享给多个消费者的归一化参数也不能只为单个消费者修改。

注意力输出投影可沿 V 分支补偿，但 GQA 中一份 KV 通道被多个查询头共享。实现用 `head_dim=128` 和 `gqa_groups=2` 建立消费者通道到生产者通道的映射，在共享组内取绝对均值的最大值，再把同一尺度广播回各消费者。视觉 fused-QKV 则只修改 V 对应的输出行切片。是否采用搜索得到的折叠由该次产物明确记录，不由局部代理误差直接推断闭环收益。

### A.3.4 精度配方、共享别名与完整索引

`quant/ptq/bake.py::inventory` 根据 safetensors 文件头建立物理键清单，并核对索引与实际 shard。量化谓词为 `.weight`、二维、$K\bmod16=0$；因此 472 个候选键既包含线性层，也包含词嵌入和位置嵌入。清单外张量按源 dtype 保持原值。

`alloc` 把模块名映射到请求方法。`fp8` 起点将动作头候选权重置为 NVFP4 RTN、backbone 置为 FP8；`head_ffn` 增加语言 gate/up，`head_lang` 再增加 q/k/v，`head_lang_vision` 再增加视觉候选矩阵；`calib` 请求所有候选矩阵使用 NVFP4。各级 NVFP4 集合嵌套。`mixed` 与 `aggr` 作为额外模块分配保留，其中动作头 FFN 按完整的 `transformer_blocks.<i>.ff.net.0.proj/2` 名称识别。

实际量化方法由校准模式与模块类型共同决定。`calibration-mode=none` 显式使用 RTN；`required` 校验完整结构、源配置、统计与缓存身份，并拒绝真正线性层缺失 Hessian 的情形；不属于 Linear 的嵌入矩阵则记录 RTN 回退。`auto` 使用可用统计。每层的请求方法、实际方法、裁剪值和回退原因都写入 `ptq_recipe.json`。

共享权重必须先于逐键写盘处理。该 checkpoint 的 `embed_tokens.weight` 与 `lm_head.weight` 在文件中各保存一份，运行时却共享同一参数。`tied_aliases` 将词嵌入定义为规范来源：先确认源文件两个张量相同，只量化规范张量，再把结果复制给别名；即使输出投影存在独立 Hessian，也不允许给同一个运行时参数产生另一份量化值。这样，加载顺序不会决定最终模型取到哪一份权重。

物理文件包含 3,455,180,928 个元素；扣除已识别共享别名的重复 311,164,928 个元素后为 3,144,016,000。前者用于完整源文件的载荷比较，后者用于已知别名去重的补充预算，两种口径都让分子与分母采用同一去重规则。这两个权重账本的共同范围是参数载荷。

写盘保留原键、shape 和 dtype，将 $Q(W)$ 存为源 BF16 张量。完成后重新读取文件头，验证键集合、索引和元素字节数，并保存源文件、实现与产物摘要。这个环节把“选择了哪一种配方”转成能够实际加载、追溯身份的模型。

## A.4 用演示数据恢复量化后的动作预测

输入是上一步的 PTQ 基座和演示窗口，输出是训练后的低秩适配器。保留基座数值不变，只更新较小的残差分支，既限制训练规模，也让恢复前后的起点可以逐张量核对。

### A.4.1 注入位置、梯度和初始化

`rl/lora_qad.py::install_lora` 在完整模型加载后，对指定范围的 Linear 安装 $A\in\mathbb R^{r\times K}$ 与 $B\in\mathbb R^{N\times r}$。该阶段沿用工程名 QAD，实际目标是演示流匹配，不调用独立教师。

```python
y = F.linear(x, m.weight, m.bias)
z = F.linear(F.linear(x, m.lora_A), m.lora_B)
return y + (z * (alpha / rank)).to(y.dtype)
```

$m.weight$ 是冻结的 PTQ 反量化权重，不在每次前向重新量化。设 $r=32,\alpha=64$，$A$ 采用 Kaiming 均匀初始化，$B=0$，所以初始残差为零，函数起点严格对应 PTQ 基座。第一步通常是 $B$ 获得非零梯度、$A$ 的梯度为零；当 $B$ 离开零点后，二者均可更新。

默认 `head` 覆盖 252 个动作头 Linear，含 38,658,048 个训练参数；`head+lang` 加入语言 q/k/v/o，覆盖 316 个 Linear；`head+lang_all` 再加入 gate/up/down，覆盖 364 个 Linear、58,580,992 个参数。以下实现说明以最后一种范围为例；实际选择由该轮恢复 manifest 记录。动作头位置嵌入属于量化候选但不属于 Linear，故不安装 LoRA。

只有 A/B 的 `requires_grad` 为真，冻结基座仍向输入传递梯度。`eval` 控制 dropout 等模块行为，`no_grad` 控制自动求导；因此语言栈保持 eval 时，低秩分支仍可接收动作损失的梯度。首个反向传播后，`install_gradient_audit` 检查动作头与所选语言范围的 B 梯度是否存在、有限且非零。B 零初始化使 A 的首步梯度为零，这是预期行为。

### A.4.2 流匹配张量与训练预算

源模型的 `action_horizon=40`、`max_action_dim=132`。动作张量 $a$、同形噪声 $z$ 和速度预测均为 $[B,40,132]$；LIBERO 的有效时间位置和控制维度由处理器提供 mask。模型采样 Beta 时间，构造 $a_t=(1-t)z+ta$，学习目标速度 $a-z$，以有效动作掩码归一化流匹配损失。

参数存储使用 FP32，前向在 BF16 autocast 下计算。`recovery_batch.py::resolve_batch` 显式区分微批量 $B_\mu$ 与累积数 $G$，单设备有效演示批量为 $B_{\mathrm{eff}}=B_\mu G$。上游 CLI 的 `global-batch-size` 实际传递累积前的批量，因此入口将其设为 $B_\mu$，并在 Trainer 初始化后验证实际值。例如 $B_\mu=1,G=16$ 与 $B_\mu=2,G=8$ 都对应 16 个演示样本。

`recovery_manifest.json` 记录基座、配方、rank、alpha、范围、批量、累积数、随机种子、训练参数量和归一化来源。`QAD_INIT_ADAPTER` 从同一冻结基座加载已有 A/B，检查形状与元数据；续训对照两支都重建优化器，使用同样的演示预算和优化器更新数。梯度检查与 manifest 共同确认实际训练的是哪一组参数。

## A.5 在学生访问的状态上加入教师监督

演示恢复之后，学生在环境中会遇到自己的动作所产生的新观测。本阶段先保存这些观测，再由冻结教师离线标注，最后让学生在同一个流插值点学习教师速度场。整个过程依次产生观察缓存、教师缓存和续训适配器。

### A.5.1 从真实策略请求保存单样本

`rl/capture_onpolicy.py::install_capture` 挂接数据 collator 和模型 `get_action`。每次策略调用先处理观测，再生成归一化动作块；采集器保存同一环境槽位的处理后观测和该次 `action_pred`，后者作为流插值的干净端点。采集不增加模型前向，不修改返回给环境的动作。

Qwen 的 `pixel_values` 第一维按图像 patch 展开，各样本的 patch 数可以不同。采集器因此保留原单环境 feature，再对选中样本独立 collate，以准确恢复该样本的图像和文本输入。浮点输入保留策略实际使用的 BF16 舍入，整数 token 与网格索引保持整数；退出 `inference_mode` 后复制缓存张量，使其可用于后续带梯度的前向。采集清单记录模型来源、统计摘要、任务文本、环境槽位、调用序号和采样上限。

训练目标只约束真实动作维度。模型仍接受 $[1,40,132]$ 的完整端点和共享插值输入；有效掩码从 checkpoint 的 LIBERO 处理器定义与统计派生，而不是对全部位置置一。本配置的有效动作窗口为 16 步、控制量为 7 维，填充时间位置和其余通用动作槽不参与蒸馏损失。

### A.5.2 教师标注与随机输入重放

`rl/opd_probe_cache.py` 加载完整的未量化教师，检查权重加载结果、16/32/4 结构和归一化身份。教师参数采用与学生训练一致的 FP32 存储、BF16 autocast，输入一次一个独立 collate 的样本。输出 `pred_actions` 是训练前向的**速度场**，不是经过四次积分后的动作块。

`probe_distill.py::replay_context` 为每个样本固定种子，保存 CPU 与模型所在 CUDA 设备的 RNG 状态，将模型置为 eval，并在结束时恢复 RNG 和每个子模块原有的 train/eval 标志。教师和学生具有相同结构、输入 shape、噪声 dtype、Beta 调度和 eval 路径；LoRA 分支本身不额外消耗随机数，因此动作头的噪声与时间采样对应到相同输入点。缓存记录这些条件，`ProbeAnchor.validate_model` 在训练开始时核对。

教师生成标签时不求梯度；学生重放时仍开启求导。缓存标签是固定常量，不需要把教师留在显存中。缓存来源字段区分 `student_rollout` 与 `demo`：前者对应采集时的学生分布，后者用于离线演示蒸馏。

### A.5.3 掩码速度 MSE 如何真正更新参数

令 $m_i$ 为有效动作掩码，教师、学生速度分别为 $v_T(P_i)$、$v_\theta(P_i)$，则

$$
\mathcal L_{\mathrm{probe},i}
=\frac{\sum_{h,d}m_{i,h,d}\big(v_\theta(P_i)_{h,d}-v_T(P_i)_{h,d}\big)^2}
{\sum_{h,d}m_{i,h,d}}.
$$

它是连续速度场的 MSE，没有离散概率分布或 KL 散度。计算后须保留到 LoRA 参数的图；`requires_grad` 检查防止只记录一个数值却没有辅助梯度。

`install_sequential_probe` 包装 `training_step`，先调用原训练步骤完成演示损失的反向传播，释放主计算图；随后在指定优化器更新的每个微批上，执行单个缓存样本的学生前向，将 $\lambda\mathcal L_{\mathrm{probe}}/G$ 反向累加到同一批参数梯度。两次反向之间没有优化器更新，因此该次更新对应

$$
\frac1G\sum_{j=1}^G\mathcal L_{\mathrm{demo},j}
+\lambda\frac1G\sum_{j=1}^G\mathcal L_{\mathrm{probe},j}.
$$

以每 4 次优化器更新添加一次为例，其余 3 次只有演示损失。更新数上的平均权重为 $\lambda/4$，但具有状态的优化器对两种调度会产生不同更新，因此运行记录同时保存权重与频率。缓存索引仅在实际执行探针时递增。实现保留原 `compute_loss` 与 `return_outputs` 协议，采用单设备 HF Trainer 的梯度累积规则。

`QAD_ACTIVATION_CHECKPOINTING=1` 可进一步启用逐语言、DiT 和 VL 块的非重入激活重算。实现只包装含可训练参数的块，保留随机状态，在学生 `eval` 模式但梯度开启时仍然有效；冻结教师的 `no_grad` 前向直接绕过重算。训练不使用生成缓存，故该模式关闭 KV cache，并在恢复 manifest 中记录设置；参数 dtype、损失和演示批量均不改变。

顺序反向降低两张计算图同时存活的需求，并没有消除额外计算。QAD 续训与 QAD+OPD 使用相同初始 A/B、演示批量、优化器更新数和新优化器；教师标注与学生探针的额外前向/反向成本另外报告。固定缓存只实现一轮学生分布蒸馏，训练中不会自动重新采集。

## A.6 将基座与适配器合并为评测 checkpoint

策略服务端读取完整模型，而训练器保存的是包含 A/B 的训练状态。导出器以冻结基座恢复完整键集合，再合并低秩残差，使输出能够直接进入已有的 GR00T 推理接口。

`rl/lora_merge_bake.py` 把原量化基座作为最终模型键集合的权威来源，仅从训练 checkpoint 提取 A/B。原因有二：训练框架可能省略运行时共享的物理别名；A、B 和基座权重也可能分散在不同 shard。导出不能以恰好出现在某个训练 shard 的键集合替代完整模型。

导出先核对恢复 manifest 的基座身份、rank、alpha、配置和归一化摘要；再根据索引查找成对 A/B，验证 $A=[r,K]$、$B=[N,r]$。对于训练 checkpoint 中保留的每个非 LoRA 张量，逐项确认其等于冻结基座。未出现的基座键仅允许显式列出的共享或未使用项，并原值保留；意外缺键、额外键、未配对残差或非有限残差都会使导出失败。

每个残差以 FP32 计算，然后写回原始基座 dtype：

```python
delta = (B.float() @ A.float()) * (alpha / rank)
W_export = (W_base.float() + delta).to(W_base.dtype)
```

合并过程保持残差原值，不再次量化。实数代数中，合并线性层与双分支等价；BF16 会改变乘加顺序和舍入位置，评测因而直接加载实际导出文件。若采用打包基座与独立 BF16 旁路部署，则将旁路载荷计入编码预算。

输出先写入独立临时目录，按基座 shard 保留全部键和 dtype，再重建 `model.safetensors.index.json`，删除所有 LoRA 专用键。处理器和统计默认来自冻结基座；另存 `merge_manifest.json`，记录源与产物摘要、残差统计和逐张量验证数量。只有重新读取并通过键集合与字节数检查后，才将目录转为正式输出。

## A.7 执行动作并记录闭环结果

输入是一份经过导出核验的 checkpoint 和固定的环境初态，输出是逐集成功布尔值、初态身份及日志。这里的主角从单次前向变成策略与环境的反复交互。

`eval/serve_recovery.py` 加载准备好的 PTQ 或合并 checkpoint，保持 `FP4VLA_QUANT=0`，防止对已准备的权重再次量化。`rl/scoped_quant.py::mark_scope` 提供另一个独立入口：从原始浮点模型出发，加载时将指定 Linear 的权重替换一次，随后恢复原线性层前向。对固定权重和固定 weight-only 量化器，量化一次与每次重复量化产生相同权重值；二者仅执行开销不同。bake 产物直接使用 checkpoint 路径。

完整模型驻留策略服务器，LIBERO 客户端经 ZMQ 发送观测并执行动作；每次执行动作块前 8 步，每 episode 最多 720 步。`run_recovery_eval.py` 串行启动任务和服务器，以一环境对应一个 episode 计数流，保存逐集布尔结果、实际分母、进程退出码、重置记录和日志摘要。缺失或超时使该任务验收失败，结果按实际完成状态保存。

环境配对由相同任务、初态索引和初始化后的模拟器状态摘要定义，开发、采集与最终评测使用不相交的初态集合。`rollout_seeded.py::install_bank_resets` 加载官方初态表，先恢复指定状态，再直接向模拟器执行 10 个全零动作稳定步骤，避免通过归一化夹爪转换改变这些零动作；同时记录 bank 文件、恢复状态和稳定后状态的摘要。本文使用 `exp/recovery_protocol_v5_exploratory_fp4.json`：开发索引 4–8、采集索引 20–23、最终评测索引 30–39，对应起始 seed 为 440000、550000、660000。每任务 seed 加 1000 倍任务索引，环境每集再加 episode 索引；服务器另按任务固定推理随机种子。执行短测使用独立 index=0、seed=770000。

`run_recovery_eval.py::validate_resets` 严格检查实际前 $N$ 个 episode 的索引、种子、三个状态或文件摘要和稳定步数。五分支汇总前，`compare_recovery.py::compare_round` 逐集比对这些摘要，并保存每任务结果、配对成功/失败的 $2\times2$ 表与探索性精确 McNemar 检验。该检验作为有限配对样本的探索性统计一并保存。

## A.8 独立的 APXInf 原生执行路径

这一条路径输入 packed 权重、激活和模型配置，输出原生推理结果。它复用前文定义的 NVFP4 数值格式，并进一步处理硬件缩放布局、矩阵输出布局和 CUDA Graph 的内存生命周期。

### A.8.1 框架职责与精度变体

APXInf 以 Rust crate 组织职责：`apxinf-core` 定义 `Tensor`、`DType`、`Shape` 与错误；`apxinf-cuda` 管理设备、stream、缓冲区和算子；`apxinf-model` 负责模型拓扑、权重加载和推理循环。FFI，即跨语言函数接口，把 Rust 包装连接到 CUDA C ABI：

```text
模型 Blocks 的 gemm_maybe_fp4
  → kernels/fp4.rs::fp4_linear
  → ffi/cublaslt.rs
  → cublaslt_fp4_adapter.cu
  → cuBLASLt 与 CUDA kernel
```

模型层决定哪些矩阵走低精度；后端验证张量合同并执行运算。适配器登记在 `apxinf-cuda/build.rs` 中；私有 FFI 的类型转换能力通过公开的薄包装提供给模型 crate。

π0.5 的 `Pi05Model<B>` 用泛型参数指定 Blocks，编译期生成各精度实现，运行时通过 `ModelVariant` 枚举选择。NVFP4 使用组合结构 `Fp4Blocks { inner: Bf16Blocks, fp4: Fp4TensorMap }`，复用 BF16 的前处理、嵌入和缓存能力。变体同时接入配置解析、权重加载和所有执行匹配点；Rust 的匹配检查约束接线完整性，数值正确性仍由独立测试保证。

### A.8.2 packed 权重、块缩放与设备加载

`quant/nvfp4_convert.py` 与 `nvfp4_convert_packed.py` 输出以下三部分：

```text
<stem>.packed.u8  # [N,K] 的 E2M1 编码，偶数列位于低四位
<stem>.scale.u8   # [N,K/16] 的 E4M3 块缩放，含物理布局补齐
manifest.json    # name、shape、block=16、tscale 等
```

每个值的解码需要数据码、块缩放与 `tscale` 三者。加载器 `Fp4DeviceWeights::from_artifact_dir` 验证二维形状、块长、字节数和有限正的 `tscale`，再上传到设备；加载时不重新量化。模型按实际加载的压缩权重建立路由，并记录 BF16 层来自主动保留还是缺少压缩产物。

缩放采用 `VEC16_UE4M3` 要求的 swizzle。令 $K_B=K/16$，行号为 $r$、块号为 $b$，则

```text
PS = 512 × ceil(KB / 4)
offset(r,b) = PS × floor(r/128) + 512 × floor(b/4)
            + 16 × (r mod 32) + 4 × (floor(r/32) mod 4) + (b mod 4)
```

完整缩放缓冲需要 `512 × ceil(KB/4) × ceil(N/128)` 字节；不满一个 tile 的尾部也要分配，超出逻辑尺寸的缩放字节必须为零。逻辑相邻与物理相邻不能混用；例如 $(r,b)=(0,0),(0,3),(32,0),(1,0)$ 的字节偏移分别为 0、3、4、16。`fp4_quant.py::swizzle_scales` 与设备端采用相同映射。

下面的 Ubuntu 记录展示打包产物的实际生成与检查。目录文件数只用于检查文件生成过程，量化覆盖量以 manifest 为准。

{{fig:shot_packed}}

### A.8.3 在线激活量化、二级缩放和输出布局

`apxinf_nvfp4_quantize_activation` 先用同一 stream 的 `cudaMemsetAsync` 将完整 scale tile 清零，再启动激活量化 kernel。每个线程读取一个 16 元素 F16 块，计算 `amax/6`、编码 E4M3 缩放、选择 E2M1 码，并写入 packed 数据与有效 swizzled scale。清零操作也被记录进 CUDA Graph，因此 arena 中原有的字节不会成为 padding 缩放。在线激活路径固定 $\tau_x=1$，与离线权重的可变 $\tau_w$ 分别定义。零缩放用于编码时采用安全除数，物理缩放仍是零；归一化采用除法，避免改为倒数乘法后在中点附近产生不同舍入。

Rust 包装将二级缩放作为权重视图的一部分传递：

```rust
pub struct Fp4WeightView<'a> {
    pub packed: &'a CudaBuffer,
    pub scale: &'a CudaBuffer,
    pub tscale: f32,
    pub rows: usize,  // N
    pub k: usize,
}
```

`fp4_linear` 要求连续二维 F16 输入，核对 $K$、packed/scale 缓冲大小和 `tscale`。激活打包、激活块缩放和输出通过 `workspace::output_buffer` 获取缓冲；存在 GraphWorkspace 时，返回的是 arena 中的持久视图。GEMM 的 64 MiB scratch 则由 `CudaContext::fp4_workspace` 单独持有，同一 context 的串行调用复用它。准备阶段调用 `apxinf_fp4_prepare_rowmajor_f16`，执行阶段调用 `apxinf_fp4_gemm_rowmajor_prepared_f16`。其计算合同是

$$
Y_{M\times N}=(\tau_x\tau_w)(q_xs_x)(q_ws_w)^\top.
$$

`fp4_plan_cache.cuh` 为每个 host thread 维护计划表，键包含当前 CUDA 设备、stream、$M,N,K$ 和 scratch 字节数。该接口的 E2M1 输入、E4M3 VEC16 缩放、FP32 compute、F16 输出及矩阵布局固定，因此这些常量不另作为缓存键。首次准备创建 cuBLASLt 句柄、描述符并查询算法；捕获阶段只允许复用已准备的计划，禁止临时创建执行资源。每次执行重新绑定本层的两个缩放指针，并将 $\tau_x\tau_w$ 作为 GEMM 的 `alpha`，不会因两层形状相同而复用上一层的尺度。准备与捕获由同一 host thread 顺序完成；没有可用算法或准备不完整时直接返回错误。

模型要求连续行主序 $[M,N]$ 输出。适配器交换两输入及其缩放，并交换 $M,N$，计算列主序 $Y^\top=W X^\top$。列主序 $[N,M]$ 的第 $(n,m)$ 个元素偏移为 $n+mN$，恰好等于行主序 $[M,N]$ 的第 $(m,n)$ 个元素偏移。由此不需要额外转置，也不能仅修改 shape 标签而忽略布局。独立探针保留的原始列主序接口与此模型接口分别命名，调用者按各自合同读取输出。

参考计算必须是 $Q(X)Q(W)^\top$，而非 $XQ(W)^\top$，因为硬件同样量化了激活。维护的 `fp4_contract_rowmajor_and_tensor_scale` 用 $(M,N,K)=(16,64,64)$ 和 $(64,128,128)$，分别测试 $\tau_w=0.03125,2.5$，覆盖正负值和零块；检查全部输出元素与按接口规定舍入的 CPU 参考。本轮 GPU 日志中四个用例的最大绝对误差均为 0。该检查同时覆盖缩放传递和非方形输出布局；它不将量化后的矩阵乘积等同于原始高精度乘积。

### A.8.4 模型路由和性能测量包含什么

`gemm_maybe_fp4` 在有压缩权重时调用 `fp4_linear_bf16`，依次执行 BF16→F16、`fp4_linear`、F16→BF16；没有对应产物则保留 BF16。π0.5 的视觉执行委托 BF16 Blocks；实际 FP4 覆盖由执行路由清点。

图路径中的五个缓冲分别是 F16 输入、E2M1 packed 输入、E4M3 scale tile、F16 输出和 BF16 输出。`fp4_graph_buffer_bytes` 按各自大小向上对齐到 256 字节，使用检查过的整数加法与乘法计算容量。`Fp4Blocks::workspace_requirements` 在 BF16 图预算之外，按实际产物的语言前缀长度、动作块长度与流积分步数增加保守容量。64 MiB scratch 由 context 独立持有，不按每个投影重复分配。这一容量预算用于保证图捕获内存充足，实际显存由运行记录给出。

`CaptureBuilder` 先在 arena 中完成准备执行并同步，再以相同的分配顺序捕获。返回的 `CapturedGraph` 持有输出、输入、模型与调制权重、backend/context 和 arena；因此即使局部 Tensor 视图离开函数，图引用的底层设备内存仍然有效。runner 按顺序更新固定输入缓冲并 replay，返回结果引用的是可复用输出，调用者若要跨后续调用保留结果，必须复制它。

准备阶段包含分配、算法选择和图捕获，稳定 replay 只提交已经记录的设备操作。类型转换、scale 清零、在线激活量化与 GEMM 仍实际执行，必须共同计时。普通 eager 调用继续分配或复用输出缓冲。`ModelRunner.execution_mode` 只读取已有准备状态，不触发推理：π0.5 未准备时返回 `unprepared`，准备就绪后返回 `graph` 或 `eager`，另有 `invalidated`、`runtime-managed` 状态；GR00T 已捕获图时返回 `cuda-graph`，否则返回 `eager`。计时记录在 warmup 后及每个样本后保存模型实际返回的字符串。

<!-- BEGIN NATIVE GRAPH VALIDATION -->
本轮设备验收包含以下四项，原始记录位于 `results/native_graph_20260929/`。每项日志的退出状态与总任务 `exit_code.txt` 均为 0。

| 检查 | 实际覆盖 | 结果 |
|---|---|---|
| 非方形矩阵乘组合 | 两个形状 × 两个非单位 `tscale`；正负值、零块、全部输出元素 | 四个用例 `max_abs=0` |
| 缩放 padding | 3 行、$K=48$；对 512 字节 scale tile 两次填入非零污染值后 replay | 全部 512 字节符合参考，包括两个方向的 padding |
| 单层图复用 | 两个同形状投影使用不同块缩放缓冲及不同 `alpha`；arena 预先污染；改变同一输入地址的内容进行四次 replay，含全零输入 | 每次全部输出的精确相等断言通过 |
| 完整 π0.5 图执行 | `nvfp4_static`，显式 `RequireGraph`；两种初始 latent 的 eager 参考，按 0、1、0 次序进行三次 graph replay | 输出均有限；每次 1600 元素，`max_abs=0` |

完整模型用例中的 1600 元素对应后处理之前的内部 $50\times32$ 动作张量。参考与待测路径使用同一 NVFP4 模型，仅改变 eager 或 graph 执行方式，用来检验图捕获、输入更新和资源复用。

`input_sha256.txt` 固定此次 wheel、补丁、算子程序与完整模型测试程序的身份；`installed_extension.json` 记录实际安装扩展的 SHA-256，并确认只读 `execution_mode` getter 存在。独立文件核验确认已安装 `.so` 与该 wheel 内扩展字节相同。CPU 编译和容量测试属于准备阶段记录，上表四项 GPU 日志才是本轮数值与图执行的设备验收依据。
<!-- END NATIVE GRAPH VALIDATION -->

引擎会把部分权重拼接成 QKV 或 gate/up。转换器先按 `[q;k;v]`、`[gate;up]` 行拼接，再进行量化，确保产物与消费者的预期形状一致。语言侧 RMSNorm 的 `(1+gamma)` 可折入相邻消费者输入通道；动作专家的运行时自适应归一化不能套用同一折叠。普通行拼接也不同于 dual-GeGLU 交错布局；遇到交错布局时，当前路由使用已有 BF16 融合算子。

Python 的 `AutoPolicy.from_pretrained` 对 π0.5 使用 `model_variant=`，对 GR00T 家族使用 `precision=` 及相应 embodiment 配置。`exp/bench_engine.py` 从这一入口分别记录加载、模型调用和完整策略调用，功耗与显存另行采样。性能记录因此同时绑定模型身份、精度路由和执行模式。

原生改动由 `patches/apxinf-fp4vla-engine.patch` 固定，构建脚本先应用或确认补丁，再生成 release wheel。安装后核对扩展路径、wheel 与 `.so` 摘要，并确认新缩放/行主序接口进入二进制。这样，源码中的实现与实际 Python 进程加载的引擎能够对应。

## A.9 从产物回查实现与测试

每个阶段都保存能定位输入与输出的 manifest。验证工具沿这些记录回查来源，再分别检查数值、梯度、序列化和设备执行。

`quant/ptq/verify_calibration.py` 提供独立的校准产物验收：通过内存映射和逐矩阵行块读取真实缓存，检查源 shard、配置、统计文件与缓存摘要，核对完整结构、窗口数、目标层覆盖和逐层行数，再验证二阶矩阵的形状、有限性、非负对角与对称性。验收只读，不重新运行采集，也不加载模型或初始化 CUDA。

数值测试、序列化测试和闭环评测分别回答不同问题。CPU 测试包括：完整 E2M1/E4M3 边界及独立编码参考；零与极小尺度；STE 前向值和梯度；GPTQ 固定尺度、可表示性及矩阵目标；AWQ 有效误差与 GQA 映射；LoRA 梯度和累积缩放；单环境输入采集；跨 shard 残差合并；共享别名和完整 checkpoint 索引。发布时的完整测试清单、实际数量与结果记录在 `paper/validation/cpu-tests.json` 和相邻日志中，其中包括填充位置不贡献损失或梯度、官方初态恢复流程及重置账本的严格检查。完整模型的行为由闭环评测记录。

独立的小模型集成检查还直接使用已安装的 HF Trainer 与 Accelerator，在累积数 $G=1,2,16$ 下比较实际钩子梯度与手工有效维度参考：A/B 梯度最大绝对差均为 0，冻结基座没有梯度。另以真实 Qwen、DiT、VL 模块构成小模型，验证可选激活重算保持输出、参数梯度和随机状态一致。这些检查均在 CPU 上进行，没有初始化 CUDA。

原生 GPU 测试另验证真实设备上的 FP4 乘法、二级缩放与布局。release wheel 的文件摘要确认 Python 使用包含这些实现的二进制。模型层成功率则来自独立的 LIBERO 评测账本。数值、执行与行为证据以各自产物身份连接。

| 阶段 | 主要函数或入口 | 应保存的产物 |
|---|---|---|
| 数值格式 | `_round_grid`、`_nvfp4_dequant`、`fp4_quant.py` 编解码 | CPU 边界测试、独立参考比较 |
| 校准 | `collector.py::accumulate`、完整模型入口 | `calib.pt`、`calib_meta.json` |
| 分配与写盘 | `inventory`、`alloc`、`tied_aliases`、`bake.py` | `ptq_recipe.json`、索引、shard 摘要 |
| 原生执行 | `Fp4DeviceWeights`、`fp4_linear`、行主序 scaled adapter | packed/scale 文件、manifest、GPU 测试与引擎日志 |
| 演示适配 | `install_lora`、`resolve_batch` | A/B checkpoint、`recovery_manifest.json` |
| 教师蒸馏 | `install_capture`、`replay_context`、`ProbeAnchor`、`install_sequential_probe` | 采集清单、教师缓存、有效 mask、训练日志 |
| 稠密导出 | `lora_merge_bake.py` | 完整 checkpoint、`merge_manifest.json` |
| 闭环评测 | `serve_recovery.py`、`rollout_seeded.py`、`run_recovery_eval.py` | 初态摘要、逐集结果、实际分母与计时 |

附录 B 给出环境、构建和分阶段命令；`paper/validation/source_refs.json` 记录本附录引用函数在当前源码中的定位及文件摘要，`paper/validation/cpu-tests.log` 保存可重跑的 CPU 检查结果。

## A.10 实现范围与辅助研究入口

前面的主线描述代码如何工作；配方和恢复超参数由冻结的开发选择记录固定，最终评测只读取这些选择。阅读产物时，以下几种量使用各自的定义。

| 对象 | 本实现采用的口径 |
|---|---|
| GR00T 恢复评测 | 对权重施加 NVFP4/FP8 数值扰动，LoRA 合并后以 PyTorch 稠密权重执行。 |
| 编码预算 | packed payload、scale、未量化参数及适用的低秩旁路；另列物理副本与已知共享别名去重口径。 |
| 磁盘与显存 | 分别读取真实 shard 字节数、进程峰值与整卡占用；激活、图 arena、scratch 和优化器按实际用途计入。 |
| 原生部署 | APXInf π0.5 实现在线激活量化与 FP4 路由；GR00T packed 基座加低秩旁路属于后续模型集成工作。 |
| 图执行验收 | 比较同一 NVFP4 模型的 eager/graph 输出；完整模型测试输出为内部 50×32 张量。 |
| 配对统计 | 对齐同任务、同官方初态与状态摘要；策略噪声按任务播种。精确 McNemar 值用于探索性分析。 |
| 数据与计算 | 校准和演示训练可能重复采样窗口；教师分支另计模拟器交互、标注和探针反向成本。 |
| 支持的训练器 | 单设备 HF Trainer 梯度累积；分布式 Apex、DeepSpeed、FSDP 的缩放规则需单独实现。 |

另有两个独立研究工具保留在仓库中。`rl/scoped_quant.py` 用于从浮点模型临时构造 weight-only 量化策略；`rl/lora_rwr.py` 用于任务加权自模仿。它们与主流程的 bake checkpoint、学生状态教师蒸馏分别选择入口。

`lora_rwr.py` 的构造器取同一环境槽位连续 8 次调用各自执行的前 8 步，形成 64 步窗口，再经 `apply_action` 变换到相对坐标和归一化动作空间，仅为 LIBERO 控制槽设置 mask。接入主模型前，需把这个 64 步窗口与 40 步容器、16 步有效监督做显式长度对齐。

RWR 按任务成功率 $w_i$ 加权，目标为 $\sum_iw_i\mathcal L_i/\sum_iw_i$；它筛选的是任务，不是具有逐 episode 成功标记的轨迹。若纳入正式对照，还需要可靠的 episode 边界与同协议采样，避免窗口跨重置。其算法是加权自模仿，未实现 PPO 的价值网络、GAE 或概率比裁剪。
