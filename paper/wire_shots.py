# -*- coding: utf-8 -*-
import io

def patch(path, pairs):
    t = io.open(path, encoding='utf-8').read()
    for old, new in pairs:
        assert old in t, f'MISS in {path}: {old[:30]}'
        t = t.replace(old, new, 1)
    io.open(path, 'w', encoding='utf-8', newline='\n').write(t)
    print('patched', path.split('/')[-1])

# ---- sec 3.1: verification screenshot after the four-layer-validation sentence ----
patch('/home/zhaosiying/codebase/fp4vla/paper/sections/03-方法.md', [
    ('收敛后才进入闭环实验。\n',
     '收敛后才进入闭环实验。该验证的算子层实跑记录见下图（最大化 WSL 终端内运行 spike/fp4_gemm_bench --verify，对 CPU fp64 参考逐元素核对）：nvfp4 的 verify_maxrel = 0.000000——GPU kernel 与参考实现在该测试形状下位级一致；同屏可见 mxfp4 的 ok=0、algos=0（heuristic 返回零算法），即 §2.2 所述格式硬约束的直接证据。\n\n{{fig:shot_verify}}\n'),
])

# ---- sec 4.2: engine benches (pi05 + gr00t), GEMM peaks, opbench ----
patch('/home/zhaosiying/codebase/fp4vla/paper/sections/04-实验.md', [
    ('{{fig:e1_latency}}\n',
     '{{fig:e1_latency}}\n\n表中两行引擎数据的实拍复现如下（最大化 WSL 终端，同一命令仅换 --model-dir）：π0.5 侧 APXInf AutoPolicy 加载 226.8 秒后 batch-1 延迟稳定在约 50 ms（model_ms_p50，与表中 49.6 ms 同源同协议）；GR00T N1.7 侧得 28.8 ms / 29.7 Hz。截图里跑的就是引擎的 serving API 本身——"评测即部署形态"在此是字面事实。\n\n{{fig:shot_pi05}}\n\n{{fig:shot_gr00t}}\n'),
    ('小投影进 FP4 只有开销没有收益。\n',
     '小投影进 FP4 只有开销没有收益。该基准的实跑截图如下（nvfp4 行在计算受限形状 417–506 TFLOPS，mxfp4 全部 ok=0）：\n\n{{fig:shot_gemm}}\n'),
    ('| GeGLU 4304×1152 | 2.20× | 2.17× | 3.57× |\n',
     '| GeGLU 4304×1152 | 2.20× | 2.17× | 3.57× |\n\n该表的实跑入口是 spike/fp4_opbench（下图为终端实拍，输出逐形状的 bf16/fp4 耗时与加速比）：\n\n{{fig:shot_opbench}}\n'),
])

# ---- appendix A.2.3: packed artifact screenshot after the loader paragraph ----
patch('/home/zhaosiying/codebase/fp4vla/paper/sections/14-附录A-核心源码走读与APXInf框架解析.md', [
    ('**注释里写明"产物不可变（QAD 冻结 scale），加载器永不重量化"**',
     '**注释里写明"产物不可变（QAD 冻结 scale），加载器永不重量化"**\n\n离线产物的实物见下图（π0.5 全部受量化线性层的 .packed.u8 权重与 swizzled .scale.u8 块缩放，共 245 个文件、约 2.4 GB；manifest 记录每个张量的布局元数据，加载器按 stem 约定直连这些文件）：\n\n{{fig:shot_packed}}'),
])
print('all figures wired')
