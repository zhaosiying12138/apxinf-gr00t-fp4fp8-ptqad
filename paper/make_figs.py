#!/usr/bin/env python3
"""Reproducible SVG figures. Charts read recorded evidence; diagrams contain no fake data."""
import html
import hashlib
from datetime import datetime, timezone
import json
import math
import pathlib
import statistics
from itertools import groupby

ROOT = pathlib.Path(__file__).resolve().parent.parent
P = ROOT/'paper'
FIGS = P/'figs'
_DATA_INPUTS = {}
_GENERATED_SVGS = set()
INK, BLUE, TEAL, AMBER = '#183c4d', '#2368a0', '#258577', '#b97824'
SUPER = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁽⁾ᵀ', '0123456789-+()T'))
SUB = dict(zip('₀₁₂₃₄₅₆₇₈₉ᵢⱼₖ', '0123456789ijk'))


def script_markup(value, size):
    """Use ordinary glyphs at explicit offsets, portable to Cairo and browsers."""
    def kind(char):return 'sup' if char in SUPER else 'sub' if char in SUB else 'normal'
    shift=0; output=[]
    for mode, chars in groupby(str(value),key=kind):
        text=''.join(chars)
        target={'sup':-size*.32,'sub':size*.23,'normal':0}[mode]
        mapping=SUPER if mode=='sup' else SUB if mode=='sub' else {}
        glyphs=''.join(mapping.get(c,c) for c in text)
        output.append(f'<tspan dy="{target-shift:g}" font-size="{size if mode=="normal" else size*.72:g}">{html.escape(glyphs)}</tspan>')
        shift=target
    return ''.join(output)

class SVG:
    def __init__(self, w, h, title, subtitle):
        self.w, self.h = w, h
        self.s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}" font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif">', '<rect width="100%" height="100%" fill="white"/>']
        self.text(30, 38, title, 23, INK, weight=700)
        self.text(30, 67, subtitle, 13, '#667c87')
    def text(self, x, y, text, size=14, color=INK, weight=400, anchor='start'):
        self.s.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{script_markup(text,size)}</text>')
    def rect(self,x,y,w,h,fill='#f1f6f8',stroke='none',rx=5):
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" rx="{rx}"/>')
    def line(self,x1,y1,x2,y2,color='#d9e4e9'):
        self.s.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" fill="none"/>')
    def save(self,name):
        output = FIGS/(name+'.svg')
        output.write_text('\n'.join(self.s+['</svg>']),encoding='utf-8')
        _GENERATED_SVGS.add(output.resolve())

def file_record(path):
    path=path.resolve()
    payload=path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}


def read(path):
    """Track the exact JSON bytes used by the charts, not a later reread."""
    resolved=(ROOT/path).resolve()
    payload=resolved.read_bytes()
    record={'path':resolved.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
    if record['path'] in _DATA_INPUTS and _DATA_INPUTS[record['path']]!=record:
        raise RuntimeError(f'Data input changed between reads: {path}')
    result=json.loads(payload.decode('utf-8-sig'))
    _DATA_INPUTS[record['path']]=record
    return result

def positive_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{field} must be finite and positive, got {value!r}')
    return value


def timing_record(record, path, metric, raw_key):
    if record.get('schema_version') != 2 or record.get('all_outputs_finite') is not True:
        raise ValueError(f'{path}: requires current finite-output schema v2')
    n = record.get('n'); values = record.get(raw_key)
    if type(n) is not int or n <= 0 or not isinstance(values, list) or len(values) != n:
        raise ValueError(f'{path}: raw timing samples/count missing or inconsistent')
    for value in values: positive_number(value, path+':'+raw_key)
    if not math.isclose(positive_number(record[metric], metric), statistics.median(values), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'{path}: reported P50 differs from actual raw samples')


def engine_record(record, path, model, variant):
    timing_record(record, path, 'total_ms_p50', 'lat_total_ms')
    timing_record(record, path, 'model_ms_p50', 'lat_model_ms')
    if (record.get('model_type') != model or record.get('variant') != variant or
        record.get('timing_scope',{}).get('entrypoint') != 'AutoPolicy.infer'):
        raise ValueError(f'{path}: engine family, variant or timing scope differs')
    proof = record.get('variant_verification',{})
    if model == 'pi05':
        if proof.get('status') != 'native_model_variant_verified' or proof.get('native_model_variant') != variant:
            raise ValueError(f'{path}: actual native variant was not verified')
    elif proof.get('status') not in ('native_model_variant_verified','explicit_precision_metadata_only') or proof.get('policy_value') != variant:
        raise ValueError(f'{path}: explicit engine precision contract missing')
    if not record.get('runtime_identity',{}).get('native_extensions') or not record.get('checkpoint_identity',{}).get('files'):
        raise ValueError(f'{path}: missing imported native extension or checkpoint identities')
    if variant == 'nvfp4_static' and len(record.get('packed_identity',{}).get('files',{})) != 198:
        raise ValueError(f'{path}: expected the complete 99-tensor NVFP4 payload identity')


def matching_native_protocol(baseline, native):
    def identity(value):
        if isinstance(value,dict):return {k:identity(v) for k,v in value.items() if k!='path'}
        if isinstance(value,list):return [identity(v) for v in value]
        return value
    for key in ('checkpoint_identity','asset_identity','seed','model_seed','device','gpu_identity','n','warmup','input_contract','timing_scope'):
        if baseline.get(key) is None or identity(baseline[key]) != identity(native.get(key)):
            raise ValueError(f'APX BF16/NVFP4 timing ratio requires matching {key}')
    for key in ('constructor_kwargs','policy_metadata'):
        left={k:v for k,v in baseline.get(key,{}).items() if k not in ('model_variant','precision')}
        right={k:v for k,v in native.get(key,{}).items() if k not in ('model_variant','precision')}
        if not left or left != right:
            raise ValueError(f'APX BF16/NVFP4 timing ratio requires matching {key} except precision')
    if baseline.get('action_shape') != native.get('action_shape') or baseline.get('sample_token_counts') != native.get('sample_token_counts'):
        raise ValueError('APX BF16/NVFP4 timing ratio requires matching action shape and input token counts')
    if baseline['runtime_identity']['native_extensions'] != native['runtime_identity']['native_extensions']:
        raise ValueError('APX BF16/NVFP4 timing ratio requires the same imported native extension')


def e1_chart():
    # Deliberately mandatory: no fallback to any previous NVFP4 execution.
    native_path='results/engine/pi05_nvfp4_ptqad_20260929.json'
    native=read(native_path)
    if native.get('model_type') != 'pi05' or native.get('variant') != 'nvfp4_static':
        raise ValueError(f'{native_path} must identify pi05 / nvfp4_static')
    engine_record(native,native_path,'pi05','nvfp4_static')
    sources=[('π0.5 / PT action chunk','results/baselines/pi05_pt_bf16_ptqad_20260929.json','latency_ms_p50','#a3b6c3'),
             ('π0.5 / APX BF16 total','results/engine/pi05_bf16_ptqad_20260929.json','total_ms_p50',BLUE),
             ('GR00T / PT get_action','results/baselines/gr00t_pt_bf16_ptqad_20260929.json','latency_ms_p50','#a3b6c3'),
             ('GR00T / APX BF16 total','results/engine/gr00t_bf16_ptqad_20260929.json','total_ms_p50',TEAL)]
    rows=[]
    for label,path,metric,color in sources:
        record=read(path)
        if path.startswith('results/baselines/'):
            timing_record(record,path,'latency_ms_p50','lat_all_ms')
            expected_entry=('PI05Policy.predict_action_chunk' if 'pi05_' in path else 'Gr00tPolicy.get_action')
            if record.get('schema_version')!=2 or record.get('dtype_config')!='bfloat16':
                raise ValueError(f'{path}: expected audited BF16-config baseline schema v2')
            if record.get('timing_scope',{}).get('entrypoint')!=expected_entry:
                raise ValueError(f'{path}: timing entrypoint differs from its plotted label')
            dtypes=record.get('parameter_dtype_elements',{})
            if not dtypes or dtypes.get('bfloat16',0)<=0 or any(type(v) is not int or v<=0 for v in dtypes.values()):
                raise ValueError(f'{path}: missing/invalid measured parameter dtype inventory')
            if not record.get('compute_dtype',{}).get('linear_io_signatures'):
                raise ValueError(f'{path}: missing observed linear compute dtypes')
        else:
            model='pi05' if 'pi05_' in path else 'gr00t'
            engine_record(record,path,model,'bf16')
            if model=='pi05': matching_native_protocol(record,native)
        value=positive_number(record[metric],path+':'+metric)
        n=positive_number(record['n'],path+':n')
        if int(n)!=n:raise ValueError(f'{path}: n must be an integer')
        rows.append((label,value,int(n),color))
    nv=positive_number(native['total_ms_p50'],native_path+':total_ms_p50')
    n=positive_number(native['n'],native_path+':n')
    if int(n)!=n:raise ValueError(f'{native_path}: n must be an integer')
    rows.append(('π0.5 / APX FP4 total',nv,int(n),AMBER))
    maximum=max(value for _,value,_,_ in rows)*1.12
    s=SVG(1120,635,'独立计时入口的 P50：调用边界分别标注','batch = 1 · ms · BF16 配置（π0.5 含 FP32 子模块）· APX total = policy.infer')
    left,width=295,615
    for tick in range(5):
        x=left+width*tick/4;s.line(x,98,x,462);s.text(x,485,f'{maximum*tick/4:.0f}',12,anchor='middle')
    for i,(label,value,count,color) in enumerate(rows):
        y=116+i*65;s.text(30,y+22,label,16)
        s.rect(left,y,width*value/maximum,31,color)
        s.text(left+width*value/maximum+10,y+14,f'{value:.2f} ms',15,color,600)
        s.text(left+width*value/maximum+10,y+34,f'n = {count}',12)
    s.text(30,523,f'同一 APX π0.5 入口：BF16 / NVFP4 的实测耗时比 = {rows[1][1]/rows[4][1]:.2f}',17,BLUE,600)
    s.rect(30,545,1060,65,'#fff6e9')
    s.text(48,571,'π0.5 PT 未使用 state-to-text；输入准备与计算内容不同，浮点入口仅分别报告耗时。',14,AMBER)
    s.text(48,594,'跨入口不计算加速比；图中耗时比只比较同一 APX π0.5 接口的 BF16 与 NVFP4 路径。',13,AMBER)
    s.save('e1_latency')


def paired_rows():
    path='paper/evidence/paired_comparison.json'
    data=read(path)
    if data.get('environment_pairing_verified') is not True:
        raise ValueError(f'{path}: environment pairing was not verified')
    order=('bf16','ptq','qad','continued_qad','qad_opd')
    arms=data['arms']
    if set(arms)!=set(order):raise ValueError(f'{path}: expected exactly five completed arms')
    reference=None;rows=[]
    for key in order:
        arm=arms[key];count=arm['count'];successes=arm['successes'];tasks=arm['per_task']
        if type(count) is not int or type(successes) is not int or not 0<=successes<=count or count<=0:
            raise ValueError(f'{key}: invalid successes/count')
        if len(tasks)!=10:raise ValueError(f'{key}: expected ten completed tasks')
        task_counts={}
        for task,result in tasks.items():
            n=result['episodes'];k=result['successes']
            if type(n) is not int or type(k) is not int or not 0<=k<=n or n<=0:
                raise ValueError(f'{key}/{task}: invalid successes/episodes')
            if not math.isclose(result['success_rate'],k/n,abs_tol=1e-12):
                raise ValueError(f'{key}/{task}: inconsistent success rate')
            task_counts[task]=n
        if sum(task_counts.values())!=count or sum(t['successes'] for t in tasks.values())!=successes:
            raise ValueError(f'{key}: per-task totals disagree')
        if len(set(task_counts.values()))!=1:
            raise ValueError(f'{key}: unequal task counts require a separate macro chart')
        if reference is None:reference=task_counts
        elif task_counts!=reference:raise ValueError(f'{key}: task denominators differ across arms')
        rate=successes/count
        macro=sum(t['success_rate'] for t in tasks.values())/10
        if not math.isclose(arm['macro_success_rate'],macro,abs_tol=1e-12):
            raise ValueError(f'{key}: macro average disagrees')
        rows.append((key,successes,count,rate))
    for key in ('qad','continued_qad'):
        difference=arms['qad_opd']['successes']/arms['qad_opd']['count']-arms[key]['successes']/arms[key]['count']
        comparison=data[f'opd_vs_{key}']
        if comparison['paired_count']!=arms[key]['count'] or not math.isclose(
                comparison['success_rate_difference_opd_minus_baseline'],difference,abs_tol=1e-12):
            raise ValueError(f'{key}: paired comparison disagrees with arm totals')
    return data,rows


def ladder_chart():
    data,rows=paired_rows()
    labels={'bf16':'BF16','ptq':'PTQ','qad':'QAD','continued_qad':'继续 QAD','qad_opd':'QAD + OPD'}
    colors={'bf16':'#7e97a8','ptq':AMBER,'qad':BLUE,'continued_qad':'#6098bf','qad_opd':TEAL}
    s=SVG(1100,635,'同协议闭环结果：量化、演示恢复与后续蒸馏','LIBERO-10 · 五臂逐集环境初态配对已验真 · 每行给出成功数 / 实际回合数')
    left,width=235,615
    for tick in range(0,101,25):
        x=left+width*tick/100;s.line(x,101,x,448);s.text(x,475,f'{tick}%',12,anchor='middle')
    for i,(key,successes,count,rate) in enumerate(rows):
        y=118+i*65;color=colors[key];s.text(30,y+22,labels[key],19,color,600)
        if rate:s.rect(left,y,width*rate,32,color)
        else:s.line(left,y,left,y+32,color)
        s.text(left+width*rate+10,y+21,f'{rate*100:.1f}%  ({successes}/{count})',16,color,600)
    delta_qad=data['opd_vs_qad']['success_rate_difference_opd_minus_baseline']*100
    delta_continued=data['opd_vs_continued_qad']['success_rate_difference_opd_minus_baseline']*100
    s.rect(30,503,1040,69,'#e8f2ee')
    s.text(48,531,f'OPD − QAD：{delta_qad:+.1f} 个百分点',18,TEAL,600)
    s.text(553,531,f'OPD − 继续 QAD：{delta_continued:+.1f} 个百分点',18,TEAL,600)
    s.text(48,557,'各任务回合数相等，总成功率等于任务宏平均；逐任务结果与配对差异见正文。',13)
    s.text(30,603,'两条续训分支共享 QAD 起点与演示更新预算；OPD 另用学生访问数据、教师标签和探针计算。',14)
    s.save('ladder')


def budget_rows():
    path='paper/evidence/recipe_inventory.json'
    data=read(path)
    order=('fp8','head_ffn','head_lang','head_lang_vision','calib')
    if tuple(data['ladder'])!=order:raise ValueError(f'{path}: unexpected nested ladder')
    residual=data['recovery_residual']
    if residual['scope']!='head+lang_all' or residual['dtype']!='bfloat16' or residual['rank']!=32:
        raise ValueError(f'{path}: unsupported recovery budget scope/dtype/rank')
    extra=positive_number(residual['target_bytes'],'residual target_bytes')
    if extra!=residual['tensor_elements']*residual['bytes_per_element'] or residual['bytes_per_element']!=2:
        raise ValueError(f'{path}: inconsistent residual byte budget')
    rows=[];sources=None;previous=0
    for key in order:
        record=data['recipes'][key];unique=record['known_tied_alias_deduplicated']
        physical_source=positive_number(record['source_tensor_bytes'],key+':physical source')
        unique_source=positive_number(unique['source_tensor_bytes'],key+':deduplicated source')
        physical_target=positive_number(record['target_full_checkpoint_bytes'],key+':physical target')
        unique_target=positive_number(unique['target_full_bytes'],key+':deduplicated target')
        if sources is None:sources=(physical_source,unique_source)
        elif sources!=(physical_source,unique_source):raise ValueError('Ladder source byte denominators changed')
        if record['nvfp4_params']<previous:raise ValueError('NVFP4 coverage decreases in nested ladder')
        previous=record['nvfp4_params']
        if not math.isclose(physical_source/physical_target,record['full_checkpoint_compression_x'],rel_tol=1e-12):
            raise ValueError(f'{key}: inconsistent physical compression')
        if not math.isclose(unique_source/unique_target,unique['full_compression_x'],rel_tol=1e-12):
            raise ValueError(f'{key}: inconsistent deduplicated compression')
        fractions=(record['fraction_of_eligible_params']['nvfp4'],unique['fraction_of_eligible']['nvfp4'])
        if any(not 0<=v<=1 for v in fractions):raise ValueError(f'{key}: invalid coverage fraction')
        rows.append({'key':key,'sources':sources,'targets':(physical_target,unique_target),'fractions':fractions})
    return data,rows,extra


def budget_ladder():
    data,rows,extra=budget_rows()
    s=SVG(1420,775,'混合量化阶梯：完整权重的目标编码预算','GB = 10⁹ 字节 · 含未量化张量与格式缩放 · 单独叠加相同的 BF16 LoRA 旁路预算')
    for x,color,label in [(30,BLUE,'物理张量口径'),(267,TEAL,'已知共享别名去重'),(555,AMBER,f'独立 BF16 旁路：{extra/1e6:.2f} MB')]:
        s.rect(x,91,18,18,color);s.text(x+28,106,label,15)
    maximum=max(max(row['targets'])+extra for row in rows)/1e9*1.07
    for panel,(heading,color) in enumerate([('物理 checkpoint：保留两份共享权重',BLUE),('已知别名去重：共享权重仅计一次',TEAL)]):
        origin=30+panel*705;left=origin+185;width=305
        source=rows[0]['sources'][panel]
        s.text(origin,150,heading,18,color,600)
        s.text(origin,178,f'BF16 源字节分母：{source/1e9:.3f} GB',14)
        for tick in range(4):
            x=left+width*tick/maximum;s.line(x,213,x,640);s.text(x,204,str(tick),12,anchor='middle')
        for i,row in enumerate(rows):
            y=231+i*82;target=row['targets'][panel];fraction=row['fractions'][panel]
            s.text(origin,y+16,row['key'],15,color,600)
            s.text(origin,y+38,f'候选 FP4：{fraction*100:.2f}%',12)
            base_width=width*(target/1e9)/maximum;extra_width=width*(extra/1e9)/maximum
            s.rect(left,y,base_width,23,color,rx=0);s.rect(left+base_width,y,extra_width,23,AMBER,rx=0)
            s.text(left+base_width+extra_width+8,y+17,f'{(target+extra)/1e9:.3f} GB',13,color,600)
            s.text(left,y+47,f'PTQ {source/target:.3f}× → 加旁路 {source/(target+extra):.3f}×',13)
    s.rect(30,665,1360,83,'#fff6e9')
    s.text(48,692,f'旁路：head+lang_all，rank = 32，{data["recovery_residual"]["linear_modules"]} 个 Linear；两种口径均完整计入旁路。',15,AMBER)
    s.text(48,716,'各配方统一叠加相同旁路以比较预算，不表示每一级均完成恢复训练；该图不包含激活、优化器、对齐和运行时工作区。',14,AMBER)
    s.text(48,738,'当前闭环采用稠密 BF16 合并 checkpoint；此处是保留 packed 基座与独立旁路时的编码预算，不是已测文件或显存压缩。',14,AMBER)
    s.save('budget_ladder')
def swizzle_layout():
    # 128 rows × 8 scale blocks: verify a bijection over two 512-byte tiles.
    def off(r,b,kb=8):return (r//128)*512*((kb+3)//4)+512*(b//4)+16*(r%32)+4*((r//32)%4)+b%4
    assert sorted(off(r,b) for r in range(128) for b in range(8)) == list(range(1024))
    s=SVG(1080,640,'NVFP4 scale：从逻辑索引到物理字节','每个逻辑 scale 管理 16 个连续权重；128 行 × 4 个 scale 块组成 512 字节 tile。')
    s.text(30,110,'逻辑 scale[r, b]：同一行的 4 个块',18,BLUE,600)
    colors=['#d8e9f4','#d9eee9','#f6e7cb','#e6def2']
    rs=[0,32,64,96]
    for i,r in enumerate(rs):
        y=132+i*58;s.text(34,y+27,f'r = {r}',15)
        for b in range(4):
            x=125+b*71;s.rect(x,y,64,40,colors[i]);s.text(x+32,y+26,f'b={b}',14,anchor='middle')
    s.text(460,206,'→',35,BLUE)
    s.text(520,110,'物理缓冲：前 16 个字节',18,BLUE,600)
    for i,r in enumerate(rs):
        for b in range(4):
            k=4*i+b;x=520+(k%8)*62;y=152+(k//8)*105
            s.rect(x,y,56,46,colors[i]);s.text(x+28,y+19,f'{off(r,b)} B',13,anchor='middle');s.text(x+28,y+37,f'{r},{b}',11,anchor='middle')
    s.text(520,375,'第 16～31 字节依次放 r=1,33,65,97 的 b=0…3。',13)
    s.text(520,399,'b=4…7 的下一个 tile 从字节 512 开始。',13)
    s.rect(30,429,1020,112)
    s.text(48,457,'KB = K / 16；P = 512 × ceil(KB / 4)；// 表示整数除法。',16,BLUE)
    s.text(48,487,'offset(r,b) = P × (r // 128) + 512 × (b // 4)',18)
    s.text(48,517,'                         + 16 × (r % 32) + 4 × ((r // 32) % 4) + (b % 4)',17)
    s.text(30,573,'例：(0,0)→0；(32,0)→4；(1,0)→16；(3,2)→50；(0,5)→513。',15)
    s.text(30,603,'这里只说明可验证的布局映射。普通行主序的同一行相邻块本来就连续；不据此猜测内部 warp 读法。',13)
    s.save('swizzle_layout')

def gptq_block():
    s=SVG(1080,840,'GPTQ 块适配：固定张量缩放，逐列补偿误差','X:[R,K]；W′:[N,K]；H、U:[K,K]。r 是输出行，i 是当前输入列，j>i 是待处理列。')
    stages=[('校准与预处理','H = XᵀX；处理零通道','阻尼后令 UᵀU = H⁻¹'),('每 16 列进入新块','用已补偿 W′ 选择块 scale','张量 scale τ 全程固定'),('量化当前列 i','按本块有效 scale 得到 q','计算未归一化误差 δ'),('补偿其余列并提交','δ 除以 U[i,i]，再乘 U[i,j]','提交当前 q，再处理 i+1')]
    for i,(title,l1,l2) in enumerate(stages):
        x=30+i*263;s.rect(x,104,237,134,'#f0f6f8');s.text(x+15,137,title,17,BLUE,600);s.text(x+15,173,l1,13);s.text(x+15,204,l2,13)
        if i<3:s.text(x+243,180,'→',20,BLUE)
    s.rect(30,260,1020,326)
    equations=[
        'τ = FP32(max(c · max|W| / 448, 2⁻¹⁴⁹))；全零矩阵取 τ = 1。',
        'b = floor(i/16)；s[r,b] = R8(c · max|W′[r,16b:16b+16]| / (6τ))',
        'σ64[r] = FP64(s[r,b]) · FP64(τ)；σ32[r] = FP32(s[r,b] · τ)',
        'q[r,i] = σ32[r] · R4(W′[r,i] / σ64[r])；σ64[r] = 0 时 q[r,i] = 0。',
        'δ[r,i] = W′[r,i] − q[r,i]',
        '对每个 j > i：W′[r,j] ← W′[r,j] − (δ[r,i] / U[i,i]) · U[i,j]',
        '最后提交：W′[r,i] ← q[r,i]。U 为上三角因子，满足 UᵀU = H⁻¹。',
    ]
    for i,line in enumerate(equations):s.text(49,296+i*41,line,16,INK)
    s.text(30,629,'先选裁剪系数 c：',17,BLUE,600)
    for i,c in enumerate(['1.00','.95','.90','.85','.80','.70','.60','.50']):
        x=240+i*97;s.rect(x,603,82,38,'#dcebf2');s.text(x+41,629,c,16,anchor='middle')
    s.text(30,667,'对 RTN 候选计算 tr(ΔW · H · ΔWᵀ)，选定 c 后固定 τ，仅运行一次 GPTQ。',16)
    s.text(30,703,'阻尼：H ← H + 0.01 · mean(diag H) · I；统计与阻尼均以当前层为单位。',14)
    s.rect(30,732,1020,82,'#fff6e9')
    s.text(48,760,'R4：E2M1 最近偶数舍入；R8：E4M3 最近偶数舍入。σ64 用于选码，σ32 用于反量化。',13,AMBER)
    s.text(48,786,'原始 W 确定 τ；补偿后的 W′ 确定块缩放 s；s 在块内固定。校准与闭环分别验收。',13,AMBER)
    s.save('gptq_block')

def recovery_protocol():
    s=SVG(1120,750,'从量化基座到 QAD，再到学生访问状态上的 OPD','开发、采集与最终评测采用不相交的官方初态索引；实线表示数据或参数依赖。')
    def box(x,y,w,h,title,lines,color=BLUE):
        s.rect(x,y,w,h,'#f0f6f8',color)
        s.text(x+16,y+30,title,18,color,600)
        for i,line in enumerate(lines):s.text(x+16,y+59+22*i,line,14)
    def arrow(points,color=BLUE):
        s.s.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="2"/>')
        (a,b),(x,y)=points[-2:]
        if x==a:s.s.append(f'<path d="M{x-5},{y-8} L{x},{y} L{x+5},{y-8}" stroke="{color}" fill="none"/>')
        else:s.s.append(f'<path d="M{x-8},{y-5} L{x},{y} L{x-8},{y+5}" stroke="{color}" fill="none"/>')
    box(30,102,270,132,'① PTQ 量化基座',['冻结低比特取值 Wq 与偏置 b','配方与缩放清单固定'])
    box(395,102,300,132,'② QAD：演示监督',['只更新低秩 A、B'])
    # Both transposes are required: F.linear(x,A)=(x A^T), then
    # F.linear(z,B)=z B^T. Keep each transpose attached to its own matrix.
    s.text(411,184,'Δy = (α/r) · (xAᵀ)Bᵀ',14)
    # Use an ordinary q to denote Wq, avoiding an ambiguous adjacent subscript
    # and superscript at different baselines in the same short expression.
    s.text(411,209,'y = x(Wq)ᵀ + b + Δy',14)
    box(790,102,300,132,'③ 学生闭环轨迹',['运行 QAD 策略并采集观测','保存完整动作端点及有效 mask'])
    arrow([(300,168),(395,168)]);arrow([(695,168),(790,168)])
    box(790,285,300,128,'④ 未量化教师标注',['师生共享噪声与时间步','缓存速度场、mask 和重放种子','教师无梯度；学生探针可求导'],TEAL)
    arrow([(940,234),(940,285)],TEAL)
    box(30,474,470,118,'⑤ 对照：继续训练 QAD',['从②的同一适配器开始；重置优化器','相同演示批量、额外步数与学习率计划'])
    box(620,474,470,118,'⑥ 实验：QAD + OPD',['演示目标 + 有效动作上的教师 MSE','主损失与探针顺序反传，控制峰值显存'],TEAL)
    arrow([(470,234),(470,260),(540,260),(540,450),(265,450),(265,474)])
    arrow([(620,234),(620,260),(725,260),(725,474)])
    arrow([(940,413),(940,474)],TEAL)
    s.text(31,303,'共享起点与演示优化预算',18,BLUE,600)
    s.text(31,334,'OPD 另用教师标注和学生探针计算。',15)
    s.text(31,361,'因此单独报告标注、训练耗时及峰值显存。',15)
    s.text(31,399,'矩阵形状：x:[M,K]；A:[r,K]；B:[N,r]',14)
    s.text(31,425,'Wq:[N,K]；b:[N]；y、Δy:[M,N]（偏置按行广播）',14)
    s.rect(30,637,1060,88,'#e8f2ee')
    s.text(52,668,'统一最终评测：BF16 · PTQ · QAD · 继续 QAD · QAD+OPD',19,TEAL,600)
    s.text(52,698,'配对任务和稳定后初态；同一执行预算。报告宏平均、实际分子分母及逐任务结果。',14)
    arrow([(265,592),(265,637)]);arrow([(855,592),(855,637)],TEAL)
    s.save('recovery_protocol')

def frontier_rows():
    data=read('paper/evidence/frontier_comparison.json')
    if data.get('version')!=1 or data.get('status')!='complete' or data.get('environment_pairing_verified') is not True:
        raise ValueError('Frontier requires a completed, paired current comparison')
    points=data['points'];names=[row['name'] for row in points]
    expected=['bf16','fp8','head_ffn','ptq','qad','continued_qad','qad_opd']
    if names!=expected or set(data['references'])!={'fp8','head_ffn'}:
        raise ValueError('Frontier requires all seven frozen current candidates')
    denominators={};residual=None
    for row in points:
        if row['count']!=100 or type(row['successes']) is not int or not 0<=row['successes']<=100:
            raise ValueError('Frontier counts must describe 100 completed heldout episodes')
        tasks=row['per_task']
        if len(tasks)!=10 or any(value['episodes']!=10 or type(value['successes']) is not int or not 0<=value['successes']<=10 or value['success_rate']!=value['successes']/10 for value in tasks.values()):
            raise ValueError('Frontier task counts/rates differ')
        if sum(value['successes'] for value in tasks.values())!=row['successes'] or not math.isclose(row['macro_success_rate'],row['successes']/100,abs_tol=1e-12):
            raise ValueError('Frontier macro or total differs from per-task outcomes')
        recovery=row['name'] in ('qad','continued_qad','qad_opd')
        if set(row['encoding_budget'])!={'physical','known_alias_deduplicated'}:
            raise ValueError('Frontier requires both encoding denominators')
        for scope,cost in row['encoding_budget'].items():
            if any(type(cost[k]) is not int or cost[k]<0 for k in ('source_bytes','base_bytes','residual_bytes','total_bytes')) or cost['source_bytes']<=0 or cost['base_bytes']<=0 or cost['total_bytes']!=cost['base_bytes']+cost['residual_bytes']:
                raise ValueError('Invalid frontier net encoding bytes')
            if not math.isclose(cost['compression_x'],cost['source_bytes']/cost['total_bytes'],rel_tol=1e-12):
                raise ValueError('Frontier compression ratio differs from net bytes')
            if denominators.setdefault(scope,cost['source_bytes'])!=cost['source_bytes']:
                raise ValueError('Frontier uses inconsistent source denominators')
            if recovery:
                if cost['residual_bytes']<=0:raise ValueError('Recovered point omits residual bytes')
                if residual is None:residual=cost['residual_bytes']
                if cost['residual_bytes']!=residual:raise ValueError('Recovered residual costs differ')
                if cost['base_bytes']!=points[3]['encoding_budget'][scope]['base_bytes']:
                    raise ValueError('Recovered point uses another PTQ base budget')
            elif cost['residual_bytes']!=0:raise ValueError('Pure PTQ/BF16 point includes a residual')
    return data,points


def ptq_frontier():
    _,points=frontier_rows()
    s=SVG(1120,990,'纯 PTQ 与恢复策略：净编码预算—闭环成功率',
          '七个冻结候选均为同协议 100 回合；描述本次观测，不代表全局 PTQ 最优或统计非劣性。')
    scope='known_alias_deduplicated';left,right,top,bottom=100,1040,124,505
    maximum=max(row['encoding_budget'][scope]['total_bytes']/1e9 for row in points)*1.06
    def x(value):return left+(right-left)*value/maximum
    def y(value):return bottom-(bottom-top)*value/100
    for value in (0,20,40,60,80,100):
        s.line(left,y(value),right,y(value));s.text(left-14,y(value)+5,str(value)+'%',13,anchor='end')
    for value in range(math.floor(maximum)+1):
        s.line(x(value),bottom,x(value),bottom+5);s.text(x(value),bottom+24,str(value),13,anchor='middle')
    s.line(left,top,left,bottom,INK);s.line(left,bottom,right,bottom,INK)
    s.text(left,103,'十任务宏平均成功率',15,BLUE,600)
    s.text(570,558,'已知共享副本去重后的净编码预算 / GB（10⁹ 字节）',15,anchor='middle')
    labels={'bf16':'BF16','fp8':'fp8 纯 PTQ','head_ffn':'head_ffn 纯 PTQ','ptq':'head_lang 纯 PTQ',
            'qad':'QAD','continued_qad':'继续 QAD','qad_opd':'QAD + OPD'}
    groups={}
    for i,row in enumerate(points):
        key=(row['encoding_budget'][scope]['total_bytes'],row['successes'])
        groups.setdefault(key,[]).append((chr(65+i),row))
    for index,((size,score),group) in enumerate(groups.items()):
        px,py=x(size/1e9),y(score);role=group[0][1]['role']
        color=INK if role=='bf16' else TEAL if role=='recovery' else BLUE
        if role=='recovery':s.s.append(f'<path d="M{px},{py-7} L{px+7},{py} L{px},{py+7} L{px-7},{py} Z" fill="{color}"/>')
        else:s.s.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{color}"/>')
        # Labels can share one marker when both measured coordinates coincide.
        offset=(-19,30,-38,49)[index%4]
        if py+offset<top-16:offset=abs(offset)+12
        s.line(px,py,px+13,py+offset-5,color)
        s.text(px+17,py+offset,','.join(code for code,_ in group),14,color,600)
    s.text(30,596,'圆点：BF16 / 纯 PTQ　　菱形：基座 + BF16 低秩残差；重合点合并字母，精确值见下表。',14)
    columns=[42,95,340,514,683,866,1050]
    headers=['点','策略','成功数','成功率','去重净 GB','物理净 GB','去重压缩']
    s.rect(30,621,1060,34,'#e7f0f5')
    for at,label in zip(columns,headers):s.text(at,644,label,14,BLUE,600,anchor='start' if at<300 else 'end')
    for i,row in enumerate(points):
        yy=682+i*34;cost=row['encoding_budget'];known=cost[scope]
        if i%2==0:s.rect(30,yy-24,1060,31,'#f5f8fa')
        values=[chr(65+i),labels[row['name']],f'{row["successes"]}/{row["count"]}',f'{100*row["macro_success_rate"]:.1f}%',
                f'{known["total_bytes"]/1e9:.6f}',f'{cost["physical"]["total_bytes"]/1e9:.6f}',f'{known["compression_x"]:.4f}×']
        for at,value in zip(columns,values):s.text(at,yy,value,14,anchor='start' if at<300 else 'end')
    s.text(30,953,'预算含格式尺度、未量化张量及恢复残差；不等于 BF16 合并文件大小、实测显存或完整原生部署。',14,AMBER)
    s.save('ptq_frontier')


def write_figure_manifest(source):
    """Only called after every figure generator succeeds in the current run."""
    required={
        'paper/evidence/paired_comparison.json',
        'paper/evidence/recipe_inventory.json',
        'paper/evidence/frontier_comparison.json',
        'results/engine/pi05_nvfp4_ptqad_20260929.json',
        'results/baselines/pi05_pt_bf16_ptqad_20260929.json',
        'results/baselines/gr00t_pt_bf16_ptqad_20260929.json',
        'results/engine/pi05_bf16_ptqad_20260929.json',
        'results/engine/gr00t_bf16_ptqad_20260929.json',
    }
    if not required.issubset(_DATA_INPUTS):
        raise RuntimeError(f'Figure generation did not consume required inputs: {sorted(required-set(_DATA_INPUTS))}')
    if file_record(pathlib.Path(__file__))!=source:
        raise RuntimeError('make_figs.py changed during generation')
    for path,record in _DATA_INPUTS.items():
        if file_record(ROOT/path)!=record:
            raise RuntimeError(f'Figure input changed during generation: {path}')
    all_svgs={path.resolve() for path in FIGS.glob('*.svg') if not path.stem.startswith('shot_')}
    expected={FIGS.joinpath(name+'.svg').resolve() for name in
              ('e1_latency','ladder','budget_ladder','swizzle_layout','gptq_block','recovery_protocol','ptq_frontier')}
    if _GENERATED_SVGS!=expected or all_svgs!=expected:
        raise RuntimeError('Unexpected or unregenerated non-screenshot SVGs; refusing a passed manifest')
    manifest={'version':1,'status':'passed','generated_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'All non-screenshot SVGs generated from the recorded current inputs in one successful run.',
              'generator':source,
              'inputs':[_DATA_INPUTS[path] for path in sorted(_DATA_INPUTS)],
              'svgs':[file_record(path) for path in sorted(all_svgs)]}
    output=P/'validation/figure-inputs.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(output)
    return output


def main():
    _DATA_INPUTS.clear();_GENERATED_SVGS.clear()
    source=file_record(pathlib.Path(__file__))
    FIGS.mkdir(exist_ok=True)
    e1_chart();ladder_chart();budget_ladder();swizzle_layout();gptq_block();recovery_protocol();ptq_frontier()
    manifest=write_figure_manifest(source)
    print(f'Generated all SVG figures; provenance: {manifest}; screenshot files untouched.')

if __name__=='__main__':main()
