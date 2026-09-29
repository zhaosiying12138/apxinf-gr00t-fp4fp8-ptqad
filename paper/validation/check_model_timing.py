#!/usr/bin/env python3
"""Independent stdlib-only review of five current timing JSON files.

Does not import model/GPU libraries, rerun inference, or rehash large weights.
Recorded checkpoint and extension identities are compared as declarations in
these exact JSON bytes; original measurement/build evidence remains necessary.
"""
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
INPUTS=(
    'results/engine/pi05_bf16_ptqad_20260929.json',
    'results/engine/pi05_nvfp4_ptqad_20260929.json',
    'results/engine/gr00t_bf16_ptqad_20260929.json',
    'results/baselines/pi05_pt_bf16_ptqad_20260929.json',
    'results/baselines/gr00t_pt_bf16_ptqad_20260929.json',
)
PRODUCERS=('exp/bench_engine.py','baselines/bench_pi05_lerobot.py','baselines/bench_gr00t_pt.py')


def require(condition,message):
    if not condition:raise ValueError(message)


def identity(path):
    return {'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def linear_percentile(values,q):
    ordered=sorted(values);position=(len(ordered)-1)*q
    left=math.floor(position);right=math.ceil(position)
    return ordered[left]+(ordered[right]-ordered[left])*(position-left)


def finite_positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def summarize(data,raw,prefix):
    values=data.get(raw)
    require(type(data.get('n')) is int and data['n']>0 and isinstance(values,list)
            and len(values)==data['n'],'Raw samples/count differ')
    require(all(finite_positive(value) for value in values),'Nonfinite or nonpositive sample')
    calculated={'p50':linear_percentile(values,.5),'p99':linear_percentile(values,.99),
                'mean':statistics.fmean(values)}
    for suffix,value in calculated.items():
        require(finite_positive(data.get(prefix+suffix)) and
                math.isclose(value,data[prefix+suffix],rel_tol=1e-10,abs_tol=1e-10),
                'Reported statistic differs: '+prefix+suffix)
    return {'raw_key':raw,'sample_count':len(values),'recomputed_ms':calculated,
            'sample_order_preserved_in_source':True}


def declared_content(value):
    if isinstance(value,dict):
        return {k:declared_content(v) for k,v in value.items() if k not in {'path','resolved_path'}}
    if isinstance(value,list):return [declared_content(v) for v in value]
    return value


def check_records(records,producer_sha):
    checks=[]
    for name,data in records.items():
        require(data.get('schema_version')==2 and data.get('all_outputs_finite') is True,
                'Current finite-output schema2 required: '+name)
        require(type(data.get('warmup')) is int and data['warmup']>=0,'Invalid warmup')
        require(data.get('quantile_method')=='numpy.percentile linear','Unknown quantile method')
        baseline='/baselines/' in name
        if baseline:
            expected='PI05Policy.predict_action_chunk' if 'pi05_' in name else 'Gr00tPolicy.get_action'
            require(data['timing_scope']['entrypoint']==expected,'Wrong baseline entrypoint')
            require(data.get('dtype_config')=='bfloat16','Wrong requested baseline precision')
            dtype=data.get('parameter_dtype_elements',{})
            require(dtype.get('bfloat16',0)>0 and all(type(n) is int and n>0 for n in dtype.values()),
                    'Invalid observed parameter dtype inventory')
            require(data.get('compute_dtype',{}).get('linear_io_signatures'),'Missing observed compute dtype')
            producer=PRODUCERS[1] if 'pi05_' in name else PRODUCERS[2]
            require(data['script_sha256']==producer_sha[producer],'Baseline producer changed')
            if 'pi05_' in name:
                require(data['shared_helper_sha256']==producer_sha[PRODUCERS[2]],'Shared baseline helper changed')
            stats=[summarize(data,'lat_all_ms','latency_ms_')]
        else:
            require(data['timing_scope']['entrypoint']=='AutoPolicy.infer','Wrong engine entrypoint')
            require(data['runtime_identity']['benchmark_script']['sha256']==producer_sha[PRODUCERS[0]],
                    'Engine producer changed')
            require(data['runtime_identity']['native_extensions'],'Missing measured extension identity')
            expected_mode='cuda-graph' if data['model_type']=='gr00t' else 'graph'
            modes=data.get('execution_mode',{})
            require(modes.get('after_warmup')==expected_mode and
                    modes.get('after_each_timed_sample')==[expected_mode]*data['n'],
                    'Missing or differing observed graph state')
            stats=[summarize(data,'lat_'+kind+'_ms',kind+'_ms_') for kind in ('model','total','wrapper')]
        checks.append({'file':name,'finite_samples':True,'statistics':stats,
                       'timing_scope':data['timing_scope'],
                       'observed_parameter_dtype_elements':data.get('parameter_dtype_elements'),
                       'observed_compute_dtype':data.get('compute_dtype'),
                       'recorded_execution_mode':data.get('execution_mode')})
    a,b=(records[INPUTS[i]] for i in (0,1))
    require(a['variant']=='bf16' and b['variant']=='nvfp4_static','Wrong same-entrypoint pair')
    keys=('checkpoint_identity','asset_identity','seed','model_seed','device','gpu_identity',
          'n','warmup','input_contract','timing_scope','action_shape','first_token_count',
          'first_observation_sha256','sample_token_counts')
    for key in keys:
        require(a.get(key) is not None and declared_content(a[key])==declared_content(b.get(key)),
                'APX pi05 pair differs: '+key)
    for key in ('constructor_kwargs','policy_metadata'):
        clean=lambda d:{k:v for k,v in d.items() if k not in ('model_variant','precision')}
        require(clean(a[key])==clean(b[key]),'APX pi05 configuration differs')
    require(a['runtime_identity']['native_extensions']==b['runtime_identity']['native_extensions'],
            'APX pi05 native extension differs')
    return checks, {'checked_record_fields':list(keys)+['constructor_kwargs_except_variant',
        'policy_metadata_except_variant','runtime_native_extensions'],
        'all_declarations_match':True,'cross_library_speedup_computed':False,
        'boundary':'Checks exact JSON declarations; does not reread private weights or infer behavioral parity.'}


def main():
    bindings={name:identity(ROOT/name) for name in (*INPUTS,*PRODUCERS)}
    checker=identity(Path(__file__))
    records={name:json.loads((ROOT/name).read_text()) for name in INPUTS}
    checks,pair=check_records(records,{name:bindings[name]['sha256'] for name in PRODUCERS})
    require(checker==identity(Path(__file__)) and all(value==identity(ROOT/name) for name,value in bindings.items()),
            'Checker/input/source changed during review')
    report={'status':'passed','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Independent stdlib linear-percentile/mean recomputation and same-entrypoint declaration comparison; no inference, CUDA import, or large-weight read.',
            'checker':checker,'input_files':[bindings[n] for n in INPUTS],
            'benchmark_source_files':[bindings[n] for n in PRODUCERS],
            'environment':{'python':platform.python_version(),'executable':sys.executable,'dependencies':'Python standard library only'},
            'checks':checks,'native_pi05_pairing':pair,
            'reproduction':'python3 -B -I -S paper/validation/check_model_timing.py'}
    output=Path(__file__).with_name('model_timing_check.json');temp=output.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(output)
    print(json.dumps({'status':'passed','inputs':len(INPUTS),'raw_latency_series':sum(len(x['statistics']) for x in checks),
                      'same_apx_pi05_declarations_match':True,'gpu':False}))

if __name__=='__main__':main()
