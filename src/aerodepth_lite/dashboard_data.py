"""Display adapters for immutable study evidence. No evaluation entry points."""
from functools import lru_cache
from pathlib import Path
import copy
import csv
import json
import math
from . import presentation_ui as backend

STUDIES=['Independent test','Validation','Historical exposed test']
MODELS={'Original':'Original','A':'Supervised A','KD':'Exploratory KD'}
COLORS={'Original':'#64748b','A':'#2563eb','KD':'#d59a22'}
POLICIES={'every':'Every frame','fixed2':'Fixed 2','fixed5':'Fixed 5','budget_fixed':'Budget fixed','v2':'V2',
          'v2_no_motion':'V2 · no motion','v2_no_visual':'V2 · no visual','v2_no_risk':'V2 · no risk','v2_no_age':'V2 · no age'}
INDEPENDENT='results/independent_test_20261030'


def stamp(path):
    path=Path(path)
    try:
        s=path.stat();return (str(path),s.st_mtime_ns,s.st_size)
    except OSError:return (str(path),None,None)


@lru_cache(maxsize=96)
def _json(identity):return backend.optional_json(identity[0])


def json_file(path):return _json(stamp(path))


@lru_cache(maxsize=24)
def _csv(identity):
    try:
        with Path(identity[0]).open(newline='') as f:return list(csv.DictReader(f))
    except OSError:return []


def csv_file(path):return _csv(stamp(path))


@lru_cache(maxsize=4)
def _independent(root,signature):return backend.independent_study(root)


def verified_independent(root):
    folder=Path(root)/INDEPENDENT
    names=['COMPLETION.json','PROTOCOL.json','EXPOSURE.json','FAILED.json','run/progress.json']
    complete=json_file(folder/'COMPLETION.json') or {}
    names+=list(complete.get('artifacts_sha256',{}))
    return _independent(str(root),tuple(stamp(folder/n) for n in names))


def study_bundle(root,config,study):
    """Normalize only recorded fields; missing timing/accounting stay absent."""
    root=Path(root);entries={};status='COMPLETED';timing=None
    if study=='Independent test':
        saved=verified_independent(root);status=saved['status'];folder=root/INDEPENDENT
        for key,value in (saved['results'] or {}).items():
            model,policy=key.split('_',1);entries[(model,policy)]=value
        timing=saved.get('timing');scope='Full trajectories · actual model calls · CPU, one FP32 pass'
    elif study=='Validation':
        folder=Path(config['results']);scheduler=json_file(folder/'scheduler_results.json') or {}
        entries={('A',policy):value for policy,value in scheduler.items()}
        students=(json_file(Path(config['comparison'])/'metrics.json') or {}).get('students',{})
        for source,model in [('epoch4','Original'),('student_A','A'),('exploratory_KD','KD')]:
            recorded=students.get(source,{}).get('scopes',{})
            if not recorded:continue
            base=copy.deepcopy(entries.get((model,'every'),{'overall':{},'per_trajectory':{}}))
            for tr,metrics in recorded.items():
                part=base['overall'] if tr=='pooled' else base['per_trajectory'].setdefault(tr,{})
                part['metrics']=metrics
            entries[(model,'every')]=base
        timing=json_file(folder/'timing_summary.json');scope='Full-split quality · recorded causal replay; timing uses separate 256-frame windows'
        if not entries:status='UNAVAILABLE'
    else:
        folder=root/'results/final_test'
        for filename,policy in [('every','every'),('fixed2','fixed2'),('fixed5','fixed5'),('fixed_validation_budget','budget_fixed'),('v2_selected','v2')]:
            result=json_file(folder/(filename+'.json'))
            if result:entries[('Original',policy)]=result
        scope='Historical exposed test · Original only · original CPU execution'
        if not entries:status='UNAVAILABLE'
    return {'study':study,'split':'validation' if study=='Validation' else 'test','entries':entries,'folder':folder,
            'status':status,'measurement':scope,'timing':timing}


def models(bundle):return [m for m in ['A','Original','KD'] if any(k[0]==m for k in bundle['entries'])]
def policies(bundle,model):return [p for m,p in bundle['entries'] if m==model]
def trajectories(bundle):
    return ['pooled',*sorted({tr for part in bundle['entries'].values() for tr in part.get('per_trajectory',{})})]
def part(bundle,model,policy,trajectory):
    result=bundle['entries'].get((model,policy),{})
    return result.get('overall',{}) if trajectory=='pooled' else result.get('per_trajectory',{}).get(trajectory,{})


def metric_rows(bundle,trajectory,model=None):
    rows=[]
    for (m,p) in bundle['entries']:
        if model is None and p!='every' or model is not None and m!=model:continue
        selected=part(bundle,m,p,trajectory)
        if not selected.get('metrics'):continue
        rows.append({'model':m,'Model':MODELS[m],'policy':p,'Policy':POLICIES.get(p,p),**selected['metrics'],
                     'Calls':selected.get('inference_calls'),'Frames':selected.get('frames'),
                     'Skip rate':selected.get('skip_rate'),'Max age':selected.get('max_prediction_age')})
    return rows


def relative_change(value,reference):
    if value is None or reference is None or not math.isfinite(value) or not math.isfinite(reference) or reference==0:return None
    return 100*(value/reference-1)


def trace_rows(bundle,model,policy):
    folder=bundle['folder']
    if bundle['study']=='Independent test':rows=csv_file(folder/'run'/f'{model}_{policy}_frames.csv')
    elif bundle['study']=='Validation':rows=json_file(folder/'policies'/f'{policy}_frames.json') or [] if model=='A' else []
    else:
        name={'budget_fixed':'fixed_validation_budget','v2':'v2_selected'}.get(policy,policy)
        rows=csv_file(folder/(name+'_frames.csv'))
    # A new projection keeps cached authoritative objects immutable.
    output=[]
    for row in rows:
        item={}
        for key,value in row.items():
            if key in ['trajectory','reason']:item[key]=value
            elif value is None or value=='':item[key]=None
            elif value is True or value=='True':item[key]=True
            elif value is False or value=='False':item[key]=False
            else:
                try:item[key]=float(value)
                except (ValueError,TypeError):item[key]=value
        if 'frame_index' not in item:item['frame_index']=item.get('position')
        output.append(item)
    return output


@lru_cache(maxsize=4)
def _features(config_text,identity):return backend.validation_features(json.loads(config_text))


def viewer_features(config):return _features(json.dumps(config,sort_keys=True),stamp(Path(config['cache'])/'features.json'))


def fixed_examples(config):
    record=json_file(Path(config['comparison'])/'example_verification.json') or {}
    return record.get('examples',[]) if record.get('passed') else []


def fixed_frame(config,trajectory,frame,model):
    """Read only the six saved validation illustrations, never test pixels."""
    import numpy as np
    from . import expanded
    if expanded.path_group(trajectory) not in expanded.VAL_GROUPS:raise ValueError('Validation illustrations only')
    matches=[r for r in fixed_examples(config) if (r['trajectory'],r['frame_index'])==(trajectory,frame)]
    if len(matches)!=1:raise ValueError('No recorded illustration for this frame')
    name=matches[0]['arrays']
    if Path(name).name!=name:raise ValueError('Unsafe illustration path')
    key={'A':'student_A','Original':'epoch4','KD':'exploratory_KD'}[model]
    with np.load(Path(config['comparison_cache'])/name) as z:
        return z['rgb'].copy(),z['gt'].copy(),z['valid'].copy(),z[key].copy(),'Recorded validation comparison example'


def scope_label(value):
    if value=='pooled':return 'Pooled trajectories'
    env,condition,name=value.split('/')
    return f'{env.split("_")[0]} · {condition.capitalize()} · {name.removeprefix("trajectory_")}'
