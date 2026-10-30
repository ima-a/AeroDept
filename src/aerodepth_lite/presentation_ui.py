"""Read-only dashboard helpers; no training/search/test evaluation entry points."""
from pathlib import Path
import json, threading, time, os, re
import numpy as np
import torch
from . import expanded as e
from .presentation import validation_inputs, ValidationGuard, read_sample, load_model, synchronize


def optional_json(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,OSError,ValueError):return None


def independent_study(repository):
    """Read saved evidence only; never open test images or invoke evaluation."""
    from .inference import file_sha256
    folder=Path(repository)/'results/independent_test_20261030'
    complete=optional_json(folder/'COMPLETION.json')
    if not complete:
        failed=optional_json(folder/'FAILED.json')
        exposed=(folder/'EXPOSURE.json').exists()
        return {'status':'INCOMPLETE' if (folder/'FAILED.json').exists() else 'IN PROGRESS' if exposed else 'NOT RUN',
                'detail':failed or optional_json(folder/'run/progress.json') or {},'results':None}
    try:
        protocol_hash=file_sha256(folder/'PROTOCOL.json')
        exposure=optional_json(folder/'EXPOSURE.json') or {}
        if complete['status']!='COMPLETED' or complete['frames']!=8820 or not complete['equal_counts']:
            raise ValueError('Invalid completion accounting')
        if protocol_hash!=complete['protocol_sha256'] or exposure.get('protocol_sha256')!=protocol_hash:
            raise ValueError('Protocol/exposure identity differs')
        for name in complete['artifacts_sha256']:
            if file_sha256(folder/name)!=complete['artifacts_sha256'][name]:
                raise ValueError('Changed result artifact: '+name)
        return {'status':'COMPLETED','detail':complete,'results':optional_json(folder/'results.json'),
                'timing':optional_json(folder/'timing.json')}
    except (OSError,ValueError,KeyError) as error:
        return {'status':'UNAVAILABLE · integrity check failed','detail':str(error),'results':None}


def load_dashboard_config(path, repository):
    """Resolve checkout-relative paths and explicit environment locations only.

    An unset external location is deliberately unavailable, never the current
    directory. This helper does no discovery and changes no numerical contract.
    """
    config=optional_json(path)
    if config is None:return None
    root=Path(repository).resolve()
    def location(value):
        expanded=os.path.expandvars(value)
        if '$' in expanded:
            name=re.sub(r'[^A-Za-z0-9_-]', '_', value)
            return str(root/'.unavailable'/name)
        p=Path(expanded).expanduser()
        return str(p if p.is_absolute() else root/p)
    for field in ['dataset_root','manifest','reservations','comparison','results','cache',
                  'comparison_cache','export_directory','delivery']:
        if field in config:config[field]=location(config[field])
    for model in config.get('models',{}).values():
        for field in ['weights','metadata']:
            if field in model:model[field]=location(model[field])
    return config


def validation_features(config):
    features=optional_json(Path(config['cache'])/'features.json')
    if features is None:return []
    if any(e.path_group(x['trajectory']) not in e.VAL_GROUPS for x in features):
        raise ValueError('Dashboard cache contains non-validation groups')
    return features


def recorded_frame(config,index,model):
    features=validation_features(config)
    if not 0<=index<len(features):raise ValueError('Frame outside validation allowlist')
    cache=Path(config['cache']);r=features[index]
    x=np.load(cache/'display_RGB_uint8.npy',mmap_mode='r')[index].copy()
    g=np.load(cache/'targets.npy',mmap_mode='r')[index,0].copy();v=np.load(cache/'valid.npy',mmap_mode='r')[index,0].copy()
    p=None;source='Recorded A full-validation prediction'
    if model=='A':p=np.load(cache/'predictions.npy',mmap_mode='r')[index,0].copy()
    else:
        verify=optional_json(Path(config['comparison'])/'example_verification.json')
        key='epoch4' if model=='Original' else 'student_A' if model=='A' else 'exploratory_KD'
        for row in (verify or {}).get('examples',[]):
            if (row['trajectory'],row['frame_index'])==(r['trajectory'],r['frame_index']):
                with np.load(Path(config['comparison_cache'])/row['arrays']) as z:
                    x=z['rgb'].copy();g=z['gt'].copy();v=z['valid'].copy();p=z[key].copy()
                source='Recorded fixed comparison example (not live inference)';break
    return x,g,v,p,source


def inference_resource(entry,device):
    model,cfg=load_model(entry,device)
    return {'model':model,'config':cfg,'lock':threading.Lock(),'state_digest':e.state_digest(model)}


def actual_inference(config,index,model_key,device,resource):
    root,records,_=validation_inputs(config)
    if not 0<=index<len(records):raise ValueError('Frame outside validation allowlist')
    guard=ValidationGuard(root,records);__import__('sys').addaudithook(guard.check)
    r=records[index];x,g,v,io_s,prep_s=read_sample(root,r,resource['config']['data'])
    with resource['lock'],torch.inference_mode(),torch.amp.autocast(device,enabled=False):
        synchronize(device);start=time.perf_counter();tensor=x[None].to(device);synchronize(device);h2d=time.perf_counter()-start
        start=time.perf_counter();pred=resource['model'](tensor);synchronize(device);model_s=time.perf_counter()-start
        start=time.perf_counter();p=pred.cpu()[0,0].numpy().copy();synchronize(device);d2h=time.perf_counter()-start
        if not np.isfinite(p).all() or np.min(p)<.1:raise ValueError('Invalid inference output')
        if e.state_digest(resource['model'])!=resource['state_digest']:raise ValueError('Model/BN mutation during inference')
    return {'key':[model_key,index,device],'rgb':x.permute(1,2,0).numpy(),'gt':g[0].numpy(),'valid':v[0].numpy(),'prediction':p,
            'source':'ACTUAL model inference on this selected validation frame',
            'observation':{'device':device,'model_only_ms':model_s*1000,'transfer_ms':(h2d+d2h)*1000,'IO_ms':io_s*1000,'preprocessing_ms':prep_s*1000,'note':'One observed call, not a repeated benchmark or playback FPS'}}


def colourize(values,mask,kind='range'):
    import matplotlib
    from matplotlib.colors import LogNorm,Normalize
    cmap=matplotlib.colormaps['viridis' if kind=='range' else 'RdBu_r']
    norm=LogNorm(.1,1000,clip=True) if kind=='range' else Normalize(-10,10,clip=True)
    rgba=cmap(norm(np.maximum(values,.1) if kind=='range' else values))
    rgba[~mask]=[.7,.7,.7,1.]
    return (rgba[:,:,:3]*255).astype('uint8')
