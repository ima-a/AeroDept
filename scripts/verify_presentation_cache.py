"""Mechanism verification on actual full-resolution predictions; no inference.

One CPU thread avoids parallel-dispatch overhead on small risk reductions.
Actual timing remains the separately frozen two-thread experiment.
"""
from pathlib import Path
from dataclasses import asdict
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from aerodepth_lite.presentation import read,write,verify_file
from aerodepth_lite.presentation_scheduler import PresentationScheduler,DISABLED
from aerodepth_lite.scheduler import near_field_risk


def main():
    config=read(ROOT/'configs/presentation_local.json');result=Path(config['results']);protocol=read(result/'PROTOCOL.json')
    for name,sha in protocol['source_sha256'].items():verify_file(ROOT/name,sha)
    if (result/'live_cache_equivalence.json').exists():raise FileExistsError('Existing mechanism evidence preserved')
    torch.set_num_threads(1);cache=Path(config['cache']);assert read(cache/'completion.json')['export_sha256']==protocol['model_export_sha256']
    features=read(cache/'features.json');gray=np.load(cache/'gray.npy',mmap_mode='r');p=np.load(cache/'predictions.npy',mmap_mode='r');report={};start=time.perf_counter()
    for policy,disabled in DISABLED.items():
        cached=PresentationScheduler(protocol['v2'],disabled,True);uncached=PresentationScheduler(protocol['v2'],disabled,False)
        previous=None;lasttr=None;source_c=source_u=None;updates=0
        for i,f in enumerate(features):
            if f['trajectory']!=lasttr:
                cached.reset();uncached.reset();previous=None;source_c=source_u=None;lasttr=f['trajectory']
            risk,available=near_field_risk(previous,10.) if 'risk' not in disabled else (0.,False)
            a=asdict(cached.decide_features(gray[i],f['navigation'],cached.risk_value,cached.risk_available))
            b=asdict(uncached.decide_features(gray[i],f['navigation'],risk,available))
            assert a==b,(policy,i,a,b)
            if a['infer']:
                source_c=source_u=i;previous=torch.tensor(np.array(p[i]));cached.update_prediction(previous,i);updates+=int('risk' not in disabled)
            assert source_c==source_u and i-source_c==a['age_after']
            if (i+1)%1000==0:print(json.dumps({'policy':policy,'frames':i+1,'seconds':time.perf_counter()-start}),flush=True)
        report[policy]={'passed':True,'frames':len(features),'compared':'all decisions/reasons/scores/source indices/ages, true risk recomputation versus explicit fresh-prediction cache updates','risk_updates':updates,'verification_CPU_threads':1,'actual_timing_CPU_threads':2}
        write(result/'live_cache_equivalence_partial.json',report)
    write(result/'live_cache_equivalence.json',report)

if __name__=='__main__':main()
