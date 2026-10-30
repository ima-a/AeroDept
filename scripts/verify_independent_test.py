"""Verify saved independent-study evidence without decoding data or running models."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse, csv, gzip, hashlib, json, math

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'results/independent_test_20261030'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def check(condition,message):
    if not condition:raise ValueError(message)


def close(actual,expected,label):
    check(math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-9),label)


# Verification
def verify(root=ROOT):
    folder=root/'results/independent_test_20261030'
    p=read(folder/'PROTOCOL.json');c=read(folder/'COMPLETION.json');exposure=read(folder/'EXPOSURE.json')
    sha=digest(folder/'PROTOCOL.json')
    check(c['status']=='COMPLETED' and c['frames']==8820 and c['equal_counts'],'Incomplete study')
    check(c['protocol_sha256']==exposure['protocol_sha256']==sha,'Protocol/exposure identity')
    times=[datetime.fromisoformat(x) for x in [p['frozen_utc'],exposure['started_utc'],c['completed_utc']]]
    check(times==sorted(times),'Exposure did not follow freeze')
    check(not (folder/'FAILED.json').exists(),'Failure record present')
    for name,h in p['source_sha256'].items():check(digest(root/name)==h,'Frozen source changed: '+name)
    for name,h in c['artifacts_sha256'].items():check(digest(folder/name)==h,'Artifact changed: '+name)
    for name,key in [('COMPATIBILITY.json','compatibility_sha256'),('INVENTORY.json','inventory_sha256'),('prepared/PREPARED.json','prepared_sha256'),('CHECKS.json','checks_sha256')]:
        check(digest(folder/name)==p[key],'Preparation identity: '+name)
    check(digest(root/p['manifest_path'])==p['manifest_sha256']==c['manifest_sha256']==exposure['manifest_sha256'],'Manifest identity')
    check(digest(root/p['validation_protocol'])==p['validation_protocol_sha256'],'Validation protocol changed')
    for key in ['A','Original']:
        check(digest(root/p['models'][key]['weights'])==p['models'][key]['sha256'],'Export identity: '+key)
        check(digest(root/p['models'][key]['metadata'])==p['models'][key]['metadata_sha256'],'Export metadata: '+key)
    manifest=json.loads(gzip.decompress((root/p['manifest_path']).read_bytes()))
    records=manifest['records'];results=read(folder/'results.json');expected_keys=[(r['trajectory_id'],r['frame_index']) for r in records]
    check(len(records)==8820 and manifest['per_trajectory']==p['test_realizations'],'Reserved counts')
    check(set(results)==set(c['policies'])=={row[0] for row in p['matrix']},'Seven-comparison identity')
    reference_counts=None;accounting={}
    for name,model,policy in p['matrix']:
        rows=list(csv.DictReader((folder/'run'/f'{name}_frames.csv').open()))
        check([(r['trajectory'],int(r['frame_index'])) for r in rows]==expected_keys,'Frame order: '+name)
        counts=[(int(r['valid_pixels']),int(r['near_valid_pixels'])) for r in rows]
        if reference_counts is None:reference_counts=counts
        check(counts==reference_counts,'Unequal current-frame counts: '+name)
        for tr,n in p['test_realizations'].items():
            rr=[r for r in rows if r['trajectory']==tr];part=results[name]['per_trajectory'][tr]
            check(len(rr)==n==part['frames'],'Trajectory count')
            source=None;previous_p10=math.inf
            calls=[i for i,r in enumerate(rr) if r['infer']=='True']
            if policy!='v2':
                if policy=='every':expected=list(range(n))
                elif policy=='fixed2':expected=list(range(0,n,2))
                elif policy=='fixed5':expected=list(range(0,n,5))
                else:
                    k=math.ceil(n*p['budget_fraction']['numerator']/p['budget_fraction']['denominator'])
                    expected=[math.ceil(j*n/k) for j in range(k)]
                check(calls==expected,'Frozen fixed comparator: '+name)
            events=[];delays=[];censored=0
            for i,r in enumerate(rr):
                if r['infer']=='True':source=int(r['index'])
                age=int(r['prediction_age'])
                check(source is not None and int(r['source_index'])==source and age==int(r['index'])-source,'Reuse/source age')
                check(0<=age<=({'every':0,'fixed2':1,'fixed5':4,'budget_fixed':1,'v2':4}[policy]),'Age bound')
                p10=float(r['gt_p10']) if r['gt_p10'] else math.inf
                event=p10<8 and (previous_p10>=8 or previous_p10-p10>=2);previous_p10=p10
                check((r['critical_event']=='True')==event,'Offline event definition')
                if event:
                    events.append(i);future=next((j for j in calls if j>=i),None)
                    if future is None:censored+=1
                    else:delays.append(future-i)
                for key in ['scheduler_ms','inference_ms','model_ms','transfer_in_ms','transfer_out_ms','shared_io_ms','shared_decode_ms']:
                    check(math.isfinite(float(r[key])) and float(r[key])>=0,'Timing not finite/nonnegative')
                close(float(r['inference_ms']),sum(float(r[k]) for k in ['model_ms','transfer_in_ms','transfer_out_ms']),'Timing boundaries')
            ev=part['critical_events']
            check(ev['count']==len(events) and ev['observed_delays_frames']==delays and ev['right_censored_count']==censored,'Event/censoring accounting')
            check(part['inference_calls']==len(calls) and part['all_reasons']==dict(Counter(r['reason'] for r in rr)),'Call/reason accounting')
            close(part['mean_prediction_age'],sum(int(r['prediction_age']) for r in rr)/n,'Mean age')
            check(part['max_prediction_age']==max(int(r['prediction_age']) for r in rr),'Maximum age')
            for key,col in [('valid_pixels','valid_pixels'),('near_valid_pixels','near_valid_pixels')]:
                check(part['metrics'][key]==sum(int(r[col]) for r in rr),'Pixel count: '+key)
            for key,col in [('inference_seconds','inference_ms'),('scheduler_seconds','scheduler_ms')]:
                close(part[key],sum(float(r[col]) for r in rr)/1000,'Time sum')
        overall=results[name]['overall'];parts=list(results[name]['per_trajectory'].values())
        check(overall['frames']==8820 and overall['inference_calls']==sum(x['inference_calls'] for x in parts),'Pooled calls')
        for metric in ['absrel','rmse','delta1','delta2','delta3','near_absrel','near_rmse','bias_m','near_bias_m','valid_floor_rate','near_floor_rate','near_prediction_ge_10m_rate']:
            count='near_valid_pixels' if metric.startswith('near_') else 'valid_pixels'
            total=sum(x['metrics'][count] for x in parts)
            exponent=2 if metric in ['rmse','near_rmse'] else 1
            value=(sum((x['metrics'][metric]**exponent)*x['metrics'][count] for x in parts if x['metrics'][count])/total)**(1/exponent)
            close(overall['metrics'][metric],value,'Pixel pooling: '+metric)
        accounting[name]={'frames':overall['frames'],'calls':overall['inference_calls'],'valid_pixels':overall['metrics']['valid_pixels'],'near_valid_pixels':overall['metrics']['near_valid_pixels'],'events':overall['critical_events']['count'],'censored':overall['critical_events']['right_censored_count']}
    table=list(csv.DictReader((folder/'comparison.csv').open()))
    check(len(table)==c['comparison_rows']==35,'Comparison rows')
    for row in table:
        result=results[row['comparison']];part=result['overall'] if row['scope']=='pooled' else result['per_trajectory'][row['scope']]
        for name,value in part['metrics'].items():
            check((row[name]=='' if value is None else math.isclose(float(row[name]),value,rel_tol=1e-12,abs_tol=1e-12)),'CSV/JSON metric: '+name)
    return {'passed':True,'protocol_sha256':sha,'completion_sha256':digest(folder/'COMPLETION.json'),'verified_artifacts':len(c['artifacts_sha256']),'verified_frozen_sources':len(p['source_sha256']),'comparison_rows':len(table),'accounting':accounting,'test_images_opened':False,'inference_calls_for_verification':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=verify();text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(text)
    print(text)

if __name__=='__main__':main()
