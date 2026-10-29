"""Apply the decision rules sealed before any TRAIN gradient was calculated."""
from pathlib import Path
from collections import defaultdict
import csv,json,math,hashlib
import numpy as np

ART=Path(__file__).resolve().parent
REPO=ART.parents[1]
RES=REPO/'results'/ART.name


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def breadth(rows):
    records=[x for row in rows for x in row['records']]
    groups=sorted({x['path_group'] for x in records})
    fractions=sorted({x['fraction_pass'] for x in records})
    fog=sorted({x['record']['trajectory_id'] for x in records if x['condition']=='foggy'})
    return {'path_groups':groups,'fraction_passes':fractions,'foggy_realizations':fog,
            'passed':len(groups)>=4 and len(fractions)>=2 and len(fog)>=1}


def quantiles(values):
    vals=[x for x in values if x is not None]
    return {'count':len(vals),'min_q25_median_q75_max':np.quantile(vals,[0,.25,.5,.75,1]).tolist()} if vals else {'count':0,'min_q25_median_q75_max':None}


def main():
    seal=json.loads((RES/'ANALYSIS_FREEZE.json').read_text())
    assert sha(ART/'summarize.py')==seal['analysis_script_sha256']
    assert sha(RES/'PLAN.json')==seal['plan_sha256']
    assert json.loads((RES/'completion.json').read_text())['passed']
    batches=json.loads((RES/'batches.json').read_text());assert len(batches)==32
    decisions={};stats={}
    for group in ['encoder','decoder','output_head']:
        info=[r for r in batches if r['counts']['near_pixels']>0 and r['groups'][group]['L2']['current']>1e-12 and r['groups'][group]['L2']['near_MAE']>1e-12]
        conflict=[r for r in info if r['groups'][group]['cosines']['current__near_MAE']<-.05]
        weak=[r for r in info if r['groups'][group]['near_partition_current_norm_ratio']<=.10]
        conflict_effect=[r for r in conflict if r['groups'][group]['cosines']['candidate__near_MAE']-r['groups'][group]['cosines']['current__near_MAE']>=.05]
        weak_effect=[r for r in weak if r['groups'][group]['weighted_near_MAE_current_norm_ratio']>=.10]
        n=len(info);ct=max(4,math.ceil(.25*n));wt=max(4,math.ceil(.5*n))
        cb,wb=breadth(conflict),breadth(weak)
        cp=n>=4 and len(conflict)>=ct and len(conflict_effect)>=math.ceil(.75*len(conflict)) and cb['passed']
        wp=n>=4 and len(weak)>=wt and len(weak_effect)>=math.ceil(.75*len(weak)) and wb['passed']
        decisions[group]={'primary':group!='output_head','informative_batches':n,'conflict_batches':[r['batch_index'] for r in conflict],
            'required_conflict_count':ct,'conflict_candidate_effect_count':len(conflict_effect),'conflict_breadth':cb,'conflict_signal_passed':cp,
            'weak_batches':[r['batch_index'] for r in weak],'required_weak_count':wt,'weak_candidate_effect_count':len(weak_effect),
            'weak_breadth':wb,'weak_signal_passed':wp,'supports_pilot_review':group!='output_head' and (cp or wp)}
        stats[group]={'parameters':batches[0]['groups'][group]['parameters'],
            'norms':{key:quantiles([r['groups'][group]['L2'][key] for r in batches]) for key in batches[0]['groups'][group]['L2']},
            'RMS':{key:quantiles([r['groups'][group]['RMS'][key] for r in batches]) for key in batches[0]['groups'][group]['RMS']},
            'cosines':{key:quantiles([r['groups'][group]['cosines'][key] for r in batches]) for key in batches[0]['groups'][group]['cosines']},
            'near_partition_ratio':quantiles([r['groups'][group]['near_partition_current_norm_ratio'] for r in batches]),
            'weighted_bonus_ratio':quantiles([r['groups'][group]['weighted_near_MAE_current_norm_ratio'] for r in batches]),
            'zero_near_gradient_with_near_gt_batches':[r['batch_index'] for r in batches if r['counts']['near_pixels']>0 and r['groups'][group]['L2']['near_MAE']<=1e-12]}
    with (RES/'sample_coverage.csv').open() as f:frames=list(csv.DictReader(f))
    coverage=[]
    for tr in sorted({x['trajectory'] for x in frames}):
        rows=[x for x in frames if x['trajectory']==tr];first=rows[0]
        nv=sum(int(x['valid_pixels']) for x in rows);nn=sum(int(x['near_pixels']) for x in rows)
        coverage.append({'trajectory':tr,'condition':first['condition'],'path_group':first['path_group'],'sample_frames':len(rows),
            'frame_indices':[int(x['frame_index']) for x in rows],'ordinal_fractions':[float(x['normalized_ordinal']) for x in rows],
            'valid_pixel_occurrences':nv,'near_pixel_occurrences':nn,'near_fraction':nn/nv if nv else 0.})
    out={'all_validity_checks_passed':True,'supports_one_teacher_pilot_review':any(x['supports_pilot_review'] for x in decisions.values()),
         'rules_unchanged':True,'rules':decisions,'gradient_statistics':stats,'sampled_realizations':coverage,
         'underlying_path_groups':sorted({r['path_group'] for r in coverage}),
         'total_valid_pixel_occurrences':sum(r['valid_pixel_occurrences'] for r in coverage),
         'total_near_pixel_occurrences':sum(r['near_pixel_occurrences'] for r in coverage),
         'batches_without_near_gt':[r['batch_index'] for r in batches if r['counts']['near_pixels']==0],
         'independence':'64 sparse frames from 20 correlated weather realizations / 12 path groups, not 64 independent flights or 32 independent statistical trials. Mixed batches cannot localize gradient causes to a trajectory.'}
    (RES/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    with (RES/'trajectory_coverage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    print(json.dumps({'supports_one_teacher_pilot_review':out['supports_one_teacher_pilot_review'],'rules':decisions},indent=2))


if __name__=='__main__':main()
