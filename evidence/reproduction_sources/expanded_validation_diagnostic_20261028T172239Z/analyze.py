"""Summarize saved VAL diagnostics and make standalone scientific figures."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import os
import time
import numpy as np

ART = Path(__file__).resolve().parent
RES = ART.parents[1] / 'results' / ART.name
os.environ['MPLCONFIGDIR'] = str(ART / 'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['epoch4', 'student_A', 'teacher_B']
LABELS = {'epoch4': 'Original epoch 4', 'student_A': 'Supervised A', 'teacher_B': 'Teacher B'}
COLORS = ['#64748b', '#0f766e', '#b45309']
Q = np.array([0, .01, .05, .25, .5, .75, .9, .95, .99, 1.])


def read(name):
    return json.loads((RES / name).read_text())


def clean(x):
    if isinstance(x, dict):return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):return [clean(v) for v in x]
    if isinstance(x, np.generic):return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):return None
    return x


def save(name, x):
    (RES / name).write_text(json.dumps(clean(x), indent=2, allow_nan=False) + '\n')


def weighted(rows):
    n = sum(r['valid_pixels'] for r in rows)
    keys = ['absrel', 'bias_m', 'mae_m', 'mean_signed_relative_error', 'mean_prediction_m',
            'mean_gt_m', 'floor_rate', 'overestimate_ge_2m_rate', 'underestimate_ge_2m_rate',
            'prediction_ge_10m_rate', 'delta1', 'delta2', 'delta3']
    out = {k: sum(r['valid_pixels'] * r[k] for r in rows if r['valid_pixels']) / n if n else None for k in keys}
    out.update(valid_pixels=n, rmse=math.sqrt(sum(r['valid_pixels'] * r['rmse']**2 for r in rows if r['valid_pixels']) / n) if n else None,
               floor_count=sum(r.get('floor_count', 0) for r in rows))
    return out


def main():
    assert read('evaluation_completion.json')['passed']
    metrics = read('metrics.json')
    trs = list(metrics['epoch4']['per_trajectory'])
    completion = read('evaluation_completion.json')
    diagnostics = {}
    quantile_rows = []
    for tr in trs:
        cache = np.load(completion['near_arrays'][tr], mmap_mode='r')
        gt = np.array(cache[:, 0], dtype=np.float64)
        out = {'count': len(gt), 'gt_quantiles_m': np.quantile(gt, Q).tolist(), 'models': {}}
        for i, name in enumerate(NAMES, 1):
            pred = np.array(cache[:, i], dtype=np.float64)
            error = pred - gt
            distribution = dict(signed_error_quantiles_m=np.quantile(error, Q).tolist(),
                                absolute_error_quantiles_m=np.quantile(np.abs(error), Q).tolist(),
                                absolute_relative_error_quantiles=np.quantile(np.abs(error) / gt, Q).tolist(),
                                prediction_quantiles_m=np.quantile(pred, Q).tolist())
            stats = metrics[name]['supplementary'][tr]['near']
            assert math.isclose(float(np.mean(error)), stats['bias_m'], abs_tol=1e-10)
            assert math.isclose(float(np.sqrt(np.mean(error**2))), stats['rmse'], abs_tol=1e-10)
            hist_edges = [-math.inf, -20., -10., -5., -2., -1., -.5, 0., .5, 1., 2., 5., 10., 20., math.inf]
            distribution['signed_error_histogram'] = {'edges_m': ['-infinity', *hist_edges[1:-1], '+infinity'], 'counts': np.histogram(error, hist_edges)[0].tolist()}
            out['models'][name] = stats | distribution
            for j, q in enumerate(Q):
                quantile_rows.append(dict(trajectory=tr, model=name, quantile=q,
                                         signed_error_m=distribution['signed_error_quantiles_m'][j],
                                         absolute_error_m=distribution['absolute_error_quantiles_m'][j],
                                         absolute_relative_error=distribution['absolute_relative_error_quantiles'][j],
                                         prediction_m=distribution['prediction_quantiles_m'][j], gt_m=out['gt_quantiles_m'][j]))
            print('EXACT_NEAR_DISTRIBUTION',tr,name,len(gt),flush=True)
            del pred,error
        diagnostics[tr] = out
        del gt,cache
    save('near_distributions.json', {'quantile_probabilities': Q.tolist(), 'scope': 'Every valid GT<10m pixel; exact FP64 arithmetic from saved FP32 predictions/GT, not a sample.', 'per_trajectory': diagnostics})
    with (RES / 'near_quantiles.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(quantile_rows[0]));w.writeheader();w.writerows(quantile_rows)

    pooled = {}
    for name in NAMES:
        pooled[name] = dict(all_valid=weighted([metrics[name]['supplementary'][tr]['all_valid'] for tr in trs]),
                            near=weighted([metrics[name]['supplementary'][tr]['near'] for tr in trs]),
                            depth_bands={band: weighted([metrics[name]['supplementary'][tr]['depth_bands'][band] for tr in trs])
                                         for band in metrics[name]['supplementary'][trs[0]]['depth_bands']})
    old=json.loads((ART.parents[1]/'results/expanded_validation/validation_metrics.json').read_text())
    for band,ref in old['depth_bands'].items():
        actual=pooled['epoch4']['depth_bands'][band]
        assert actual['valid_pixels']==ref['valid_pixels']
        assert abs(actual['absrel']-ref['absrel'])<1e-5 and abs(actual['rmse']-ref['rmse'])<1e-3
    save('pooled_supplementary.json',pooled)

    frames=[]
    with (RES/'per_frame.csv').open() as f:
        for r in csv.DictReader(f):
            frames.append({k:(v if k in ('model','trajectory') else float(v) if v else None) for k,v in r.items()})
    by={name:{tr:sorted([r for r in frames if r['model']==name and r['trajectory']==tr],key=lambda r:r['frame_index']) for tr in trs} for name in NAMES}
    concentration={};selections=[];top_rows=[]
    for tr in trs:
        paired=[]
        for old,a,t in zip(by['epoch4'][tr],by['student_A'][tr],by['teacher_B'][tr]):
            assert old['frame_index']==a['frame_index']==t['frame_index']
            paired.append(dict(frame_index=int(t['frame_index']),near_pixels=int(t['near_valid_pixels']),
                               teacher_sse=t['near_sse'],student_sse=a['near_sse'],original_sse=old['near_sse'],
                               excess_sse=t['near_sse']-a['near_sse'],teacher_near_absrel=t['near_absrel'],
                               teacher_bias=t['near_bias_m'],student_bias=a['near_bias_m']))
        active=[r for r in paired if r['near_pixels']>0]
        top=sorted(active,key=lambda r:(-r['teacher_sse'],r['frame_index']))
        excess=sorted(active,key=lambda r:(-r['excess_sse'],r['frame_index']))
        total=sum(r['teacher_sse'] for r in active);positive_excess=sum(max(0,r['excess_sse']) for r in active)
        concentration[tr]=dict(frames=len(paired),frames_with_near=len(active),frames_without_near=len(paired)-len(active),
                               teacher_near_sse=total,teacher_minus_A_near_sse=sum(r['excess_sse'] for r in paired),
                               frames_teacher_near_sse_worse_than_A=sum(r['excess_sse']>0 for r in active),
                               teacher_sse_share={f'top_{n}_frames':sum(r['teacher_sse'] for r in top[:n])/total for n in [1,5,10,25,50]},
                               positive_excess_sse_share={f'top_{n}_frames':sum(max(0,r['excess_sse']) for r in excess[:n])/positive_excess if positive_excess else None for n in [1,5,10,25,50]},
                               top_teacher_frames=top[:10],top_regression_frames=excess[:10])
        eligible=[r for r in paired if r['near_pixels']>=64]
        median=np.median([r['teacher_near_absrel'] for r in eligible])
        typical=min(eligible,key=lambda r:(abs(r['teacher_near_absrel']-median),r['frame_index']))
        poor=min(eligible,key=lambda r:(-r['teacher_near_absrel'],r['frame_index']))
        regression=min(eligible,key=lambda r:(-r['excess_sse'],r['frame_index']))
        seen=set()
        for reason,r in [('typical',typical),('poor',poor),('regression',regression)]:
            if r['frame_index'] not in seen:
                selections.append(dict(trajectory=tr,reason=reason,**r));seen.add(r['frame_index'])
        for rank,r in enumerate(excess[:25],1):top_rows.append(dict(trajectory=tr,rank=rank,**r))
    save('concentration.json',concentration)
    save('example_selection.json',{'predeclared_rule':'PLAN.json; no frame omitted from metrics; >=64 near pixels applies only to illustrations','examples':selections})
    with (RES/'top_regression_frames.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(top_rows[0]));w.writeheader();w.writerows(top_rows)

    packed=np.load(RES/'near_spatial_maps.npz');regional=[]
    regions=[(r,c,slice(y0,y1),slice(x0,x1)) for r,(y0,y1) in enumerate(zip([0,64,128],[64,128,192]))
             for c,(x0,x1) in enumerate(zip([0,106,213],[106,213,320]))]
    for tr in trs:
        key=tr.replace('/','__');count=packed[key+'__count']
        for name in NAMES:
            maps=packed[key+'__'+name]
            for r,c,ys,xs in regions:
                n=int(count[ys,xs].sum());sums=maps[:,ys,xs].sum(axis=(1,2))
                regional.append(dict(trajectory=tr,model=name,row=r,column=c,near_pixels=n,
                                     near_bias_m=sums[0]/n if n else None,near_rmse=math.sqrt(sums[1]/n) if n else None,
                                     near_absrel=sums[2]/n if n else None,near_floor_rate=sums[3]/n if n else None,
                                     near_sse=sums[1]))
    with (RES/'spatial_regions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(regional[0]));w.writeheader();w.writerows(regional)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,tr in enumerate(trs):
        for i,name in enumerate(NAMES):
            d=diagnostics[tr]['models'][name]
            axes[0,col].plot(d['signed_error_quantiles_m'],Q,label=LABELS[name],color=COLORS[i])
            axes[1,col].plot(d['absolute_relative_error_quantiles'],Q,color=COLORS[i])
        axes[0,col].set_title(tr.split('/')[0]+' / '+tr.split('/')[1]);axes[0,col].set_xlabel('Near signed error (prediction − GT), m')
        axes[1,col].set_xlabel('Near absolute relative error');axes[0,col].set_ylabel('Quantile probability');axes[1,col].set_ylabel('Quantile probability')
        axes[0,col].axvline(0,color='black',lw=.5);axes[0,col].legend()
        axes[0,col].set_xlim(min(diagnostics[tr]['models'][n]['signed_error_quantiles_m'][1] for n in NAMES),
                            max(diagnostics[tr]['models'][n]['signed_error_quantiles_m'][-2] for n in NAMES))
        axes[1,col].set_xlim(0,max(diagnostics[tr]['models'][n]['absolute_relative_error_quantiles'][-2] for n in NAMES))
    fig.suptitle('Exact all-near-pixel quantile knots; axes show 1–99% span; full tails in CSV')
    fig.tight_layout();fig.savefig(RES/'near_error_distributions.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for col,scope in enumerate(['pooled',*trs]):
        for i,name in enumerate(NAMES):
            d=pooled[name]['depth_bands'] if scope=='pooled' else metrics[name]['supplementary'][scope]['depth_bands']
            x=np.arange(len(d))+(i-1)*.25
            for row,metric in enumerate(['absrel','rmse']):axes[row,col].bar(x,[v[metric] for v in d.values()],width=.24,label=LABELS[name],color=COLORS[i])
        for row in range(2):axes[row,col].set_xticks(np.arange(len(d)),list(d),rotation=35,ha='right')
        axes[0,col].set_title(scope if scope=='pooled' else scope.split('/')[0]+' / '+scope.split('/')[1])
        axes[0,col].set_ylabel('AbsRel');axes[1,col].set_ylabel('RMSE, m (log axis)');axes[1,col].set_yscale('log')
    axes[0,0].legend();fig.suptitle('Unchanged GT depth bands; validation only');fig.tight_layout();fig.savefig(RES/'depth_bands.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for row,tr in enumerate(trs):
        key=tr.replace('/','__');cnt=packed[key+'__count']
        for col,name in enumerate(NAMES):
            err=packed[key+'__'+name][0];bias=np.divide(err,cnt,out=np.full(err.shape,np.nan),where=cnt>0)
            im=axes[row,col].imshow(bias,cmap='RdBu_r',vmin=-10,vmax=10)
            axes[row,col].set_title(tr.split('/')[0]+' '+LABELS[name]);axes[row,col].set_xticks([0,106,213,319]);axes[row,col].set_yticks([0,64,128,191])
            fig.colorbar(im,ax=axes[row,col],label='Mean near error, m; colours clipped ±10')
    fig.suptitle('GT<10m valid-pixel spatial bias; blank = no near observations');fig.tight_layout();fig.savefig(RES/'near_spatial_bias.png',dpi=160);plt.close(fig)

    history=[]
    with (RES/'saved_history.csv').open() as f:history=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for name,color in [('student_A',COLORS[1]),('teacher_B',COLORS[2])]:
        rr=[r for r in history if r['model']==name];x=[int(r['epoch']) for r in rr]
        for ax,k in zip(axes,['pooled_absrel','Kite_near_absrel','Kite_near_rmse']):
            ax.plot(x,[float(r[k]) for r in rr],marker='.',label=LABELS[name],color=color)
        selected=next(r for r in rr if r['selected_by_frozen_pooled_absrel']=='True')
        for ax,k in zip(axes,['pooled_absrel','Kite_near_absrel','Kite_near_rmse']):ax.scatter([int(selected['epoch'])],[float(selected[k])],s=80,facecolors='none',edgecolors=color)
    for ax,title in zip(axes,['Pooled validation AbsRel','Kite foggy near AbsRel','Kite foggy near RMSE, m']):
        ax.set_title(title);ax.set_xlabel('Completed epoch');ax.legend()
    fig.suptitle('Saved histories; circles = unchanged selected checkpoints; no alternate promotion')
    fig.tight_layout();fig.savefig(RES/'training_history.png',dpi=160);plt.close(fig)
    save('analysis_completion.json',{'passed':True,'exact_near_pixels':sum(d['count'] for d in diagnostics.values()),'deterministic_examples':len(selections),'source':'Existing validation predictions/cache and saved histories only','planning_or_training_launched':False})
    print('ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
