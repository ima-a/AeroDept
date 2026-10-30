"""Render tables and plots from verified saved results; never reevaluate images."""
from pathlib import Path
import csv, json
from verify_independent_test import verify

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/independent_test_20261030'


def table(rows,columns):
    def value(x):
        if x is None or x=='':return 'undefined'
        if isinstance(x,float):return f'{x:.6f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(title for _,title in columns)+' |','|'+'|'.join('---' for _ in columns)+'|',*['| '+' | '.join(value(row.get(key)) for key,_ in columns)+' |' for row in rows]])


def main():
    verify(ROOT)
    if (FOLDER/'REPORT.md').exists() or (FOLDER/'figures').exists():raise FileExistsError('Rendered study evidence already exists')
    results=json.loads((FOLDER/'results.json').read_text());protocol=json.loads((FOLDER/'PROTOCOL.json').read_text())
    completion=json.loads((FOLDER/'COMPLETION.json').read_text());timing=json.loads((FOLDER/'timing.json').read_text())
    rows=[]
    for name,result in results.items():
        for scope,part in [('pooled',result['overall']),*result['per_trajectory'].items()]:
            rows.append({'comparison':name,'scope':scope,**part['metrics'],**{k:v for k,v in part.items() if k not in ['metrics','critical_events','all_reasons','inference_reasons']},**{'events':part['critical_events']['count'],'fresh_on_event':part['critical_events']['fresh_on_event'],'missed_update_rate':part['critical_events']['missed_update_rate'],'mean_delay':part['critical_events']['mean_observed_delay_frames'],'max_delay':part['critical_events']['max_observed_delay_frames'],'censored':part['critical_events']['right_censored_count']}})
    scopes=['pooled',*results['A_every']['per_trajectory']]
    lines=['# Independent reserved final test · completed',
        'One frozen study, seven comparisons, 8,820 current frames per comparison. A epoch 9 remained the preselected primary model; Original epoch 4 and exploratory KD epoch 9 were fixed secondary comparisons. No training, tuning or post-test selection occurred.',
        '## Inputs, freeze and exposure',
        'Eight CRC-verified archives supplied 17,640 images / 8,820 RGB-range pairs. The four HDF5 streams were reused from healthy rescue storage. Archive/member hashes, sensor correspondence and underlying-path/weather-equivalent disjointness were verified before freezing. No pre-freeze scene previews or depth-distribution analysis occurred. Downloads originals remain intact; staging is external to the repository. No failed-HDD mount, repair or write was used.',
        table([{'trajectory':k,'frames':v} for k,v in protocol['test_realizations'].items()],[('trajectory','Reserved realization'),('frames','Frames')]),
        f"Protocol SHA-256: `{completion['protocol_sha256']}`. Manifest SHA-256: `{completion['manifest_sha256']}`. Frozen UTC: {protocol['frozen_utc']}. Exposure UTC: {json.loads((FOLDER/'EXPOSURE.json').read_text())['started_utc']}. Completed UTC: {completion['completed_utc']}.",
        table([{'model':k,'epoch':v['selected_epoch'],'sha':v['sha256']} for k,v in protocol['models'].items()],[('model','Model'),('epoch','Epoch'),('sha','Weights SHA-256')]),
        'The exclusive exposure ledger preceded decoding. The runner rejects existing exposure/output and has no automatic resume/retry. The original blocked proposal/specification and historical 4,436-frame test remain unchanged. The compatibility receipt permits inference from the exact rejected-teacher KD checkpoint with AST-equivalent source; it does not accept the teacher or permit training resume.',
        '## Metrics and quality',
        '192×320 full-frame RGB, one in-model ImageNet normalization, Euclidean range metres, FP32, valid finite GT [0.1,1000] m and near GT <10 m. No new crop, scaling or upper prediction clamp. Nonfinite predictions are fatal even outside the valid mask. AbsRel and strict ratio delta thresholds are pooled by valid pixels; RMSE comes from pooled squared error. Every current frame is scored, including reused maps. Counts agree across all seven comparisons. Classification precision/recall/F1 are not depth-regression metrics.',
    ]
    quality=[('comparison','Model / policy'),('absrel','AbsRel ↓'),('rmse','RMSE m ↓'),('delta1','δ1 ↑'),('delta2','δ2 ↑'),('delta3','δ3 ↑'),('near_absrel','Near AbsRel ↓'),('near_rmse','Near RMSE m ↓')]
    for scope in scopes:lines.extend(['### '+scope,table([r for r in rows if r['scope']==scope],quality)])
    lines+=['### Pixel counts (identical across all comparisons)',table([r for r in rows if r['comparison']=='A_every'],[('scope','Scope'),('frames','Frames'),('valid_pixels','Valid pixels'),('near_valid_pixels','Near pixels')]),'### Signed bias and floor diagnostics','Bias is P−GT in metres. Floor is P≤0.101 m on valid pixels; near-floor uses near pixels. Near ≥10 is the fraction of near-GT pixels predicted at least 10 m. Undefined denominators remain null.']
    for scope in scopes:
        lines += ['**'+scope+'**',table([r for r in rows if r['scope']==scope],[('comparison','Model / policy'),('bias_m','Bias m'),('near_bias_m','Near bias m'),('valid_floor_count','Floor pixels'),('valid_floor_rate','Floor fraction'),('near_floor_rate','Near floor fraction'),('near_prediction_ge_10m_rate','Near predicted ≥10 fraction')])]
    lines+=['## Frozen contrasts and interpretation','Signed relative error change is (candidate/reference−1)×100; negative values mean lower error. These are descriptive comparisons from one run, not confidence intervals.']
    changes=[]
    for scope in scopes:
        part=lambda key:results[key]['overall'] if scope=='pooled' else results[key]['per_trajectory'][scope]
        for candidate,reference in [('A_v2','A_budget_fixed'),('A_every','Original_every'),('KD_every','A_every')]:
            a=part(candidate);b=part(reference)
            changes.append({'scope':scope,'contrast':candidate+' vs '+reference,**{k:100*(a['metrics'][k]/b['metrics'][k]-1) for k in ['absrel','rmse','near_absrel','near_rmse']},'calls':a['inference_calls']-b['inference_calls']})
    lines.append(table(changes,[('scope','Scope'),('contrast','Contrast'),('absrel','AbsRel change %'),('rmse','RMSE change %'),('near_absrel','Near AbsRel change %'),('near_rmse','Near RMSE change %'),('calls','Call difference')]))
    a=results['A_every']['overall']['metrics'];o=results['Original_every']['overall']['metrics'];kd=results['KD_every']['overall']['metrics']
    direction=lambda x:'lower' if x<0 else 'higher' if x>0 else 'unchanged'
    lines.append(f"A versus Original: pooled AbsRel is {direction(a['absrel']-o['absrel'])} ({100*(a['absrel']/o['absrel']-1):+.2f}%) and near RMSE is {direction(a['near_rmse']-o['near_rmse'])} ({100*(a['near_rmse']/o['near_rmse']-1):+.2f}%). Original remains the predeclared near-field reference; this does not guarantee it ranks first on every new path.")
    lines.append(f"KD versus A: pooled AbsRel is {direction(kd['absrel']-a['absrel'])} ({100*(kd['absrel']/a['absrel']-1):+.2f}%) and near RMSE is {direction(kd['near_rmse']-a['near_rmse'])} ({100*(kd['near_rmse']/a['near_rmse']-1):+.2f}%). On validation both were worse (+1.59% and +1.42% respectively). Any improvement here is a test reversal and does not erase the negative validation result or failed teacher gate.")
    lines.append('Inspect all four subgroup rows: pooled pixels can conceal path regressions. V2 and budget-fixed need not use equal test calls; the comparator was frozen from validation, independently of observed adaptive calls. Reuse can incidentally reduce error on correlated frames, so lower average error is not evidence of correct hazard detection. No model/policy reselection follows these outcomes.')
    lines.append('Subgroup caution: A regresses on Kite sunny, with AbsRel +10.80% and near RMSE +79.29% versus Original. V2 near RMSE is worse than budget-fixed on Kite foggy, Kite sunny and PLE spring; its PLE winter improvement produces the small pooled near-RMSE benefit. The sunny trajectory has zero defined critical events, so event rates/delays there remain undefined rather than zero-error claims.')
    lines+=['## Calls, freshness and events','Events are GT p10<8 m on entry or a ≥2 m drop from the previous frame, resetting per path. GT events are offline labels only. Delay is time to the next fresh update in the same path; terminal unresolved events are right-censored. No-event rates/delays stay undefined. A missed update is not a missed hazard detection.']
    for scope in scopes:
        lines+=['### '+scope,table([r for r in rows if r['scope']==scope],[('comparison','Model / policy'),('inference_calls','Calls'),('skip_rate','Skip fraction'),('mean_prediction_age','Mean age'),('max_prediction_age','Max age'),('events','Events'),('fresh_on_event','Fresh on event'),('missed_update_rate','Missed-update fraction'),('mean_delay','Mean delay frames'),('max_delay','Max delay'),('censored','Censored')])]
    lines+=['All inference/skip reasons and individual observed/censored events remain in `results.json` and each `run/*_frames.csv`. Source indices, ages and current-frame pixel counts are retained for every comparison; the verifier checks their consistency.','## Actual measured timing',
        'Single chronological traversal, rotated comparison order, CPU FP32 batch 1, two Torch threads. Ten synthetic warmups per model were excluded before exposure. Each comparison executed its own real fresh model calls, with no full prediction lookup cache. Inference includes transfer-in, forward and transfer-out; scheduler includes reset/decision and fresh risk-cache update. Compute is inference + scheduler. Shared I/O/decode are measured once per frame. Offline scores, events and CSV writes are excluded from per-policy compute. Whole-matrix wall includes all seven policies and cannot be reported as single-policy throughput. OS cache/load are uncontrolled; one timing pass has no repeatability interval.']
    for scope in scopes:
        lines+=['**'+scope+'**',table([r for r in rows if r['scope']==scope],[('comparison','Model / policy'),('inference_seconds','Inference s'),('scheduler_seconds','Scheduler s'),('compute_seconds','Compute s'),('model_seconds','Forward s'),('transfer_in_seconds','Transfer-in s'),('transfer_out_seconds','Transfer-out s'),('model_call_median_ms','Median forward ms')])]
    lines += [table([timing['shared']],[('shared_io_seconds','Shared I/O s'),('shared_decode_seconds','Shared decode/resize s'),('matrix_wall_seconds','Whole seven-comparison wall s')]),'Hardware/runtime: `'+json.dumps(timing['hardware'],sort_keys=True)+'`. CUDA was unavailable to this runtime; historical validation CUDA timings remain separately labelled. No energy/battery savings or flight-safety claim is supported.','## Limits and evidence',
        'Four reserved simulated paths, one seed, validation-selected checkpoints and scheduler settings, correlated neighbouring frames/weather and possible shared scenery. Pixel counts are not independent samples. The original model used different training data/budget. The teacher failed its original near-field gate; KD remains exploratory. The historical exposed test stays quarantined and separate. No training, tuning, teacher repair, KD rerun, split redraw or automatic exposed-study retry occurred.',
        '[Ready protocol](PROTOCOL.json) · [Exposure](EXPOSURE.json) · [Completion](COMPLETION.json) · [Full comparison CSV](comparison.csv) · [Full results/reasons/events](results.json) · [Timing](timing.json) · [Inventory](INVENTORY.json) · [Preparation](prepared/PREPARED.json) · [Compatibility](COMPATIBILITY.json).',
        'Read-only audit: `python scripts/verify_independent_test.py`. See [reproduction requirements](../../docs/independent_final_test.md), [consolidated report](../../docs/FINAL_REPORT.md), and [code guide](../../CODE_GUIDE.md).']
    (FOLDER/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
    plot_results(results)
    print('Rendered REPORT.md and figures/quality_calls.png from saved evidence')


def plot_results(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['absrel','near_rmse'],['Pooled AbsRel ↓','Near RMSE (m) ↓']):
        for name in ['A_every','A_fixed2','A_fixed5','A_budget_fixed','A_v2']:
            part=results[name]['overall'];ax.scatter(part['inference_calls'],part['metrics'][metric],s=65)
            ax.annotate(name.replace('A_',''),(part['inference_calls'],part['metrics'][metric]),xytext=((5,-17) if name=='A_fixed2' else (5,10)),textcoords='offset points',fontsize=9)
        ax.set(xlabel='Actual fresh model calls / 8,820 frames',ylabel=title,title=title);ax.grid(alpha=.2);ax.margins(.2)
    fig.suptitle('Independent test · frozen A policies · one CPU pass')
    (FOLDER/'figures').mkdir(exist_ok=True);fig.savefig(FOLDER/'figures/quality_calls.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
