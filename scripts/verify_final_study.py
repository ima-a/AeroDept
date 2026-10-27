"""Audit saved final-study accounting without reading images or invoking a model."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=Path('artifacts/final_study/test'))
    parser.add_argument('--protocol', type=Path, default=Path('results/final_test/protocol.json'))
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve existing verification')
    p = read(a.protocol); completion = read(a.study_dir/'completed.json')
    assert sha(a.protocol) == completion['protocol_sha256']
    assert sha(a.protocol) == a.protocol.with_suffix('.sha256').read_text().split()[0]
    assert sha(Path(p['model']['weights'])) == p['model']['sha256']
    source = read(a.study_dir/'execution.json')['source_hashes']
    assert all(sha(Path(name)) == expected for name, expected in source.items())
    baseline = read(a.study_dir/'every.json')
    checks, total_calls = {}, 0
    for policy in completion['policies']:
        result = read(a.study_dir/f'{policy}.json')
        with (a.study_dir/f'{policy}_frames.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == completion['frames'] == p['manifest']['test_frames']
        assert [int(r['index']) for r in rows] == list(range(len(rows)))
        calls = sum(r['infer']=='True' for r in rows); total_calls += calls
        assert calls == result['overall']['inference_calls']
        assert result['overall']['metrics']['valid_pixels'] == baseline['overall']['metrics']['valid_pixels']
        assert result['overall']['metrics']['near_valid_pixels'] == baseline['overall']['metrics']['near_valid_pixels']
        for tr, x in result['per_trajectory'].items():
            rr = [r for r in rows if r['trajectory']==tr]; n = len(rr)
            assert [int(r['position']) for r in rr] == list(range(n))
            assert n == x['frames'] and sum(r['infer']=='True' for r in rr) == x['inference_calls']
            source_index, previous_p10 = None, math.inf
            k = (n*p['fixed_budget']['numerator'] + p['fixed_budget']['denominator']-1)//p['fixed_budget']['denominator']
            expected_positions = {(i*n+k-1)//k for i in range(k)}
            events, delays, censored = [], [], []
            for pos, r in enumerate(rr):
                infer = r['infer']=='True'
                if policy != 'v2_selected':
                    expected = pos in expected_positions if policy=='fixed_validation_budget' else pos % {'every':1,'fixed2':2,'fixed5':5}[policy]==0
                    assert infer == expected
                if infer:source_index = int(r['index'])
                age = int(r['index'])-source_index
                assert age == int(r['prediction_age']) and int(r['source_index'])==source_index
                assert age <= (0 if policy=='every' else 1 if policy in ['fixed2','fixed_validation_budget'] else 4)
                assert (float(r['inference_ms'])>0) == infer and float(r['scheduler_ms'])>=0
                current = float(r['gt_p10']) if r['gt_p10'] else math.inf
                event = current<8 and (previous_p10>=8 or previous_p10-current>=2)
                assert event == (r['critical_event']=='True'); previous_p10=current
                if event:events.append(pos)
            for pos in events:
                next_call=next((j for j in range(pos,n) if rr[j]['infer']=='True'),None)
                if next_call is None:censored.append(pos)
                else:delays.append(next_call-pos)
            ev=x['critical_events'];fresh=sum(rr[j]['infer']=='True' for j in events)
            assert ev['count']==len(events) and ev['fresh_on_event']==fresh
            assert ev['observed_delays_frames']==delays and ev['right_censored_count']==len(censored)
            assert ev['missed_update_rate']==((len(events)-fresh)/len(events) if events else None)
            assert x['max_prediction_age']==max(int(r['prediction_age']) for r in rr)
            assert math.isclose(x['mean_prediction_age'],sum(int(r['prediction_age']) for r in rr)/n)
            for field, column in [('scheduler_seconds','scheduler_ms'),('inference_seconds','inference_ms')]:
                assert math.isclose(x[field],sum(float(r[column]) for r in rr)/1000,abs_tol=1e-9)
            for metric in ['valid_pixels','near_valid_pixels']:
                assert x['metrics'][metric]==baseline['per_trajectory'][tr]['metrics'][metric]
        # Independent pooled arithmetic check from reported per-trajectory metrics/counts.
        per=list(result['per_trajectory'].values());overall=result['overall']['metrics']
        for key, weight, square in [('absrel','valid_pixels',False),('rmse','valid_pixels',True),
                                    ('near_absrel','near_valid_pixels',False),('near_rmse','near_valid_pixels',True),
                                    ('delta1','valid_pixels',False),('delta2','valid_pixels',False),('delta3','valid_pixels',False)]:
            total=sum(v['metrics'][weight] for v in per)
            if total:
                pooled=sum(v['metrics'][weight]*(v['metrics'][key]**2 if square else v['metrics'][key]) for v in per)/total
                assert math.isclose(overall[key],math.sqrt(pooled) if square else pooled,rel_tol=1e-12)
        checks[policy]={'frames':len(rows),'calls':calls,'valid_pixels':overall['valid_pixels'],
                        'near_valid_pixels':overall['near_valid_pixels'],'passed':True}
    assert total_calls==completion['actual_measured_model_calls']
    a.output.write_text(json.dumps({'passed':True,'protocol_sha256':sha(a.protocol),'actual_measured_calls':total_calls,
                                   'excluded_warmup_calls':completion['excluded_warmup_calls'],'policies':checks,
                                   'checks':'Model/source identity; frame coverage; independent fixed cadence; call/source/age accounting; GT event definition and within-trajectory delays/censoring; component timings; equal pixel counts; pooled metrics. No image/depth reads or model calls.'},indent=2)+'\n')
    print('Saved-result accounting PASS:', total_calls, 'actual measured calls')


if __name__=='__main__':
    main()
