"""Saved-result comparison and figures only; no model or data loaders."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json,csv,os,math,hashlib,subprocess
ART=Path(__file__).resolve().parent;REPO=ART.parents[1];RES=REPO/'results'/ART.name
os.environ['MPLCONFIGDIR']=str(ART/'mpl-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(RES/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
 with (RES/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pooled(rows):
 n=sum(r['valid_pixels'] for r in rows)
 if not n:return {'valid_pixels':0,'floor_count':0,**{k:None for k in ['absrel','rmse','bias_m','floor_rate','delta1','delta2','delta3']}}
 result={k:sum(r['valid_pixels']*r[k] for r in rows if r['valid_pixels'])/n for k in rows[0] if k not in ['valid_pixels','floor_count','rmse']}
 result.update(valid_pixels=n,floor_count=sum(r.get('floor_count',0) for r in rows),rmse=math.sqrt(sum(r['valid_pixels']*r['rmse']**2 for r in rows if r['valid_pixels'])/n))
 return result
P=load(RES/'PLAN.json');OLD=Path(P['reuse_root']);B=Path(P['KD_bundle']);assert load(RES/'completion.json')['passed']
old=load(OLD/'metrics.json');oldsupp=load(OLD/'pooled_supplementary.json');kd=load(RES/'KD_computed_metrics.json')
raw={k:old[k] for k in ['epoch4','student_A','teacher_B']};raw['exploratory_KD']=kd
names=['epoch4','student_A','exploratory_KD'];labels={'epoch4':'Original epoch4','student_A':'Supervised A epoch9','exploratory_KD':'Exploratory KD epoch9','teacher_B':'Rejected teacher B epoch3'}
trs=list(old['epoch4']['per_trajectory']);scopes=['pooled',*trs];bands=[x[0] for x in P['contract']['depth_bands']]
metrics={};table=[];bandrows=[]
for name in [*names,'teacher_B']:
 data=raw[name]
 supplementary=oldsupp[name] if name!='exploratory_KD' else {key:pooled([data['supplementary'][tr][key] for tr in trs]) for key in ['all_valid','near']}
 if name=='exploratory_KD':supplementary['depth_bands']={b:pooled([data['supplementary'][tr]['depth_bands'][b] for tr in trs]) for b in bands}
 byscope={}
 for scope in scopes:
  core=data['overall'] if scope=='pooled' else data['per_trajectory'][scope];supp=supplementary if scope=='pooled' else data['supplementary'][scope]
  row={**core,'near_bias_m':supp['near']['bias_m'],'valid_floor_count':supp['all_valid']['floor_count'],'valid_floor_rate':supp['all_valid']['floor_rate'],'near_floor_rate':supp['near']['floor_rate'],'near_prediction_ge_10m_rate':supp['near'].get('prediction_ge_10m_rate'),'depth_bands':supp['depth_bands']}
  assert row['valid_pixels']==supp['all_valid']['valid_pixels'] and row['near_valid_pixels']==supp['near']['valid_pixels']
  assert row['valid_pixels']==sum(x['valid_pixels'] for x in row['depth_bands'].values())
  assert row['near_valid_pixels']==row['depth_bands'][bands[0]]['valid_pixels']
  byscope[scope]=row
  if name in names:
   table.append({'scope':scope,'model':name,**{k:v for k,v in row.items() if k!='depth_bands'}})
   for band,d in row['depth_bands'].items():bandrows.append({'scope':scope,'model':name,'band':band,**{key:d.get(key) for key in ['valid_pixels','absrel','rmse','delta1','delta2','delta3','bias_m','floor_count','floor_rate']}})
 metrics[name]={'selected_checkpoint':P['checkpoints'].get(name,load(OLD/'identities.json')['teacher_B']),'reuse':name!='exploratory_KD','scopes':byscope}
for scope in scopes:
 assert len({(metrics[name]['scopes'][scope]['valid_pixels'],metrics[name]['scopes'][scope]['near_valid_pixels']) for name in metrics})==1
save('metrics.json',{'students':{k:metrics[k] for k in names},'teacher_context_reused_no_inference':metrics['teacher_B'],'contract':P['contract'],'undefined_representation':None,'CSV_undefined':'empty field represents JSON null'})
csvsave('comparison.csv',table);csvsave('depth_band_metrics.csv',bandrows)
comparisons=[]
for newer,base,title in [('exploratory_KD','student_A','KD_minus_A'),('student_A','epoch4','A_minus_original')]:
 for scope in scopes:
  a=metrics[base]['scopes'][scope];b=metrics[newer]['scopes'][scope]
  for key in [k for k in a if k!='depth_bands']:
   v,w=a[key],b[key];delta=None if v is None or w is None else w-v
   pct=None if delta is None or v==0 else 100*delta/abs(v)
   comparisons.append({'comparison':title,'scope':scope,'metric':key,'baseline':v,'new':w,'absolute_change':delta,'relative_change_percent_abs_baseline':pct})
 for scope in scopes:
  for band in bands:
   a=metrics[base]['scopes'][scope]['depth_bands'][band];b=metrics[newer]['scopes'][scope]['depth_bands'][band]
   for key in ['absrel','rmse','delta1','delta2','delta3','bias_m','floor_rate']:
    v,w=a.get(key),b.get(key);delta=None if v is None or w is None else w-v;pct=None if delta is None or v==0 else 100*delta/abs(v)
    comparisons.append({'comparison':title,'scope':scope+' / '+band,'metric':key,'baseline':v,'new':w,'absolute_change':delta,'relative_change_percent_abs_baseline':pct})
csvsave('changes.csv',comparisons);save('changes.json',{'convention':'new minus baseline; relative percent uses abs(baseline), undefined when baseline zero/null. Error improvement negative; accuracy improvement positive. Rate differences are fractions (multiply by100 for percentage points). Signed-bias relative change is not an error improvement score.','rows':comparisons})
selection=load(RES/'frame_selection.json');examples=load(RES/'example_verification.json')['examples'];lookup={(x['trajectory'],x['frame_index']):x for x in examples}
fig,axes=plt.subplots(6,8,figsize=(25,13));fig.subplots_adjust(left=.105,right=.985,top=.925,bottom=.13,wspace=.025,hspace=.25)
range_cmap=plt.get_cmap('viridis').copy();range_cmap.set_bad('#bcbcbc');error_cmap=plt.get_cmap('RdBu_r').copy();error_cmap.set_bad('#bcbcbc')
records=[]
for row,s in enumerate(selection['examples']):
 item=lookup[(s['trajectory'],s['frame_index'])]
 with np.load(ART/item['arrays']) as z:arr={k:z[k] for k in z.files}
 gt=arr['gt'];valid=arr['valid'];near=valid&(gt<10);axes[row,0].imshow(arr['rgb'])
 scope='Kite foggy' if s['trajectory'].startswith('Kite') else 'PLE spring'
 axes[row,0].set_ylabel(f'{scope} / {s["frame_index"]}\nPrior {s["reason"]}; near n={int(near.sum()):,}',fontsize=8)
 range_im=axes[row,1].imshow(np.ma.masked_where(~valid,gt),cmap=range_cmap,norm=LogNorm(vmin=.1,vmax=1000))
 for col,name in enumerate(names,2):
  axes[row,col].imshow(np.ma.masked_where(~valid,arr[name]),cmap=range_cmap,norm=LogNorm(vmin=.1,vmax=1000))
  e=(arr[name].astype(np.float64)-gt.astype(np.float64))[near]
  axes[row,col].text(.02,.04,f'Near RMSE {math.sqrt(float(np.mean(e*e))):.2f}m; bias {float(e.mean()):+.2f}m',transform=axes[row,col].transAxes,fontsize=6,color='white',bbox={'facecolor':'black','alpha':.65,'pad':1})
 for col in range(1,5):axes[row,col].contour(near,levels=[.5],colors='#ffff00',linewidths=.4)
 for col,name in enumerate(names,5):
  error_im=axes[row,col].imshow(np.ma.masked_where(~near,arr[name].astype(np.float64)-gt.astype(np.float64)),cmap=error_cmap,vmin=-10,vmax=10)
 for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
 records.append({'trajectory':s['trajectory'],'frame_index':s['frame_index'],'reason_from_prior_teacher_audit':s['reason'],'near_pixels':int(near.sum()),'input_sha256':item['input_sha256'],'cache_sha256':sha(ART/item['arrays'])})
for col,title in enumerate(['RGB','GT range','Original range','A range','Exploratory KD range','Original near error','A near error','KD near error']):axes[0,col].set_title(title,fontsize=10)
fig.colorbar(range_im,cax=fig.add_axes([.14,.073,.36,.012]),orientation='horizontal',label='Euclidean range, m; log colour 0.1–1000; yellow outline = valid GT<10m')
fig.colorbar(error_im,cax=fig.add_axes([.59,.073,.34,.012]),orientation='horizontal',label='Prediction − GT, m on GT<10m; colours clipped ±10; grey = outside mask')
fig.suptitle('Three selected students on identical fixed validation frames — KD is exploratory with rejected teacher\nFrames fixed by the earlier teacher audit; no KD-based selection. Values used in metrics are never visually clipped.',fontsize=13)
fig.text(.5,.008,'MidAir — Fonder & Van Droogenbroeck, CC BY-NC-SA 4.0. Validation only; no test imagery or model promotion.',ha='center',fontsize=9)
fig.savefig(RES/'figures/student_contact_sheet.png',dpi=180);plt.close(fig)
save('visualization_record.json',{'examples':records,'scales':{k:v for k,v in selection.items() if k not in ['examples']},'figures':['figures/student_contact_sheet.png'],'cached_RGB_GT_prediction_only_from_allowed_validation':True})
colours=['#606c86','#12827b','#d87822'];scope_labels=['Pooled','Kite foggy','PLE spring']
fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for col,scope in enumerate(scopes):
 for i,name in enumerate(names):
  d=metrics[name]['scopes'][scope]['depth_bands'];ys=[d[band].get('absrel') for band in bands];zs=[d[band].get('rmse') for band in bands]
  axs[0,col].plot(range(5),ys,'o-',label=labels[name],color=colours[i]);axs[1,col].plot(range(5),zs,'o-',color=colours[i])
 axs[0,col].set_title(scope_labels[col]);axs[1,col].set_yscale('log')
 for ax in axs[:,col]:ax.set_xticks(range(5),['<10','10–50','50–100','100–300','300–1000']);ax.set_xlabel('GT range band, m');ax.grid(alpha=.2)
axs[0,0].set_ylabel('AbsRel');axs[1,0].set_ylabel('RMSE, m (log)');axs[0,0].legend(fontsize=8);fig.suptitle('Existing GT bands; identical counts across selected students; validation only')
fig.savefig(RES/'figures/depth_bands.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,(key,title) in zip(axs.ravel(),[('absrel','Overall AbsRel'),('rmse','Overall RMSE, m'),('near_absrel','Near AbsRel'),('near_rmse','Near RMSE, m')]):
 for i,name in enumerate(names):ax.bar(np.arange(3)+(i-1)*.24,[metrics[name]['scopes'][sc][key] for sc in scopes],width=.24,label=labels[name],color=colours[i])
 ax.set_xticks(range(3),scope_labels);ax.set_title(title);ax.grid(axis='y',alpha=.2)
axs[0,0].legend(fontsize=8);fig.suptitle('Overall versus near-field trade-offs; lower error is better; KD exploratory')
fig.savefig(RES/'figures/overall_near_tradeoffs.png',dpi=180);plt.close(fig)
# Detailed report uses saved results only; original scientific evidence is untouched.
fmt=lambda x,d=6:'null' if x is None else f'{x:.{d}f}'
lines=['# Final selected-student validation comparison','',f"Completed {datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5,minutes=30))).isoformat()} (IST). Validation only; no training, teacher repair, scheduler tuning, test access, new checkpoint selection or automatic promotion.",'','The official selections remain original epoch4, supervised A epoch9 and exploratory KD epoch9. Checkpoint byte hashes, saved selected-epoch records, parent sources/model contracts and KD completion/authorization were verified before any pixel inference. Actual paths come from the prior audit and recovered exploratory authorization, not the unmounted failed HDD. Handoff is the dated startup record; completion.json and10-epoch history provide the newer completion truth.','', 'Original/A/teacher complete metrics and supplementary statistics were reused from the checksum-verified prior validation audit. Only KD needed a full4410-frame inference pass for bias, floor and depth bands. Its recomputed standard scores were compared to the existing selected checkpoint record. Original/A were replayed only on the six original two-frame batches (12 frames/model), to create the contact sheet and verify existing per-frame scores; teacher was not evaluated. No other checkpoint was loaded for selection.','', 'All inference uses FP32 eval/inference mode, batch2, no autocast/TF32,192×320 full frame, unchanged normalization, GT decoding and finite inclusive[0.1,1000]m mask before nearest resize, near GT<10m. Delta rates use strict max(P/G,G/P)<1.25^k; metrics pool pixels and squared errors, rather than averaging frame/path RMSE. Floor rate is prediction<=0.101m divided by valid GT pixels. Near bias is mean(P−GT) on valid GT<10m, positive meaning overestimated range. Undefined JSON values are null; corresponding CSV cells are blank. Colour limits never clip the underlying metric predictions.','', '| Scope | Model | AbsRel | RMSE m | δ1 | δ2 | δ3 | Near AbsRel | Near RMSE m |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for scope,short in zip(scopes,scope_labels):
 for name in names:
  d=metrics[name]['scopes'][scope];lines.append('| '+ ' | '.join([short,labels[name],*[fmt(d[k]) for k in ['absrel','rmse','delta1','delta2','delta3','near_absrel','near_rmse']]])+' |')
lines+=['','| Scope | Model | Valid pixels | Near pixels | Near bias m | Valid floor count | Valid floor % |','|---|---|---:|---:|---:|---:|---:|']
for scope,short in zip(scopes,scope_labels):
 for name in names:
  d=metrics[name]['scopes'][scope];lines.append(f"| {short} | {labels[name]} | {d['valid_pixels']:,} | {d['near_valid_pixels']:,} | {fmt(d['near_bias_m'])} | {d['valid_floor_count']:,} | {100*d['valid_floor_rate']:.6f} |")
lines+=['','Pixel counts match exactly across all models, cached audit, KD saved record and frozen validation reference: pooled212,903,814/32,267,840 valid/near; Kite114,412,073/894,287; PLE98,491,741/31,373,553. Band counts partition each valid count exactly. Only83 consecutive Kite frames1526–1608 contain near GT; their GT range is6.715–9.992m. They supply2.77% of pooled near pixels;97.23% come from PLE. These correlated frames and two familiar trajectories are not independent generalization evidence. No count discrepancy was omitted.','', '| Scope | Change | Δ AbsRel (relative %) | Δ RMSE m (relative %) | Δ Near AbsRel (relative %) | Δ Near RMSE m (relative %) | Δ Near bias m | Δ Floor percentage points |','|---|---|---:|---:|---:|---:|---:|---:|']
for scope,short in zip(scopes,scope_labels):
 for new,base,title in [('exploratory_KD','student_A','KD − A'),('student_A','epoch4','A − original')]:
  dd=[]
  for key in ['absrel','rmse','near_absrel','near_rmse']:
   v=metrics[base]['scopes'][scope][key];w=metrics[new]['scopes'][scope][key];dd.append(f'{w-v:+.6f} ({100*(w-v)/abs(v):+.3f}%)')
  v=metrics[base]['scopes'][scope];w=metrics[new]['scopes'][scope]
  lines.append('| '+' | '.join([short,title,*dd,f"{w['near_bias_m']-v['near_bias_m']:+.6f}",f"{100*(w['valid_floor_rate']-v['valid_floor_rate']):+.6f}"])+' |')
lines+=['','All requested absolute and relative changes, including delta rates, counts, bias, floor rates and every band, are in changes.csv/changes.json. Relative changes divide by absolute baseline and stay null for zero/null baselines; a signed bias change is not an error-improvement score. Rate differences in CSV are fractions; display tables convert floor differences to percentage points.','', '| Scope | GT band m | Valid pixels | Original AbsRel / RMSE m | A AbsRel / RMSE m | Exploratory KD AbsRel / RMSE m |','|---|---|---:|---:|---:|---:|']
for scope,short in zip(scopes,scope_labels):
 for band in bands:
  dd=[metrics[name]['scopes'][scope]['depth_bands'][band] for name in names];assert len({v['valid_pixels'] for v in dd})==1
  cells=[f"{fmt(v.get('absrel'))} / {fmt(v.get('rmse'))}" for v in dd]
  lines.append('| '+' | '.join([short,band,f"{dd[0]['valid_pixels']:,}",*cells])+' |')
lines+=['','Full band δ1/2/3, bias and floor data are in depth_band_metrics.csv and machine-readable metrics.json. The first band[0.1,10) is exactly the existing near subset; final band includes1000m.','', 'Teacher context (reused; no teacher inference): selected B epoch3 remains rejected under the original gate.','', '| Scope | Teacher AbsRel | Teacher RMSE m | Teacher near AbsRel | Teacher near RMSE m | Teacher near bias m |','|---|---:|---:|---:|---:|---:|']
for scope,short in zip(scopes,scope_labels):
 d=metrics['teacher_B']['scopes'][scope];lines.append('| '+' | '.join([short,*[fmt(d[k]) for k in ['absrel','rmse','near_absrel','near_rmse','near_bias_m']]])+' |')
lines+=['','Known rejection: B Kite near AbsRel0.742757 and RMSE6.636090m exceed A0.504336 and4.695176m despite better pooled/overall metrics. The original failed gate/hash and protocol remain unchanged. Exploratory KD is a separately authorized rejected-teacher run and does not retrospectively become the original accepted-teacher KD arm.','', '**Figures and interpretation:** [contact sheet](figures/student_contact_sheet.png), [overall/near trade-offs](figures/overall_near_tradeoffs.png), [depth bands](figures/depth_bands.png). The six exact teacher-audit frames were fixed before KD existed: Kite1567 typical,1534 poor,1577 regression; PLE445 typical,1691 poor,1928 regression. These are selected stress examples, not a representative validation sample. Frame-selection records include original batch partners and input/cache hashes. Shared log range colour[0.1,1000]m, signed near-error colours±10m, invalid/outside-near grey, near boundary yellow. MidAir attribution is retained.','', '**Measured findings:** See the tables for trade-offs. Pooled overall selection rewards A relative to the original but does not preserve its near-field accuracy. KD must be judged separately on each trajectory and near group, not inferred from the rejected teacher’s pooled advantage. A lower signed bias alone can coexist with larger error variance/RMSE. The gradient probe did not support a teacher-repair run; this comparison does not overturn it. No causal claim that distillation, fog, BN or a specific loss term alone produced any subgroup change follows from one trained seed and these familiar two validation paths.','', '**Demonstration proposal:** Use supervised A epoch9 as the primary full-depth visualization candidate because it has the lowest pooled validation AbsRel among the three students and improves overall RMSE over the original. Keep the original epoch4 as the explicit near-field reference/toggle: A sacrifices near accuracy, especially in fog, so do not label it a proximity-safety upgrade. KD remains an exploratory comparison rather than a preferred default on pooled AbsRel. This is a recommendation only; no export, published-weight change, config promotion or deployment was made. The demo should visibly show model identity, metric range units, stale/fresh state if scheduling is used, and validation-only/fog/near limitations; it is not a navigation or collision-avoidance system.','', '**Exact remaining work (none executed or newly authorized here):**','', '1. Scheduler ablations: retain completed original V1/V2 and negative results unchanged. For the chosen new student, preregister a separate validation-only causal replay using unchanged event/mask definitions, no privileged future/GT scheduler inputs, per-sequence reset and the frozen policy settings. Compare every-frame, fixed2, fixed5, original validation-budget fixed rule, and V2; include RGB-only, motion-only and combined controls to isolate the previously joint changes. Report calls, stale-depth pooled/per-path/near errors and event-update delays. Freeze any new validation selection before test; no scheduler search or retraining is authorized now. Historical epoch4 policy results do not establish A/KD scheduler behavior.', '2. Timing: measure the selected exact export on the intended laptop/runtime, FP32 eval batch1, fixed threads/device, synchronized CUDA when used, identical excluded warmups and at least3 alternating measured repetitions for adaptive versus fixed comparisons. Separate model-only inference, scheduler overhead, preprocessing/I/O and end-to-end latency; report distribution, memory, power/thermal/storage conditions and call counts. Historical CPU scheduler timing and short training benchmarks do not substitute for this matched GPU/CPU deployment timing. No energy/battery claims without direct measurements.', '3. Final testing: preserve original exposed test study/ledger and do not reuse it as untouched selection data. Verify healthy storage/acquisition for reserved independent new groups Kite0007/0008, PLE0008/0009 without inspecting their RGB/depth during planning; current expanded TRAIN/VAL manifest has zero test records. Freeze final model/checkpoint, preprocessing, policies, comparisons, analysis and one-use exposure protocol before separately authorized test access. Acquisition availability and expected8820 frames must be verified, not assumed or forced. A separate guarded test manifest/receipt and exposure ledger are prerequisites. No new test result exists from this task.', '4. Demonstration: after a human accepts the explicit trade-off, create a separate pinned inference export/config and model card, verify byte/state identity and input/output parity, wire it to the visualization with honest stale/fresh indicators and caveats, then perform the bounded inference smoke/timing above on permitted inputs. Keep published original weights untouched unless a later explicit promotion is authorized. Teacher online inference is not needed to deploy any student. There is no remaining authorized training step.', '', 'Provenance: PLAN/FREEZE, source_inputs, identities, configurations, runtime, completion, saved-reference differences, example verification, visualization record, execution record and preservation verification accompany this report. All original checkpoints/protocols/gates/audits remain separate. The checks verify full model/BN state unchanged and persistent bytes before/after; training and optimizer instances=0. Exact commands and any runtime differences are recorded.']
(RES/'REPORT.md').write_text('\n'.join(lines)+'\n')
save('analysis_completion.json',{'passed':True,'student_models':names,'scopes':scopes,'full_KD_frames':4410,'cached_original_A_teacher_metrics_reused':True,'figures':3,'selection_changes':0,'demonstration_recommendation':'supervised_A_epoch9_with_original_epoch4_near_reference','promotion_performed':False})
print(json.dumps({'report':str(RES/'REPORT.md'),'pooled':{n:metrics[n]['scopes']['pooled']|{'depth_bands':'see metrics.json'} for n in names}},indent=2))
