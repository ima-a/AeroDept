"""Local artifact-driven AeroDepth presentation; explicit inference button only."""
from pathlib import Path
import os,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import streamlit as st
import pandas as pd
import numpy as np
import torch
from aerodepth_lite.presentation_ui import optional_json,load_dashboard_config,validation_features,recorded_frame,inference_resource,actual_inference,colourize

st.set_page_config(page_title='AeroDepth-Lite · Research studio',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1440px}
[data-testid="stSidebar"]{background:#0b1422;border-right:1px solid #223249}
[data-testid="stMetric"]{background:linear-gradient(135deg,#152439,#101b2c);border:1px solid #2b4058;border-radius:14px;padding:17px;min-height:130px}
[data-testid="stMetricLabel"]{color:#b4c5d8;font-size:.85rem}
[data-testid="stMetricValue"]{font-size:2rem;font-variant-numeric:tabular-nums}
h1{letter-spacing:-.045em;font-weight:750}h2,h3{letter-spacing:-.025em}
[data-testid="stTabs"] [role="tablist"]{gap:1rem;border-bottom:1px solid #2b4058}
[data-testid="stTabs"] button{font-weight:600;padding:.8rem .1rem;white-space:nowrap}
.eyebrow{color:#49d8be;letter-spacing:.19em;font-size:11px;font-weight:750;margin-bottom:.5rem}
.status-row{display:flex;gap:.5rem;flex-wrap:wrap;margin:.3rem 0 1.25rem}
.status-pill{border:1px solid #355268;border-radius:50px;padding:5px 11px;color:#b9cbdc;background:#101d2c;font-size:12px}
.status-pill.ready{color:#76e2c9;border-color:#255e58}.status-pill.pending{color:#e9ce8a;border-color:#655631}
.legend{display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#b4c5d8;font-size:12px;margin:.7rem 0}
.range-scale{display:inline-block;width:130px;height:9px;border-radius:6px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)}
.error-scale{display:inline-block;width:130px;height:9px;border-radius:6px;background:linear-gradient(90deg,#2166ac,#f7f7f7,#b2182b)}
@media(max-width:760px){.block-container{padding-top:1rem}[data-testid="stMetric"]{min-height:110px}[data-testid="stMetricValue"]{font-size:1.5rem}}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">AERODEPTH-LITE / RESEARCH STUDIO</div>',unsafe_allow_html=True)
st.title('See the depth. Understand the trade-off.')
st.caption('Monocular camera range · causal inference scheduling · a local, evidence-led demonstration')
# Keep this laptop's prepared config; relocated checkouts use portable paths.
local=ROOT/'configs/presentation_local.json';local_data=optional_json(local)
local_matches=local_data and Path(local_data.get('results','/')).is_relative_to(ROOT)
default_config=local if local_matches else ROOT/'configs/presentation.json'
config_path=Path(os.environ.get('AERODEPTH_PRESENTATION_CONFIG',default_config))
cfg=load_dashboard_config(config_path,ROOT)
if cfg is None:
    st.warning('Presentation configuration unavailable. Set AERODEPTH_PRESENTATION_CONFIG to a prepared local JSON config.');st.stop()
result=Path(cfg['results']);comparison=Path(cfg['comparison']);saved=optional_json(comparison/'metrics.json')
features=validation_features(cfg)
metrics=(saved or {}).get('students',{})
labels={'epoch4':'Original epoch 4','student_A':'Supervised A epoch 9','exploratory_KD':'Exploratory KD epoch 9'}
scopes=['pooled','Kite_training/foggy/trajectory_2003','PLE_training/spring/trajectory_5000']
scope_labels={'pooled':'Pooled','Kite_training/foggy/trajectory_2003':'Kite · foggy','PLE_training/spring/trajectory_5000':'PLE · spring'}
results=optional_json(result/'scheduler_results.json');timing=optional_json(result/'timing_summary.json');protocol=optional_json(result/'PROTOCOL.json')
summary=optional_json(result/'model_summary.json')
timing_active=(optional_json(result/'timing_activity.json') or {}).get('active',False)
delivery=Path(cfg.get('delivery',ROOT/'.unavailable/delivery'))
acquisition=optional_json(delivery/'acquisition_readiness.json')
final_status='NOT RUN · acquisition blocked' if acquisition else 'Pending · no completed independent study'
validation_status='VALIDATION · COMPLETED' if metrics and results else 'VALIDATION · RESULTS UNAVAILABLE'
st.markdown('<div class="status-row"><span class="status-pill ready">'+validation_status+'</span><span class="status-pill">FP32 · METRES</span><span class="status-pill pending">INDEPENDENT TEST · NOT RUN</span></div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown('### The demonstration')
    st.caption('AeroDepth-Lite · local research')
    st.markdown('**1 · Compare**\n\nKeep pooled and near-field errors together.\n\n**2 · Explore**\n\nView a validation frame, then explicitly run inference.\n\n**3 · Schedule**\n\nSee when an old prediction is reused.')
    st.divider()
    st.markdown('**Model roles**\n\n🟢 **A** · general-depth default\n\n🔵 **Original** · near-field reference\n\n🟠 **KD** · exploratory negative result')
    st.divider()
    st.caption('Independent final study')
    st.markdown('**'+final_status+'**')
    st.caption('Synthetic MidAir data and simulated navigation sensors. This is not a flight-safety demonstration.')
    written_report=delivery/'FINAL_REPORT.md'
    if not written_report.is_file():written_report=ROOT/'docs/FINAL_REPORT.md'
    if written_report.is_file():
        st.download_button('Download final written report',written_report.read_bytes(),file_name='AeroDepth_Lite_Final_Report.md')


def metric_rows(scope):
    return [{'Student':labels[n],**d['scopes'][scope]} for n,d in metrics.items() if scope in d['scopes']]


def unavailable(message='This result is unavailable or pending.'):
    st.info(message)


def show_depth(x,g,v,p,source):
    st.caption(source)
    cols=st.columns(4)
    with cols[0]:st.image(x,caption='Validation RGB',width='stretch')
    with cols[1]:st.image(colourize(g,v),caption='GT range · metres',width='stretch')
    if p is None:
        with cols[2]:unavailable('Recorded prediction unavailable for this model/frame. Use the actual inference button.')
    else:
        with cols[2]:st.image(colourize(p,v),caption='Predicted range · metres',width='stretch')
        with cols[3]:st.image(colourize(p-g,v&(g<10),'error'),caption='Signed error on GT <10 m',width='stretch')
    st.markdown('<div class="legend"><span>RANGE · log metres</span><span>0.1</span><span class="range-scale"></span><span>1000</span><span>NEAR ERROR · metres</span><span>−10</span><span class="error-scale"></span><span>+10</span></div>',unsafe_allow_html=True)
    st.caption('Near error: blue under / red over. Grey is outside the mask. Identical scales across models; colour clipping does not alter metrics.')


tabs=st.tabs(['Overview','Model comparison','Depth demonstration','Scheduler','Evidence'])
with tabs[0]:
    st.subheader('One small model. Three different outcomes.')
    st.write('The objective is to estimate metric range from drone RGB and examine when a model prediction can be reused, with measured quality/compute trade-offs. A ResNet-50 range teacher was trained to guide a MobileNetV3-Small student. The teacher failed its original acceptance gate; the later KD run is a separately authorized exploratory comparison with a negative result.')
    if metrics:
        old=metrics['epoch4']['scopes']['pooled'];a=metrics['student_A']['scopes']['pooled'];kd=metrics['exploratory_KD']['scopes']['pooled']
        cols=st.columns(3)
        cols[0].metric('Default demonstration', 'Supervised A',f"{100*(a['absrel']/old['absrel']-1):.2f}% pooled AbsRel",delta_color='inverse')
        cols[1].metric('Near-field trade-off',f"{a['near_rmse']:.3f} m",f"{100*(a['near_rmse']/old['near_rmse']-1):+.2f}% near RMSE",delta_color='inverse')
        cols[2].metric('KD vs supervised A',f"{kd['absrel']:.6f}",f"{100*(kd['absrel']/a['absrel']-1):+.2f}% AbsRel",delta_color='inverse')
        st.warning('A improves pooled overall accuracy while worsening near-field RMSE. Original epoch 4 remains the near-field reference. Fresh inference can still be wrong.')
    else:unavailable('Verified model-comparison metrics unavailable.')
    st.markdown('**Architecture:** RGB → ImageNet-normalized MobileNetV3-Small encoder → skip decoder → positive Euclidean camera range in metres. Teacher: ResNet-50 encoder + metric-range decoder; teacher is not needed for student inference.')
    assignments=optional_json(result/'dataset_assignments.json')
    if assignments:
        with st.expander('Dataset assignments · grouped paths, not independent weather samples',expanded=False):
            st.dataframe(pd.DataFrame(assignments['rows']).astype(str),hide_index=True,width='stretch')
    else:unavailable('Dataset assignment artifact unavailable.')
    if summary:
        with st.expander('Pinned model identities · selected checkpoints and inference exports'):
            st.dataframe(pd.DataFrame(summary['students']),hide_index=True,width='stretch')
    st.markdown('**Study status**')
    st.dataframe(pd.DataFrame([
        {'Study':'Student comparison','Status':'Completed' if metrics else 'Unavailable'},
        {'Study':'Nine A validation scheduler policies','Status':'Completed' if results else 'Pending'},
        {'Study':'Actual CPU / CUDA timing','Status':'Completed' if timing else 'Pending'},
        {'Study':'Original teacher gate','Status':'Rejected · preserved'},
        {'Study':'Independent reserved final test','Status':final_status},
    ]),hide_index=True,width='stretch')
    if acquisition:
        with st.expander('Why final testing is blocked · exact acquisition checklist'):
            st.write(acquisition['human_action'])
            st.dataframe(pd.DataFrame([{'Environment / condition':' / '.join(x['realization'].split('/')[:2]),'Official page ID':x['official_page_ID'],'Internal ID':x['realization'].split('/')[-1],'Needed':'Left RGB + depth archives'} for x in acquisition['requested']]),hide_index=True,width='stretch')
    st.caption('Only two familiar validation paths. Weather/scene samples are correlated. No measured energy, battery or flight-safety improvement is claimed.')

with tabs[1]:
    st.subheader('Keep overall and near-field results together.')
    scope=st.selectbox('Comparison scope',scopes,format_func=lambda s:scope_labels[s])
    rows=metric_rows(scope)
    if rows:
        columns=['Student','absrel','rmse','delta1','delta2','delta3','near_absrel','near_rmse','near_bias_m','valid_floor_rate','valid_pixels','near_valid_pixels']
        display_names={'absrel':'AbsRel ↓','rmse':'RMSE (m) ↓','delta1':'δ1 ↑','delta2':'δ2 ↑','delta3':'δ3 ↑','near_absrel':'Near AbsRel ↓','near_rmse':'Near RMSE (m) ↓','near_bias_m':'Near bias (m)','valid_floor_rate':'Floor rate','valid_pixels':'Valid pixels','near_valid_pixels':'Near pixels'}
        table=pd.DataFrame(rows)[columns].rename(columns=display_names)
        st.dataframe(table,hide_index=True,width='stretch',column_config={display_names[k]:st.column_config.NumberColumn(format='%.4f') for k in ['absrel','rmse','delta1','delta2','delta3','near_absrel','near_rmse','near_bias_m','valid_floor_rate']})
        st.caption('Lower error is better; delta accuracies higher is better. Near bias is signed P−GT in metres. Floor rate is a fraction on valid GT pixels, prediction ≤0.101 m. Undefined values stay null.')
        chart=pd.DataFrame([{'Student':r['Student'],'Overall AbsRel':r['absrel'],'Near AbsRel':r['near_absrel']} for r in rows]).set_index('Student')
        cols=st.columns(2)
        with cols[0]:st.markdown('**Overall versus near AbsRel**');st.bar_chart(chart,color=['#51cfb7','#f1bd6a'],height=270)
        with cols[1]:
            st.markdown('**Near-field RMSE · metres**')
            st.bar_chart(pd.DataFrame([{'Student':r['Student'],'Near RMSE (m)':r['near_rmse']} for r in rows]).set_index('Student'),color='#f1bd6a',height=270)
    else:unavailable()
    history=optional_json(result/'training_curves.json')
    if history:
        with st.expander('Recorded training histories · different data/budgets, not a controlled architecture ranking'):
            h=pd.DataFrame(history['rows']);wide=h.pivot(index='epoch',columns='model',values='validation_absrel');st.line_chart(wide)
            st.caption(history['precision_note'])
    else:unavailable('Training-curve records unavailable.')
    with st.expander('Depth bands and detailed comparison figures'):
        for filename in ['figures/overall_near_tradeoffs.png','figures/depth_bands.png']:
            if (comparison/filename).is_file():st.image(str(comparison/filename),width='stretch')

with tabs[2]:
    st.subheader('Recorded frames, or a real forward pass.')
    st.caption('MidAir imagery: Fonder & Van Droogenbroeck · CC BY-NC-SA 4.0 · validation inputs only.')
    controls=st.columns([2,1])
    with controls[0]:student=st.selectbox('Student model',['A','Original','KD'],format_func=lambda n:cfg['models'][n]['label'])
    with controls[1]:device=st.selectbox('Actual inference device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
    if not features:
        unavailable('Validation playback cache unavailable. Prepare the guarded cache first.')
    else:
        trajectories=list(dict.fromkeys(f['trajectory'] for f in features));tr=st.selectbox('Validation trajectory',trajectories)
        available=[f for f in features if f['trajectory']==tr]
        if st.session_state.get('frame_trajectory')!=tr:
            st.session_state['frame_trajectory']=tr;st.session_state['validation_frame']=min(1567,len(available)-1)
        examples=optional_json(comparison/'frame_selection.json')
        choices=[x for x in (examples or {}).get('examples',[]) if x['trajectory']==tr]
        if choices:
            presets=st.columns(len(choices))
            def choose_frame(value):st.session_state['validation_frame']=value
            for col,example in zip(presets,choices):
                col.button(example['reason'].capitalize()+' · frame '+str(example['frame_index']),key='preset_'+tr+example['reason'],on_click=choose_frame,args=(example['frame_index'],),help='Use this already selected validation illustration. Does not run inference.')
        frame=st.slider('Validation frame',min_value=0,max_value=len(available)-1,step=1,key='validation_frame')
        idx=available[frame]['index'];st.caption(f"{tr} / frame {available[frame]['frame_index']} · validation only")
        exists=Path(cfg['dataset_root']).is_dir() and Path(cfg['models'][student]['weights']).is_file()
        if not exists:unavailable('Dataset or selected model unavailable; recorded evidence remains viewable.')
        if timing_active:st.info('Controlled timing is running. Actual inference and playback are temporarily paused to avoid competing work.')
        if st.button('Run actual model inference',disabled=not exists or timing_active,type='primary'):
            try:
                with st.spinner('Running the selected model on this validation frame…'):
                    resource=st.cache_resource(show_spinner=False)(inference_resource)(cfg['models'][student],device)
                    st.session_state['actual']=actual_inference(cfg,idx,student,device,resource)
                st.success('Actual inference completed; model parameters and BN buffers unchanged.')
            except Exception as ex:st.error(f'Inference unavailable: {ex}')
        actual=st.session_state.get('actual')
        if actual and actual['key']==[student,idx,device]:
            show_depth(actual['rgb'],actual['gt'],actual['valid'],actual['prediction'],actual['source'])
            with st.expander('This forward pass · observed latency, not a repeated benchmark'):st.json(actual['observation'])
        else:
            try:show_depth(*recorded_frame(cfg,idx,student))
            except Exception as ex:unavailable('Recorded frame unavailable: '+str(ex))
        if student=='A' and results:
            st.divider();st.markdown('**Recorded causal replay**')
            playback_policy=st.selectbox('Recorded playback policy',list(results))
            rate=st.slider('Recorded playback rate (frames/s)',1,8,3)
            trace=optional_json(result/'policies'/f'{playback_policy}_frames.json')
            if not trace:unavailable('Recorded policy timeline unavailable; playback is disabled.')
            play=st.checkbox('Play recorded A replay',value=False,disabled=timing_active or not trace) and not timing_active and bool(trace)
            @st.fragment(run_every=1/rate if play else None)
            def player():
                if not play:return
                key=(tr,playback_policy)
                if st.session_state.get('playback_key')!=key:st.session_state['playback_key']=key;st.session_state['playback_index']=idx
                position=st.session_state['playback_index'];row=trace[position]
                if row['trajectory']!=tr:position=available[0]['index'];row=trace[position]
                x,g,v,_,_=recorded_frame(cfg,position,'A');cache=np.load(Path(cfg['cache'])/'predictions.npy',mmap_mode='r');p=cache[row['source_index'],0]
                status=st.columns(3);status[0].metric('Prediction', 'Fresh' if row['infer'] else 'Reused');status[1].metric('Prediction age',str(row['prediction_age'])+' frames');status[2].metric('Source frame',str(features[row['source_index']]['frame_index']))
                cols=st.columns(2);cols[0].image(x,caption=f"Recorded RGB · frame {row['frame_index']}",width='stretch');cols[1].image(colourize(p,v),caption=f"{'Fresh update' if row['infer'] else 'Reused prediction'} · age {row['prediction_age']} · {row['reason']}",width='stretch')
                st.caption(f'Requested playback {rate} frames/s. PRECOMPUTED replay, no live model call; this is not measured inference performance.')
                st.session_state['playback_index']=position+1 if position+1<=available[-1]['index'] else available[0]['index']
            player()
        elif student!='A':st.caption('Continuous scheduler playback is prepared for A. Original/KD recorded predictions exist only for the six fixed comparison examples; other frames require the explicit inference button.')

with tabs[3]:
    st.subheader('An update is a decision, not a detection.')
    study=st.selectbox('Scheduler study',['New supervised A validation','Historical epoch-4 V1','Historical epoch-4 V2'])
    if study.startswith('Historical'):
        folder=ROOT/'results'/('scheduler_v1' if study.endswith('V1') else 'scheduler_v2')
        st.warning('Historical epoch-4 study. These results do not measure supervised A or exploratory KD.')
        if (folder/'comparison.csv').is_file():st.dataframe(pd.read_csv(folder/'comparison.csv'),hide_index=True,width='stretch')
        else:unavailable()
        if (folder/'REPORT.md').is_file():st.markdown((folder/'REPORT.md').read_text())
    elif results:
        policy=st.selectbox('Evaluated A policy',list(results));row=results[policy]['overall']
        st.caption('Full-split quality is causal counterfactual replay over verified A predictions; policy calls are replay decisions. Live timings below execute the model on separate fixed windows.')
        cols=st.columns(4)
        cols[0].metric('Fresh calls',str(row['inference_calls']));cols[1].metric('Frames reused',f"{100*row['skip_rate']:.1f}%");cols[2].metric('Near RMSE',f"{row['metrics']['near_rmse']:.3f} m");cols[3].metric('Maximum age',str(row['max_prediction_age']))
        st.caption('Ablations remove both signal weight and associated forced triggers, without renormalizing survivors. Their call budgets also change. First frame and hard age bound are retained; age-only duplicates fixed-5 and was not rerun.')
        st.json({'quality':row['metrics'],'events':row['critical_events'],'inference_reasons':row['inference_reasons']},expanded=False)
        trace=optional_json(result/'policies'/f'{policy}_frames.json')
        if trace:
            timeline_tr=st.selectbox('Timeline trajectory',list(results[policy]['per_trajectory']))
            df=pd.DataFrame([r for r in trace if r['trajectory']==timeline_tr]);st.line_chart(df.set_index('frame_index')[['prediction_age']],color='#51cfb7',height=230)
            with st.expander('Per-frame decisions · source, age and reason'):st.dataframe(df[['frame_index','infer','source_index','prediction_age','reason']],hide_index=True,width='stretch')
        if timing:
            match='v2_cached' if policy=='v2' else policy
            t=[r for r in timing['rows'] if r['policy']==match]
            if t:
                st.markdown('**Actual live timing — bounded windows, not cached-replay speed**')
                table=[{'device':r['device'],'repetitions':r['repetitions'],'actual_calls':str(r['actual_calls']),'end_to_end_median_s':r['end_to_end_s']['median'],'end_to_end_min_s':r['end_to_end_s']['min'],'end_to_end_max_s':r['end_to_end_s']['max'],'model_call_median_ms':r['model_call_median_ms']['median'],'scheduler_median_s':r['scheduler_s']['median']} for r in t]
                st.dataframe(pd.DataFrame(table),hide_index=True,width='stretch');st.caption(timing['scope']+' '+timing['end_to_end'])
            else:unavailable('Actual timing unavailable for this ablation; its offline replay duration is not substituted.')
        else:unavailable('Actual repeated timing pending.')
    else:unavailable('New A scheduler evaluation pending.')

with tabs[4]:
    st.subheader('Trace every number back to a record.')
    with st.expander('Artifact and model identities · full SHA-256 and local locations'):
        st.json({'models':cfg['models'],'comparison':cfg['comparison'],'presentation_results':cfg['results'],'dataset_root':cfg['dataset_root']})
    columns=st.columns(2)
    for col,(folder,names) in zip(columns,[(result,['PROTOCOL.json','scheduler_comparison.csv','timing_individual.csv','live_cache_equivalence.json','REPORT.md','FINAL_TEST_PROPOSAL.json','figures/A_scheduler_quality.png','figures/A_live_timing.png']),(comparison,['comparison.csv','depth_band_metrics.csv','changes.csv','REPORT.md','figures/student_contact_sheet.png'])]):
        with col:
            st.markdown('**'+('A validation and timing' if folder==result else 'Student comparison')+'**')
            for name in names:
                path=folder/name
                if path.is_file():st.download_button(f'Download {folder.name} / {name}',path.read_bytes(),file_name=path.name,key=str(path),width='stretch')
                else:st.caption(f'{name}: unavailable / pending')
    with st.expander('Launch and validation reproduction commands'):
        st.code('python -m pip install -e ".[dashboard]"\npython -m streamlit run scripts/dashboard.py\n\n# Fresh frozen config/output only; completed evidence must not be overwritten.\npython scripts/presentation.py cache --config /path/to/fresh_config.json\npython scripts/presentation.py quality --config /path/to/fresh_config.json\npython scripts/presentation.py cache-check --config /path/to/fresh_config.json\npython scripts/presentation.py timing --config /path/to/fresh_config.json',language='bash')
    proposal=optional_json(result/'FINAL_TEST_PROPOSAL.json')
    if proposal:st.json(proposal,expanded=False)
    if (result/'FINAL_TEST_COMMAND.txt').is_file():st.code((result/'FINAL_TEST_COMMAND.txt').read_text(),language='bash')
    st.caption('No UI action trains, tunes, benchmarks or evaluates test data. Final testing requires a separate frozen exposure protocol and an explicit opt-in CLI invocation. Missing results are shown as unavailable or pending.')
