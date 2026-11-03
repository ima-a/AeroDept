"""Native Streamlit research views, separate from evidence loading and inference."""
from pathlib import Path
import io
import json
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from . import dashboard_data as data
from . import presentation_ui as backend

PAGES=['Results','Model comparison','Depth viewer','Scheduler','Artifacts']
METRICS={'absrel':'AbsRel ↓','rmse':'RMSE (m) ↓','delta1':'δ1 ↑','delta2':'δ2 ↑','delta3':'δ3 ↑',
         'near_absrel':'Near AbsRel ↓','near_rmse':'Near RMSE (m) ↓'}


def number(value,precision=4):
    if value is None or isinstance(value,float) and not math.isfinite(value):return '—'
    return f'{value:,.{precision}f}' if isinstance(value,(float,np.floating)) else f'{value:,}' if isinstance(value,int) else str(value)


def choice(label,options,key,**kwargs):
    if st.session_state.get(key) not in options:st.session_state[key]=options[0]
    return st.selectbox(label,options,key=key,**kwargs)


def context(root,cfg,page):
    with st.container(key='context'):
        cols=st.columns([1.5,1.15,1.7,1.1])
        with cols[0]:study=choice('Study / split',data.STUDIES,'study')
        bundle=data.study_bundle(root,cfg,study);available=data.models(bundle)
        if not available:
            st.info('Saved evidence unavailable for this study. Choose another study or restore the recorded artifacts.');return bundle,None,None,None
        allowed=['A'] if page=='Scheduler' and study=='Validation' and 'A' in available else available
        previous=st.session_state.get('model','A')
        if previous not in allowed:st.caption('Supported model for this context: '+', '.join(data.MODELS[m] for m in allowed)+'.')
        with cols[1]:model=choice('Model',allowed,'model',format_func=data.MODELS.get,disabled=len(allowed)==1)
        with cols[2]:trajectory=choice('Trajectory',data.trajectories(bundle),'trajectory',format_func=data.scope_label)
        with cols[3]:
            if page=='Model comparison':policy=choice('Policy',['every'],'comparison_policy',format_func=data.POLICIES.get,disabled=True)
            else:policy=choice('Policy',data.policies(bundle,model),'policy',format_func=lambda p:data.POLICIES.get(p,p))
    st.caption(f"{study} / {bundle['split']} · {data.MODELS[model] if page!='Model comparison' else 'Available models'} · {data.POLICIES.get(policy,policy)} · {data.scope_label(trajectory)} · {bundle['measurement']}")
    return bundle,model,policy,trajectory


def metric_table(rows,model_comparison=False):
    if not rows:
        st.info('No recorded metrics for this selection.');return
    columns=['Model'] if model_comparison else ['Policy']
    columns+=list(METRICS)+([] if model_comparison else ['Calls','Skip rate','Max age'])
    frame=pd.DataFrame(rows).reindex(columns=columns)
    config={name:st.column_config.NumberColumn(label,format='%.6f',help='Higher is better' if name.startswith('delta') else 'Lower is better') for name,label in METRICS.items()}
    config.update({'Calls':st.column_config.NumberColumn(format='%d'),'Skip rate':st.column_config.NumberColumn(format='percent'),'Max age':st.column_config.NumberColumn('Max age (frames)',format='%d')})
    st.dataframe(frame,hide_index=True,width='stretch',column_config=config)


def paired_charts(rows,axis='Model'):
    if not rows:return
    frame=pd.DataFrame(rows);cols=st.columns(2)
    present=[m for m in data.MODELS if m in set(frame['model'])]
    scale=alt.Scale(domain=[data.MODELS[m] for m in present],range=[data.COLORS[m] for m in present])
    for col,metric,title in [(cols[0],'absrel','Overall AbsRel ↓'),(cols[1],'near_rmse','Near RMSE (m) ↓')]:
        with col:
            st.markdown('**'+title+'**')
            chart=alt.Chart(frame).mark_bar(cornerRadiusEnd=2,size=18).encode(
                x=alt.X(metric+':Q',title=title,scale=alt.Scale(zero=True)),
                y=alt.Y(axis+':N',title=None,sort=None,axis=alt.Axis(labelLimit=160)),
                color=alt.Color('Model:N',scale=scale,legend=alt.Legend(title=None,orient='bottom')),
                tooltip=[axis,alt.Tooltip(metric+':Q',title=title,format='.6f')]).properties(height=max(220,min(330,38*len(rows)+100)))
            st.altair_chart(chart,width='stretch')


def summary(part,labels=None):
    metrics=part.get('metrics',{});cols=st.columns(4)
    for col,key in zip(cols,['absrel','rmse','near_rmse','delta1']):col.metric(METRICS[key],number(metrics.get(key),6))


def download_table(rows,name):
    st.download_button('Export selected table',pd.DataFrame(rows).drop(columns=['model','policy'],errors='ignore').to_csv(index=False).encode(),file_name=name,mime='text/csv',key='selected_export')


def teacher_gate_panel(bundle):
    if bundle['study']!='Validation' or 'Teacher' not in data.models(bundle):return
    gate=bundle.get('teacher_gate') or {}
    if gate.get('teacher_gate_passed') is False:
        st.warning('ResNet-50 teacher failed its acceptance gate. Better pooled error did not preserve near-field quality on every validation path.')
    else:
        st.info('Teacher acceptance-gate record unavailable.' if not gate else 'Teacher gate: '+str(gate.get('teacher_gate_passed')))
    with st.expander('Teacher gate details · both validation trajectories'):
        saved=gate.get('recomputed_from_saved_metrics',{})
        st.write(saved.get('rule','No saved gate rule available.'))
        failures=[]
        for tr in data.trajectories(bundle):
            if tr=='pooled':continue
            teacher=data.part(bundle,'Teacher','every',tr).get('metrics',{})
            baseline=data.part(bundle,'A','every',tr).get('metrics',{})
            for key in ['absrel','rmse','near_absrel','near_rmse']:
                a,b=teacher.get(key),baseline.get(key)
                if a is not None and b is not None and a>b:
                    failures.append({'Trajectory':data.scope_label(tr),'Failed metric':METRICS[key],
                                     'ResNet-50 teacher':a,'Supervised A':b})
        if failures:st.dataframe(pd.DataFrame(failures),hide_index=True,width='stretch')
        st.caption('Saved FP32 validation audit; no new inference. Teacher trained for 20 epochs and selected epoch 3; A trained for 10 and selected epoch 9. These are not matched training budgets. KD remains exploratory with the rejected teacher. ResNet-50 has no independent-test result.')


# Results
def results_page(root,cfg):
    bundle,model,policy,trajectory=context(root,cfg,'Results')
    if not model:return
    selected=data.part(bundle,model,policy,trajectory);summary(selected)
    if model=='Teacher':teacher_gate_panel(bundle)
    st.subheader('Policy results')
    st.caption(data.MODELS[model]+' · same trajectory scope · lower errors are better')
    rows=data.metric_rows(bundle,trajectory,model);paired_charts(rows,'Policy');metric_table(rows);download_table(rows,'selected_policy_results.csv')
    with st.expander('Study status and limitations'):
        st.write('Status: '+bundle['status']+'. Metrics pool valid pixels; near means GT <10 m. Adjacent frames and weather variants are correlated. Differences describe one study and do not establish statistical significance.')
        st.write('A is the preselected general-depth model; Original is the near-field reference. KD remains exploratory. Validation and test rankings differ. No energy, battery or flight-safety benefit is measured.')
        st.caption('Protocol and completion records are available in Artifacts.')


# Comparison
def comparison_page(root,cfg):
    bundle,model,policy,trajectory=context(root,cfg,'Model comparison')
    if not model:return
    rows=data.metric_rows(bundle,trajectory)
    if bundle['study']=='Historical exposed test':st.info('Historical study: only Original was evaluated. A and KD are unavailable on this split.')
    else:
        expected=['Original','A','KD']+(['Teacher'] if bundle['study']=='Validation' else [])
        missing=[data.MODELS[m] for m in expected if m not in {r['model'] for r in rows}]
        if missing:st.info('Model comparison unavailable for '+', '.join(missing)+': saved comparison artifact is missing.')
    metric_table(rows,True);paired_charts(rows)
    teacher_gate_panel(bundle)
    reference=choice('Reference model',[r['model'] for r in rows],'reference_model',format_func=data.MODELS.get)
    ref=next(r for r in rows if r['model']==reference)
    changes=[{'Model':r['Model'],'Δ AbsRel (%)':data.relative_change(r.get('absrel'),ref.get('absrel')),'Δ RMSE (%)':data.relative_change(r.get('rmse'),ref.get('rmse')),'Δ Near RMSE (%)':data.relative_change(r.get('near_rmse'),ref.get('near_rmse'))} for r in rows]
    st.caption('Relative difference versus '+data.MODELS[reference]+' · negative means lower error · descriptive, not a significance test')
    st.dataframe(pd.DataFrame(changes),hide_index=True,width='stretch',column_config={k:st.column_config.NumberColumn(format='%+.2f%%') for k in changes[0] if k!='Model'})
    download_table(rows,'selected_model_comparison.csv')
    with st.expander('Training histories · validation evidence'):
        history=data.json_file(Path(cfg['results'])/'training_curves.json')
        if not history:st.info('Training history unavailable.')
        else:
            frame=pd.DataFrame(history['rows']);st.line_chart(frame.pivot(index='epoch',columns='model',values='validation_absrel'),x_label='Training epoch',y_label='Validation AbsRel')
            st.caption(history['precision_note']);st.caption('Different data, budgets and precision: these curves are not a controlled architecture ranking.')


def timeline_chart(rows,frame_window=None,cursor=None):
    frame=pd.DataFrame(rows)
    if frame.empty or 'frame_index' not in frame:st.info('Recorded timeline unavailable.');return
    if frame_window:frame=frame[frame.frame_index.between(*frame_window)]
    events=[]
    for _,r in frame.iterrows():
        if r.get('infer'):events.append({'frame_index':r.frame_index,'Event':'Fresh inference'})
        if r.get('critical_event'):events.append({'frame_index':r.frame_index,'Event':'Offline GT event'})
    domain=[float(frame.frame_index.min()),float(frame.frame_index.max())]
    x=alt.X('frame_index:Q',title='Frame index',scale=alt.Scale(domain=domain))
    top=alt.Chart(pd.DataFrame(events,columns=['frame_index','Event'])).mark_tick(thickness=1.3,size=16).encode(x=x,y=alt.Y('Event:N',title=None),color=alt.Color('Event:N',scale=alt.Scale(domain=['Fresh inference','Offline GT event'],range=['#2563eb','#d59a22']),legend=None),tooltip=['frame_index','Event']).properties(height=55)
    age=alt.Chart(frame).mark_line(color='#2563eb',interpolate='step-after').encode(x=x,y=alt.Y('prediction_age:Q',title='Age (frames)',scale=alt.Scale(zero=True)),tooltip=['frame_index','prediction_age','reason']).properties(height=100)
    panels=[top,age]
    signals=[k for k in ['motion','visual','risk','score'] if k in frame and frame[k].notna().any()]
    if signals:
        values=frame.melt(id_vars=['frame_index'],value_vars=signals,var_name='Signal',value_name='Value')
        panels.append(alt.Chart(values).mark_line().encode(x=x,y=alt.Y('Value:Q',title='Recorded score'),color=alt.Color('Signal:N',legend=alt.Legend(orient='bottom')),tooltip=['frame_index','Signal','Value']).properties(height=100))
    if cursor is not None:
        for i,panel in enumerate(panels):
            rule=alt.Chart(pd.DataFrame({'frame_index':[cursor]})).mark_rule(color='#475569',strokeDash=[3,3]).encode(x=x)
            panels[i]=panel+rule
    st.altair_chart(alt.vconcat(*panels,spacing=8).resolve_scale(x='shared'),width='stretch')
    if not signals:st.caption('Signal values were not recorded for this study; no telemetry is inferred.')


# Scheduler
def scheduler_page(root,cfg):
    bundle,model,policy,trajectory=context(root,cfg,'Scheduler')
    if not model:return
    selected=data.part(bundle,model,policy,trajectory);events=selected.get('critical_events',{})
    cols=st.columns(4)
    cols[0].metric('Replay calls' if bundle['study']=='Validation' else 'Actual calls',number(selected.get('inference_calls')))
    cols[1].metric('Frames reused',number(100*selected['skip_rate'],1)+'%' if selected.get('skip_rate') is not None else '—')
    cols[2].metric('Mean / max age',number(selected.get('mean_prediction_age'),2)+' / '+number(selected.get('max_prediction_age')))
    cols[3].metric('Max event delay (frames)',number(events.get('max_observed_delay_frames')))
    st.caption(f"AbsRel {number(selected.get('metrics',{}).get('absrel'),6)} · Near RMSE {number(selected.get('metrics',{}).get('near_rmse'),6)} m · Events {number(events.get('count'))} · Right-censored {number(events.get('right_censored_count'))}")
    options=[tr for tr in data.trajectories(bundle) if tr!='pooled'] if trajectory=='pooled' else [trajectory]
    tr=choice('Timeline trajectory',options,'timeline_trajectory',format_func=data.scope_label,disabled=len(options)==1)
    rows=[r for r in data.trace_rows(bundle,model,policy) if r['trajectory']==tr]
    if rows:
        n=int(max(r['frame_index'] for r in rows));key='window_'+bundle['study']+model+policy+tr
        window=st.slider('Frame window',0,n,(0,n),key=key)
        timeline_chart(rows,window)
        with st.expander('Frame inspector and event definitions'):
            frame=st.number_input('Inspect frame',0,n,0,step=1);row=next((r for r in rows if r['frame_index']==frame),{})
            st.dataframe(pd.DataFrame([{k:row.get(k) for k in ['frame_index','infer','source_index','prediction_age','reason','critical_event']}]),hide_index=True,width='stretch')
            st.caption('GT events are offline scoring labels. Delay is to the next fresh update in the same trajectory. Zero-event delays/rates are unavailable; unresolved terminal events are right-censored. A fresh update is not a detection guarantee.')
    else:st.info('Recorded scheduler trace unavailable for this model/context.')
    with st.expander('Measured execution · hardware and boundaries',expanded=False):
        if bundle['study']=='Validation':
            timing=bundle['timing']
            st.caption('Quality above: full trajectories, causal replay. Timing below: separate 256-frame windows, three repetitions per device. These call budgets differ from full-split quality.')
            matching='v2_cached' if policy=='v2' else policy
            rows=[{'Hardware':r['device'].upper(),'Repetitions':r['repetitions'],'Calls / window run':str(r['actual_calls']),'End-to-end median (s)':r['end_to_end_s']['median'],'Min (s)':r['end_to_end_s']['min'],'Max (s)':r['end_to_end_s']['max'],'Median forward (ms)':r['model_call_median_ms']['median']} for r in (timing or {}).get('rows',[]) if r['policy']==matching]
            if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch');st.caption(timing['end_to_end'])
            else:st.info('No measured live timing for this policy; replay duration is not inference performance.')
        else:
            st.caption('CPU · batch 1 FP32 · two Torch threads · one measured pass · selected trajectory scope. Compute = inference + scheduler; scoring/logging excluded. OS load/cache uncontrolled.')
            st.dataframe(pd.DataFrame([{'Inference (s)':selected.get('inference_seconds'),'Scheduler (s)':selected.get('scheduler_seconds'),'Compute (s)':selected.get('compute_seconds'),'Median forward (ms)':selected.get('model_call_median_ms')}]),hide_index=True,width='stretch')
            st.caption('Independent-test shared I/O/decode were measured once per frame across seven comparisons. Matrix wall time is not single-policy end-to-end throughput.')
    if model=='A':
        v2=data.part(bundle,model,'v2',trajectory).get('inference_calls');fixed=data.part(bundle,model,'budget_fixed',trajectory).get('inference_calls')
        if v2 is not None and fixed is not None:st.caption(f'V2 {v2:,} calls · budget-fixed {fixed:,} calls · '+('equal call counts in this quality scope' if v2==fixed else 'different call budgets'))


def depth_panels(x,g,v,p):
    with st.container(key='viewer_canvas'):
        for cols,items in [(st.columns(2),[(x,'RGB · validation'),(backend.colourize(g,v),'Ground truth · range (m)')]),
                           (st.columns(2),[(backend.colourize(p,v) if p is not None else None,'Prediction · range (m)'),(backend.colourize(p-g,v&(g<10),'error') if p is not None else None,'Signed near error · P−GT (m)')])]:
            for col,(image,label) in zip(cols,items):
                with col:
                    st.caption(label)
                    if image is None:st.caption('Recorded prediction unavailable; use explicit inference if inputs are present.')
                    else:st.image(image,width='stretch')
    st.caption('Range: viridis, log 0.1–1000 m · Error: blue under / red over, −10 to +10 m · Grey: outside mask. Display clipping never changes metrics.')


# Viewer
def viewer_page(root,cfg):
    st.caption('Validation inputs only · MidAir, Fonder & Van Droogenbroeck · CC BY-NC-SA 4.0')
    st.caption('Student predictions: Original, A and KD. For ResNet-50 teacher metrics and the gate explanation, choose Model comparison → Validation.')
    features=data.viewer_features(cfg);examples=data.fixed_examples(cfg)
    trajectories=list(dict.fromkeys(r['trajectory'] for r in features or examples))
    if not trajectories:
        st.info('Validation cache unavailable. Connect the configured healthy storage to view recorded frames. Saved result pages remain available.');return
    controls=st.columns([1.6,1.1,1])
    with controls[0]:tr=choice('Validation trajectory',trajectories,'viewer_trajectory',format_func=data.scope_label)
    with controls[1]:model=choice('Model',['A','Original','KD'],'model',format_func=data.MODELS.get)
    import torch
    with controls[2]:device=choice('Device',['cpu']+(['cuda'] if torch.cuda.is_available() else []),'device')
    available=[r for r in features if r['trajectory']==tr];fixed=[r for r in examples if r['trajectory']==tr]
    context_key=(tr,model)
    default_frame=fixed[0]['frame_index'] if fixed else 0
    if st.session_state.get('viewer_context')!=context_key:
        st.session_state['viewer_context']=context_key
        st.session_state['viewer_position']=default_frame;st.session_state['viewer_frame']=default_frame
        st.session_state['play_cursor']=default_frame;st.session_state['play']=False
    if 'viewer_frame' not in st.session_state:
        st.session_state['viewer_frame']=st.session_state.get('viewer_position',st.session_state.get('play_cursor',default_frame))
    valid_frames=range(len(available)) if available else [r['frame_index'] for r in fixed]
    if st.session_state['viewer_frame'] not in valid_frames:st.session_state['viewer_frame']=default_frame
    st.session_state.setdefault('play',False)
    if not available:st.info('Continuous validation cache unavailable. Showing saved comparison examples; continuous replay is disabled.')
    presets=st.columns(max(1,len(fixed))+1)
    selections=data.json_file(Path(cfg['comparison'])/'frame_selection.json') or {}
    reasons={(r['trajectory'],r['frame_index']):r['reason'].capitalize() for r in selections.get('examples',[])}
    def seek(value):
        st.session_state['viewer_frame']=value;st.session_state['viewer_position']=value;st.session_state['play_cursor']=value;st.session_state['play']=False
    for col,row in zip(presets,fixed):col.button(reasons.get((tr,row['frame_index']),'Example')+' · '+str(row['frame_index']),on_click=seek,args=(row['frame_index'],),key='preset_'+str(row['frame_index']))
    # All forwards remain explicit; fragments below only read saved validation arrays.
    selected_frame=st.session_state['viewer_frame']
    index=next((r['index'] for r in available if r['frame_index']==selected_frame),None)
    record_index={}
    if index is None and Path(cfg['dataset_root']).is_dir():
        # Authoritative manifest order supplies the index without any image read.
        from .presentation import validation_inputs
        try:
            _,records,_=validation_inputs(cfg)
            record_index={(r.trajectory_id,r.frame_index):i for i,r in enumerate(records)}
            index=record_index.get((tr,selected_frame))
        except (OSError,ValueError,KeyError):index=None
    timing=(data.json_file(Path(cfg['results'])/'timing_activity.json') or {}).get('active',False)
    ready=Path(cfg['dataset_root']).is_dir() and Path(cfg['models'][model]['weights']).is_file() and index is not None
    with presets[-1]:run=st.button('Run actual inference',type='primary',disabled=not ready or timing,help='One explicit FP32/eval forward on the selected validation frame.')
    if not Path(cfg['models'][model]['weights']).is_file():st.caption('Selected model checkpoint unavailable; recorded comparisons remain viewable.')
    elif not ready:st.caption('Raw validation inputs unavailable; actual inference is disabled.')
    if run:
        try:
            with st.spinner('Running validation inference…'):
                resource=st.cache_resource(show_spinner=False)(backend.inference_resource)(cfg['models'][model],device)
                st.session_state['actual']=backend.actual_inference(cfg,index,model,device,resource)
            st.success('Actual inference complete · model and BatchNorm state unchanged.')
        except Exception as error:st.error('Inference unavailable: '+str(error))
    playback=st.columns([1.2,1,1])
    with playback[0]:policy=choice('Recorded policy',list((data.json_file(Path(cfg['results'])/'scheduler_results.json') or {'every':{}})),'viewer_policy',format_func=data.POLICIES.get,disabled=model!='A' or not available)
    trace=data.json_file(Path(cfg['results'])/'policies'/f'{policy}_frames.json') if model=='A' and available else None
    with playback[1]:rate=st.select_slider('Playback rate (frames/s)',[1,2,3,4,6,8],value=3,disabled=not trace)
    with playback[2]:play=st.checkbox('Play recorded replay',key='play',disabled=not trace or timing)
    if model!='A':st.caption('Continuous policy replay is available for A. Original/KD have six fixed recorded examples and optional explicit inference.')
    def on_seek():
        st.session_state['viewer_position']=st.session_state['viewer_frame'];st.session_state['play_cursor']=st.session_state['viewer_frame']
    @st.fragment(run_every=1/rate if play and trace else None)
    def frame_workspace():
        if play and trace:st.session_state['viewer_frame']=st.session_state.get('play_cursor',0)
        if available:
            frame=st.slider('Validation frame',0,len(available)-1,key='viewer_frame',on_change=on_seek)
            idx=available[frame]['index']
        else:
            options=[r['frame_index'] for r in fixed];frame=choice('Recorded validation frame',options,'viewer_frame');idx=record_index.get((tr,frame))
        st.session_state['viewer_position']=frame
        actual=st.session_state.get('actual');source_frame=frame;age=0;reason='recorded example'
        try:
            fresh=actual and actual['key']==[model,idx,device] and not play
            if fresh:
                x,g,v,p=actual['rgb'],actual['gt'],actual['valid'],actual['prediction'];source='Actual fresh inference · '+device.upper()
            elif available:
                x,g,v,p,source=backend.recorded_frame(cfg,idx,model)
                if trace:
                    row=trace[idx];source_frame=features[row['source_index']]['frame_index'];age=row['prediction_age'];reason=row['reason']
                    cache=np.load(Path(cfg['cache'])/'predictions.npy',mmap_mode='r');p=cache[row['source_index'],0]
                    source='Recorded policy replay · '+('fresh update' if row['infer'] else 'reused prediction')
            else:x,g,v,p,source=data.fixed_frame(cfg,tr,frame,model)
            st.caption(f'{source} · frame {frame} · source frame {source_frame} · age {age} · {reason}')
            depth_panels(x,g,v,p)
            if fresh:
                observation=actual['observation']
                st.dataframe(pd.DataFrame([{k:observation[k] for k in ['device','model_only_ms','transfer_ms','IO_ms','preprocessing_ms']}]),hide_index=True,width='stretch')
                st.caption('One observed call; loading, browser rendering and replay are outside the forward timing.')
            if p is not None:
                from PIL import Image
                buffer=io.BytesIO();Image.fromarray(backend.colourize(p,v)).save(buffer,format='PNG')
                st.download_button('Download displayed prediction image',buffer.getvalue(),file_name=f'validation_{model}_{frame:06d}.png',mime='image/png')
            if trace:
                rows=[r for r in trace if r['trajectory']==tr and abs(r['frame_index']-frame)<=60]
                timeline_chart(rows,cursor=frame)
                st.caption(f'Recorded replay · requested {rate} frames/s · cached display rate, not neural inference throughput.')
            if play and trace:st.session_state['play_cursor']=(frame+1)%len(available)
        except (OSError,ValueError,KeyError) as error:st.info('Recorded validation assets unavailable: '+str(error))
    frame_workspace()


# Artifacts
def artifacts_page(root,cfg):
    study=choice('Study / split',data.STUDIES,'study');bundle=data.study_bundle(root,cfg,study);folder=bundle['folder']
    st.caption(study+' · study-wide artifacts · '+bundle['status'])
    tabs=st.tabs(['Reports','Tables','Figures','Protocols','Models'])
    groups=[
        [('Research report',folder/'REPORT.md'),('Consolidated final report',Path(root)/'docs/FINAL_REPORT.md')],
        [('Comparison table',folder/('scheduler_comparison.csv' if study=='Validation' else 'comparison.csv'))],
        [('Quality and calls',folder/'figures/quality_calls.png')] if study=='Independent test' else [('Scheduler quality',folder/'figures/A_scheduler_quality.png'),('Live timing',folder/'figures/A_live_timing.png')] if study=='Validation' else [('Quality and calls',folder/'quality_vs_calls.png')],
        [('Frozen protocol',folder/('protocol.json' if study=='Historical exposed test' else 'PROTOCOL.json')),('Exposure ledger',folder/'EXPOSURE.json'),('Completion receipt',folder/('completed.json' if study=='Historical exposed test' else 'COMPLETION.json'))],
        [('Original metadata',Path(cfg['models']['Original']['metadata'])),('Supervised A metadata',Path(cfg['models']['A']['metadata']))]]
    if study=='Validation':groups[1].append(('Student model comparison',Path(cfg['comparison'])/'comparison.csv'))
    if cfg['models'].get('KD',{}).get('metadata'):
        groups[4].append(('Exploratory KD metadata',Path(cfg['models']['KD']['metadata'])))
    if study=='Validation' and cfg['models'].get('Teacher',{}).get('metadata'):
        groups[4].append(('ResNet-50 teacher metadata',Path(cfg['models']['Teacher']['metadata'])))
        groups[3].append(('Teacher gate verification',Path(root)/'results/expanded_validation_diagnostic_20261028T172239Z/gate_verification.json'))
    for tab,items in zip(tabs,groups):
        with tab:
            for label,path in items:
                if path.is_file():
                    if path.suffix=='.png':st.image(str(path),width='stretch')
                    st.download_button('Download '+label.lower(),path.read_bytes(),file_name=study.lower().replace(' ','_')+'_'+path.name,key='artifact_'+str(path))
                else:st.caption(label+': unavailable')
    with tabs[4]:st.caption('Inference export identities; these do not add evaluated models to the selected study.')
    with st.expander('Technical inspector · identities and protocol'):
        st.json({'study':study,'models':{data.MODELS.get(k,item.get('label') or k):{field:value for field,value in item.items() if field in ['sha256','metadata_sha256','selected_epoch','kind']} for k,item in cfg['models'].items()}},expanded=False)
        protocol=data.json_file(folder/('protocol.json' if study=='Historical exposed test' else 'PROTOCOL.json'))
        if protocol:st.json(protocol,expanded=False)
    with st.expander('Historical scheduler archives · Original only'):
        for name in ['scheduler_v1','scheduler_v2']:
            for filename in ['REPORT.md','comparison.csv']:
                path=Path(root)/'results'/name/filename
                if path.is_file():st.download_button('Download '+name+' '+filename,path.read_bytes(),file_name=name+'_'+filename,key='archive_'+str(path))
    with st.expander('Launch and data availability'):
        st.code('python -m streamlit run scripts/dashboard.py',language='bash')
        st.caption('Student and ResNet-50 teacher inference weights are included. Teacher metrics are validation-only; the depth viewer supports the three students. Validation viewing/replay needs the configured data/cache. No dashboard action trains, tunes or evaluates test inputs.')
