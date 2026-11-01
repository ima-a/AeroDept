"""Read-only AeroDepth research application; explicit guarded validation inference."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import streamlit as st
from aerodepth_lite import dashboard_views as views
from aerodepth_lite.dashboard_style import CSS
from aerodepth_lite.presentation_ui import optional_json,load_dashboard_config

st.set_page_config(page_title='AeroDepth-Lite · Evaluation',layout='wide',initial_sidebar_state='auto')
st.markdown(CSS,unsafe_allow_html=True)
local=ROOT/'configs/presentation_local.json';local_data=optional_json(local)
local_matches=local_data and Path(local_data.get('results','/')).is_relative_to(ROOT)
default=local if local_matches else ROOT/'configs/presentation.json'
cfg=load_dashboard_config(Path(os.environ.get('AERODEPTH_PRESENTATION_CONFIG',default)),ROOT)
with st.sidebar:
    st.markdown('**AeroDepth-Lite**')
    st.caption('Evaluation workspace')
    page=st.radio('Navigation',views.PAGES,key='page',label_visibility='collapsed')
    st.divider()
    st.caption('Metric range · FP32')
    st.caption('Saved research evidence')
st.title(page)
if cfg is None:
    st.info('Presentation configuration unavailable. Set AERODEPTH_PRESENTATION_CONFIG to a prepared config.');st.stop()
render={'Results':views.results_page,'Model comparison':views.comparison_page,'Depth viewer':views.viewer_page,'Scheduler':views.scheduler_page,'Artifacts':views.artifacts_page}
render[page](ROOT,cfg)
