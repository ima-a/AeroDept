"""Application-only styling; model and metric colours are defined in the adapter."""
CSS='''<style>
html,body,[data-testid="stApp"] {font-family: Inter, ui-sans-serif, system-ui, sans-serif;}
.block-container {max-width:1440px;padding:1.5rem 2rem 2.5rem;}
h1 {font-size:1.5rem!important;font-weight:650!important;letter-spacing:-.035em;margin:0 0 .25rem!important;}
h2 {font-size:1.05rem!important;font-weight:650!important;letter-spacing:-.015em;}
h3 {font-size:.95rem!important;font-weight:600!important;}
p,li,label,button,[data-testid="stCaptionContainer"] {font-size:14px;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#475569!important;}
[data-testid="stSidebar"] {background:#f8fafc;border-right:1px solid #e2e8f0;}
[data-testid="stSidebar"] .block-container {padding:1.4rem 1rem;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e2e8f0;border-radius:6px;padding:12px 15px;min-height:88px;}
[data-testid="stMetricValue"] {font-size:1.6rem;font-weight:600;font-variant-numeric:tabular-nums;}
[data-testid="stMetricLabel"] {color:#526174;font-size:12px;}
[data-testid="stDataFrame"] {font-variant-numeric:tabular-nums;}
.st-key-viewer_canvas {background:#111827;border:1px solid #263348;border-radius:6px;padding:12px;}
.st-key-viewer_canvas [data-testid="stCaptionContainer"] {color:#d1dae6!important;}
.st-key-viewer_canvas p {color:#d1dae6!important;}
.st-key-context {background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px 12px 0;margin-bottom:8px;}
.st-key-context [data-testid="stVerticalBlock"] {gap:.35rem;}
[data-testid="stVerticalBlock"] {gap:.75rem;}
[data-testid="stExpander"] {border-color:#e2e8f0;border-radius:6px;}
button:focus-visible,[role="radio"]:focus-visible {outline:2px solid #2563eb!important;outline-offset:3px;}
@media (min-width:900px) {[data-testid="stSidebar"] {min-width:205px;max-width:205px;}}
@media (max-width:640px) {.block-container {padding:4.5rem .8rem 2rem;}h1{font-size:1.3rem!important;}[data-testid="stMetricValue"]{font-size:1.3rem;}}
</style>'''
