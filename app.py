"""
EyeX — Explainable Retinal AI  (dark app-style UI)
Single-file app with three pages: Upload & Quality · AI Analysis · Clinical Report.
Model / inference / report logic is unchanged; only the interface layer is new.

Tip: for the best dark rendering, add .streamlit/config.toml with
    [theme]
    base = "dark"
"""

import io
import os
import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference.predict import predict_image
from inference.classification_models import predict_all_models, calculate_model_agreement
from inference.lesion_detector import IDRiDLesionDetector
from reports.pdf_report import generate_pdf_report, generate_report_id
from processors.image_quality import assess_image_quality

st.set_page_config(page_title="EyeX | Explainable Retinal AI", page_icon="👁️",
                   layout="wide", initial_sidebar_state="expanded")

# ======================================================
# Helpers
# ======================================================
def h(text):
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def md(text):
    st.markdown(h(text), unsafe_allow_html=True)


def section(title, sub=""):
    md(f'<div class="x-title">{title}</div>' + (f'<div class="x-sub">{sub}</div>' if sub else ""))


@st.cache_resource
def load_lesion_detector():
    return IDRiDLesionDetector()


PAGES = ["🩻  Upload & Quality", "🧠  AI Analysis", "📄  Clinical Report"]
P_UPLOAD, P_ANALYSIS, P_REPORT = PAGES


def go(page):
    st.session_state.page = page


CLASS_ACCENTS = {
    "Healthy": {"color": "#34D399", "label": "Healthy"},
    "DR": {"color": "#FBBF24", "label": "Diabetic Retinopathy"},
    "Glaucoma": {"color": "#5EEAD4", "label": "Glaucoma"},
    "AMD": {"color": "#3FA3A8", "label": "Age-Related Macular Degeneration"},
}
DEFAULT_ACCENT = {"color": "#7FA0D6", "label": "Result"}


def get_accent(key):
    return CLASS_ACCENTS.get(str(key).strip(), DEFAULT_ACCENT)


# ======================================================
# Session state
# ======================================================
for key in ["quality_result", "quality_image_name", "analysis_result", "classification_comparison",
            "lesion_result", "report_path", "report_id", "image", "image_format", "image_name"]:
    st.session_state.setdefault(key, None)
st.session_state.setdefault("page", P_UPLOAD)

RESET_KEYS = ["quality_result", "analysis_result", "classification_comparison",
              "lesion_result", "report_path", "report_id"]


# ======================================================
# Logo
# ======================================================
def logo_svg(size=56):
    return f"""
    <svg class="x-logo" width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <defs>
    <linearGradient id="lgBg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6B2A45"/><stop offset="1" stop-color="#14050C"/></linearGradient>
    <linearGradient id="lgSky" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#EAF0FB"/><stop offset="1" stop-color="#7FA0D6"/></linearGradient>
    </defs>
    <rect x="1.5" y="1.5" width="61" height="61" rx="20" fill="url(#lgBg)" stroke="url(#lgSky)" stroke-opacity=".55" stroke-width="1.5"/>
    <path d="M9 32 Q32 9 55 32 Q32 55 9 32Z" fill="none" stroke="url(#lgSky)" stroke-width="3.2" stroke-linejoin="round"/>
    <circle class="x-lring" cx="32" cy="32" r="11.5" fill="none" stroke="#3FA3A8" stroke-width="1.8" stroke-dasharray="3 5" stroke-linecap="round"/>
    <path d="M32 22.5 L34.6 29.4 L41.5 32 L34.6 34.6 L32 41.5 L29.4 34.6 L22.5 32 L29.4 29.4Z" fill="url(#lgSky)"/>
    <circle cx="32" cy="32" r="2" fill="#14050C"/>
    </svg>"""


def wordmark(size=44, tag=""):
    return (f'<div><div class="x-wm" style="--s:{size}px"><span class="e">Eye</span><span class="x">X</span></div>'
            + (f'<div class="x-wm-tag">{tag}</div>' if tag else "") + '</div>')


# ======================================================
# Global styling
# ======================================================
md("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Manrope:wght@400;500;600;700&display=swap');
@property --p{syntax:'<integer>';inherits:true;initial-value:0;}
:root{--bg:#14050C;--panel:#2A0B18;--panel2:#42142A;--line:#6B2A45;--txt:#FFF4EA;--mut:#CDAEA2;
 --amber:#A9C1EA;--cyan:#7FA0D6;--rose:#3FA3A8;--grad:linear-gradient(120deg,#A9C1EA,#3FA3A8 50%,#8E2B4F);}
html,body,.stApp,[class*="css"]{font-family:'Manrope',sans-serif;}
.stApp{background:var(--bg);color:var(--txt);}
.stApp p,.stApp label,.stApp span,.stApp li,.stApp div[data-testid="stCaptionContainer"]{color:inherit;}
.stApp [data-testid="stCaptionContainer"]{color:var(--mut);}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{position:relative;z-index:1;padding-top:1rem;padding-bottom:3rem;max-width:1380px;}
h1,h2,h3,.x-title,.x-brand,.x-big{font-family:'Outfit',sans-serif;}

/* ---------- animated backdrop ---------- */
.stApp::before{content:"";position:fixed;inset:-20%;z-index:0;pointer-events:none;
 background:radial-gradient(520px 420px at 15% 20%,rgba(127,160,214,.18),transparent 70%),
  radial-gradient(560px 440px at 85% 15%,rgba(63,163,168,.15),transparent 70%),
  radial-gradient(600px 480px at 60% 90%,rgba(169,193,234,.13),transparent 70%);
 animation:drift 22s ease-in-out infinite alternate;}
.stApp::after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
 background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
 background-size:46px 46px;mask-image:radial-gradient(ellipse at 50% 30%,#000 20%,transparent 75%);
 animation:grid 14s linear infinite;}
@keyframes drift{0%{transform:translate(0,0) scale(1)}50%{transform:translate(4%,-3%) scale(1.08)}100%{transform:translate(-4%,3%) scale(1)}}
@keyframes grid{to{background-position:46px 46px}}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F0409,#260B16);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] *{color:#F6E6D8;}
.x-brand{display:flex;align-items:center;gap:10px;font-size:26px;font-weight:800;}
.x-brand em{font-style:normal;color:var(--amber);}
.x-side-sub{font-size:12px;color:var(--mut) !important;margin:2px 0 18px;}
.x-status{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:14px;background:rgba(255,255,255,.03);border:1px solid var(--line);margin-bottom:8px;font-size:13px;}
.x-dot{width:9px;height:9px;border-radius:50%;background:#7A3550;flex:none;}
.x-dot.on{background:#34D399;box-shadow:0 0 0 0 rgba(52,211,153,.7);animation:blip 1.8s infinite;}
.x-dot.now{background:var(--amber);animation:blip2 1.4s infinite;}
@keyframes blip{70%{box-shadow:0 0 0 9px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
@keyframes blip2{70%{box-shadow:0 0 0 9px rgba(169,193,234,0)}100%{box-shadow:0 0 0 0 rgba(169,193,234,0)}}
.x-meta-l{font-size:11px;color:var(--mut) !important;margin-top:14px;}
.x-meta-v{font-size:13.5px;font-weight:600;}

/* ---------- top nav (radio as pills) ---------- */
div[role="radiogroup"]{gap:6px;background:rgba(36,10,20,.8);backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:999px;padding:6px;width:fit-content;max-width:100%;flex-wrap:wrap;}
div[role="radiogroup"] label{padding:9px 20px;border-radius:999px;cursor:pointer;transition:all .3s cubic-bezier(.2,.8,.2,1);margin:0 !important;}
div[role="radiogroup"] label>div:first-child{display:none;}
div[role="radiogroup"] label p{font-weight:600;font-size:14px;color:var(--mut);}
div[role="radiogroup"] label:hover{background:rgba(255,255,255,.06);}
div[role="radiogroup"] label:has(input:checked){background:linear-gradient(120deg,#A9C1EA,#3FA3A8);box-shadow:0 8px 24px rgba(63,163,168,.35);transform:translateY(-1px);}
div[role="radiogroup"] label:has(input:checked) p{color:#2A0A12;}

/* ---------- hero ---------- */
.x-hero{position:relative;display:grid;grid-template-columns:1.15fr .85fr;gap:20px;align-items:center;margin:22px 0 14px;padding:10px 6px;animation:rise .9s cubic-bezier(.2,.8,.2,1) both;}
@keyframes rise{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}
.x-h1{font-family:'Outfit',sans-serif;font-size:clamp(38px,5.2vw,72px);font-weight:800;line-height:1.02;letter-spacing:-2px;margin:14px 0 14px;}
.x-h1 .g{background:var(--grad);background-size:220% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 6s linear infinite;}
@keyframes shine{to{background-position:220% 0}}
.x-lead{font-size:16px;line-height:1.7;color:#E2C9BA;max-width:560px;}
.x-pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px;}
.x-pill{font-size:12.5px;font-weight:600;padding:7px 14px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.04);animation:rise .8s both;}
.x-orbit{position:relative;height:380px;display:flex;align-items:center;justify-content:center;}
.x-orbit svg{width:340px;height:340px;filter:drop-shadow(0 0 40px rgba(127,160,214,.28));animation:bob 7s ease-in-out infinite;}
@keyframes bob{50%{transform:translateY(-14px) rotate(2deg)}}
.x-sweep{transform-origin:100px 100px;animation:spin 5s linear infinite;}
.x-ring2{transform-origin:100px 100px;animation:spin 18s linear infinite reverse;}
@keyframes spin{to{transform:rotate(360deg)}}
.x-pulse{transform-origin:100px 100px;animation:pulse 3s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(1)}50%{opacity:.95;transform:scale(1.05)}}
.x-chip{position:absolute;padding:10px 14px;border-radius:16px;font-size:13px;font-weight:700;background:rgba(52,16,31,.85);border:1px solid var(--line);backdrop-filter:blur(10px);box-shadow:0 14px 30px rgba(0,0,0,.45);animation:floaty 6s ease-in-out infinite;}
.x-chip small{display:block;font-weight:500;color:var(--mut);font-size:11px;}
.x-chip.a{top:30px;left:0;transform:rotate(-5deg);}
.x-chip.b{bottom:50px;left:-10px;transform:rotate(3deg);animation-delay:-2s;}
.x-chip.c{top:110px;right:-6px;transform:rotate(5deg);animation-delay:-4s;}
@keyframes floaty{50%{translate:0 -12px}}
.x-ticker{overflow:hidden;border-block:1px solid var(--line);margin:8px 0 26px;mask-image:linear-gradient(90deg,transparent,#000 12%,#000 88%,transparent);}
.x-track{display:flex;gap:46px;width:max-content;padding:12px 0;animation:marq 28s linear infinite;font-family:'Outfit',sans-serif;font-weight:600;font-size:15px;color:var(--mut);}
.x-track span b{color:var(--amber);margin-right:8px;}
@keyframes marq{to{transform:translateX(-50%)}}
.x-logo{filter:drop-shadow(0 6px 18px rgba(169,193,234,.4));}
.x-lring{transform-origin:32px 32px;animation:spin 12s linear infinite;}
@keyframes scan{50%{transform:translateY(32px)}}

/* ---------- cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:24px !important;border:1px solid var(--line) !important;
 background:linear-gradient(160deg,rgba(52,16,31,.9),rgba(26,6,12,.9)) !important;backdrop-filter:blur(10px);
 box-shadow:0 18px 44px rgba(0,0,0,.4);transition:border-color .35s,transform .35s,box-shadow .35s;animation:rise .8s cubic-bezier(.2,.8,.2,1) both;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{border-color:rgba(127,160,214,.45) !important;box-shadow:0 22px 54px rgba(127,160,214,.12);}
.x-title{font-size:24px;font-weight:700;letter-spacing:-.4px;}
.x-sub{font-size:14px;color:var(--mut);margin:2px 0 16px;}
.x-label{font-size:13px;font-weight:600;color:var(--mut);margin-bottom:6px;}
.x-big{font-size:42px;font-weight:800;line-height:1.1;letter-spacing:-1px;}
.x-chipline{display:inline-block;font-size:12px;font-weight:700;padding:6px 14px;border-radius:999px;margin-top:12px;}
.x-info{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}
.x-info div{font-size:12.5px;padding:8px 12px;border-radius:12px;background:rgba(255,255,255,.04);border:1px solid var(--line);word-break:break-all;}
.x-info b{color:var(--mut);font-weight:600;margin-right:6px;}

/* ---------- quality tiles ---------- */
.x-qgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:6px;}
.x-qt{position:relative;padding:16px;border-radius:18px;background:rgba(255,255,255,.035);border:1px solid var(--line);overflow:hidden;animation:pop .6s cubic-bezier(.2,.9,.3,1.3) both;}
.x-qt:nth-child(2){margin-top:14px;animation-delay:.1s}.x-qt:nth-child(3){margin-top:-14px;animation-delay:.2s}.x-qt:nth-child(4){animation-delay:.3s}
@keyframes pop{from{opacity:0;transform:scale(.85) translateY(14px)}to{opacity:1;transform:none}}
.x-qt::after{content:"";position:absolute;left:0;top:0;height:3px;width:100%;background:var(--c);}
.x-qt .v{font-family:'Outfit',sans-serif;font-size:26px;font-weight:700;margin:4px 0;}
.x-qt .s{font-size:12px;font-weight:600;color:var(--c);}
.x-verdict{display:flex;align-items:center;gap:12px;padding:14px 16px;border-radius:16px;font-weight:700;font-size:14px;margin-bottom:14px;border:1px solid var(--c);background:color-mix(in srgb,var(--c) 12%,transparent);color:var(--c);}

/* ---------- ring gauge ---------- */
.x-ring{--c:#7FA0D6;position:relative;width:170px;height:170px;border-radius:50%;margin:4px auto 0;
 background:conic-gradient(var(--c) calc(var(--p)*1%),#43182B 0);animation:fill 1.6s cubic-bezier(.2,.8,.2,1) both;
 counter-reset:n var(--p);display:grid;place-items:center;box-shadow:0 0 40px color-mix(in srgb,var(--c) 30%,transparent);}
.x-ring::before{content:"";position:absolute;inset:14px;border-radius:50%;background:#1A060E;}
.x-ring b{position:relative;font-family:'Outfit',sans-serif;font-size:38px;font-weight:800;}
.x-ring b::after{content:counter(n) "%";}
@keyframes fill{from{--p:0}}
.x-bar{height:10px;border-radius:999px;background:#43182B;overflow:hidden;margin-top:8px;}
.x-bar>div{height:100%;border-radius:999px;background:var(--grad);animation:grow 1.3s cubic-bezier(.2,.8,.2,1) both;position:relative;}
.x-bar>div::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.55),transparent);animation:glint 2.2s infinite;}
@keyframes grow{from{width:0 !important}}
@keyframes glint{from{transform:translateX(-100%)}to{transform:translateX(100%)}}

/* ---------- model cards ---------- */
.x-model{padding:16px 18px;border-radius:18px;background:rgba(255,255,255,.035);border:1px solid var(--line);margin-bottom:12px;transition:transform .3s,border-color .3s;animation:slide .7s cubic-bezier(.2,.8,.2,1) both;}
.x-model:nth-child(2){margin-left:26px;animation-delay:.12s}.x-model:nth-child(3){margin-left:52px;animation-delay:.24s}
.x-model:hover{transform:translateX(8px);border-color:var(--amber);}
@keyframes slide{from{opacity:0;transform:translateX(-30px)}to{opacity:1;transform:none}}
.x-model .n{font-family:'Outfit',sans-serif;font-weight:700;font-size:16px;display:flex;justify-content:space-between;gap:10px;}
.x-model .n span{font-size:12px;font-weight:700;padding:4px 12px;border-radius:999px;background:rgba(127,160,214,.14);color:var(--cyan);}
.x-model .c{font-size:12.5px;color:var(--mut);margin-top:8px;}
.x-perf{display:grid;grid-template-columns:90px 1fr 56px;gap:10px;align-items:center;font-size:12.5px;margin-top:8px;}
.x-perf .x-bar{margin:0;height:7px;}

/* ---------- lesions ---------- */
.x-lesion{padding:18px;border-radius:20px;background:rgba(255,255,255,.035);border:1px solid var(--line);border-top:4px solid var(--c);min-height:150px;transition:transform .3s,box-shadow .3s;animation:pop .7s both;}
.x-lesion:hover{transform:translateY(-8px) rotate(-1deg);box-shadow:0 18px 34px color-mix(in srgb,var(--c) 25%,transparent);}
.x-lesion .nm{font-size:12px;font-weight:700;color:var(--mut);}
.x-lesion .st{font-family:'Outfit',sans-serif;font-size:23px;font-weight:800;margin:6px 0 10px;}
.x-lesion .ln{font-size:13px;color:#E2C9BA;line-height:1.7;}
.x-legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:12px;color:var(--mut);}
.x-legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;}

/* ---------- gradcam ---------- */
.x-ex-t{font-family:'Outfit',sans-serif;font-size:16px;font-weight:700;}
.x-ex-d{font-size:12.5px;color:var(--mut);margin-bottom:10px;}
[data-testid="stImage"] img{border-radius:16px;transition:transform .45s cubic-bezier(.2,.8,.2,1),box-shadow .45s;}
[data-testid="stImage"] img:hover{transform:scale(1.04) rotate(.6deg);box-shadow:0 20px 44px rgba(0,0,0,.6);}
.x-tip{font-size:13.5px;margin-bottom:8px;line-height:1.55;color:#EFDACB;}

/* ---------- report ---------- */
.x-check{display:flex;align-items:center;gap:14px;padding:14px 16px;border-radius:16px;background:rgba(255,255,255,.035);border:1px solid var(--line);margin-bottom:10px;font-weight:600;font-size:14px;animation:slide .6s both;}
.x-check .ic{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;background:#43182B;font-size:13px;flex:none;}
.x-check.ok .ic{background:#064E3B;color:#34D399;box-shadow:0 0 18px rgba(52,211,153,.4);}
.x-check small{display:block;color:var(--mut);font-weight:500;font-size:12px;}
.x-doc{position:relative;border-radius:22px;padding:26px;background:linear-gradient(150deg,#4A1428,#260B16);border:1px solid var(--line);overflow:hidden;transform:rotate(1.5deg);transition:transform .4s;}
.x-doc:hover{transform:rotate(0) scale(1.02);}
.x-doc::after{content:"";position:absolute;left:0;right:0;height:60px;top:-60px;background:linear-gradient(180deg,transparent,rgba(127,160,214,.25),transparent);animation:docscan 3.6s linear infinite;}
@keyframes docscan{to{top:110%}}
.x-doc .ln{height:9px;border-radius:6px;background:#6B2A45;margin:12px 0;}
.x-disc{background:linear-gradient(120deg,rgba(169,193,234,.12),rgba(63,163,168,.08));border:1px solid rgba(169,193,234,.4);border-radius:18px;padding:18px 22px;margin-top:10px;}
.x-disc b{color:var(--amber);font-size:14px;}
.x-disc div{font-size:13px;color:#C9D6EE;line-height:1.6;margin-top:4px;}
.x-empty{text-align:center;padding:40px 10px;color:var(--mut);font-size:15px;}
.x-empty span{font-size:46px;display:block;animation:floaty 4s ease-in-out infinite;margin-bottom:6px;}
.x-footer{text-align:center;margin-top:40px;padding-top:20px;border-top:1px solid var(--line);color:var(--mut);font-size:12.5px;}

/* ---------- widgets ---------- */
div.stButton>button,div.stDownloadButton>button{border-radius:16px;font-weight:700;padding:.75rem 1.2rem;border:1px solid var(--line);color:#fff;
 background:linear-gradient(120deg,#4A1428,#6B1D3A);transition:all .3s cubic-bezier(.2,.8,.2,1);position:relative;overflow:hidden;}
div.stButton>button:hover,div.stDownloadButton>button:hover{transform:translateY(-3px) scale(1.01);color:#fff;border-color:var(--cyan);box-shadow:0 14px 30px rgba(127,160,214,.25);}
div.stButton>button[kind="primary"],div.stDownloadButton>button{background:linear-gradient(120deg,#A9C1EA,#3FA3A8);color:#2A0A12;border:none;box-shadow:0 12px 30px rgba(63,163,168,.35);}
div.stButton>button[kind="primary"]:hover,div.stDownloadButton>button:hover{color:#2A0A12;box-shadow:0 18px 40px rgba(63,163,168,.5);}
div.stButton>button:disabled{opacity:.45;}
[data-testid="stFileUploaderDropzone"]{border:2px dashed #8A2D4F;border-radius:22px;background:radial-gradient(circle at 50% 0,rgba(127,160,214,.12),transparent 70%),#180711;transition:all .3s;animation:breathe 3.5s ease-in-out infinite;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--amber);background:#2A0E1C;}
@keyframes breathe{50%{border-color:#8E2B4F;box-shadow:0 0 28px rgba(127,160,214,.18)}}
[data-testid="stFileUploaderDropzone"] *{color:#EFDACB !important;}
[data-testid="stFileUploaderDropzone"] button{background:#4A1428;border:1px solid var(--line);color:#fff !important;}
[data-testid="stAlert"]{border-radius:16px;background:rgba(52,16,31,.9);border:1px solid var(--line);}
[data-testid="stAlert"] *{color:var(--txt) !important;}
details{background:rgba(36,10,20,.8) !important;border:1px solid var(--line) !important;border-radius:18px !important;}
details summary p{color:var(--txt);}
hr{border-color:var(--line) !important;}
@media (max-width:900px){.x-hero{grid-template-columns:1fr}.x-orbit{height:300px}.x-model{margin-left:0 !important}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none !important;transition:none !important;}}

/* ---------- nerve background ---------- */
.x-bg{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;}
.x-bg svg{position:absolute;width:100%;height:100%;}
.x-nv{fill:none;stroke:url(#nvg);stroke-width:1.6;stroke-linecap:round;stroke-dasharray:6 14;animation:flow 9s linear infinite;opacity:.55;}
.x-nv.t{stroke-width:.9;opacity:.35;animation-duration:14s;}
@keyframes flow{to{stroke-dashoffset:-400}}
.x-nd{fill:#A9C1EA;animation:twinkle 3.5s ease-in-out infinite;}
@keyframes twinkle{0%,100%{opacity:.15}50%{opacity:.9}}
.x-iris{transform-origin:center;transform-box:fill-box;animation:spin 60s linear infinite;}
/* ---------- wordmark ---------- */
.x-wm{display:inline-flex;align-items:center;gap:2px;font-family:'Outfit',sans-serif;font-weight:800;font-size:var(--s,44px);line-height:1;letter-spacing:-.04em;position:relative;}
.x-wm .e{background:linear-gradient(180deg,#fff,#EAF0FB);-webkit-background-clip:text;background-clip:text;color:transparent;}
.x-wm .x{position:relative;margin-left:2px;font-size:1.22em;background:linear-gradient(135deg,#E4EDFB,#A9C1EA 45%,#3FA3A8);-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(0 0 14px rgba(169,193,234,.55));animation:glow 3s ease-in-out infinite;}
.x-wm .x::after{content:"";position:absolute;inset:-10% -18%;border:1.5px dashed rgba(169,193,234,.55);border-radius:50%;animation:spin 10s linear infinite;}
@keyframes glow{50%{filter:drop-shadow(0 0 26px rgba(63,163,168,.8))}}
.x-wm-tag{font-size:12px;color:var(--mut);margin-top:6px;letter-spacing:.04em;}
.x-bar-top{display:flex;align-items:center;gap:14px;}
/* ---------- consensus + model cards ---------- */
.x-cons{display:flex;align-items:center;gap:24px;flex-wrap:wrap;padding:22px 26px;border-radius:26px;margin-bottom:16px;border:1px solid rgba(169,193,234,.35);
 background:linear-gradient(120deg,rgba(169,193,234,.14),rgba(142,43,79,.18) 60%,rgba(63,163,168,.1));position:relative;overflow:hidden;animation:rise .8s both;}
.x-cons::after{content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent 30%,rgba(255,255,255,.08) 50%,transparent 70%);animation:glint 4s infinite;}
.x-votes{display:flex;gap:10px;}
.x-vote{width:46px;height:46px;border-radius:50%;display:grid;place-items:center;font-weight:800;background:#43182B;border:2px solid #7A3550;}
.x-vote.y{background:radial-gradient(circle,#DCE8FB,#5E86C7);border-color:#E4EDFB;color:#2A0A12;box-shadow:0 0 22px rgba(169,193,234,.6);animation:blip2 2s infinite;}
.x-mgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
.x-mc{position:relative;padding:24px 20px 20px;border-radius:26px;background:linear-gradient(170deg,rgba(74,20,40,.9),rgba(24,7,17,.95));border:1px solid var(--line);text-align:center;overflow:hidden;transition:transform .4s cubic-bezier(.2,.8,.2,1),border-color .4s,box-shadow .4s;animation:pop .8s cubic-bezier(.2,.9,.3,1.2) both;}
.x-mc:nth-child(2){margin-top:26px;animation-delay:.15s}.x-mc:nth-child(3){animation-delay:.3s}
.x-mc:hover{transform:translateY(-10px);border-color:var(--amber);box-shadow:0 24px 50px rgba(169,193,234,.15);}
.x-mc.win::before{content:"Majority";position:absolute;top:14px;right:-30px;transform:rotate(38deg);background:var(--amber);color:#2A0A12;font-size:10.5px;font-weight:800;padding:3px 34px;}
.x-mc .nm{font-family:'Outfit',sans-serif;font-weight:700;font-size:17px;margin-bottom:12px;}
.x-mc .pr{display:inline-block;margin:14px 0 4px;font-weight:800;font-size:13px;padding:6px 16px;border-radius:999px;background:rgba(127,160,214,.16);color:#DCE6F8;border:1px solid rgba(127,160,214,.35);}
.x-ring.sm{width:116px;height:116px;}.x-ring.sm::before{inset:10px;}.x-ring.sm b{font-size:25px;}
.x-mc .x-perf{grid-template-columns:62px 1fr 48px;text-align:left;}
.x-mc .tt{font-size:11.5px;color:var(--mut);margin-top:14px;text-align:left;}
@media (max-width:900px){.x-mgrid{grid-template-columns:1fr}.x-mc:nth-child(2){margin-top:0}}
/* ---------- big gradcam ---------- */
.x-stage{position:relative;border-radius:28px;padding:14px;background:radial-gradient(circle at 50% 0,rgba(142,43,79,.25),transparent 70%),#180711;border:1px solid var(--line);}
.x-stage [data-testid="stImage"] img,.st-key-gc_stage [data-testid="stImage"] img{max-height:78vh;object-fit:contain;margin:0 auto;display:block;}
/* ---------- on-screen report ---------- */
.x-sheet{border-radius:28px;padding:34px 38px;background:linear-gradient(160deg,#42142A,#180711);border:1px solid rgba(169,193,234,.35);box-shadow:0 30px 70px rgba(0,0,0,.55);animation:rise .7s both;}
.x-sheet h4{font-family:'Outfit',sans-serif;font-size:18px;margin:26px 0 10px;color:var(--amber);}
.x-kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;}
.x-kv div{padding:12px 14px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid var(--line);font-size:13px;}
.x-kv b{display:block;font-size:11.5px;color:var(--mut);font-weight:600;margin-bottom:3px;}
</style>
""")


md("""
<div class="x-bg"><svg viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="nvg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#A9C1EA"/><stop offset=".5" stop-color="#3FA3A8"/><stop offset="1" stop-color="#8E2B4F"/></linearGradient>
<radialGradient id="irg" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#8E2B4F" stop-opacity=".0"/><stop offset=".7" stop-color="#8E2B4F" stop-opacity=".16"/><stop offset="1" stop-color="#A9C1EA" stop-opacity=".22"/></radialGradient></defs>
<g class="x-iris" opacity=".9"><circle cx="1230" cy="170" r="260" fill="url(#irg)"/><circle cx="1230" cy="170" r="200" fill="none" stroke="#A9C1EA" stroke-opacity=".18" stroke-dasharray="2 10"/><circle cx="1230" cy="170" r="120" fill="none" stroke="#3FA3A8" stroke-opacity=".2"/><circle cx="1230" cy="170" r="52" fill="#14050C" stroke="#A9C1EA" stroke-opacity=".35"/></g>
<path d="M-40 640 C260 560 420 760 760 700 S1180 560 1480 640" fill="none" stroke="#A9C1EA" stroke-opacity=".08" stroke-width="2"/>
<path d="M-40 700 C300 780 520 520 860 600 S1200 800 1480 720" fill="none" stroke="#3FA3A8" stroke-opacity=".08" stroke-width="2"/>
<g><path class="x-nv" d="M1230 170 C1080 230 980 190 860 300 S640 330 520 450 S300 520 160 640"/>
<path class="x-nv" d="M1230 170 C1130 300 1180 420 1050 520 S900 700 760 800"/>
<path class="x-nv t" d="M860 300 C830 220 760 170 700 90 S600 20 560 -20"/>
<path class="x-nv t" d="M520 450 C430 400 330 420 250 340 S120 280 40 300"/>
<path class="x-nv t" d="M1050 520 C1150 590 1300 590 1400 680"/>
<path class="x-nv t" d="M980 190 C960 120 1000 60 960 -10"/>
<path class="x-nv" d="M-20 120 C140 200 220 120 340 220 S520 260 600 200"/>
<path class="x-nv t" d="M160 640 C120 720 140 800 90 920"/><path class="x-nv t" d="M760 800 C690 760 600 790 520 860"/></g>
<g><circle class="x-nd" cx="860" cy="300" r="4"/><circle class="x-nd" cx="520" cy="450" r="4" style="animation-delay:-1s"/><circle class="x-nd" cx="1050" cy="520" r="4" style="animation-delay:-2s"/><circle class="x-nd" cx="340" cy="220" r="3" style="animation-delay:-.5s"/><circle class="x-nd" cx="160" cy="640" r="4" style="animation-delay:-2.5s"/><circle class="x-nd" cx="760" cy="800" r="3" style="animation-delay:-1.6s"/></g>
</svg></div>""")

# ======================================================
# Sidebar
# ======================================================
def _status(label, state):
    cls = "on" if state == "done" else ("now" if state == "now" else "")
    return f'<div class="x-status"><span class="x-dot {cls}"></span>{label}</div>'


with st.sidebar:
    md(f'<div class="x-brand">{logo_svg(40)}{wordmark(34)}</div>'
       '<div class="x-side-sub">AI-assisted retinal analysis</div>')
    s = st.session_state
    flags = [s.image is not None, s.quality_result is not None,
             s.analysis_result is not None, bool(s.report_path)]
    nxt = next((i for i, d in enumerate(flags) if not d), -1)
    md("".join(_status(n, "done" if flags[i] else ("now" if i == nxt else ""))
               for i, n in enumerate(["Image uploaded", "Quality checked", "Retina analyzed", "Report ready"])))
    for label, value in [("Models", "EfficientNet-B3 · ConvNeXt-Tiny · Swin-Tiny"),
                         ("Classes", "Healthy, DR, Glaucoma, AMD"),
                         ("Explainability", "Grad-CAM + lesion segmentation"),
                         ("Purpose", "Research &amp; education")]:
        md(f'<div class="x-meta-l">{label}</div><div class="x-meta-v">{value}</div>')

# ======================================================
# Top bar + navigation
# ======================================================
_b, _n = st.columns([1, 2.6], vertical_alignment="center")
with _b:
    md(f'<div class="x-bar-top">{logo_svg(42)}{wordmark(36)}</div>')
with _n:
    st.radio("Navigate", PAGES, key="page", horizontal=True, label_visibility="collapsed")
page = st.session_state.page


def empty_state(icon, text, btn_label=None, target=None, key=None):
    md(f'<div class="x-empty"><span>{icon}</span>{text}</div>')
    if btn_label:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.button(btn_label, on_click=go, args=(target,), key=key, use_container_width=True)


# ======================================================
# PAGE 1: Upload & Quality
# ======================================================
def page_upload():
    if st.session_state.image is None:
        md(f"""
        <div class="x-hero">
        <div>
        {wordmark(78, "Explainable retinal AI")}
        <div class="x-h1">See what the model <span class="g">sees.</span></div>
        <div class="x-lead">Upload a retinal fundus photo. EyeX checks the image first, classifies it with three deep-learning models, then shows where each decision came from.</div>
        <div class="x-pills"><div class="x-pill" style="animation-delay:.2s">3 AI models</div><div class="x-pill" style="animation-delay:.3s">Quality gate</div><div class="x-pill" style="animation-delay:.4s">Grad-CAM</div><div class="x-pill" style="animation-delay:.5s">Lesion maps</div></div>
        </div>
        <div class="x-orbit">
        <div class="x-chip a">Quality gate<small>sharpness · contrast</small></div>
        <div class="x-chip b">Grad-CAM<small>attention heatmap</small></div>
        <div class="x-chip c">3 models<small>majority vote</small></div>
        <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
        <defs>
        <radialGradient id="fg" cx=".42" cy=".42" r=".65"><stop offset="0" stop-color="#A9C1EA"/><stop offset=".55" stop-color="#B45309"/><stop offset="1" stop-color="#3B0F08"/></radialGradient>
        <linearGradient id="sw" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#7FA0D6" stop-opacity="0"/><stop offset="1" stop-color="#7FA0D6" stop-opacity=".7"/></linearGradient>
        </defs>
        <g class="x-ring2"><circle cx="100" cy="100" r="95" fill="none" stroke="#7FA0D6" stroke-opacity=".5" stroke-dasharray="2 8"/><circle cx="100" cy="5" r="3.5" fill="#A9C1EA"/><circle cx="195" cy="100" r="2.5" fill="#7FA0D6"/></g>
        <circle class="x-pulse" cx="100" cy="100" r="84" fill="none" stroke="#3FA3A8" stroke-width="1.4"/>
        <circle cx="100" cy="100" r="76" fill="url(#fg)"/>
        <g stroke="#7F1D1D" stroke-width="2" fill="none" stroke-linecap="round" opacity=".85">
        <path d="M128 92 C108 80 90 62 70 50"/><path d="M128 92 C110 100 92 118 76 146"/>
        <path d="M128 92 C146 80 156 68 164 52"/><path d="M128 92 C146 106 156 122 160 142"/></g>
        <circle cx="128" cy="92" r="11" fill="#FDE68A"/><circle cx="76" cy="100" r="7" fill="#451A03" opacity=".7"/>
        <g class="x-sweep"><path d="M100 100 L100 8 A92 92 0 0 1 188 72 Z" fill="url(#sw)"/></g>
        </svg></div></div>
        <div class="x-ticker"><div class="x-track">
        <span><b>◆</b>Image quality check</span><span><b>◆</b>EfficientNet-B3</span><span><b>◆</b>ConvNeXt-Tiny</span><span><b>◆</b>Swin-Tiny</span><span><b>◆</b>Grad-CAM explanations</span><span><b>◆</b>IDRiD lesion segmentation</span><span><b>◆</b>PDF screening report</span>
        <span><b>◆</b>Image quality check</span><span><b>◆</b>EfficientNet-B3</span><span><b>◆</b>ConvNeXt-Tiny</span><span><b>◆</b>Swin-Tiny</span><span><b>◆</b>Grad-CAM explanations</span><span><b>◆</b>IDRiD lesion segmentation</span><span><b>◆</b>PDF screening report</span>
        </div></div>""")

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        with st.container(border=True):
            section("Upload a fundus image", "A clear, centered photograph gives the best results.")
            uploaded_file = st.file_uploader("Upload fundus image",
                                             type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                                             label_visibility="collapsed")
            st.caption("JPG, JPEG, PNG, BMP or TIFF.")

            if uploaded_file is not None and st.session_state.image_name != uploaded_file.name:
                try:
                    img = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                    for k in RESET_KEYS:
                        st.session_state[k] = None
                    st.session_state.image = img
                    st.session_state.image_name = uploaded_file.name
                    st.session_state.quality_image_name = uploaded_file.name
                    st.session_state.image_format = (uploaded_file.type or "unknown").split("/")[-1].upper()
                except (UnidentifiedImageError, OSError):
                    st.error("Unable to process this image. Upload a valid fundus image.")

        image = st.session_state.image
        if image is not None:
            with st.container(border=True):
                st.image(image, use_container_width=True)
                md(f"""<div class="x-info"><div><b>File</b>{st.session_state.image_name}</div>
                <div><b>Size</b>{image.width} × {image.height}px</div>
                <div><b>Format</b>{st.session_state.image_format}</div></div>""")

    with right:
        image = st.session_state.image
        with st.container(border=True):
            section("Image quality", "We check the photo before any AI model sees it.")
            if image is None:
                empty_state("🩻", "Upload an image to unlock the quality check.")
            else:
                if st.button("Check image quality", type="primary", use_container_width=True):
                    os.makedirs("results", exist_ok=True)
                    quality_temp_path = "results/quality_check_image.jpg"
                    try:
                        image.save(quality_temp_path)
                        with st.spinner("Checking retinal image quality..."):
                            st.session_state.quality_result = assess_image_quality(quality_temp_path)
                        st.session_state.analysis_result = None
                        st.session_state.classification_comparison = None
                        st.session_state.lesion_result = None
                        st.session_state.report_path = None
                        st.session_state.report_id = None
                    except Exception:
                        st.error("Unable to assess image quality.")
                        st.session_state.quality_result = None

                qr = st.session_state.quality_result
                if qr is not None:
                    quality = qr["quality"]
                    vc, vt = {"Good": ("#34D399", "Good quality. This image is suitable for AI analysis."),
                              "Fair": ("#FBBF24", "Fair quality. Analysis is possible, but a clearer image is better.")
                              }.get(quality, ("#3FA3A8", "Poor quality. Upload a clearer fundus image."))
                    md(f'<div class="x-verdict" style="--c:{vc}">{vt}</div>')

                    def tile(name, value, ok, bad):
                        c = "#34D399" if ok else "#3FA3A8"
                        return (f'<div class="x-qt" style="--c:{c}"><div class="x-label">{name}</div>'
                                f'<div class="v">{value}</div><div class="s">{"Acceptable" if ok else bad}</div></div>')

                    md('<div class="x-qgrid">'
                       + tile("Sharpness", f"{qr['sharpness']:.2f}", qr["sharpness_ok"], "Low sharpness")
                       + tile("Brightness", f"{qr['brightness']:.2f}", qr["brightness_ok"], "Poor brightness")
                       + tile("Contrast", f"{qr['contrast']:.2f}", qr["contrast_ok"], "Low contrast")
                       + tile("Resolution", f"{qr['width']} × {qr['height']}", qr["resolution_ok"], "Low resolution")
                       + '</div>')
                    st.write("")
                    passed = qr["passed_checks"]
                    md(f'<div class="x-label">Checks passed: {passed} of 4</div>'
                       f'<div class="x-bar"><div style="width:{passed / 4 * 100:.0f}%"></div></div>')
                    st.write("")
                    if quality == "Poor":
                        st.error("AI analysis is disabled for poor-quality images.")
                    else:
                        st.button("Continue to AI analysis", on_click=go, args=(P_ANALYSIS,),
                                  key="to_analysis", type="primary", use_container_width=True)


# ======================================================
# PAGE 2: AI Analysis
# ======================================================
def page_analysis():
    image = st.session_state.image
    qr = st.session_state.quality_result

    if image is None:
        empty_state("👁️", "No image yet. Upload a fundus photo first.", "Go to upload", P_UPLOAD, "a_up")
        return
    if qr is None:
        empty_state("🩻", "Run the image quality check before the AI analysis.", "Go to quality check", P_UPLOAD, "a_q")
        return

    quality = qr["quality"]
    top_l, top_r = st.columns([1, 1.4], gap="large")
    with top_l:
        with st.container(border=True):
            st.image(image, use_container_width=True)
    with top_r:
        with st.container(border=True):
            section("Analyze the retina",
                    "Runs EfficientNet-B3 with Grad-CAM, then compares ConvNeXt-Tiny and Swin-Tiny.")
            if quality == "Poor":
                st.error("AI analysis is disabled because the image quality is poor. Upload a clearer image.")
                analyze_clicked = False
            else:
                if quality == "Fair":
                    st.warning("Fair image quality. You can continue, but a clearer image is recommended.")
                analyze_clicked = st.button("Analyze retina", type="primary", use_container_width=True)

    if analyze_clicked:
        os.makedirs("results", exist_ok=True)
        temp_path = "results/uploaded_fundus_image.jpg"
        try:
            image.save(temp_path)
        except OSError:
            st.error("Unable to process this image. Upload a valid fundus image.")
            st.stop()
        try:
            with st.spinner("Analyzing retina: running EfficientNet-B3 and generating Grad-CAM..."):
                result = predict_image(temp_path)
        except Exception:
            st.error("Unable to process this image. Upload a valid fundus image.")
            st.stop()
        st.session_state.analysis_result = result
        st.session_state.report_path = None
        st.session_state.report_id = None

        try:
            with st.spinner("Running EfficientNet-B3, ConvNeXt-Tiny and Swin-Tiny comparison..."):
                model_results = predict_all_models(temp_path)
                model_agreement = calculate_model_agreement(model_results)
            st.session_state.classification_comparison = {"models": model_results, "agreement": model_agreement}
        except Exception as comparison_error:
            st.session_state.classification_comparison = None
            st.error("Unable to complete the three-model classification comparison.")
            st.caption(f"Technical detail: {comparison_error}")

        if result["disease"] == "DR":
            try:
                with st.spinner("Diabetic Retinopathy detected. Running IDRiD lesion analysis..."):
                    st.session_state.lesion_result = load_lesion_detector().predict(temp_path, threshold=0.5)
            except Exception as lesion_error:
                st.session_state.lesion_result = None
                st.error("Unable to complete IDRiD lesion analysis.")
                st.caption(f"Technical detail: {lesion_error}")
        else:
            st.session_state.lesion_result = None

    result = st.session_state.analysis_result
    if result is None:
        return

    accent = get_accent(result["disease"])
    conf = min(max(result["confidence"], 0.0), 1.0)
    comparison = st.session_state.classification_comparison

    # ---------- headline bento ----------
    st.write("")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        with st.container(border=True):
            md(f"""<div style="border-left:5px solid {accent['color']};padding-left:18px;">
            <div class="x-label">Predicted condition</div>
            <div class="x-big" style="color:{accent['color']}">{result['disease']}</div>
            <div class="x-chipline" style="background:{accent['color']}22;color:{accent['color']}">{accent['label']}</div>
            <div class="x-bar" style="margin-top:22px"><div style="width:{conf * 100:.1f}%"></div></div>
            <div class="x-label" style="margin-top:8px">Confidence {conf * 100:.2f}%</div></div>""")
            if comparison is not None:
                ag = comparison["agreement"]
                md(f"""<div class="x-info" style="margin-top:18px"><div><b>Model agreement</b>{ag["agreement_count"]}/{ag["total_models"]} ({ag["agreement_ratio"] * 100:.0f}%)</div>
                <div><b>Majority prediction</b>{ag["final_prediction"]}</div></div>""")
    with c2:
        with st.container(border=True):
            md(f"""<div class="x-label" style="text-align:center">Model confidence</div>
            <div class="x-ring" style="--p:{round(conf * 100)};--c:{accent['color']}"><b></b></div>""")
            st.caption("Prediction generated. This is not a clinical diagnosis.")

    # ---------- Grad-CAM ----------
    st.write("")
    section("Where the model looked", "Grad-CAM highlights the regions that contributed most to the prediction.")
    views = {"Overlay": ("Prediction overlay", "Activation regions laid over the original fundus.", result["overlay"]),
             "Heatmap": ("Grad-CAM heatmap", "Brighter areas mean stronger model activation.", result["heatmap"]),
             "Original": ("Original fundus", "The image provided to the model.", result["original_image"])}
    view = st.radio("Grad-CAM view", ["Overlay", "Heatmap", "Original", "All three"], horizontal=True,
                    key="gc_view", label_visibility="collapsed")
    if view == "All three":
        cols = st.columns(3, gap="medium")
        for col, (t, d, img) in zip(cols, views.values()):
            with col:
                with st.container(border=True):
                    md(f'<div class="x-ex-t">{t}</div><div class="x-ex-d">{d}</div>')
                    st.image(img, use_container_width=True)
    else:
        t, d, img = views[view]
        gl, gr = st.columns([2.2, 1], gap="large")
        with gl:
            with st.container(border=True, key="gc_stage"):
                st.image(img, use_container_width=True)
        with gr:
            with st.container(border=True):
                md(f'<div class="x-ex-t" style="font-size:20px">{t}</div><div class="x-ex-d">{d}</div>'
                   '<div class="x-tip">🔥 Brighter regions show stronger activation.</div>'
                   '<div class="x-tip">🎯 The overlay shows where the model focused.</div>'
                   '<div class="x-tip">⚕️ Attention is not a clinical diagnosis.</div>')
                st.caption("Switch views above, or choose All three to compare side by side.")

    # ---------- Model comparison ----------
    if comparison is not None:
        st.write("")
        section("Three models, one verdict", "How each model read this image, and how each performed on the test set.")
        model_results = comparison["models"]
        ag = comparison["agreement"]
        votes = "".join(f'<div class="x-vote {"y" if i < ag["agreement_count"] else ""}">{"✓" if i < ag["agreement_count"] else "–"}</div>'
                        for i in range(ag["total_models"]))
        md(f"""<div class="x-cons"><div class="x-votes">{votes}</div>
        <div><div class="x-label">Consensus</div><div class="x-big" style="font-size:34px">{ag["final_prediction"]}</div></div>
        <div style="margin-left:auto"><div class="x-label">Agreement</div>
        <div class="x-big" style="font-size:34px">{ag["agreement_count"]}/{ag["total_models"]} <span style="font-size:15px;color:var(--mut)">{ag["agreement_ratio"] * 100:.0f}% agree</span></div></div></div>""")
        perf = {"EfficientNet-B3": (84.22, 77.21), "ConvNeXt-Tiny": (78.50, 74.31), "Swin-Tiny": (79.40, 75.87)}
        cards = ""
        for name in ["EfficientNet-B3", "ConvNeXt-Tiny", "Swin-Tiny"]:
            mr = model_results.get(name)
            if mr is None:
                continue
            cf = mr.get("confidence", 0.0) * 100
            pred = mr.get("prediction", "Not available")
            win = "win" if str(pred) == str(ag["final_prediction"]) else ""
            acc, f1 = perf[name]
            cards += (f'<div class="x-mc {win}"><div class="nm">{name}</div>'
                      f'<div class="x-ring sm" style="--p:{round(cf)};--c:{get_accent(pred)["color"]}"><b></b></div>'
                      f'<div class="pr">{pred}</div><div class="x-label" style="margin-top:6px">{cf:.2f}% on this image</div>'
                      f'<div class="tt">Test-set performance</div>'
                      f'<div class="x-perf"><span>Accuracy</span><div class="x-bar"><div style="width:{acc}%"></div></div><span>{acc:.1f}%</span></div>'
                      f'<div class="x-perf"><span>Macro F1</span><div class="x-bar"><div style="width:{f1}%"></div></div><span>{f1:.1f}%</span></div></div>')
        if cards:
            md(f'<div class="x-mgrid">{cards}</div>')
        else:
            st.warning("Model comparison results are not available.")

    # ---------- Lesions ----------
    lesion_result = st.session_state.lesion_result
    if result["disease"] == "DR" and lesion_result is not None:
        st.write("")
        with st.container(border=True):
            section("Diabetic retinopathy lesions",
                    "Segmentation from the trained IDRiD U-Net. These are AI findings, not a diagnosis.")
            stats = lesion_result["statistics"]
            cols = st.columns(4, gap="medium")
            for i, (col, (title, color)) in enumerate(zip(cols, [
                    ("Microaneurysms", "#FF3B30"), ("Haemorrhages", "#5EEAD4"),
                    ("Hard Exudates", "#FFD60A"), ("Soft Exudates", "#34C759")])):
                item = stats[title]
                with col:
                    md(f"""<div class="x-lesion" style="--c:{color};animation-delay:{i * .12}s;margin-top:{(i % 2) * 18}px">
                    <div class="nm">{title}</div>
                    <div class="st">{"Detected" if item["detected"] else "Not detected"}</div>
                    <div class="ln"><b>Area:</b> {item["percentage"]:.3f}%<br><b>Regions:</b> {item["regions"]:,}</div></div>""")
            st.write("")
            overlay = load_lesion_detector().create_overlay(lesion_result, alpha=0.28)
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                st.image(overlay, caption="AI-detected retinal lesion regions", use_container_width=True)
            md("""<div class="x-legend"><span><i style="background:#FF3B30"></i>Microaneurysms</span>
            <span><i style="background:#5EEAD4"></i>Haemorrhages</span>
            <span><i style="background:#FFD60A"></i>Hard Exudates</span>
            <span><i style="background:#34C759"></i>Soft Exudates</span></div>""")

    st.write("")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.button("Continue to clinical report", on_click=go, args=(P_REPORT,), key="to_report",
                  type="primary", use_container_width=True)


# ======================================================
# PAGE 3: Clinical Report
# ======================================================
def page_report():
    s = st.session_state
    result = s.analysis_result
    if result is None:
        empty_state("📄", "Run the AI analysis first. The report is built from its results.",
                    "Go to AI analysis", P_ANALYSIS, "r_a")
        return

    lesion_result = s.lesion_result
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        with st.container(border=True):
            section("Clinical screening report", "Everything below goes into the PDF.")
            items = [
                ("Image quality", "Sharpness, brightness, contrast, resolution", s.quality_result is not None),
                ("Classification", f"{result['disease']} at {result['confidence'] * 100:.1f}% confidence", True),
                ("Model comparison", "EfficientNet-B3, ConvNeXt-Tiny, Swin-Tiny", s.classification_comparison is not None),
                ("Grad-CAM explanation", "Heatmap and overlay", True),
                ("Lesion analysis", "Included for DR cases", result["disease"] == "DR" and lesion_result is not None),
            ]
            md("".join(
                f'<div class="x-check {"ok" if ok else ""}" style="animation-delay:{i * .1}s"><div class="ic">{"✓" if ok else "–"}</div>'
                f'<div>{name}<small>{desc if ok else "Not available for this image"}</small></div></div>'
                for i, (name, desc, ok) in enumerate(items)))

            b1, b2 = st.columns(2)
            with b1:
                if st.button("View report on screen", use_container_width=True):
                    st.session_state.show_report = not st.session_state.get("show_report", False)
            with b2:
                report_clicked = st.button("Generate PDF report", type="primary", use_container_width=True)
            if report_clicked:
                try:
                    report_id = generate_report_id()
                    report_directory = os.path.join("results", "reports")
                    os.makedirs(report_directory, exist_ok=True)
                    report_path = os.path.join(report_directory, f"{report_id}.pdf")
                    report_image_path = "results/uploaded_fundus_image.jpg"

                    report_lesion_overlay = None
                    if result["disease"] == "DR" and lesion_result is not None:
                        report_lesion_overlay = load_lesion_detector().create_overlay(lesion_result, alpha=0.28)

                    with st.spinner("Generating EyeX clinical screening report..."):
                        generate_pdf_report(
                            output_path=report_path,
                            image_path=report_image_path,
                            quality_result=s.quality_result,
                            analysis_result=result,
                            classification_comparison=s.classification_comparison,
                            lesion_result=lesion_result,
                            lesion_overlay=report_lesion_overlay,
                            report_id=report_id,
                        )
                    s.report_path = report_path
                    s.report_id = report_id
                    st.success(f"Report generated. Report ID: {report_id}")
                except Exception as report_error:
                    s.report_path = None
                    s.report_id = None
                    st.error("Unable to generate the clinical report.")
                    st.caption(f"Technical detail: {report_error}")

            if s.report_path and os.path.exists(s.report_path):
                with open(s.report_path, "rb") as report_file:
                    report_bytes = report_file.read()
                st.download_button("Download clinical report", data=report_bytes,
                                   file_name=f"{s.report_id}.pdf", mime="application/pdf",
                                   use_container_width=True)

    with right:
        ready = bool(s.report_path)
        md(f"""<div class="x-doc"><div class="x-label">{"Report " + str(s.report_id) if ready else "Report preview"}</div>
        <div class="x-big" style="font-size:30px">EyeX screening</div>
        <div class="ln" style="width:70%"></div><div class="ln"></div><div class="ln" style="width:85%"></div>
        <div class="x-chipline" style="background:{get_accent(result['disease'])['color']}22;color:{get_accent(result['disease'])['color']}">{result['disease']}</div>
        <div class="ln" style="width:55%"></div><div class="ln"></div><div class="ln" style="width:75%"></div></div>""")

    if st.session_state.get("show_report"):
        import datetime
        qr = s.quality_result or {}
        acc = get_accent(result["disease"])
        qhtml = ""
        if qr:
            qhtml = (f'<h4>Image quality: {qr["quality"]}</h4><div class="x-kv">'
                     f'<div><b>Sharpness</b>{qr["sharpness"]:.2f}</div><div><b>Brightness</b>{qr["brightness"]:.2f}</div>'
                     f'<div><b>Contrast</b>{qr["contrast"]:.2f}</div><div><b>Resolution</b>{qr["width"]} × {qr["height"]}</div>'
                     f'<div><b>Checks passed</b>{qr["passed_checks"]} of 4</div></div>')
        chtml = ""
        cmp_ = s.classification_comparison
        if cmp_:
            ag = cmp_["agreement"]
            rows = "".join(f'<div><b>{n}</b>{m.get("prediction", "n/a")} · {m.get("confidence", 0.0) * 100:.1f}%</div>'
                           for n, m in cmp_["models"].items())
            chtml = f'<h4>Model comparison: {ag["agreement_count"]}/{ag["total_models"]} agree on {ag["final_prediction"]}</h4><div class="x-kv">{rows}</div>'
        lhtml = ""
        if lesion_result is not None and result["disease"] == "DR":
            rows = "".join(f'<div><b>{n}</b>{"Detected" if v["detected"] else "Not detected"} · {v["percentage"]:.3f}% · {v["regions"]:,} regions</div>'
                           for n, v in lesion_result["statistics"].items())
            lhtml = f'<h4>Lesion analysis</h4><div class="x-kv">{rows}</div>'
        st.write("")
        md(f"""<div class="x-sheet"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px">
        {wordmark(40, "Clinical screening report")}
        <div class="x-kv" style="min-width:280px"><div><b>Report ID</b>{s.report_id or "Not yet generated"}</div><div><b>Date</b>{datetime.date.today():%d %b %Y}</div></div></div>
        <h4>Result</h4><div class="x-kv"><div><b>Predicted condition</b><span style="color:{acc['color']};font-weight:800;font-size:17px">{result['disease']}</span></div>
        <div><b>Classification</b>{acc['label']}</div><div><b>Confidence</b>{result['confidence'] * 100:.2f}%</div></div>
        {qhtml}{chtml}{lhtml}</div>""")
        rc1, rc2 = st.columns(2, gap="medium")
        with rc1:
            st.image(result["original_image"], caption="Original fundus", use_container_width=True)
        with rc2:
            st.image(result["overlay"], caption="Grad-CAM overlay", use_container_width=True)
        if lesion_result is not None and result["disease"] == "DR":
            st.image(load_lesion_detector().create_overlay(lesion_result, alpha=0.28),
                     caption="AI-detected lesion regions", use_container_width=True)

    md("""<div class="x-disc"><b>Research &amp; educational prototype</b>
    <div>EyeX is intended for research and education. Model predictions are not a medical diagnosis or a
    substitute for evaluation by a qualified ophthalmologist.</div></div>""")


# ======================================================
# Router + footer
# ======================================================
{P_UPLOAD: page_upload, P_ANALYSIS: page_analysis, P_REPORT: page_report}[page]()

md(f"""<div class="x-footer">{logo_svg(30)}<div style="margin-top:6px">EyeX: Explainable Retinal AI. Research prototype.</div></div>""")