# -*- coding: utf-8 -*-
# ============================================================
#  ScoreSense AI  -  Premium Aurora Edition
#  Run: streamlit run premium_app.py
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="ScoreSense AI | Premium",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  AURORA GLASS CSS
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --aurora-1: #6E44FF;
    --aurora-2: #0DE2EA;
    --aurora-3: #F9A03F;
    --bg-dark: #07070B;
    --panel-bg: rgba(20, 20, 35, 0.5);
    --sidebar-bg: rgba(12, 12, 18, 0.85);
    --border-light: rgba(255, 255, 255, 0.08);
    --glass-blur: blur(24px);
    --text-main: #FFFFFF;
    --text-muted: #9CA3AF;
  }

  /* Core App Overrides */
  [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark);
    background-image: 
      radial-gradient(circle at 10% 20%, rgba(110, 68, 255, 0.15) 0%, transparent 40%),
      radial-gradient(circle at 90% 40%, rgba(13, 226, 234, 0.15) 0%, transparent 40%),
      radial-gradient(circle at 50% 90%, rgba(249, 160, 63, 0.1) 0%, transparent 50%);
    background-attachment: fixed;
    color: var(--text-main);
    font-family: 'DM Sans', sans-serif;
  }
  
  [data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    backdrop-filter: var(--glass-blur);
    border-right: 1px solid var(--border-light);
  }

  h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; }

  /* Premium Header Elements */
  .premium-header {
    margin-top: 10px;
    margin-bottom: 40px;
    padding: 40px;
    border-radius: 24px;
    background: var(--panel-bg);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border-light);
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
  }
  .premium-header::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  }
  .title-glow {
    font-family: 'Outfit', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF, var(--aurora-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
  }
  .subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    font-weight: 400;
    margin-top: 5px;
  }
  
  /* Circle Gauge SVG */
  .gauge-container { display: flex; justify-content: center; align-items: center; margin: 30px 0; }
  .gauge-svg { width: 280px; height: 280px; transform: rotate(-90deg); filter: drop-shadow(0 0 20px rgba(0,0,0,0.5)); }
  .gauge-track { fill: none; stroke: rgba(255,255,255,0.05); stroke-width: 12; }
  .gauge-fill {
    fill: none; stroke-width: 12; stroke-linecap: round;
    transition: stroke-dasharray 1.5s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .gauge-text-val {
    font-family: 'Outfit'; font-size: 4rem; font-weight: 800; fill: #fff;
    transform: rotate(90deg); transform-origin: center; dominant-baseline: middle; text-anchor: middle;
  }
  .gauge-text-lbl {
    font-family: 'DM Sans'; font-size: 1rem; fill: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
    transform: rotate(90deg) translate(0, 45px); transform-origin: center; dominant-baseline: middle; text-anchor: middle;
  }
  
  /* Feature Cards */
  .feature-card {
    background: var(--panel-bg);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border-light);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }
  .feature-card:hover { transform: translateY(-5px); border-color: rgba(255,255,255,0.2); box-shadow: 0 15px 30px rgba(0,0,0,0.4); }
  .fc-icon { font-size: 2rem; margin-bottom: 12px; }
  .fc-title { font-family: 'Outfit'; font-weight: 700; font-size: 1.1rem; color: #fff; margin-bottom: 6px; }
  .fc-desc { font-size: 0.9rem; color: var(--text-muted); line-height: 1.5; }
  .fc-impact { position: absolute; top: 20px; right: 20px; background: rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 12px; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #fff; font-weight: 600; }

  /* Generic UI components overrides */
  div[data-testid="stSlider"] > div > div > div > div { background: linear-gradient(90deg, var(--aurora-1), var(--aurora-2)) !important; }
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--aurora-1), var(--aurora-2)) !important;
    border: none !important; color: white !important; font-family: 'Outfit' !important; font-weight: 700 !important;
    font-size: 1.1rem !important; padding: 24px !important; border-radius: 16px !important;
    box-shadow: 0 8px 25px rgba(13, 226, 234, 0.3) !important; transition: all 0.3s ease !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover { transform: scale(1.02) !important; box-shadow: 0 12px 35px rgba(13, 226, 234, 0.5) !important; }

  #MainMenu, footer { visibility: hidden; }
  header { background-color: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  MODEL CONFIGURATION
# ============================================================
@st.cache_resource
def load_model():
    with open("scoresense_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    pkg      = load_model()
    model    = pkg["model"]
    FEATURES = pkg["features"]
except FileNotFoundError:
    st.error("scoresense_model.pkl not found. Please train the model first.")
    st.stop()

BENCHMARKS = {
    "study_hours":    5.8, "sleep_hours":    7.4, "screen_time":    3.9,
    "active_time":    3.3, "social_time":    4.2, "attendance_pct": 87.0,
    "stressed":       0,
}

GRADE_CONFIG = {
    "A (Excellent)": {"color": "#0DE2EA", "glow": "rgba(13,226,234,0.6)", "emoji": "🏆", "msg": "Elite Performance"},
    "B (Good)":      {"color": "#6E44FF", "glow": "rgba(110,68,255,0.6)", "emoji": "👍", "msg": "Strong Trajectory"},
    "C (Average)":   {"color": "#F9A03F", "glow": "rgba(249,160,63,0.6)", "emoji": "📘", "msg": "Baseline Operation"},
    "D (Below Avg)": {"color": "#EF4444", "glow": "rgba(239,68,68,0.6)", "emoji": "⚠️", "msg": "Warning: Deviation"},
    "F (Fail)":      {"color": "#991B1B", "glow": "rgba(153,27,27,0.6)", "emoji": "❗", "msg": "System Critical"},
}

def grade_category(score):
    if score >= 85:   return "A (Excellent)"
    elif score >= 70: return "B (Good)"
    elif score >= 55: return "C (Average)"
    elif score >= 40: return "D (Below Avg)"
    else:             return "F (Fail)"

# ============================================================
#  FEATURE LOGIC
# ============================================================
def build_features(d: dict):
    screen_time  = d["screen_time"]
    active_time  = d["active_time"]
    social_time  = d["social_time"]
    stressed_bin = 1 if d["stressed"] == "Yes" else 0
    gender_enc   = 1 if d["gender"] == "Male" else 0
    study_hours  = d["study_hours"]
    sleep_hours  = d["sleep_hours"]
    attendance   = d["attendance_pct"]

    mobile_est   = round(screen_time * 0.60, 1)
    tv_est       = round(screen_time * 0.40, 1)
    exercise_est = round(active_time * 0.60, 1)
    extra_est    = round(active_time * 0.40, 1)

    study_efficiency   = study_hours / (screen_time + 1)
    health_score       = (sleep_hours / 8 + exercise_est / 2 - stressed_bin * 0.4) / 3 * 100
    sleep_adequate     = int(sleep_hours >= 7)
    high_screen        = int(screen_time > 5)
    study_x_attendance = study_hours * attendance / 100
    stress_adj_study   = study_hours * (1 - stressed_bin * 0.25)

    feature_map = {
        "sleep_hours": sleep_hours, "study_hours": study_hours,
        "screen_time": screen_time, "active_time": active_time,
        "social_time": social_time, "attendance_pct": attendance,
        "stressed": stressed_bin,   "gender_encoded": gender_enc,
        "study_efficiency": study_efficiency, "health_score": health_score,
        "sleep_adequate": sleep_adequate, "high_screen": high_screen,
        "study_x_attendance": study_x_attendance, "stress_adj_study": stress_adj_study,
    }
    row = pd.DataFrame([[feature_map[f] for f in FEATURES]], columns=FEATURES)
    expanded = {"mobile_est": mobile_est, "tv_est": tv_est, "exercise_est": exercise_est, "extra_est": extra_est}
    return row, expanded

def predict(d: dict):
    row, expanded = build_features(d)
    score = float(model.predict(row)[0])
    score = max(30.0, min(100.0, score))
    return round(score, 2), grade_category(score), expanded

def make_recommendations(d: dict, expanded: dict):
    recs = []
    gap = BENCHMARKS["study_hours"] - d["study_hours"]
    if gap > 0.5:
        recs.append({"title": "Neural Focus Protocol", "detail": f"Increase focus output to {BENCHMARKS['study_hours']:.1f}h/day. Add {gap:.1f}h using intensive Pomodoro sessions.", "impact": 0.40, "icon": "🧠"})
    if d["sleep_hours"] < 7:
        recs.append({"title": "Regenerative Cycle Alert", "detail": f"Current cycle: {d['sleep_hours']:.1f}h. Sub-optimal. Aim for 7-9h to maximize memory consolidation.", "impact": 0.20, "icon": "🛌"})
    elif d["sleep_hours"] > 9.5:
        recs.append({"title": "Oversleep Warning", "detail": "Extended stasis detected. Reduce below 9h to prevent grogginess and cognitive lag.", "impact": 0.10, "icon": "🛌"})
    if d["screen_time"] > BENCHMARKS["screen_time"] + 0.8:
        excess = d["screen_time"] - BENCHMARKS["screen_time"]
        recs.append({"title": "Digital Detoxification", "detail": f"High digital saturation detected (~{d['screen_time']:.1f}h). Reduce by {excess:.1f}h. Initiate app limits.", "impact": 0.15, "icon": "📵"})
    if d["attendance_pct"] < BENCHMARKS["attendance_pct"] - 5:
        recs.append({"title": "Presence Calibration", "detail": f"Network attendance at {d['attendance_pct']:.0f}%. Target node connection rate: {BENCHMARKS['attendance_pct']:.0f}%.", "impact": 0.18, "icon": "📡"})
    if d["active_time"] < BENCHMARKS["active_time"] - 0.4:
        recs.append({"title": "Kinetic Optimization", "detail": f"Physical input low ({d['active_time']:.1f}h). Increase BDNF flow via kinetic activities.", "impact": 0.08, "icon": "⚡"})
    if d["stressed"] == "Yes":
        recs.append({"title": "Stress Response Purge", "detail": "High cortisol levels detected. Run meditation or decompression protocols to preserve hippocampal integrity.", "impact": 0.12, "icon": "🧘"})

    recs.sort(key=lambda x: x["impact"], reverse=True)
    return recs

# ============================================================
#  HOLOGRAPHIC RADAR CHART
# ============================================================
def holographic_radar_chart(d: dict):
    stressed_bin = 1 if d["stressed"] == "Yes" else 0
    categories = ["Study", "Sleep", "Class", "Active", "Digital\nDetox", "Zen"]
    def norm(val, lo, hi): return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    cv = [
        norm(d["study_hours"], 0, 10), norm(d["sleep_hours"], 3, 10), norm(d["attendance_pct"], 40, 100),
        norm(d["active_time"], 0, 8), norm(12 - d["screen_time"], 2, 12), norm(1 - stressed_bin, 0, 1)
    ]
    HEALTHY = {
        "study_hours": min(d["study_hours"] + 1.5, 8.0), "sleep_hours": max(min(d["sleep_hours"], 9.0), 7.0),
        "attendance_pct": min(d["attendance_pct"] + 10, 95.0), "active_time": min(d["active_time"] + 1.0, 5.0),
        "screen_time": max(d["screen_time"] - 1.5, 2.0), "stressed": 0,
    }
    pv = [
        norm(HEALTHY["study_hours"], 0, 10), norm(HEALTHY["sleep_hours"], 3, 10), norm(HEALTHY["attendance_pct"], 40, 100),
        norm(HEALTHY["active_time"], 0, 8), norm(12 - HEALTHY["screen_time"], 2, 12), 1.0
    ]
    pv = [max(c, p) for c, p in zip(cv, pv)]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    cv += [cv[0]]; pv += [pv[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)

    # Holographic Grids
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r]*(N+1), color="#0DE2EA", linewidth=0.5, alpha=0.3)
    for ang in angles[:-1]:
        ax.plot([ang, ang], [0, 1], color="#0DE2EA", linewidth=0.5, alpha=0.3)

    # Potential
    ax.fill(angles, pv, color="#6E44FF", alpha=0.1)
    ax.plot(angles, pv, color="#6E44FF", linewidth=2, linestyle='dotted')
    
    # Current
    ax.fill(angles, cv, color="#0DE2EA", alpha=0.25)
    ax.plot(angles, cv, color="#0DE2EA", linewidth=3)
    ax.scatter(angles[:-1], cv[:-1], color="#FFFFFF", s=50, edgecolors="#0DE2EA", zorder=10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color="#FFFFFF")
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    
    fig.tight_layout()
    return fig

# ============================================================
#  LAYOUT: SIDEBAR & MAIN AREA
# ============================================================
st.sidebar.markdown("""<div style="text-align:center; padding: 20px 0;"><h2 style="margin:0; font-family:'Outfit'; font-weight:800; background:linear-gradient(90deg,#fff,#0DE2EA); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">SCORESENSE</h2><p style="color:#9CA3AF; letter-spacing:4px; font-size:0.7rem; text-transform:uppercase;">Input Matrix</p></div>""", unsafe_allow_html=True)
st.sidebar.markdown('---')

with st.sidebar:
    gender = st.selectbox("Operative Type (Gender)", ["Male", "Female"])
    study_hours = st.slider("Neural Focus (Study Hours)", 0.0, 12.0, 4.5, 0.5)
    sleep_hours = st.slider("Stasis Cycle (Sleep Hours)", 3.0, 12.0, 7.0, 0.5)
    screen_time = st.slider("Digital Saturation (Screen Time)", 0.0, 14.0, 5.5, 0.5)
    active_time = st.slider("Kinetic Output (Active Time)", 0.0, 10.0, 2.7, 0.5)
    social_time = st.slider("Network Bandwidth (Social Time)", 0.0, 12.0, 4.5, 0.5)
    attendance_pct = st.slider("Node Connection (Attendance %)", 40.0, 100.0, 78.0, 1.0)
    stressed = st.radio("Cortisol Overload? (Stress)", ["No", "Yes"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("INITIALIZE PREDICTION", width='stretch', type="primary")

# ============================================================
#  MAIN CANVAS
# ============================================================
st.markdown("""
<div class="premium-header">
  <div style="display:flex; justify-content:space-between; align-items:flex-end;">
    <div>
      <h1 class="title-glow">ScoreSense <i>Premium</i></h1>
      <div class="subtitle">Next-Generation Academic Prediction & Neural Optimization Model.</div>
    </div>
    <div style="font-family:'JetBrains Mono'; color:#0DE2EA; font-size:0.8rem; letter-spacing:0.1em; opacity:0.8; text-align:right;">
      MODEL STATUS: ONLINE<br>ALGORITHM: GRADIENT BOOSTING
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not predict_btn:
    st.markdown("""
    <div style="text-align:center; margin-top:80px; animation: fadeUp 1s ease-out;">
      <h1 style="font-size:4rem; margin:0; opacity:0.8;">🌌</h1>
      <h3 style="color:#FFF; opacity:0.6; font-weight:400; margin-top:20px;">Awaiting Input Parameters...</h3>
      <p style="color:var(--text-muted); font-size:0.95rem;">Configure the matrix on the sidebar and press <strong style="color:var(--aurora-2)">INITIALIZE</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    student = {"gender": gender, "study_hours": study_hours, "sleep_hours": sleep_hours, "screen_time": screen_time, "active_time": active_time, "social_time": social_time, "attendance_pct": attendance_pct, "stressed": stressed}
    score, grade, expanded = predict(student)
    cfg = GRADE_CONFIG[grade]
    
    circumference = 2 * np.pi * 130
    dash_offset = circumference - (score / 100.0) * circumference
    
    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.markdown(f"""
        <h3 style="margin-top:0; color:#fff;">Score Projection</h3>
        <p style="color:var(--text-muted); margin-bottom:20px;">Predicted examination outcome based on current parameters.</p>
        <div class="gauge-container" style="background:var(--panel-bg); backdrop-filter:var(--glass-blur); border:1px solid var(--border-light); border-radius:24px; padding:40px 0;">
          <svg class="gauge-svg" style="filter: drop-shadow(0 0 15px {cfg['glow']});">
            <circle class="gauge-track" cx="140" cy="140" r="130"></circle>
            <circle class="gauge-fill" cx="140" cy="140" r="130" stroke="{cfg['color']}" 
                    stroke-dasharray="{circumference}" stroke-dashoffset="{dash_offset}"></circle>
            <text class="gauge-text-val" x="140" y="140">{score}</text>
            <text class="gauge-text-lbl" x="140" y="140">{cfg['msg']} {cfg['emoji']}</text>
          </svg>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<br><h3 style="color:#fff;">Holographic Profile</h3>', unsafe_allow_html=True)
        st.pyplot(holographic_radar_chart(student), width="stretch")

    with col2:
        recs = make_recommendations(student, expanded)
        st.markdown(f'<h3 style="margin-top:0; color:#fff;">Actionable Protocols ({len(recs)})</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--text-muted); margin-bottom:20px;">AI-derived instructions to reach maximum potential.</p>', unsafe_allow_html=True)
        
        if not recs:
            st.markdown("""<div class="feature-card" style="border-color:#0DE2EA; text-align:center;"><div class="fc-icon">🌟</div><div class="fc-title">Optimum Vector Reached</div><div class="fc-desc">Your parameters are flawlessly aligned. Maintain current operational logic to secure top performance over time.</div></div>""", unsafe_allow_html=True)
        else:
            for rec in recs:
                st.markdown(f"""
                <div class="feature-card">
                  <div class="fc-icon">{rec["icon"]}</div>
                  <div class="fc-title">{rec["title"]}</div>
                  <div class="fc-desc">{rec["detail"]}</div>
                  <div class="fc-impact">Yield: +{rec['impact']:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
