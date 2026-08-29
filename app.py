# -*- coding: utf-8 -*-
# ============================================================
#  ScoreSense AI  -  Official Web Interface
#  Run: streamlit run app.py
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

st.set_page_config(
    page_title="ScoreSense AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
#  PROFESSIONAL CSS
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ═══════════════════════════════════════
     ROOT TOKENS
  ═══════════════════════════════════════ */
  :root {
    --bg-base:      #060910;
    --bg-surface:   #0c1220;
    --bg-elevated:  #111827;
    --bg-card:      #141e2e;
    --border:       rgba(255,255,255,0.07);
    --border-glow:  rgba(99,179,237,0.25);
    --text-primary: #f0f4ff;
    --text-sec:     #8899bb;
    --text-muted:   #445577;
    --accent:       #3b82f6;
    --accent-glow:  rgba(59,130,246,0.35);
    --gold:         #f59e0b;
    --gold-glow:    rgba(245,158,11,0.3);
    --green:        #10b981;
    --red:          #ef4444;
  }

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg-base) !important;
  }

  /* ═══════════════════════════════════════
     HEADER BANNER
  ═══════════════════════════════════════ */
  .ss-header {
    position: relative;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 0 0 20px 20px;
    padding: 28px 36px 24px;
    margin-bottom: 28px;
    overflow: hidden;
  }
  .ss-header::before {
    content: 'SS';
    position: absolute;
    right: 30px; top: 50%;
    transform: translateY(-50%);
    font-family: 'Syne', sans-serif;
    font-size: 9rem;
    font-weight: 800;
    color: rgba(59,130,246,0.04);
    letter-spacing: -8px;
    pointer-events: none;
    line-height: 1;
  }
  .ss-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
  }
  .ss-header-title span { color: var(--accent); }
  .ss-header-sub {
    color: var(--text-sec);
    font-size: 0.88rem;
    margin: 0 0 14px 0;
    font-weight: 400;
  }
  .ss-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(59,130,246,0.1);
    color: #93c5fd;
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-right: 6px;
    margin-bottom: 4px;
  }
  .ss-badge-gold {
    background: rgba(245,158,11,0.1);
    color: #fcd34d;
    border-color: rgba(245,158,11,0.2);
  }
  .ss-badge-green {
    background: rgba(16,185,129,0.1);
    color: #6ee7b7;
    border-color: rgba(16,185,129,0.2);
  }

  /* ═══════════════════════════════════════
     LEFT PANEL
  ═══════════════════════════════════════ */
  .input-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 20px;
  }
  .input-panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 4px 0;
    letter-spacing: 0.02em;
  }
  .input-panel-sub {
    color: var(--text-muted);
    font-size: 0.74rem;
    margin: 0 0 18px 0;
  }
  .divider-thin {
    height: 1px;
    background: var(--border);
    margin: 14px 0;
  }

  /* Field labels */
  .f-label {
    color: var(--text-sec);
    font-size: 0.69rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 14px 0 1px 0;
  }
  .f-hint {
    color: var(--text-muted);
    font-size: 0.71rem;
    margin: 0 0 4px 0;
  }
  .f-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.13);
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #7dd3fc;
    margin: 3px 0 8px 0;
  }

  /* ═══════════════════════════════════════
     SCORE CARD
  ═══════════════════════════════════════ */
  .score-card {
    position: relative;
    border-radius: 18px;
    padding: 32px 28px;
    overflow: hidden;
    margin-bottom: 16px;
  }
  .score-card-noise {
    position: absolute;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    opacity: 0.4;
  }
  .score-card-orb {
    position: absolute;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    top: -60px; right: -60px;
  }
  .score-card-orb2 {
    position: absolute;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
    bottom: -40px; left: 20px;
  }
  .score-card-emoji {
    font-size: 2.4rem;
    margin-bottom: 4px;
    display: block;
  }
  .score-card-num {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
    margin: 0 0 2px 0;
    letter-spacing: -2px;
  }
  .score-card-grade {
    font-size: 1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.8);
    letter-spacing: 0.04em;
    margin: 6px 0 4px 0;
  }
  .score-card-msg {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
    margin: 0;
  }
  .score-card-ci {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,0,0,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.6);
    margin-top: 12px;
  }

  /* ═══════════════════════════════════════
     METRIC TILES
  ═══════════════════════════════════════ */
  .metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
  }
  .metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 12px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .metric-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
  }
  .mt-blue::after  { background: var(--accent); }
  .mt-gray::after  { background: var(--text-muted); }
  .mt-green::after { background: var(--green); }
  .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 3px;
  }
  .metric-lbl {
    font-size: 0.64rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 4px;
  }
  .metric-delta-pos { font-size: 0.73rem; color: var(--green); font-weight: 600; }
  .metric-delta-neg { font-size: 0.73rem; color: var(--red); font-weight: 600; }
  .metric-delta-neu { font-size: 0.73rem; color: var(--text-sec); font-weight: 500; }

  /* ═══════════════════════════════════════
     PROGRESS BAR
  ═══════════════════════════════════════ */
  .prog-track {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0 4px;
  }
  .prog-fill {
    height: 100%;
    border-radius: 8px;
    position: relative;
  }
  .prog-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 40px; height: 100%;
    background: rgba(255,255,255,0.3);
    border-radius: 0 8px 8px 0;
    filter: blur(4px);
  }
  .prog-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 14px;
  }

  /* ═══════════════════════════════════════
     SECTION HEADING
  ═══════════════════════════════════════ */
  .sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0 12px 0;
  }
  .sec-head-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
    flex-shrink: 0;
  }
  .sec-head-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .sec-head-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
  }

  /* ═══════════════════════════════════════
     HABIT TABLE
  ═══════════════════════════════════════ */
  .habit-table { width: 100%; }
  .habit-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .habit-row:last-child { border-bottom: none; }
  .h-feat {
    font-size: 0.78rem;
    color: var(--text-sec);
    min-width: 90px;
  }
  .h-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-primary);
    font-weight: 500;
  }
  .h-tgt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    color: var(--green);
    opacity: 0.7;
  }
  .h-warn { color: #fbbf24; }

  /* ═══════════════════════════════════════
     RECOMMENDATIONS
  ═══════════════════════════════════════ */
  .rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
  }
  .rec-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 3px 0 0 3px;
  }
  .rec-high::before   { background: var(--red); }
  .rec-medium::before { background: var(--gold); }
  .rec-low::before    { background: var(--accent); }
  .rec-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .rec-impact-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    padding: 1px 7px;
    border-radius: 20px;
    font-weight: 600;
    margin-left: auto;
  }
  .rec-high   .rec-impact-badge { background: rgba(239,68,68,0.15);  color: #fca5a5; }
  .rec-medium .rec-impact-badge { background: rgba(245,158,11,0.15); color: #fcd34d; }
  .rec-low    .rec-impact-badge { background: rgba(59,130,246,0.15); color: #93c5fd; }
  .rec-body {
    font-size: 0.8rem;
    color: var(--text-sec);
    line-height: 1.55;
  }

  /* ═══════════════════════════════════════
     EMPTY STATE
  ═══════════════════════════════════════ */
  .empty-hero {
    text-align: center;
    padding: 36px 24px 20px;
  }
  .empty-hero-icon {
    font-size: 3.5rem;
    margin-bottom: 14px;
    display: block;
  }
  .empty-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }
  .empty-hero-sub {
    color: var(--text-sec);
    font-size: 0.88rem;
    line-height: 1.6;
    max-width: 360px;
    margin: 0 auto 24px;
  }
  .feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 24px;
  }
  .feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    text-align: left;
  }
  .feature-card-icon { font-size: 1.4rem; margin-bottom: 6px; }
  .feature-card-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 3px;
  }
  .feature-card-desc {
    font-size: 0.74rem;
    color: var(--text-muted);
    line-height: 1.45;
  }

  /* ═══════════════════════════════════════
     PREDICT BUTTON
  ═══════════════════════════════════════ */
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 16px 20px !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.3),
                0 4px 20px rgba(37,99,235,0.4) !important;
    transition: all 0.25s ease !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 0 0 1px rgba(59,130,246,0.5),
                0 8px 30px rgba(37,99,235,0.6) !important;
    transform: translateY(-2px) !important;
  }

  /* ═══════════════════════════════════════
     SLIDER / RADIO TWEAKS
  ═══════════════════════════════════════ */
  div[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
  }
  div[data-testid="stNumberInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    text-align: center !important;
  }

  /* ═══════════════════════════════════════
     HIDE CHROME
  ═══════════════════════════════════════ */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  LOAD MODEL
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
    st.error("scoresense_model.pkl not found. Please run the notebook first to train the model.")
    st.stop()


# ============================================================
#  CONSTANTS
# ============================================================
BENCHMARKS = {
    "study_hours":    5.8,
    "sleep_hours":    7.4,
    "screen_time":    3.9,
    "active_time":    3.3,
    "social_time":    4.2,
    "attendance_pct": 87.0,
    "stressed":       0,
}

GRADE_CONFIG = {
    "A (Excellent)": {"color": "#059669", "bg": "linear-gradient(135deg,#059669,#047857)", "emoji": "🏆", "msg": "Outstanding performance!"},
    "B (Good)":      {"color": "#2563eb", "bg": "linear-gradient(135deg,#2563eb,#1d4ed8)", "emoji": "👍", "msg": "Good work, keep it up!"},
    "C (Average)":   {"color": "#d97706", "bg": "linear-gradient(135deg,#d97706,#b45309)", "emoji": "📘", "msg": "Room for improvement."},
    "D (Below Avg)": {"color": "#ea580c", "bg": "linear-gradient(135deg,#ea580c,#c2410c)", "emoji": "⚠️",  "msg": "Needs focused effort."},
    "F (Fail)":      {"color": "#dc2626", "bg": "linear-gradient(135deg,#dc2626,#b91c1c)", "emoji": "❗", "msg": "Immediate action required!"},
}

REC_ICONS = {
    "high":   ("🔴", "#ef4444"),
    "medium": ("🟡", "#f59e0b"),
    "low":    ("🔵", "#3b82f6"),
}

def grade_category(score):
    if score >= 85:   return "A (Excellent)"
    elif score >= 70: return "B (Good)"
    elif score >= 55: return "C (Average)"
    elif score >= 40: return "D (Below Avg)"
    else:             return "F (Fail)"


# ============================================================
#  FEATURE BUILDER & PREDICTOR
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
    expanded = {"mobile_est": mobile_est, "tv_est": tv_est,
                "exercise_est": exercise_est, "extra_est": extra_est}
    return row, expanded

def predict(d: dict):
    row, expanded = build_features(d)
    score = float(model.predict(row)[0])
    score = max(30.0, min(100.0, score))
    return round(score, 2), grade_category(score), expanded


# ============================================================
#  RECOMMENDATIONS
# ============================================================
def make_recommendations(d: dict, expanded: dict):
    recs = []
    gap = BENCHMARKS["study_hours"] - d["study_hours"]
    if gap > 0.5:
        recs.append({"icon": "📚", "title": "Increase Study Time",
            "detail": (f"You study {d['study_hours']:.1f}h/day. "
                       f"Aim for {BENCHMARKS['study_hours']:.1f}h. "
                       f"Add {gap:.1f}h using Pomodoro (25-min focused blocks + 5-min breaks)."),
            "impact": 0.40, "level": "high"})

    if d["sleep_hours"] < 7:
        recs.append({"icon": "😴", "title": "Improve Sleep Duration",
            "detail": (f"You sleep {d['sleep_hours']:.1f}h. Aim for 7-9 hours. "
                       "Poor sleep reduces memory consolidation by up to 40%."),
            "impact": 0.20, "level": "medium"})
    elif d["sleep_hours"] > 9.5:
        recs.append({"icon": "😴", "title": "Reduce Oversleeping",
            "detail": "Sleeping 10+ hours causes grogginess. Target 7-9h for peak cognitive performance.",
            "impact": 0.10, "level": "low"})

    if d["screen_time"] > BENCHMARKS["screen_time"] + 0.8:
        excess = d["screen_time"] - BENCHMARKS["screen_time"]
        recs.append({"icon": "📱", "title": "Cut Down Screen Time",
            "detail": (f"Your screen time: {d['screen_time']:.1f}h "
                       f"(~{expanded['mobile_est']:.1f}h mobile + ~{expanded['tv_est']:.1f}h TV). "
                       f"Reduce by {excess:.1f}h — try grayscale mode and app timers."),
            "impact": 0.15, "level": "medium"})

    if d["attendance_pct"] < BENCHMARKS["attendance_pct"] - 5:
        recs.append({"icon": "🏫", "title": "Improve Class Attendance",
            "detail": (f"Attendance: {d['attendance_pct']:.0f}%. "
                       f"Target: {BENCHMARKS['attendance_pct']:.0f}%. "
                       "Every missed class = 3x more self-study needed to catch up."),
            "impact": 0.18, "level": "medium"})

    if d["active_time"] < BENCHMARKS["active_time"] - 0.4:
        recs.append({"icon": "🏃", "title": "Increase Active Time",
            "detail": (f"Active time: {d['active_time']:.1f}h "
                       f"(~{expanded['exercise_est']:.1f}h exercise + ~{expanded['extra_est']:.1f}h extracurricular). "
                       "Exercise increases BDNF — a brain protein that improves learning speed."),
            "impact": 0.08, "level": "low"})

    if d["stressed"] == "Yes":
        recs.append({"icon": "🧘", "title": "Manage Stress",
            "detail": "Chronic stress shrinks the hippocampus (memory centre). "
                      "Try: 10-min daily meditation, time-blocking, or speaking to a counselor.",
            "impact": 0.12, "level": "low"})

    recs.sort(key=lambda x: x["impact"], reverse=True)
    return recs


# ============================================================
#  RADAR CHART  —  You Now vs Your Potential
# ============================================================
def radar_chart(d: dict):
    """Radar: You Now (orange solid) vs Your Potential (blue dashed). Dark-native."""
    stressed_bin = 1 if d["stressed"] == "Yes" else 0
    categories   = ["Study Hours", "Sleep Hours", "Attendance","Active Time", "Low Screen", "Not Stressed"]

    def norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    current_vals = [
        norm(d["study_hours"],        0,  10),
        norm(d["sleep_hours"],        3,  10),
        norm(d["attendance_pct"],    40, 100),
        norm(d["active_time"],        0,   8),
        norm(12 - d["screen_time"],   2,  12),
        norm(1 - stressed_bin,        0,   1),
    ]
    HEALTHY = {
        "study_hours":    min(d["study_hours"] + 1.5, 8.0),
        "sleep_hours":    max(min(d["sleep_hours"], 9.0), 7.0),
        "attendance_pct": min(d["attendance_pct"] + 10, 95.0),
        "active_time":    min(d["active_time"] + 1.0, 5.0),
        "screen_time":    max(d["screen_time"] - 1.5, 2.0),
        "stressed":       0,
    }
    potential_vals = [
        norm(HEALTHY["study_hours"],        0,  10),
        norm(HEALTHY["sleep_hours"],        3,  10),
        norm(HEALTHY["attendance_pct"],    40, 100),
        norm(HEALTHY["active_time"],        0,   8),
        norm(12 - HEALTHY["screen_time"],   2,  12),
        1.0,
    ]
    potential_vals = [max(c, p) for c, p in zip(current_vals, potential_vals)]

    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    cv     = current_vals   + [current_vals[0]]
    pv     = potential_vals + [potential_vals[0]]

    BG = "#0c1220"

    # Larger figure for proper visibility
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                           facecolor=BG)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Grid rings with subtle glow
    for r in [0.25, 0.5, 0.75, 1.0]:
        ring_a = [r] * (N + 1)
        lw  = 0.8 if r < 1.0 else 1.2
        alp = 0.18 if r < 1.0 else 0.30
        ax.plot(angles, ring_a, color="white", linewidth=lw, alpha=alp)

    # Spokes
    for ang in angles[:-1]:
        ax.plot([ang, ang], [0, 1], color="white", linewidth=0.5, alpha=0.10)

    # Potential zone (blue dashed)
    ax.fill(angles, pv, color="#3b82f6", alpha=0.12)
    ax.plot(angles, pv, color="#60a5fa", linewidth=2.2,
            linestyle="--", dashes=(7, 3))
    # Dots on potential
    ax.scatter(angles[:-1], potential_vals,
               color="#93c5fd", s=28, zorder=4, linewidths=0)

    # Current zone (orange solid)
    ax.fill(angles, cv, color="#f97316", alpha=0.22)
    ax.plot(angles, cv, color="#fb923c", linewidth=2.8)
    # Dots on current with glow effect
    ax.scatter(angles[:-1], current_vals,
               color="#fbbf24", s=55, zorder=6,
               linewidths=1.8, edgecolors="#f97316")

    # Ring percentage labels
    for r, lbl in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
        ax.text(np.pi / 2, r + 0.05, lbl, ha="center", va="bottom",
                fontsize=7, color="#445577", fontweight="500")

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, color="#94a3b8",
                       fontweight="600", linespacing=1.3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # Legend
    p_now  = mpatches.Patch(facecolor="#fb923c", alpha=0.85,
                             edgecolor="none", label="You Now")
    p_pot  = mpatches.Patch(facecolor="#60a5fa", alpha=0.65,
                             edgecolor="none", label="Your Potential")
    legend = ax.legend(
        handles=[p_now, p_pot],
        loc="upper right", bbox_to_anchor=(1.45, 1.22),
        fontsize=9, framealpha=0.25,
        facecolor="#0c1220", edgecolor=(1,1,1,0.10),
        labelcolor="#e2e8f0"
    )

    ax.set_title("You Now  vs  Your Potential",
                 fontsize=11, fontweight="bold",
                 color="#e2e8f0", pad=24, loc="center")

    fig.tight_layout(pad=2.0)
    return fig


# ============================================================
#  FEATURE IMPACT CHART  (shown before prediction)
# ============================================================
def impact_chart():
    """Horizontal bar chart — dark themed, showing feature weights."""
    labels  = ["Study Hours", "Sleep Quality", "Attendance",
               "Screen Time", "Stress", "Active Time", "Social Time"]
    impacts = [0.40, 0.20, 0.18, 0.15, 0.12, 0.08, 0.05]
    colors  = ["#10b981", "#8b5cf6", "#3b82f6", "#ef4444",
               "#f97316", "#06b6d4", "#64748b"]
    BG = "#0c1220"

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    bars = ax.barh(labels, impacts, color=colors, height=0.58,
                   edgecolor="none", alpha=0.88)

    for bar, val in zip(bars, impacts):
        ax.text(val + 0.007, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=9,
                color="#e2e8f0", fontweight="700")

    ax.set_xlabel("Relative Impact on Exam Score",
                  color="#445577", fontsize=8.5)
    ax.set_title("What Drives Your Score?", fontsize=12,
                 fontweight="bold", color="#f0f4ff", pad=14)
    ax.set_xlim(0, 0.58)
    ax.invert_yaxis()
    ax.tick_params(colors="#8899bb", labelsize=9.5, left=False)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=(1,1,1,0.05), linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    fig.tight_layout(pad=1.8)
    return fig


# ============================================================
#  SYNCED SLIDER + NUMBER INPUT
# ============================================================
def slider_with_input(label, key, min_val, max_val, default, step,
                      hint=None, estimate_html=None):
    val_key    = f"val_{key}"
    slider_key = f"sl_{key}"
    num_key    = f"ni_{key}"

    if val_key not in st.session_state:
        st.session_state[val_key] = float(default)

    def on_slider():
        st.session_state[val_key] = st.session_state[slider_key]

    def on_num():
        st.session_state[val_key] = float(st.session_state[num_key])

    st.markdown(f'<p class="field-label">{label}</p>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<p class="field-hint">{hint}</p>', unsafe_allow_html=True)

    col_s, col_n = st.columns([3, 1])
    with col_s:
        st.slider(f"_s_{key}", min_value=float(min_val), max_value=float(max_val),
                  value=float(st.session_state[val_key]), step=float(step),
                  label_visibility="collapsed", key=slider_key, on_change=on_slider)
    with col_n:
        st.number_input(f"_n_{key}", min_value=float(min_val), max_value=float(max_val),
                        value=float(st.session_state[val_key]), step=float(step),
                        label_visibility="collapsed", key=num_key, on_change=on_num)

    if estimate_html:
        st.markdown(f'<div class="field-estimate">{estimate_html}</div>',
                    unsafe_allow_html=True)

    return float(st.session_state[val_key])


# ============================================================
#  HEADER
# ============================================================
st.markdown("""
<div class="ss-header">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
    <span style="font-size:2.2rem;filter:drop-shadow(0 0 14px rgba(59,130,246,0.7));">🎓</span>
    <div>
      <div class="ss-header-title">Score<span>Sense</span> AI</div>
      <div class="ss-header-sub">Intelligent Academic Scoring &amp; Personalised Recommendation Engine</div>
    </div>
  </div>
  <div>
    <span class="ss-badge">⚡ ML Powered</span>
    <span class="ss-badge ss-badge-gold">🏅 Gradient Boosting</span>
    <span class="ss-badge ss-badge-green">📊 5,000 Students</span>
    <span class="ss-badge">🔮 Real-Time Prediction</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  MAIN LAYOUT
# ============================================================
left_col, right_col = st.columns([1, 1.65], gap="large")

# ── LEFT PANEL ────────────────────────────────────────────────
with left_col:
    st.markdown('### 📋 Your Daily Habits')
    st.caption('Drag the slider **or** type a value in the box on the right.')
    st.markdown('---')

    # Gender
    st.markdown('<p class="f-label">Gender</p>', unsafe_allow_html=True)
    gender = st.radio("_g", ["Male", "Female"], horizontal=True, label_visibility="collapsed")

    # Study
    study_hours = slider_with_input(
        "Study Hours / Day", "study", 0.0, 12.0, 4.5, 0.5,
        hint="Active studying only — not scrolling or passive reading."
    )

    # Sleep
    sleep_hours = slider_with_input(
        "Sleep Hours / Day", "sleep", 3.0, 12.0, 7.0, 0.5,
        hint="Average hours of sleep per night."
    )

    # Screen Time
    screen_time = slider_with_input(
        "Screen Time / Day (Mobile + TV)", "screen", 0.0, 14.0, 5.5, 0.5,
        hint="Combined phone + TV/streaming hours.",
        estimate_html=f"📱 Mobile ~{round(st.session_state.get('val_screen',5.5)*0.6,1)}h &nbsp;|&nbsp; 📺 TV ~{round(st.session_state.get('val_screen',5.5)*0.4,1)}h"
    )

    # Active Time
    active_time = slider_with_input(
        "Active Time / Day (Exercise + Extra)", "active", 0.0, 10.0, 2.7, 0.5,
        hint="Physical workout + clubs/hobbies combined.",
        estimate_html=f"🏃 Exercise ~{round(st.session_state.get('val_active',2.7)*0.6,1)}h &nbsp;|&nbsp; 🎨 Extra ~{round(st.session_state.get('val_active',2.7)*0.4,1)}h"
    )

    # Social Time
    social_time = slider_with_input(
        "Social Time / Day (Friends + Family)", "social", 0.0, 12.0, 4.5, 0.5,
        hint="Time with friends and family combined.",
        estimate_html=f"👫 Friends ~{round(st.session_state.get('val_social',4.5)*0.5,1)}h &nbsp;|&nbsp; 🏠 Family ~{round(st.session_state.get('val_social',4.5)*0.5,1)}h"
    )

    # Attendance
    attendance_pct = slider_with_input(
        "Attendance (%)", "attend", 40.0, 100.0, 78.0, 1.0,
        hint="Percentage of classes you attend."
    )

    # Stressed
    st.markdown('<p class="f-label">Experiencing Stress?</p>', unsafe_allow_html=True)
    st.markdown('<p class="f-hint">From studies or personal life</p>', unsafe_allow_html=True)
    stressed = st.selectbox("_st", ["No", "Yes"], label_visibility="collapsed")

    st.markdown("")

    predict_btn = st.button(
        "🔮  Predict My Score & Get Recommendations",
        use_container_width=True, type="primary"
    )


# ── RIGHT PANEL ───────────────────────────────────────────────
with right_col:

    if predict_btn:
        student = {
            "gender": gender, "study_hours": study_hours,
            "sleep_hours": sleep_hours, "screen_time": screen_time,
            "active_time": active_time, "social_time": social_time,
            "attendance_pct": attendance_pct, "stressed": stressed,
        }
        score, grade, expanded = predict(student)
        cfg   = GRADE_CONFIG[grade]
        rmse  = pkg.get("metadata", {}).get("rmse", 4.0)
        ci_lo = max(30, round(score - 1.96 * rmse, 1))
        ci_hi = min(100, round(score + 1.96 * rmse, 1))
        recs  = make_recommendations(student, expanded)
        potential_score = min(100, round(score + len(recs) * 4.5, 1))

        # ── SCORE CARD ────────────────────────────────────────
        st.markdown(
            f'<div class="score-card" style="background:{cfg["bg"]};">'
            f'  <div class="score-card-noise"></div>'
            f'  <div class="score-card-orb"></div>'
            f'  <div class="score-card-orb2"></div>'
            f'  <span class="score-card-emoji">{cfg["emoji"]}</span>'
            f'  <div class="score-card-num">{score}</div>'
            f'  <div class="score-card-grade">{grade}</div>'
            f'  <div class="score-card-msg">{cfg["msg"]}</div>'
            f'  <div><span class="score-card-ci">📊 95% CI: {ci_lo} – {ci_hi}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── METRIC TILES ──────────────────────────────────────
        avg_delta = score - 65
        pot_delta = potential_score - score
        avg_sign  = "pos" if avg_delta >= 0 else "neg"
        avg_word  = "above avg" if avg_delta >= 0 else "below avg"

        st.markdown(
            f'<div class="metric-grid">'
            f'  <div class="metric-tile mt-blue">'
            f'    <div class="metric-lbl">Your Score</div>'
            f'    <div class="metric-val">{score}</div>'
            f'    <div class="metric-delta-{avg_sign}">{avg_delta:+.1f} {avg_word}</div>'
            f'  </div>'
            f'  <div class="metric-tile mt-gray">'
            f'    <div class="metric-lbl">Class Average</div>'
            f'    <div class="metric-val">~65</div>'
            f'    <div class="metric-delta-neu">baseline</div>'
            f'  </div>'
            f'  <div class="metric-tile mt-green">'
            f'    <div class="metric-lbl">Your Potential</div>'
            f'    <div class="metric-val">~{potential_score}</div>'
            f'    <div class="metric-delta-pos">+{pot_delta:.1f} pts possible</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Progress bar
        pct = int(score)
        st.markdown(
            f'<div class="prog-track">'
            f'  <div class="prog-fill" style="width:{pct}%;background:{cfg["color"]};"></div>'
            f'</div>'
            f'<div class="prog-labels"><span>0</span><span>Score: {score}</span><span>100</span></div>',
            unsafe_allow_html=True
        )

        # ── RADAR + HABIT TABLE ───────────────────────────────
        st.markdown(
            '<div class="sec-head"><div class="sec-head-dot"></div>'
            '<div class="sec-head-text">Habit Profile</div>'
            '<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )

        rc1, rc2 = st.columns([1.3, 1])
        with rc1:
            st.pyplot(radar_chart(student), use_container_width=True)

        with rc2:
            st.markdown(
                '<div class="sec-head" style="margin-top:8px"><div class="sec-head-dot" style="background:#10b981;box-shadow:0 0 8px rgba(16,185,129,0.4);"></div>'
                '<div class="sec-head-text">You vs Target</div>'
                '<div class="sec-head-line"></div></div>',
                unsafe_allow_html=True
            )

            rows = [
                ("Gender",       gender,                   "—"),
                ("Study/Day",    f"{study_hours}h",        f"{BENCHMARKS['study_hours']}h",
                 study_hours < BENCHMARKS["study_hours"] - 0.5),
                ("Sleep/Day",    f"{sleep_hours}h",        "7–9h",
                 sleep_hours < 7 or sleep_hours > 9.5),
                ("Screen Time",  f"{screen_time}h",        f"<{BENCHMARKS['screen_time']}h",
                 screen_time > BENCHMARKS["screen_time"] + 0.8),
                ("Active Time",  f"{active_time}h",        f"{BENCHMARKS['active_time']}h",
                 active_time < BENCHMARKS["active_time"] - 0.4),
                ("Social Time",  f"{social_time}h",        f"~{BENCHMARKS['social_time']}h", False),
                ("Attendance",   f"{attendance_pct:.0f}%", f"{BENCHMARKS['attendance_pct']:.0f}%",
                 attendance_pct < BENCHMARKS["attendance_pct"] - 5),
                ("Stress",       stressed,                 "No", stressed == "Yes"),
            ]

            html_rows = ""
            for row in rows:
                if len(row) == 3:
                    feat, val, tgt, warn_flag = row[0], row[1], row[2], False
                else:
                    feat, val, tgt, warn_flag = row
                warn = " ⚠️" if warn_flag else ""
                val_class = "h-val h-warn" if warn_flag else "h-val"
                html_rows += (
                    f'<div class="habit-row">'
                    f'<span class="h-feat">{feat}</span>'
                    f'<span class="{val_class}">{val}{warn}</span>'
                    f'<span class="h-tgt">/ {tgt}</span>'
                    f'</div>'
                )
            st.markdown(f'<div class="habit-table">{html_rows}</div>', unsafe_allow_html=True)

        # ── RECOMMENDATIONS ───────────────────────────────────
        st.markdown(
            '<div class="sec-head"><div class="sec-head-dot" style="background:#f59e0b;box-shadow:0 0 8px rgba(245,158,11,0.4);"></div>'
            '<div class="sec-head-text">💡 Personalised Recommendations</div>'
            '<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )

        if not recs:
            st.success(
                "🌟 **Exceptional habits!** You're already performing at the top level. "
                "Maintain this consistency — consider peer tutoring or mentoring others."
            )
        else:
            for i, rec in enumerate(recs, 1):
                lvl = rec.get("level", "low")
                st.markdown(
                    f'<div class="rec-card rec-{lvl}">'
                    f'  <div class="rec-title">'
                    f'    {rec["icon"]} {i}. {rec["title"]}'
                    f'    <span class="rec-impact-badge">Impact {rec["impact"]:.0%}</span>'
                    f'  </div>'
                    f'  <div class="rec-body">{rec["detail"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── EMPTY STATE ───────────────────────────────────────────
    else:
        st.markdown("""
        <div class="empty-hero">
          <span class="empty-hero-icon">🎯</span>
          <div class="empty-hero-title">Ready to Predict?</div>
          <div class="empty-hero-sub">
            Enter your daily habits on the left and click the button
            to receive your AI-powered exam score prediction with
            personalised action plan.
          </div>
        </div>
        <div class="feature-grid">
          <div class="feature-card">
            <div class="feature-card-icon">🎯</div>
            <div class="feature-card-title">Predicted Score</div>
            <div class="feature-card-desc">ML-powered prediction with 95% confidence interval</div>
          </div>
          <div class="feature-card">
            <div class="feature-card-icon">📊</div>
            <div class="feature-card-title">Habit Radar</div>
            <div class="feature-card-desc">You Now vs Your Potential — see your growth gap</div>
          </div>
          <div class="feature-card">
            <div class="feature-card-icon">💡</div>
            <div class="feature-card-title">Action Plan</div>
            <div class="feature-card-desc">Ranked recommendations sorted by score impact</div>
          </div>
          <div class="feature-card">
            <div class="feature-card-icon">📈</div>
            <div class="feature-card-title">Growth Metrics</div>
            <div class="feature-card-desc">See exactly how many points you can gain</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="sec-head"><div class="sec-head-dot"></div>'
            '<div class="sec-head-text">What Drives Your Score?</div>'
            '<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )
        st.pyplot(impact_chart(), use_container_width=True)