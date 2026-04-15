"""
RCA_EfficientNet · Bearing Fault Diagnosis Dashboard
Production-ready Streamlit dashboard — v2.0
Live Prediction removed · All charts animated on click-to-reveal
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from graphviz import Digraph

from utils import (
    CLASS_NAMES,
    load_model_comparison,
    load_roc_data,
    load_pr_data,
    load_confusion_matrix,
    MODELS_DIR,
    ASSETS_DIR,
    METRICS_DIR,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RCA_EfficientNet | Bearing Fault Diagnosis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

  :root {
    --bg:        #000000;
    --surface:   rgba(15,23,42,0.75);
    --border:    rgba(56,189,248,0.18);
    --blue:      #38bdf8;
    --green:     #22c55e;
    --purple:    #a78bfa;
    --red:       #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }

  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
  }

  [data-testid="stSidebar"] {
    background: rgba(5,10,25,0.97) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  #MainMenu, footer, header { visibility: hidden; }

  /* ── Glassmorphism card ── */
  .glass-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 4px 32px rgba(56,189,248,0.06);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
    margin-bottom: 1.2rem;
  }
  .glass-card:hover {
    border-color: rgba(56,189,248,0.38);
    box-shadow: 0 8px 48px rgba(56,189,248,0.12);
  }

  /* ── Typography ── */
  .grad-title {
    font-family: var(--font-head) !important;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(135deg, var(--blue) 0%, var(--purple) 55%, var(--green) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.3rem;
  }
  .grad-subtitle {
    font-family: var(--font-head) !important;
    font-size: 1.05rem;
    color: var(--muted);
    letter-spacing: 0.04em;
  }
  .section-title {
    font-family: var(--font-head) !important;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--blue);
    border-left: 3px solid var(--purple);
    padding-left: 0.75rem;
    margin-bottom: 1rem;
  }

  /* ── KPI tiles ── */
  .kpi-row { display: flex; gap: 1.2rem; flex-wrap: wrap; margin: 1.2rem 0; }
  .kpi-tile {
    flex: 1 1 160px;
    background: rgba(56,189,248,0.05);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .kpi-tile:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(56,189,248,0.15);
  }
  .kpi-tile .kpi-val {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--blue), var(--green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .kpi-tile .kpi-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.25rem;
  }

  /* ── Badges ── */
  .badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-blue   { background: rgba(56,189,248,0.15); color: var(--blue);   border: 1px solid rgba(56,189,248,0.35); }
  .badge-green  { background: rgba(34,197,94,0.15);  color: var(--green);  border: 1px solid rgba(34,197,94,0.35); }
  .badge-purple { background: rgba(167,139,250,0.15);color: var(--purple); border: 1px solid rgba(167,139,250,0.35); }

  /* ── Pipeline ── */
  .pipeline { display: flex; align-items: center; gap: 0; overflow-x: auto; padding: 1rem 0; }
  .pipe-node {
    flex-shrink: 0;
    background: rgba(56,189,248,0.07);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    text-align: center;
    min-width: 120px;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .pipe-node:hover { transform: translateY(-3px); box-shadow: 0 6px 24px rgba(56,189,248,0.15); }
  .pipe-node.rca {
    background: rgba(167,139,250,0.12);
    border-color: var(--purple);
    box-shadow: 0 0 18px rgba(167,139,250,0.25);
  }
  .pipe-node .node-icon { font-size: 1.5rem; }
  .pipe-node .node-label { font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); margin-top: 0.3rem; }
  .pipe-arrow { color: var(--blue); font-size: 1.3rem; padding: 0 0.3rem; flex-shrink: 0; }

  /* ── Ablation cards ── */
  .ablation-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
  .ablation-card {
    flex: 1 1 180px;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid var(--border);
    background: var(--surface);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .ablation-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(56,189,248,0.12); }
  .ablation-card .ab-model { font-size: 0.85rem; font-weight: 600; color: var(--blue); margin-bottom: 0.4rem; }
  .ablation-card .ab-val   { font-size: 1.6rem; font-weight: 700; color: var(--text); }
  .ablation-card .ab-f1    { font-size: 0.8rem; color: var(--green); margin-top: 0.1rem; }
  .ablation-card.best      { border-color: var(--purple); box-shadow: 0 0 20px rgba(167,139,250,0.2); }

  /* ── Sidebar nav ── */
  div[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem;
    padding: 0.55rem 1rem;
    border-radius: 8px;
    transition: all 0.2s;
  }
  div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(56,189,248,0.08) !important;
    border-color: var(--border) !important;
    color: var(--blue) !important;
  }

  /* ── Reveal button ── */
  .stButton > button {
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(167,139,250,0.12)) !important;
    border: 1px solid var(--border) !important;
    color: var(--blue) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    transition: all 0.25s ease;
    letter-spacing: 0.06em;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(167,139,250,0.22)) !important;
    border-color: var(--blue) !important;
    box-shadow: 0 0 20px rgba(56,189,248,0.2);
    transform: translateY(-1px);
  }

  /* ── Overrides ── */
  .stSelectbox > div, .stFileUploader > div { background: var(--surface) !important; }
  [data-baseweb="select"] * { background: #0f1a2e !important; color: var(--text) !important; }
  .stProgress > div > div { background: linear-gradient(90deg, var(--blue), var(--purple)) !important; }

  /* ── Reveal animation ── */
  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .reveal-wrap { animation: fadeSlideIn 0.55s cubic-bezier(0.22,1,0.36,1) both; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,18,40,0.7)",
    font=dict(family="JetBrains Mono, monospace", color="#e2e8f0", size=11),
    margin=dict(l=40, r=20, t=50, b=40),
    transition=dict(duration=600, easing="cubic-in-out"),
)

def apply_theme(fig):
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(56,189,248,0.08)", zerolinecolor="rgba(56,189,248,0.1)")
    fig.update_yaxes(gridcolor="rgba(56,189,248,0.08)", zerolinecolor="rgba(56,189,248,0.1)")
    return fig

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
PAGES = [
    ("🏠", "Home"),
    ("🏗️", "Model Architecture"),
    ("📊", "Model Comparison"),
    ("📈", "Evaluation Metrics"),
    ("🔲", "Confusion Matrix"),
    ("🧪", "Ablation Study"),
]

with st.sidebar:
    st.markdown('<div class="grad-title" style="font-size:1.35rem;margin-bottom:0.2rem;">⚙️ RCA_</div>', unsafe_allow_html=True)
    st.markdown('<div class="grad-subtitle" style="font-size:0.72rem;margin-bottom:1.2rem;">Bearing Fault Diagnosis</div>', unsafe_allow_html=True)
    st.divider()

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for icon, label in PAGES:
        active = "🔹 " if st.session_state.page == label else "   "
        if st.button(f"{active}{icon}  {label}", key=f"nav_{label}"):
            st.session_state.page = label

    st.divider()
    st.markdown('<p style="color:#475569;font-size:0.68rem;text-align:center;">v2.0 · RCA_EfficientNet<br>Bearing Surface Imaging</p>', unsafe_allow_html=True)

page = st.session_state.page

# ─── Helper: animated reveal wrapper ─────────────────────────────────────────
def reveal_section(key: str, button_label: str, content_fn):
    """
    Shows a button. On click, sets session_state[key]=True and
    calls content_fn() wrapped in a fade-slide animation div.
    Once revealed, stays revealed until page changes.
    """
    state_key = f"_revealed_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    if not st.session_state[state_key]:
        if st.button(button_label, key=f"btn_{key}"):
            st.session_state[state_key] = True
            st.rerun()
    else:
        with st.container():
            content_fn()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown('<div class="grad-title">RCA_EfficientNet</div>', unsafe_allow_html=True)
    st.markdown('<div class="grad-subtitle">Residual Channel Attention-Based Bearing Fault Diagnosis System</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Row ──
    try:
        df_kpi = load_model_comparison()
    except Exception as e:
        st.error(str(e))
        st.stop()

    rca = df_kpi[df_kpi["Model"] == "RCA_EfficientNet"].iloc[0]
    kpi_cols = st.columns(4)
    kpis = [
        ("Accuracy",  f"{rca['Accuracy']:.2f}%"),
        ("F1 Score",  f"{rca['F1 Score']:.2f}%"),
        ("Precision", f"{rca['Precision']:.2f}%"),
        ("Recall",    f"{rca['Recall']:.2f}%"),
    ]
    for col, (label, value) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-val">{value}</div>
              <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── System Overview ──
    st.markdown("""
    <div class="glass-card">
      <div class="section-title">System Overview</div>
      <p style="line-height:1.9;color:#cbd5e1;font-size:0.92rem;">
        This system performs <strong style="color:#38bdf8">image-based bearing fault diagnosis</strong>
        using deep learning. Instead of relying on vibration signals, it directly analyses
        <em>bearing surface images</em> to detect and classify faults with high accuracy.
      </p>
      <p style="line-height:1.9;color:#cbd5e1;font-size:0.92rem;margin-top:0.8rem;">
        The proposed <strong style="color:#22c55e">RCA_EfficientNet</strong> model integrates
        channel attention with EfficientNet to enhance feature representation and improve
        fault discrimination in complex industrial environments.
      </p>
      <div style="margin-top:1rem;">
        <span class="badge badge-blue">Image-Based</span>&nbsp;
        <span class="badge badge-purple">Deep Learning</span>&nbsp;
        <span class="badge badge-green">Real-Time Inference</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Grid ──
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">Input</div>
          <p style="color:#cbd5e1;font-size:0.9rem;">Bearing surface images (RGB, 224×224) captured via standard imaging devices.</p>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">Model Core</div>
          <p style="color:#cbd5e1;font-size:0.9rem;">EfficientNet backbone enhanced with Residual Channel Attention for feature refinement.</p>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">Output</div>
          <p style="color:#cbd5e1;font-size:0.9rem;">Multi-class fault classification: Normal, Inner, Outer, Ball, Cage.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Features ──
    st.markdown("""
    <div class="glass-card">
      <div class="section-title">Key Features</div>
      <ul style="color:#cbd5e1;font-size:0.9rem;line-height:2;padding-left:1.2rem;">
        <li><strong style="color:#38bdf8">Image-based diagnosis</strong> — no sensor dependency</li>
        <li><strong style="color:#a78bfa">Channel attention</strong> — enhanced feature selection</li>
        <li><strong style="color:#22c55e">EfficientNet backbone</strong> — lightweight and scalable</li>
        <li>Robust multi-class classification</li>
        <li>Optimised for real-time deployment</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-bottom:0.5rem;">📊 Accuracy Overview</div>', unsafe_allow_html=True)

    # ── Animated Accuracy Chart ──
    def render_home_chart():
        colors = ["#475569", "#64748b", "#94a3b8", "#38bdf8", "#a78bfa"]
        fig = go.Figure()
        for i, row in df_kpi.iterrows():
            color = "#a78bfa" if row["Model"] == "RCA_EfficientNet" else colors[i % len(colors)]
            fig.add_bar(
                name=row["Model"],
                x=[row["Model"]],
                y=[row["Accuracy"]],
                marker_color=color,
                width=0.5,
                text=f"{row['Accuracy']:.2f}%",
                textposition="outside",
            )
        fig.update_layout(**PLOT_LAYOUT)
        fig.update_yaxes(range=[80, 102], title="Accuracy (%)",
                         gridcolor="rgba(56,189,248,0.08)")
        fig.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    reveal_section("home_accuracy", "▶ Load Accuracy Chart", render_home_chart)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Architecture":
    st.markdown('<div class="grad-title" style="font-size:2rem;">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="grad-subtitle">RCA_EfficientNet: Residual Channel Attention augmented EfficientNet-B0</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    # Pipeline diagram — static (structural, not a chart)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Forward Pass Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
      <div class="pipe-node"><div class="node-icon">🖼️</div><div style="font-size:0.85rem;color:#38bdf8;font-weight:600;">Input</div><div class="node-label">224×224 RGB</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="node-icon">🔧</div><div style="font-size:0.85rem;color:#38bdf8;font-weight:600;">Preprocessing</div><div class="node-label">Resize + Normalize</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="node-icon">🧠</div><div style="font-size:0.85rem;color:#38bdf8;font-weight:600;">EfficientNet-B0</div><div class="node-label">Feature Extractor</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node rca"><div class="node-icon">⚡</div><div style="font-size:0.85rem;color:#a78bfa;font-weight:600;">RCA Block</div><div class="node-label" style="color:#a78bfa;">Attention Module</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="node-icon">📊</div><div style="font-size:0.85rem;color:#38bdf8;font-weight:600;">GAP + Dropout</div><div class="node-label">Pooling</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="node-icon">🎯</div><div style="font-size:0.85rem;color:#22c55e;font-weight:600;">Classifier</div><div class="node-label">5 Classes</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="node-icon">✅</div><div style="font-size:0.85rem;color:#22c55e;font-weight:600;">Softmax</div><div class="node-label">Fault Probabilities</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">EfficientNet-B0 Backbone</div>
          <p style="color:#cbd5e1;font-size:0.84rem;line-height:1.8;">
            EfficientNet-B0 uses <strong style="color:#38bdf8">compound coefficient scaling</strong>
            to jointly scale depth, width, and resolution. Its MBConv blocks with depthwise-separable
            convolutions provide an excellent feature representation of bearing surface texture and
            geometric defects with minimal parameters (~5.3M).
          </p>
          <ul style="color:#94a3b8;font-size:0.82rem;line-height:2;padding-left:1.2rem;">
            <li>7 MBConv blocks with squeeze-excitation</li>
            <li>Output feature map: 1280 channels</li>
            <li>Trained with ImageNet initialisation</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">Preprocessing Stage</div>
          <p style="color:#cbd5e1;font-size:0.84rem;line-height:1.8;">
            All images are resized to <strong style="color:#38bdf8">224×224</strong> pixels and
            normalised with ImageNet statistics
            <code style="color:#a78bfa">μ=[0.485,0.456,0.406]</code>
            <code style="color:#a78bfa">σ=[0.229,0.224,0.225]</code>.
            No data augmentation is applied at inference time.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(167,139,250,0.35);box-shadow:0 0 24px rgba(167,139,250,0.12);">
          <div class="section-title" style="color:#a78bfa;border-color:#22c55e;">⚡ RCA Module</div>
          <p style="color:#cbd5e1;font-size:0.84rem;line-height:1.8;">
            The <strong style="color:#a78bfa">RCA block</strong> applies channel-wise attention on top of
            EfficientNet features, using both average and max pooling to capture holistic and salient
            channel statistics. A residual connection preserves gradient flow.
          </p>
          <div style="background:rgba(167,139,250,0.07);border-radius:8px;padding:0.9rem;margin-top:0.8rem;font-family:monospace;font-size:0.78rem;color:#e2e8f0;">
            <span style="color:#a78bfa">x</span> → Conv3×3 → BN → ReLU → Conv3×3 → BN<br>
            &nbsp;&nbsp;&nbsp;↓<br>
            AvgPool + MaxPool → FC → FC → Sigmoid<br>
            &nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            &nbsp;&nbsp;&nbsp;⊗ channel attention ← <span style="color:#a78bfa">scale</span><br>
            &nbsp;&nbsp;&nbsp;↓<br>
            <span style="color:#22c55e">out</span> = ReLU(<span style="color:#a78bfa">x</span> + attended)
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">Classifier Head</div>
          <p style="color:#cbd5e1;font-size:0.84rem;line-height:1.8;">
            Global Average Pooling → Dropout(0.3) → Linear(1280 → 5).
            The 5-way softmax output corresponds to:
            <strong style="color:#38bdf8">Normal, Inner Race, Outer Race, Ball, Cage</strong> faults.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Animated parameter count chart ──
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Architecture Parameter Breakdown</div>', unsafe_allow_html=True)

    def render_param_chart():
        components = ["EfficientNet-B0\nBackbone", "RCA\nAttention", "GAP +\nDropout", "Classifier\nHead"]
        params     = [5.27, 0.39, 0.0, 0.006]
        colors_p   = ["#38bdf8", "#a78bfa", "#22c55e", "#f87171"]
        fig = go.Figure(go.Bar(
            x=components, y=params,
            marker_color=colors_p,
            text=[f"{v:.3f}M" if v > 0 else "<0.001M" for v in params],
            textposition="outside",
        ))
        fig.update_layout(**PLOT_LAYOUT)
        fig.update_yaxes(title="Parameters (Millions)", gridcolor="rgba(56,189,248,0.08)")
        fig.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
        fig.update_layout(showlegend=False, title="Parameter Distribution Across Modules")
        st.plotly_chart(fig, use_container_width=True)

    reveal_section("arch_params", "▶ Load Parameter Chart", render_param_chart)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown('<div class="grad-title" style="font-size:2rem;">📊 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="grad-subtitle">Benchmarking RCA_EfficientNet against baseline architectures.</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    try:
        df = load_model_comparison()
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()

    # ── Grouped Bar Chart ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Accuracy & F1 Score Comparison</div>', unsafe_allow_html=True)

    def render_grouped_bar():
        bar_clr_acc = ["#475569" if m != "RCA_EfficientNet" else "#38bdf8" for m in df["Model"]]
        bar_clr_f1  = ["#334155" if m != "RCA_EfficientNet" else "#a78bfa" for m in df["Model"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy", x=df["Model"], y=df["Accuracy"],
                             marker_color=bar_clr_acc,
                             text=df["Accuracy"].apply(lambda v: f"{v:.2f}%"),
                             textposition="outside"))
        fig.add_trace(go.Bar(name="F1 Score", x=df["Model"], y=df["F1 Score"],
                             marker_color=bar_clr_f1,
                             text=df["F1 Score"].apply(lambda v: f"{v:.2f}%"),
                             textposition="outside"))
        fig.update_layout(**PLOT_LAYOUT)
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08))
        fig.update_yaxes(range=[80, 103], title="Score (%)", gridcolor="rgba(56,189,248,0.08)")
        fig.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    reveal_section("comp_grouped", "▶ Load Accuracy & F1 Chart", render_grouped_bar)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Radar Chart ──
    if {"Precision", "Recall"}.issubset(df.columns):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Multi-Metric Radar</div>', unsafe_allow_html=True)

        def render_radar():
            metrics  = ["Accuracy", "F1 Score", "Precision", "Recall"]
            colors_r = ["#475569", "#64748b", "#94a3b8", "#38bdf8", "#a78bfa"]
            fig2 = go.Figure()
            for i, row in df.iterrows():
                vals = [row[m] for m in metrics] + [row[metrics[0]]]
                fig2.add_trace(go.Scatterpolar(
                    r=vals, theta=metrics + [metrics[0]],
                    fill="toself", name=row["Model"],
                    line_color=colors_r[i % len(colors_r)],
                    opacity=0.75 if row["Model"] == "RCA_EfficientNet" else 0.35,
                ))
            fig2.update_layout(**PLOT_LAYOUT)
            fig2.update_layout(
                polar=dict(
                    bgcolor="rgba(10,18,40,0.7)",
                    radialaxis=dict(range=[80, 100], gridcolor="rgba(56,189,248,0.15)", color="#64748b"),
                    angularaxis=dict(gridcolor="rgba(56,189,248,0.15)", color="#94a3b8"),
                )
            )
            st.plotly_chart(fig2, use_container_width=True)

        reveal_section("comp_radar", "▶ Load Radar Chart", render_radar)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Precision / Recall breakdown bar ──
    if {"Precision", "Recall"}.issubset(df.columns):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Precision vs Recall Breakdown</div>', unsafe_allow_html=True)

        def render_prec_recall():
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name="Precision", x=df["Model"], y=df["Precision"],
                                  marker_color="#22c55e",
                                  text=df["Precision"].apply(lambda v: f"{v:.2f}%"),
                                  textposition="outside"))
            fig3.add_trace(go.Bar(name="Recall", x=df["Model"], y=df["Recall"],
                                  marker_color="#f87171",
                                  text=df["Recall"].apply(lambda v: f"{v:.2f}%"),
                                  textposition="outside"))
            fig3.update_layout(**PLOT_LAYOUT)
            fig3.update_layout(barmode="group", legend=dict(orientation="h", y=1.08))
            fig3.update_yaxes(range=[80, 103], title="Score (%)", gridcolor="rgba(56,189,248,0.08)")
            fig3.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
            st.plotly_chart(fig3, use_container_width=True)

        reveal_section("comp_pr_bar", "▶ Load Precision vs Recall Chart", render_prec_recall)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Score Table ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Full Score Table</div>', unsafe_allow_html=True)
    numeric_cols = [c for c in df.columns if c != "Model"]
    st.dataframe(
        df.style
          .highlight_max(subset=["Accuracy", "F1 Score"], color="#1a2e1a")
          .format({c: "{:.2f}%" for c in numeric_cols}),
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Evaluation Metrics":
    st.markdown('<div class="grad-title" style="font-size:2rem;">📈 Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="grad-subtitle">ROC and Precision-Recall curves loaded from /metrics.</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    model_sel = st.selectbox("Select Model", ["CNN", "ResNet50", "EfficientNet", "DeiT", "RCA_EfficientNet"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)

        try:
            roc_df = load_roc_data(model_sel)
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        auc = np.trapz(roc_df["TPR"], roc_df["FPR"])

        def render_roc():
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_df["FPR"], y=roc_df["TPR"],
                mode="lines", name="ROC",
                line=dict(color="#38bdf8", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.06)",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#475569", dash="dash"), name="Random",
            ))
            fig_roc.update_layout(
                title=f"AUC = {auc:.4f}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                **PLOT_LAYOUT,
            )
            fig_roc.update_xaxes(range=[0, 1], gridcolor="rgba(56,189,248,0.08)")
            fig_roc.update_yaxes(range=[0, 1], gridcolor="rgba(56,189,248,0.08)")
            st.plotly_chart(fig_roc, use_container_width=True)

        reveal_section(f"roc_{model_sel}", "▶ Load ROC Curve", render_roc)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Precision-Recall Curve</div>', unsafe_allow_html=True)

        try:
            pr_df = load_pr_data(model_sel)
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        ap = abs(np.trapz(pr_df["Precision"], pr_df["Recall"]))

        def render_pr():
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=pr_df["Recall"], y=pr_df["Precision"],
                mode="lines", name="PR",
                line=dict(color="#a78bfa", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.06)",
            ))
            fig_pr.update_layout(
                title=f"AP = {ap:.4f}",
                xaxis_title="Recall",
                yaxis_title="Precision",
                **PLOT_LAYOUT,
            )
            fig_pr.update_xaxes(range=[0, 1], gridcolor="rgba(56,189,248,0.08)")
            fig_pr.update_yaxes(range=[0, 1], gridcolor="rgba(56,189,248,0.08)")
            st.plotly_chart(fig_pr, use_container_width=True)

        reveal_section(f"pr_{model_sel}", "▶ Load PR Curve", render_pr)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── AUC summary across all models ──
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AUC Summary — All Models</div>', unsafe_allow_html=True)

    def render_auc_summary():
        model_list = ["CNN", "ResNet50", "EfficientNet", "DeiT", "RCA_EfficientNet"]
        auc_vals, ap_vals, labels_ok = [], [], []
        for m in model_list:
            try:
                r = load_roc_data(m)
                p = load_pr_data(m)
                auc_vals.append(np.trapz(r["TPR"], r["FPR"]))
                ap_vals.append(abs(np.trapz(p["Precision"], p["Recall"])))
                labels_ok.append(m)
            except Exception:
                pass
        if not labels_ok:
            st.info("No ROC/PR data found for any model.")
            return
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Bar(name="AUC", x=labels_ok, y=auc_vals,
                                 marker_color=["#a78bfa" if m == "RCA_EfficientNet" else "#38bdf8" for m in labels_ok],
                                 text=[f"{v:.4f}" for v in auc_vals], textposition="outside"))
        fig_auc.add_trace(go.Bar(name="AP", x=labels_ok, y=ap_vals,
                                 marker_color=["#22c55e" if m == "RCA_EfficientNet" else "#64748b" for m in labels_ok],
                                 text=[f"{v:.4f}" for v in ap_vals], textposition="outside"))
        fig_auc.update_layout(**PLOT_LAYOUT)
        fig_auc.update_layout(barmode="group", legend=dict(orientation="h", y=1.08))
        fig_auc.update_yaxes(range=[0, 1.1], title="Score", gridcolor="rgba(56,189,248,0.08)")
        fig_auc.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
        st.plotly_chart(fig_auc, use_container_width=True)

    reveal_section("auc_summary", "▶ Load AUC Summary Chart", render_auc_summary)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Confusion Matrix":
    st.markdown('<div class="grad-title" style="font-size:2rem;">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="grad-subtitle">Per-class prediction breakdown.</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    model_sel = st.selectbox(
        "Select Model",
        ["CNN", "ResNet50", "EfficientNet", "DeiT", "RCA_EfficientNet"],
        index=4,
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    try:
        cm_df = load_confusion_matrix(model_sel)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    labels    = list(cm_df.columns)
    row_sums  = cm_df.values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    z_norm    = (cm_df.values / row_sums) * 100
    text_grid = [
        [f"{cm_df.values[r][c]}<br>({z_norm[r][c]:.1f}%)" for c in range(len(labels))]
        for r in range(len(labels))
    ]

    def render_cm():
        fig_cm = go.Figure(go.Heatmap(
            z=z_norm, x=labels, y=labels,
            text=text_grid, texttemplate="%{text}",
            colorscale=[[0, "#0f172a"], [0.5, "#1e3a5f"], [1, "#38bdf8"]],
            showscale=True,
            colorbar=dict(title="Recall %", tickfont=dict(color="#e2e8f0")),
        ))
        fig_cm.update_layout(**PLOT_LAYOUT)
        fig_cm.update_xaxes(title="Predicted Label", tickangle=-30)
        fig_cm.update_yaxes(title="True Label", autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)

    reveal_section(f"cm_{model_sel}", "▶ Load Confusion Matrix", render_cm)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-class accuracy bar ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Per-Class Recall</div>', unsafe_allow_html=True)

    def render_per_class():
        per_class_recall = [z_norm[i][i] for i in range(len(labels))]
        colors_pc = ["#22c55e" if v >= 95 else "#38bdf8" if v >= 85 else "#f87171" for v in per_class_recall]
        fig_pc = go.Figure(go.Bar(
            x=labels, y=per_class_recall,
            marker_color=colors_pc,
            text=[f"{v:.1f}%" for v in per_class_recall],
            textposition="outside",
        ))
        fig_pc.update_layout(**PLOT_LAYOUT, showlegend=False,
                             title="Per-Class Diagonal Recall (%)")
        fig_pc.update_yaxes(range=[0, 110], title="Recall (%)", gridcolor="rgba(56,189,248,0.08)")
        fig_pc.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
        st.plotly_chart(fig_pc, use_container_width=True)

    reveal_section(f"recall_{model_sel}", "▶ Load Per-Class Recall", render_per_class)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Ablation Study":

    # ── Title ──
    st.markdown(
        '<div class="grad-title" style="font-size:2rem;">🧪 Ablation Study</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="grad-subtitle">Incremental contribution of each architectural component.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Load Data ──
    ABLATION_CSV = os.path.join("metrics", "ablation.csv")

    try:
        ab_df = pd.read_csv(ABLATION_CSV)
        required_cols = {"Model", "Accuracy", "F1", "Description"}
        if not required_cols.issubset(ab_df.columns):
            raise ValueError(f"Missing columns: {required_cols - set(ab_df.columns)}")
    except Exception as e:
        st.error(f"Ablation data error: {e}")
        st.stop()

    ablation_data = ab_df.to_dict("records")
    base_acc = ablation_data[0]["Accuracy"]
    best_idx = len(ablation_data) - 1

    # ════════════════════════════════════════
    # 🔷 Ablation Cards — one st.markdown per card to avoid HTML parse bugs
    # ════════════════════════════════════════
    st.markdown('<div class="section-title">Component Summary</div>', unsafe_allow_html=True)

    cols = st.columns(len(ablation_data))

    for i, (col, row) in enumerate(zip(cols, ablation_data)):
        is_best  = (i == best_idx)
        delta    = "baseline" if i == 0 else f"+{row['Accuracy'] - base_acc:.2f}%"
        delta_color = "#22c55e" if i > 0 else "#64748b"

        border   = "rgba(167,139,250,0.55)" if is_best else "rgba(56,189,248,0.18)"
        glow     = "0 0 22px rgba(167,139,250,0.22)" if is_best else "none"
        bg       = "rgba(167,139,250,0.07)" if is_best else "rgba(15,23,42,0.75)"
        badge    = '<span class="badge badge-purple" style="margin-top:0.5rem;display:inline-block;">★ Best</span>' if is_best else ""

        with col:
            st.markdown(
                f"""
                <div style="
                    background:{bg};
                    border:1px solid {border};
                    border-radius:14px;
                    padding:1.2rem 1rem;
                    text-align:center;
                    box-shadow:{glow};
                    transition:transform 0.2s,box-shadow 0.2s;
                    height:100%;
                ">
                    <div style="font-size:0.8rem;font-weight:600;color:#38bdf8;
                                letter-spacing:0.04em;margin-bottom:0.5rem;">
                        {row['Model']}
                    </div>
                    <div style="font-size:1.9rem;font-weight:700;color:#e2e8f0;
                                line-height:1.1;">
                        {row['Accuracy']:.2f}%
                    </div>
                    <div style="font-size:0.78rem;color:#22c55e;margin-top:0.3rem;">
                        F1: {row['F1']:.2f}%
                    </div>
                    <div style="font-size:0.75rem;color:{delta_color};margin-top:0.2rem;">
                        Δ {delta}
                    </div>
                    {badge}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # 📋 Component Description Table
    # ════════════════════════════════════════
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Component Breakdown</div>', unsafe_allow_html=True)

    ab_display = ab_df[["Model", "Accuracy", "F1", "Description"]].copy()
    ab_display.insert(
        3, "Δ Accuracy",
        ab_df["Accuracy"].diff().fillna(0).apply(
            lambda x: "—" if x == 0 else f"+{x:.2f}%"
        ),
    )
    ab_display = ab_display.rename(columns={"F1": "F1 Score"})

    st.dataframe(
        ab_display.style.highlight_max(
            subset=["Accuracy", "F1 Score"], color="#1a2e1a"
        ).format({"Accuracy": "{:.2f}%", "F1 Score": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # 📈 Performance Progression
    # ════════════════════════════════════════
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Performance Progression</div>', unsafe_allow_html=True)

    models = [d["Model"] for d in ablation_data]
    accs   = [d["Accuracy"] for d in ablation_data]
    f1s    = [d["F1"] for d in ablation_data]

    def render_progression():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=models, y=accs,
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(
                size=[14 if m == ablation_data[best_idx]["Model"] else 9 for m in models],
                color=["#a78bfa" if m == ablation_data[best_idx]["Model"] else "#38bdf8" for m in models],
                line=dict(color="#e2e8f0", width=1.5),
            ),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.05)",
        ))
        fig.add_trace(go.Scatter(
            x=models, y=f1s,
            mode="lines+markers",
            name="F1 Score",
            line=dict(color="#a78bfa", width=2.5, dash="dot"),
            marker=dict(size=9),
        ))
        fig.update_layout(**PLOT_LAYOUT)
        fig.update_yaxes(title="Score (%)", range=[85, 100], gridcolor="rgba(56,189,248,0.08)")
        fig.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
        fig.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    reveal_section("ablation_progress", "▶ Load Performance Progression", render_progression)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # 📊 Delta Gain Chart
    # ════════════════════════════════════════
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Accuracy Gain per Component</div>', unsafe_allow_html=True)

    def render_delta():
        deltas = [0.0] + [
            ablation_data[i]["Accuracy"] - ablation_data[i - 1]["Accuracy"]
            for i in range(1, len(ablation_data))
        ]
        fig = go.Figure(go.Bar(
            x=models,
            y=deltas,
            text=["baseline" if d == 0 else f"+{d:.2f}%" for d in deltas],
            textposition="outside",
            marker_color=[
                "#a78bfa" if m == ablation_data[best_idx]["Model"]
                else "#22c55e" if d > 0
                else "#f87171"
                for m, d in zip(models, deltas)
            ],
            width=0.5,
        ))
        fig.update_layout(**PLOT_LAYOUT, showlegend=False)
        fig.update_yaxes(title="Δ Accuracy (%)", gridcolor="rgba(56,189,248,0.08)")
        fig.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    reveal_section("ablation_delta", "▶ Load Delta Gain", render_delta)
    st.markdown("</div>", unsafe_allow_html=True)