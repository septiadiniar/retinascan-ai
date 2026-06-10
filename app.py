"""
app.py — Streamlit App: Klasifikasi Penyakit Retina OCT
Jalankan: streamlit run app.py
"""

import os
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch

# Import dari models.py
from models import (
    DEVICE, CLASS_NAMES, NUM_CLASSES, CLASS_INFO, SEVERITY,
    MODEL_BUILDERS, MODEL_FILES, MODEL_DESC,
    load_model, predict,
    AttentionRollout, rollout_to_heatmap,
    GradCAM, cam_to_heatmap, overlay_heatmap,
    IMG_SIZE,
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg-primary   : #0a0f1e;
    --bg-card      : #111827;
    --bg-card2     : #1a2235;
    --accent-cyan  : #06b6d4;
    --accent-teal  : #14b8a6;
    --accent-red   : #ef4444;
    --accent-green : #22c55e;
    --accent-yellow: #eab308;
    --text-primary : #f1f5f9;
    --text-muted   : #94a3b8;
    --border       : rgba(6,182,212,0.2);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #0c1a2e 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #06b6d4, #14b8a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
}
.badge-device {
    display: inline-block;
    background: rgba(6,182,212,0.15);
    border: 1px solid rgba(6,182,212,0.4);
    color: var(--accent-cyan);
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 500;
    margin-top: 0.6rem;
    letter-spacing: 0.05em;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 0.8rem;
}

/* Diagnosis result */
.diagnosis-card {
    background: linear-gradient(135deg, #111827, #0c1a2e);
    border: 2px solid var(--accent-cyan);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 30px rgba(6,182,212,0.1);
}
.diagnosis-class {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent-cyan);
    margin: 0;
    line-height: 1;
}
.diagnosis-full {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 0.3rem 0 0 0;
}
.confidence-big {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
}
.severity-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
    letter-spacing: 0.03em;
}

/* Progress bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.55rem;
}
.prob-label {
    font-size: 0.78rem;
    font-weight: 500;
    width: 64px;
    flex-shrink: 0;
    color: var(--text-primary);
}
.prob-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}
.prob-val {
    font-size: 0.75rem;
    color: var(--text-muted);
    width: 44px;
    text-align: right;
    flex-shrink: 0;
}

/* Info box */
.info-box {
    background: rgba(6,182,212,0.06);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin-top: 0.8rem;
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1626 !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #06b6d4, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 20px rgba(6,182,212,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(6,182,212,0.45) !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(6,182,212,0.15) !important;
    color: var(--accent-cyan) !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: rgba(6,182,212,0.03) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 1.2rem 0 !important;
}

/* Warning / info */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* Hide default streamlit elements */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(model_name, ckpt_path=None):
    return load_model(model_name, ckpt_path)


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#111827', dpi=120)
    buf.seek(0)
    return Image.open(buf)


def make_prob_bars(all_probs, pred_class):
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    html = '<div style="margin-top:0.5rem">'
    for cls, prob in sorted_probs:
        is_top = cls == pred_class
        pct    = prob * 100
        bar_color = '#06b6d4' if is_top else '#334155'
        label_color = '#f1f5f9' if is_top else '#94a3b8'
        weight = '600' if is_top else '400'
        emoji  = CLASS_INFO[cls][1]
        html += f"""
        <div class="prob-row">
            <span class="prob-label" style="color:{label_color};font-weight:{weight}">
                {emoji} {cls}
            </span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{pct:.1f}%;background:{bar_color}"></div>
            </div>
            <span class="prob-val">{pct:.1f}%</span>
        </div>"""
    html += '</div>'
    return html


def make_attention_fig(orig_img, attn_mask):
    """Buat figure 3-panel untuk Attention Rollout."""
    img_np   = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heatmap  = rollout_to_heatmap(attn_mask, IMG_SIZE)
    overlaid = overlay_heatmap(img_np, heatmap, alpha=0.55)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#111827')
    titles = ['Citra Asli', 'Attention Map', 'Overlay']
    imgs   = [img_np, attn_mask, overlaid]
    cmaps  = [None, 'hot', None]

    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color='#94a3b8', fontsize=10, pad=6)
        ax.axis('off')
        ax.set_facecolor('#111827')

    plt.tight_layout(pad=0.5)
    return fig


def make_gradcam_fig(orig_img, cam):
    """Buat figure 3-panel untuk Grad-CAM."""
    img_np   = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heatmap  = cam_to_heatmap(cam, IMG_SIZE)
    overlaid = overlay_heatmap(img_np, heatmap, alpha=0.55)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#111827')
    titles = ['Citra Asli', 'Grad-CAM', 'Overlay']
    imgs   = [img_np, cam, overlaid]
    cmaps  = [None, 'jet', None]

    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color='#94a3b8', fontsize=10, pad=6)
        ax.axis('off')
        ax.set_facecolor('#111827')

    plt.tight_layout(pad=0.5)
    return fig


def make_confidence_gauge(confidence, pred_class):
    """Gauge chart untuk confidence."""
    fig, ax = plt.subplots(figsize=(4, 2.2),
                            subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    theta = np.linspace(0, np.pi, 200)
    ax.fill_between(theta, 0.6, 1.0, color='#1e293b', zorder=1)

    # Warna berdasarkan confidence
    if confidence >= 0.85:
        color = '#22c55e'
    elif confidence >= 0.65:
        color = '#eab308'
    else:
        color = '#ef4444'

    fill_theta = np.linspace(0, np.pi * confidence, 200)
    ax.fill_between(fill_theta, 0.6, 1.0, color=color, alpha=0.85, zorder=2)

    # Needle
    needle_angle = np.pi * confidence
    ax.plot([needle_angle, needle_angle], [0, 0.95],
            color='white', linewidth=2.5, zorder=3)
    ax.plot(needle_angle, 0.95, 'o', color='white', markersize=5, zorder=4)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'],
                        color='#64748b', fontsize=7)
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    ax.text(np.pi/2, 0.25, f'{confidence*100:.1f}%',
            ha='center', va='center', fontsize=18,
            fontweight='bold', color=color,
            transform=ax.transData)

    plt.tight_layout(pad=0)
    return fig


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem">
        <div style="font-size:2.5rem">🔬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                    font-weight:800;color:#06b6d4">RetinaScan AI</div>
        <div style="font-size:0.72rem;color:#64748b;margin-top:2px">
            Powered by DINOv2 + Attention Rollout
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Konfigurasi Model")

    selected_model = st.selectbox(
        "Pilih Model",
        list(MODEL_BUILDERS.keys()),
        index=0,
        help="DINOv2 adalah model utama dengan performa terbaik"
    )

    st.markdown(f"""
    <div style="background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.2);
                border-radius:8px;padding:0.7rem 0.9rem;margin:0.5rem 0 1rem 0;
                font-size:0.75rem;color:#94a3b8;line-height:1.6">
        {MODEL_DESC[selected_model]}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Checkpoint (Opsional)")
    ckpt_file = st.file_uploader(
        "Upload file .pth",
        type=['pth', 'pt'],
        help="Jika tidak diupload, model akan menggunakan pretrained backbone (demo mode)"
    )

    ckpt_path = None
    if ckpt_file:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            f.write(ckpt_file.read())
            ckpt_path = f.name
        st.success(f"✅ Checkpoint loaded: {ckpt_file.name}")

    st.markdown("---")
    st.markdown("### 🔧 Pengaturan Visualisasi")

    show_rollout = st.toggle("Tampilkan Attention Rollout", value=True)
    show_gradcam = st.toggle("Tampilkan Grad-CAM", value=True)
    discard_ratio = st.slider(
        "Attention Discard Ratio",
        min_value=0.5, max_value=0.99, value=0.9, step=0.05,
        help="Makin tinggi = makin fokus ke area utama"
    )
    overlay_alpha = st.slider(
        "Overlay Opacity", min_value=0.3, max_value=0.9, value=0.55, step=0.05
    )

    st.markdown("---")

    # Device info
    device_str = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    gpu_name   = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—"
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#64748b;line-height:1.8">
        <div>🖥️ <b style="color:#94a3b8">Device:</b> {device_str}</div>
        <div>🎮 <b style="color:#94a3b8">GPU:</b> {gpu_name}</div>
        <div>🏷️ <b style="color:#94a3b8">Kelas:</b> {NUM_CLASSES} penyakit</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.68rem;color:#475569;line-height:1.6">
        ⚠️ <b>Disclaimer:</b> Aplikasi ini hanya untuk tujuan edukasi dan penelitian.
        Bukan pengganti diagnosis medis profesional.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
    <p class="hero-title">🔬 RetinaScan AI</p>
    <p class="hero-sub">
        Klasifikasi Penyakit Retina Berbasis Citra OCT
        menggunakan Fine-Tuned DINOv2 Vision Transformer
    </p>
    <span class="badge-device">{'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Mode'}</span>
    &nbsp;
    <span class="badge-device">Model: {selected_model}</span>
    &nbsp;
    <span class="badge-device">{NUM_CLASSES} Kelas Penyakit</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Upload & Classify
# ──────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown('<div class="card-title">📤 Upload Citra OCT</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop atau klik untuk upload",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        label_visibility='collapsed'
    )

    if uploaded:
        pil_image = Image.open(uploaded).convert('RGB')
        st.image(pil_image, caption="Citra yang diupload", use_column_width=True)

        w, h = pil_image.size
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#64748b;margin-top:0.5rem">
            📐 Ukuran: {w}×{h}px &nbsp;|&nbsp;
            📁 Format: {uploaded.type.split('/')[-1].upper()} &nbsp;|&nbsp;
            💾 Size: {uploaded.size/1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍 Analisis Sekarang", use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#475569">
            <div style="font-size:3rem;margin-bottom:0.8rem">🫁</div>
            <div style="font-size:0.85rem;line-height:1.8">
                Upload citra OCT retina<br>
                (JPG, PNG, BMP, TIFF)
            </div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

with col_result:
    if uploaded and run_btn:
        with st.spinner(f"🧠 Memuat {selected_model}..."):
            model, loaded_from_ckpt = get_model(selected_model, ckpt_path)

        with st.spinner("🔍 Menganalisis citra..."):
            pred_class, confidence, all_probs, top3 = predict(model, pil_image)

        st.session_state.last_result = {
            'pred_class': pred_class,
            'confidence': confidence,
            'all_probs' : all_probs,
            'top3'      : top3,
            'pil_image' : pil_image,
            'model'     : model,
            'model_name': selected_model,
        }

        # ── Diagnosis Card ──────────────────────
        full_name   = CLASS_INFO[pred_class][0]
        emoji       = CLASS_INFO[pred_class][1]
        description = CLASS_INFO[pred_class][2]
        sev_label, sev_color = SEVERITY[pred_class]

        conf_color = '#22c55e' if confidence >= 0.85 else \
                     '#eab308' if confidence >= 0.65 else '#ef4444'

        st.markdown(f"""
        <div class="diagnosis-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div style="font-size:0.72rem;color:#64748b;
                                letter-spacing:0.1em;text-transform:uppercase;
                                margin-bottom:0.4rem">HASIL DIAGNOSIS</div>
                    <p class="diagnosis-class">{emoji} {pred_class}</p>
                    <p class="diagnosis-full">{full_name}</p>
                    <span class="severity-badge"
                          style="background:{sev_color}22;color:{sev_color};
                                 border:1px solid {sev_color}66">
                        ● {sev_label}
                    </span>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.72rem;color:#64748b;
                                letter-spacing:0.1em;text-transform:uppercase;
                                margin-bottom:0.4rem">CONFIDENCE</div>
                    <div class="confidence-big"
                         style="color:{conf_color}">{confidence*100:.1f}%</div>
                    <div style="font-size:0.72rem;color:#64748b;margin-top:0.2rem">
                        {'Model: checkpoint ✅' if loaded_from_ckpt else 'Demo mode ⚠️'}
                    </div>
                </div>
            </div>
            <div class="info-box">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top 3 Prediksi ──────────────────────
        st.markdown('<div class="card-title">🏆 Top 3 Prediksi</div>', unsafe_allow_html=True)
        for i, (cls, prob) in enumerate(top3):
            medal = ['🥇', '🥈', '🥉'][i]
            full  = CLASS_INFO[cls][0]
            color = '#06b6d4' if i == 0 else '#94a3b8'
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        margin-bottom:0.5rem;padding:0.6rem 0.8rem;
                        background:#1a2235;border-radius:8px;
                        border:1px solid {'rgba(6,182,212,0.3)' if i==0 else 'rgba(255,255,255,0.05)'}">
                <span style="font-size:1.2rem">{medal}</span>
                <div style="flex:1">
                    <div style="font-size:0.85rem;font-weight:600;
                                color:{color}">{CLASS_INFO[cls][1]} {cls}</div>
                    <div style="font-size:0.7rem;color:#64748b">{full}</div>
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:1rem;
                            font-weight:700;color:{color}">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="height:350px;display:flex;align-items:center;
                    justify-content:center;text-align:center">
            <div>
                <div style="font-size:4rem;margin-bottom:1rem;opacity:0.3">📊</div>
                <div style="color:#475569;font-size:0.9rem">
                    Hasil analisis akan muncul di sini<br>setelah citra diupload
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Distribusi Probabilitas (full width)
# ──────────────────────────────────────────────
if st.session_state.last_result and uploaded:
    res = st.session_state.last_result

    st.markdown("---")
    st.markdown('<div class="card-title">📊 Distribusi Probabilitas Semua Kelas</div>',
                unsafe_allow_html=True)

    prob_html = make_prob_bars(res['all_probs'], res['pred_class'])
    st.markdown(prob_html, unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # Tabs: Visualisasi
    # ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-title">🧠 Explainability & Visualisasi</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "👁️ Attention Rollout",
        "🔥 Grad-CAM",
        "📈 Confidence Gauge",
        "ℹ️ Info Penyakit",
    ])

    # ── Tab 1: Attention Rollout ────────────────
    with tab1:
        is_vit = 'ViT' in res['model_name'] or 'DINOv2' in res['model_name']

        if not show_rollout:
            st.info("Attention Rollout dinonaktifkan di sidebar.")
        elif not is_vit:
            st.warning(f"⚠️ Attention Rollout hanya tersedia untuk model berbasis ViT. "
                       f"Model **{res['model_name']}** adalah CNN — gunakan tab Grad-CAM.")
        else:
            with st.spinner("🔄 Menghitung Attention Rollout..."):
                try:
                    rollout = AttentionRollout(res['model'], discard_ratio=discard_ratio)
                    attn_mask = rollout(res['pil_image'])
                    rollout.remove_hooks()
                except Exception as e:
                    attn_mask = None
                    st.error(f"Error saat Attention Rollout: {e}")

            if attn_mask is not None:
                fig = make_attention_fig(res['pil_image'], attn_mask)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown("""
                <div class="info-box">
                    <b>Attention Rollout</b> (Abnar & Zuidema, 2020) menelusuri aliran
                    informasi melalui semua layer attention ViT secara rekursif.
                    Area terang menunjukkan region yang paling diperhatikan model
                    saat membuat prediksi.
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 2: Grad-CAM ─────────────────────────
    with tab2:
        if not show_gradcam:
            st.info("Grad-CAM dinonaktifkan di sidebar.")
        else:
            with st.spinner("🔄 Menghitung Grad-CAM..."):
                try:
                    grad_cam = GradCAM(res['model'], res['model_name'])
                    pred_idx = list(res['all_probs'].values()).index(
                        max(res['all_probs'].values())
                    )
                    cam = grad_cam(res['pil_image'], class_idx=pred_idx)
                except Exception as e:
                    cam = None
                    st.error(f"Error saat Grad-CAM: {e}")

            if cam is not None:
                fig = make_gradcam_fig(res['pil_image'], cam)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown("""
                <div class="info-box">
                    <b>Grad-CAM</b> (Selvaraju et al., 2017) menggunakan gradient
                    dari kelas target terhadap feature map terakhir untuk menghasilkan
                    localization map. Area merah/kuning = area paling diskriminatif
                    untuk prediksi kelas tersebut.
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 3: Confidence Gauge ─────────────────
    with tab3:
        col_g1, col_g2, col_g3 = st.columns([1, 1.5, 1])
        with col_g2:
            gauge_fig = make_confidence_gauge(res['confidence'], res['pred_class'])
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close(gauge_fig)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        conf = res['confidence']
        top3 = res['top3']

        m1.metric("🎯 Confidence",   f"{conf*100:.2f}%")
        m2.metric("🔢 Prediksi",     res['pred_class'])
        m3.metric("📊 #2 Kandidat",  f"{top3[1][0]} ({top3[1][1]*100:.1f}%)")
        m4.metric("📉 Selisih Top-2",f"{(top3[0][1]-top3[1][1])*100:.1f}%")

        # Bar chart semua kelas
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')

        cls_list = CLASS_NAMES
        vals     = [res['all_probs'][c] * 100 for c in cls_list]
        colors   = ['#06b6d4' if c == res['pred_class'] else '#1e3a4a'
                    for c in cls_list]
        bars = ax.bar(cls_list, vals, color=colors,
              edgecolor='#06b6d4', linewidth=0.8, alpha=0.9)

        for bar, val in zip(bars, vals):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=8, color='#94a3b8')

        ax.set_xlabel('Kelas Penyakit', color='#64748b', fontsize=9)
        ax.set_ylabel('Probabilitas (%)', color='#64748b', fontsize=9)
        ax.set_title('Distribusi Probabilitas', color='#94a3b8', fontsize=10)
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.spines[['top','right','left','bottom']].set_color('#1e293b')
        ax.grid(axis='y', color='#1e293b', linewidth=0.5)
        ax.set_ylim([0, max(vals) * 1.2])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Tab 4: Info Penyakit ─────────────────────
    with tab4:
        pred_cls = res['pred_class']
        full_name, emoji, desc = CLASS_INFO[pred_cls]
        sev_label, sev_color = SEVERITY[pred_cls]

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#111827,#0c1a2e);
                    border:1px solid rgba(6,182,212,0.25);border-radius:14px;
                    padding:1.8rem 2rem;margin-bottom:1.2rem">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">{emoji}</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                        font-weight:800;color:#06b6d4">{pred_cls}</div>
            <div style="font-size:0.95rem;color:#94a3b8;margin-bottom:0.8rem">
                {full_name}
            </div>
            <span style="background:{sev_color}22;color:{sev_color};
                         border:1px solid {sev_color}55;border-radius:20px;
                         padding:3px 14px;font-size:0.8rem;font-weight:600">
                Tingkat Keparahan: {sev_label}
            </span>
            <div class="info-box" style="margin-top:1rem">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Semua kelas dalam grid
        st.markdown('<div class="card-title">📚 Referensi Semua Kelas</div>',
                    unsafe_allow_html=True)

        cols = st.columns(4)
        for i, cls in enumerate(CLASS_NAMES):
            full, em, dsc = CLASS_INFO[cls]
            sv_label, sv_color = SEVERITY[cls]
            is_current = cls == pred_cls
            border = f'border:1px solid {sv_color}55;' if is_current else \
                     'border:1px solid rgba(255,255,255,0.05);'

            with cols[i % 4]:
                st.markdown(f"""
                <div style="background:#1a2235;{border}border-radius:10px;
                            padding:0.9rem;margin-bottom:0.7rem">
                    <div style="font-size:1.5rem">{em}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.85rem;
                                font-weight:700;color:{'#06b6d4' if is_current else '#f1f5f9'};
                                margin-top:0.3rem">{cls}</div>
                    <div style="font-size:0.68rem;color:#64748b;margin:0.2rem 0">{full}</div>
                    <span style="background:{sv_color}22;color:{sv_color};
                                 border-radius:10px;padding:1px 8px;font-size:0.65rem">
                        {sv_label}
                    </span>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem;color:#334155;font-size:0.72rem;line-height:1.8">
    <b style="color:#475569">RetinaScan AI</b> &nbsp;·&nbsp;
    DINOv2 ViT-B/14 Fine-Tuned on Retinal OCT C8 Dataset &nbsp;·&nbsp;
    Attention Rollout (Abnar & Zuidema, 2020) &nbsp;·&nbsp;
    Grad-CAM (Selvaraju et al., 2017)<br>
    ⚠️ Untuk keperluan penelitian dan edukasi — bukan pengganti diagnosis medis profesional
</div>
""", unsafe_allow_html=True)
