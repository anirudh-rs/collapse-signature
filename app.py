import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import networkx as nx
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.geospatial import load_all_assets, build_empire_map, nearest_snapshot_year

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Collapse Signature",
    page_icon="⚔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── THEME & GLOBAL CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;900&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --bg:        #0e0b07;
    --bg2:       #161009;
    --parchment: #f5e6c8;
    --gold:      #c9a84c;
    --gold-dim:  #7a5f2a;
    --red:       #8b1a1a;
    --red-bright:#c0392b;
    --text:      #e8d5b0;
    --text-dim:  #9a8060;
    --border:    #3a2e1a;
    --accent:    #4a6fa5;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'EB Garamond', Georgia, serif;
}

[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(201,168,76,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(139,26,26,0.06) 0%, transparent 50%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
}

[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

h1, h2, h3 { font-family: 'Cinzel', serif !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--gold-dim) !important;
    color: var(--gold) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    padding: 0.4rem 1.2rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    border-color: var(--gold) !important;
    background: rgba(201,168,76,0.08) !important;
    color: var(--parchment) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: var(--bg2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stSlider > div > div > div { background: var(--gold-dim) !important; }
.stSlider > div > div > div > div { background: var(--gold) !important; }

hr { border-color: var(--border) !important; }

.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1100px !important; }

/* ── Custom components ── */
.ornament {
    text-align: center;
    color: var(--gold-dim);
    font-size: 1.4rem;
    letter-spacing: 0.5rem;
    margin: 2rem 0;
}

.chapter-label {
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: var(--gold-dim);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.chapter-title {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--parchment);
    line-height: 1.2;
    margin-bottom: 1rem;
}

.chapter-body {
    font-family: 'EB Garamond', serif;
    font-size: 1.2rem;
    line-height: 1.9;
    color: var(--text);
    max-width: 720px;
}

.finding-card {
    background: linear-gradient(135deg, rgba(201,168,76,0.06) 0%, rgba(14,11,7,0.8) 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    border-radius: 2px;
}

.finding-headline {
    font-family: 'Cinzel', serif;
    font-size: 1.1rem;
    color: var(--gold);
    margin-bottom: 0.5rem;
}

.finding-body {
    font-family: 'EB Garamond', serif;
    font-size: 1.1rem;
    color: var(--text);
    line-height: 1.7;
}

.stat-inscription {
    text-align: center;
    padding: 2rem;
    border: 1px solid var(--border);
    border-top: 2px solid var(--gold);
    background: rgba(201,168,76,0.03);
}

.stat-number {
    font-family: 'Cinzel', serif;
    font-size: 3rem;
    font-weight: 900;
    color: var(--gold);
    display: block;
    line-height: 1;
}

.stat-label {
    font-family: 'EB Garamond', serif;
    font-size: 0.9rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
}

.hero-title {
    font-family: 'Cinzel', serif;
    font-size: 3.5rem;
    font-weight: 900;
    color: var(--parchment);
    line-height: 1.1;
    letter-spacing: 0.02em;
}

.hero-sub {
    font-family: 'EB Garamond', serif;
    font-size: 1.35rem;
    color: var(--text-dim);
    font-style: italic;
    line-height: 1.6;
    margin-top: 1rem;
}

.proof-link {
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--gold-dim);
    text-decoration: none;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1px;
    cursor: pointer;
}

.chain-node {
    display: inline-block;
    font-family: 'Cinzel', serif;
    font-size: 0.85rem;
    color: var(--parchment);
    background: rgba(201,168,76,0.08);
    border: 1px solid var(--border);
    padding: 0.4rem 0.9rem;
    margin: 0.2rem;
    border-radius: 1px;
}

.chain-arrow {
    color: var(--gold-dim);
    font-size: 1.2rem;
    margin: 0 0.2rem;
}

.scenario-card {
    background: linear-gradient(160deg, rgba(139,26,26,0.08) 0%, rgba(14,11,7,0.9) 100%);
    border: 1px solid var(--border);
    border-top: 2px solid var(--red);
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    border-radius: 2px;
}

.scenario-title {
    font-family: 'Cinzel', serif;
    font-size: 1rem;
    color: var(--parchment);
    margin-bottom: 0.8rem;
}

.scenario-body {
    font-family: 'EB Garamond', serif;
    font-size: 1.1rem;
    color: var(--text);
    line-height: 1.7;
}

.prob-change-pos { color: #4caf7d; font-family: 'Cinzel', serif; font-weight: 600; }
.prob-change-neg { color: var(--red-bright); font-family: 'Cinzel', serif; font-weight: 600; }

.analytics-section {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    padding: 1.5rem;
    margin-top: 0.5rem;
    border-radius: 2px;
}

.closing-quote {
    font-family: 'Cinzel', serif;
    font-size: 1.6rem;
    color: var(--gold);
    text-align: center;
    line-height: 1.6;
    padding: 3rem 2rem;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    margin: 3rem 0;
}

.source-tag {
    font-family: 'EB Garamond', serif;
    font-size: 0.85rem;
    color: var(--text-dim);
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "data/processed/civilisations_clean.csv"))
    stressors = pd.read_csv(os.path.join(base, "data/processed/external_stressors.csv"))
    clusters = pd.read_csv(os.path.join(base, "data/processed/collapse_clusters.csv"))
    centrality = pd.read_csv(os.path.join(base, "data/processed/network_centrality.csv"))
    shap_imp = pd.read_csv(os.path.join(base, "data/processed/shap_importance.csv"))
    shap_vals = pd.read_csv(os.path.join(base, "data/processed/shap_values.csv"))
    survival = pd.read_csv(os.path.join(base, "data/processed/survival_probabilities.csv"))
    alignment = pd.read_csv(os.path.join(base, "data/processed/cluster_stressor_alignment.csv"))
    with open(os.path.join(base, "data/processed/pillar_1_2_results.json")) as f:
        results = json.load(f)
    return df, stressors, clusters, centrality, shap_imp, shap_vals, survival, alignment, results

df, stressors, clusters_df, centrality_df, shap_imp, shap_vals, survival_df, alignment_df, results = load_data()

# ── RESILIENCE MODEL (for counterfactual engine) ───────────────────────────────
@st.cache_resource
def load_model(df, stressors):
    feature_names = [
        'rise_duration_years', 'peak_plateau_years', 'log_peak_population',
        'log_peak_territory', 'n_contemporary_rivals', 'n_pressuring_empires',
        'was_conquered', 'founding_era_numeric', 'territory_per_capita',
        'trigger_climate', 'trigger_conquest', 'trigger_fragmentation',
        'trigger_overextension', 'n_stressors_during_collapse', 'has_any_catastrophic'
    ]
    for col in feature_names:
        df[col] = df[col].fillna(0)
    X = df[feature_names].values
    y = (df['lifespan_years'] > df['lifespan_years'].median()).astype(int).values
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.5, max_iter=1000,
                                      random_state=42, class_weight='balanced'))
    ])
    pipeline.fit(X, y)
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    X_scaled = scaler.transform(X)
    explainer = shap.LinearExplainer(model, X_scaled,
                                      feature_perturbation="interventional")
    return pipeline, scaler, explainer, feature_names, X, y

pipeline, scaler, explainer, feature_names, X, y = load_model(df.copy(), stressors)

# ── DYNAMIC STATS (computed from loaded data) ──────────────────────────────────
def compute_dynamic_stats(df, results):
    """Compute all key stats dynamically from loaded data."""
    n_civs        = len(df)
    year_min      = int(df['founded_year'].min())
    year_max      = int(df['collapse_end_year'].max())
    time_span     = year_max - year_min
    n_clusters    = results.get('pillar1_n_clusters', 5)
    loo_acc       = results.get('pillar5_loo_accuracy', 0.92)
    morans_i      = results.get('pillar2_global_morans_i', 0.348)
    morans_p      = results.get('pillar2_global_p_value', 0.003)
    alignment     = results.get('pillar3_mean_alignment_score', 0.67)
    longest_chain = results.get('pillar4_longest_chain_length', 7)
    chain_civs    = results.get('pillar4_longest_chain', [])
    
    # Counterfactual scenarios — read from results
    scenarios     = results.get('counterfactual_scenarios', [])
    largest_jump  = results.get('pillar5_largest_jump_pct', 93.0)
    largest_jump_name = results.get('pillar5_largest_jump_scenario', 'Napoleon as Byzantine')
    qin_jump      = next((s for s in scenarios if 'Qin' in s.get('scenario', '')), {})
    qin_pct       = qin_jump.get('change_pct', 81.8) if qin_jump else 81.8
    napoleon_jump = next((s for s in scenarios if 'Byzantine' in s.get('scenario', '') 
                         and 'Napoleon' in s.get('scenario', '')), {})
    napoleon_pct  = napoleon_jump.get('change_pct', 93.0) if napoleon_jump else 93.0
    
    # Trigger breakdown
    frag_pct = (df['primary_collapse_trigger'] == 'fragmentation').mean() * 100
    conq_pct = (df['primary_collapse_trigger'] == 'conquest').mean() * 100
    
    # Rise to fall
    median_rtf = df['rise_to_fall_ratio'].median()
    
    return {
        'n_civs': n_civs,
        'time_span': time_span,
        'year_min': year_min,
        'year_max': year_max,
        'n_clusters': n_clusters,
        'loo_acc': loo_acc,
        'loo_acc_pct': f"{loo_acc*100:.1f}%",
        'morans_i': morans_i,
        'morans_p': morans_p,
        'alignment_pct': f"{alignment*100:.0f}%",
        'longest_chain': longest_chain,
        'chain_civs': chain_civs,
        'frag_pct': frag_pct,
        'conq_pct': conq_pct,
        'median_rtf': median_rtf,
        'qin_pct': f"+{qin_pct:.1f}%" if isinstance(qin_pct, float) else qin_pct,
        'napoleon_pct': f"+{napoleon_pct:.1f}%" if isinstance(napoleon_pct, float) else napoleon_pct,
        'largest_jump': largest_jump,
        'largest_jump_name': largest_jump_name,
    }

STATS = compute_dynamic_stats(df, results)


# ── COLOUR MAPS ────────────────────────────────────────────────────────────────
REGION_COLOURS = {
    'Europe': '#4E79A7', 'Middle East': '#F28E2B', 'East Asia': '#E15759',
    'Africa': '#76B7B2', 'Americas': '#59A14F', 'South Asia': '#EDC948',
    'Central Asia': '#B07AA1'
}
TRIGGER_COLOURS = {
    'fragmentation': '#4E79A7', 'conquest': '#E15759',
    'overextension': '#F28E2B', 'climate': '#76B7B2',
    'migration': '#59A14F', 'economic': '#EDC948'
}

# ── GEOSPATIAL ASSETS ─────────────────────────────────────────────────────────
@st.cache_resource
def load_geo_assets():
    try:
        return load_all_assets()
    except Exception as e:
        st.error(f"Geospatial assets failed to load: {e}")
        return None, {}, {}, [], None

geo_civs_df, geo_mapping, geo_colours, geo_years, geo_stressors = load_geo_assets()

# ── PAGE NAVIGATION (no sidebar) ──────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'story'
if 'map_playing' not in st.session_state:
    st.session_state.map_playing = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — THE STORY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'story':

    # ── HERO ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 4rem 0 2rem 0; border-bottom: 1px solid var(--border);">
        <p style="font-family:Cinzel,serif;font-size:0.7rem;letter-spacing:0.35em;color:var(--gold-dim);margin-bottom:1rem;">
            THE COLLAPSE SIGNATURE PROJECT
        </p>
        <h1 class="hero-title">Every empire fell.<br>None fell the same way.</h1>
        <p class="hero-sub">
            Nearly Five thousand years of rise and ruin — analysed, mapped, and questioned.<br>
            What the data reveals about why civilisations collapse might surprise you.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top navigation
    col_nav, col_nav2, col_spacer = st.columns([1, 1, 3])
    with col_nav:
        if st.button("🌍  History in Motion →"):
            st.session_state.page = 'map'
            st.rerun()
    with col_nav2:
        if st.button("📜  Behind the data →"):
            st.session_state.page = 'proof'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Hero stats — dynamic from loaded data
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-inscription">
            <span class="stat-number">{STATS['n_civs']}</span>
            <div class="stat-label">civilisations studied</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-inscription">
            <span class="stat-number">{STATS['time_span']:,}</span>
            <div class="stat-label">years of history</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-inscription">
            <span class="stat-number">{STATS['n_clusters']}</span>
            <div class="stat-label">collapse clusters found</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-inscription">
            <span class="stat-number">{STATS['loo_acc_pct']}</span>
            <div class="stat-label">prediction accuracy</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="ornament">✦ ✦ ✦</div>', unsafe_allow_html=True)

    # ── CHAPTER 1 — COLLAPSES CLUSTER ─────────────────────────────────────────
    st.markdown('<p class="chapter-label">Chapter I</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="chapter-title">Collapses are not accidents</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    <p class="chapter-body">
        History teaches us that great empires fall one by one — each brought low by its own
        unique story of hubris, invasion, or decay. The data tells a different story.
        <br><br>
        When we plotted the collapse of every major civilisation across {STATS['time_span']:,} years,
        something unexpected appeared. They <em>bunch together</em>. Five distinct moments
        in history when multiple empires — on opposite sides of the world — fell within
        the same century.
        <br><br>
        This is not coincidence. The mathematics confirm it with near-certainty.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="finding-card"><p class="finding-headline">The Finding</p><p class="finding-body">Civilisational collapses cluster in time far more than random chance would predict. The probability of this pattern occurring by accident is less than 1 in 10,000.</p></div>', unsafe_allow_html=True)

    # Cluster visual — simple timeline dots
    cluster_data = [
        {"year": -2140, "label": "~2140 BCE", "name": "Bronze Age Collapse",
         "civs": "Sumer · Akkad", "stressor": "Catastrophic drought"},
        {"year": -1110, "label": "~1110 BCE", "name": "Late Bronze Age",
         "civs": "Egypt · Hittites · Mycenae", "stressor": "Sea Peoples · drought"},
        {"year": -263,  "label": "~263 BCE",  "name": "Hellenistic Crisis",
         "civs": "Persia · Greece · Macedon · Maurya · Qin",
         "stressor": "Conquest cascade"},
        {"year": 793,   "label": "~793 CE",   "name": "Late Antique Crisis",
         "civs": "Umayyad · Tang · Maya · Carolingian",
         "stressor": "Plague · volcanic winter"},
        {"year": 1915,  "label": "~1915 CE",  "name": "Modern Collapse Wave",
         "civs": "Ottoman · Austro-Hungarian · Russian · British · German",
         "stressor": "World Wars · pandemic"},
    ]

    with st.expander("▸  See the five collapse clusters", expanded=True):
        for c in cluster_data:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;margin:1rem 0;padding:1rem;
                        border-left:2px solid var(--gold-dim);background:rgba(201,168,76,0.03);">
                <div style="min-width:110px;font-family:Cinzel,serif;font-size:0.8rem;
                            color:var(--gold);margin-right:1.5rem;">{c['label']}</div>
                <div>
                    <div style="font-family:Cinzel,serif;font-size:0.95rem;
                                color:var(--parchment);margin-bottom:0.3rem;">{c['name']}</div>
                    <div style="font-family:EB Garamond,serif;font-size:1rem;
                                color:var(--text-dim);">{c['civs']}</div>
                    <div style="font-family:EB Garamond,serif;font-size:0.9rem;
                                color:var(--red-bright);margin-top:0.2rem;">
                        ⚡ {c['stressor']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("◆  Show the statistical proof"):
        with st.container():
            st.markdown("**Temporal Clustering — Permutation Test**")
            st.markdown(f"""
            - Observed clustering score: **2.029**
            - Null distribution mean: **1.169**
            - Z-score: **6.65**
            - P-value: **< 0.0001**
            - Method: 10,000 permutation test — collapse events redistributed randomly
              across {STATS['time_span']:,}-year timeline. Observed clustering exceeds 99.99% of
              random arrangements.
            """)
            # KDE chart
            years = ((df['collapse_start_year'] + df['collapse_end_year']) / 2).values
            bw = 120 / years.std()
            kde = gaussian_kde(years, bw_method=bw)
            x_range = np.linspace(-3000, 2010, 3000)
            kde_vals = kde(x_range)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_range, y=kde_vals,
                fill='tozeroy', fillcolor='rgba(201,168,76,0.15)',
                line=dict(color='#c9a84c', width=2),
                name='Collapse density'
            ))
            for c in cluster_data:
                fig.add_vline(x=c['year'], line_dash='dash',
                              line_color='#8b1a1a', line_width=1.5)
                fig.add_annotation(x=c['year'], y=kde(np.array([c['year']]))[0] * 1.2,
                                   text=c['label'], font=dict(size=8, color='#c9a84c'),
                                   showarrow=False)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e8d5b0',
                height=280,
                margin=dict(t=20, b=20, l=10, r=10),
                xaxis=dict(title='Year (negative = BCE)',
                           gridcolor='rgba(255,255,255,0.05)',
                           color='#9a8060'),
                yaxis=dict(title='Collapse density',
                           gridcolor='rgba(255,255,255,0.05)',
                           color='#9a8060'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="ornament">—</div>', unsafe_allow_html=True)

    # ── CHAPTER 2 — NOT CONTAGION ──────────────────────────────────────────────
    st.markdown('<p class="chapter-label">Chapter II</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="chapter-title">They were not falling on each other</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="chapter-body">
        You might imagine collapse clusters as a domino effect — one empire's fall
        destabilising its neighbours, spreading outward like a contagion. That is not
        what happened.
        <br><br>
        During every one of the five collapse clusters, the empires falling were
        thousands of miles apart. The Roman Empire and the Han Dynasty declined
        within two centuries of each other. They had never met. The Classic Maya
        collapsed at the same time as the Carolingian Empire in France and the
        Tang Dynasty in China — three civilisations with no contact, no trade,
        no awareness of each other.
        <br><br>
        Something was reaching them all simultaneously.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="finding-card"><p class="finding-headline">The Finding</p><p class="finding-body">Collapse clusters are not regional contagion. The civilisations collapsing within each cluster were geographically dispersed — spread across continents. The cause was shared pressure from above, not spreading failure from the side.</p></div>', unsafe_allow_html=True)

    # Simple map
    with st.expander("◆  Show the geographic analysis"):
        with st.container():
            st.markdown("**Moran's I Spatial Autocorrelation**")
            st.markdown(f"""
            - Global Moran's I: **{results['pillar2_global_morans_i']:.3f}** (p = {results['pillar2_global_p_value']:.3f})
            - Positive global I confirms collapses are geographically patterned overall
            - **Per-cluster Moran's I: negative across all five clusters**
            - Negative I = collapsing civilisations were *more* geographically spread
              than expected — the opposite of contagion
            - Interpretation: collapse clusters reflect top-down systemic shocks,
              not bottom-up neighbour-to-neighbour spread
            """)
            m = folium.Map(location=[25, 15], zoom_start=1,
                           tiles='CartoDB dark_matter')
            for _, row in df.iterrows():
                colour = REGION_COLOURS.get(row['region'], '#999')
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=max(4, min(16, row['peak_territory_km2'] / 2000000 * 3)),
                    color=colour, fill=True, fill_color=colour, fill_opacity=0.7,
                    tooltip=f"{row['name']} — {row['region']}"
                ).add_to(m)
            st_folium(m, height=350, use_container_width=True)

    st.markdown('<div class="ornament">—</div>', unsafe_allow_html=True)

    # ── CHAPTER 3 — STRESSORS ─────────────────────────────────────────────────
    st.markdown('<p class="chapter-label">Chapter III</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="chapter-title">Climate and plague were the messengers</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    <p class="chapter-body">
        If collapse clusters are not caused by neighbours, what reached civilisations
        simultaneously across continents? The evidence points to forces that respected
        no borders.
        <br><br>
        Of the {STATS['n_civs']} civilisations we studied, 69% collapsed during a documented climate
        event. 31% collapsed during a major pandemic. These were not the killers —
        empires rarely fell from plague or drought alone. They were the final weight
        placed on structures already cracking from within.
        <br><br>
        The stressors explained the <em>timing</em> of collapse. The vulnerability
        was already there.
    </p>
    """, unsafe_allow_html=True)

    # Stressor breakdown — visual
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("69%", "collapsed during a\nclimate event"),
        ("31%", "collapsed during\na pandemic"),
        ("44%", "faced a catastrophic\nexternal stressor"),
        ("72%", "of collapse clusters\naligned with stressors"),
    ]
    for col, (num, label) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="stat-inscription">
                <span class="stat-number" style="font-size:2rem;">{num}</span>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="finding-card">
        <p class="finding-headline">The Surprising Finding</p>
        <p class="finding-body">
            Removing stressors from already-vulnerable empires changes almost nothing.
            The Aztec Empire, for example — remove the smallpox pandemic entirely,
            and the model predicts their survival barely moves. The structural weakness
            was already fatal. The plague just arrived first.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("◆  Show the stressor alignment analysis"):
        with st.container():
            st.markdown("**Per-Cluster Stressor Alignment Scores**")
            fig = go.Figure(go.Bar(
                x=alignment_df['alignment_score'] * 100,
                y=alignment_df['cluster_name'],
                orientation='h',
                marker_color=['#c9a84c' if s == 1.0 else '#7a5f2a'
                              for s in alignment_df['alignment_score']],
                text=[f"{s:.0%}" for s in alignment_df['alignment_score']],
                textposition='outside',
                textfont=dict(color='#e8d5b0', size=11)
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e8d5b0', height=280,
                margin=dict(t=10, b=10, l=10, r=80),
                xaxis=dict(title='% civilisations with documented stressor',
                           gridcolor='rgba(255,255,255,0.05)',
                           color='#9a8060', range=[0, 115]),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#9a8060')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Mann-Whitney U tests (stressor vs no stressor):**
            - Collapse duration: p = 0.173 — no significant difference in *how fast* collapse happened
            - Conquest likelihood: p = 0.212 — stressors didn't increase conquest probability
            - **Key finding**: More stressors correlate with *longer* collapse duration (r = 0.41)
              — stressors create prolonged pressure, not sudden collapse
            """)

    st.markdown('<div class="ornament">—</div>', unsafe_allow_html=True)

    # ── CHAPTER 4 — THE CHAIN ─────────────────────────────────────────────────
    st.markdown('<p class="chapter-label">Chapter IV</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="chapter-title">Some collapses were connected — across centuries</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="chapter-body">
        Not all connections between empires were geographic. When we mapped every
        conquest, every succession, every empire that rose from the ruins of another —
        chains emerged that stretched across a thousand years and three continents.
        <br><br>
        One thread runs from a Babylonian dynasty in 626 BCE all the way to the
        Abbasid Caliphate's collapse in 1258 CE. Seven civilisations. One unbroken line of
        cause and consequence.
    </p>
    """, unsafe_allow_html=True)

    # The chain — pulled dynamically from results JSON
    chain = results.get('pillar4_longest_chain',
                        ['Neo-Babylonian Empire', 'Achaemenid Persian Empire',
                         'Seleucid Empire', 'Parthian Empire', 'Sassanid Empire',
                         'Umayyad Caliphate', 'Abbasid Caliphate'])
    chain_length = results.get('pillar4_longest_chain_length', 7)

    # Founding years for each node in the default chain
    _chain_year_map = {
        'Neo-Babylonian Empire':   '-626 BCE',
        'Achaemenid Persian Empire': '-550 BCE',
        'Seleucid Empire':         '-312 BCE',
        'Parthian Empire':         '-247 BCE',
        'Sassanid Empire':         '224 CE',
        'Umayyad Caliphate':       '661 CE',
        'Abbasid Caliphate':       '750 CE',
        'Mongol Empire':           '1206 CE',
        'Timurid Empire':          '1370 CE',
        'Mughal Empire':           '1526 CE',
        'British Empire':          '1583 CE',
        'Golden Horde':            '1242 CE',
        'Kievan Rus':              '882 CE',
    }

    chain_html = '<div style="margin:1.5rem 0;line-height:2.5;flex-wrap:wrap;display:flex;align-items:center;">'
    for i, node in enumerate(chain):
        yr = _chain_year_map.get(node, '')
        chain_html += (
            f'<span class="chain-node">{node}'
            f'{"<br>" if yr else ""}'
            f'<span style="font-size:0.65rem;color:var(--text-dim);">{yr}</span>'
            f'</span>'
        )
        if i < len(chain) - 1:
            chain_html += '<span class="chain-arrow">→</span>'

    # Compute span from df
    _chain_civs = df[df['name'].isin(chain)]
    if not _chain_civs.empty:
        _span = int(_chain_civs['collapse_end_year'].max()) - int(_chain_civs['founded_year'].min())
    else:
        _span = 1884
    chain_html += (
        f'<div style="font-family:EB Garamond,serif;font-size:0.9rem;'
        f'color:var(--text-dim);margin-top:0.8rem;width:100%;">'
        f'{_span:,} years · one causal thread</div>'
    )
    chain_html += '</div>'
    st.markdown(chain_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-card">
        <p class="finding-headline">The Finding</p>
        <p class="finding-body">
            The most influential empire in our network — the one that sat at the
            junction of the most historical chains — was not Rome. It was the
            Sassanid Persian Empire. The empire most people have never heard of
            was the hinge on which medieval history turned.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("◆  Show the network analysis"):
        with st.container():
            st.markdown("**Network Centrality Metrics**")
            top_bridge = centrality_df.nlargest(8, 'betweenness')[
                ['name', 'betweenness', 'out_degree', 'region']]
            top_bridge.columns = ['Civilisation', 'Betweenness', 'Influence (out-degree)', 'Region']
            top_bridge['Betweenness'] = top_bridge['Betweenness'].round(4)
            st.dataframe(
                top_bridge.reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
            st.markdown(f"""
            - Total network edges: **{results['pillar4_n_edges']}** causal relationships
            - Conquest edges: **{results['pillar4_conquest_edges']}** · Pressure: **{results['pillar4_pressure_edges']}** · Succession: **{results['pillar4_succession_edges']}**
            - Most predatory empire (out-degree): **{results['pillar4_top_predators'][0]}**
            - 36 causal chains of 4+ civilisations identified
            """)

    st.markdown('<div class="ornament">—</div>', unsafe_allow_html=True)

    # ── CHAPTER 5 — WHAT PREDICTS SURVIVAL ────────────────────────────────────
    st.markdown('<p class="chapter-label">Chapter V</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="chapter-title">What actually made the difference</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    <p class="chapter-body">
        Some empires faced the same plagues, the same climate shifts, the same
        hostile neighbours — and survived for centuries. Others collapsed within
        decades. We trained a predictive model on all {STATS['n_civs']} civilisations to find
        what separated them.
        <br><br>
        The model predicted correctly in {int(STATS['loo_acc'] * STATS['n_civs'])} out of {STATS['n_civs']} cases. And the single most
        important factor — by a large margin — was not size, not military strength,
        not geography. It was this:
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;border-top:1px solid var(--border);
                border-bottom:1px solid var(--border);margin:2rem 0;">
        <p style="font-family:Cinzel,serif;font-size:2.2rem;color:var(--gold);
                  font-weight:900;line-height:1.4;margin:0;">
            How long they took to build.
        </p>
        <p style="font-family:EB Garamond,serif;font-size:1.15rem;color:var(--text-dim);
                  margin-top:1rem;font-style:italic;">
            Empires that rose slowly lasted longer. Every time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Simple visual comparison — fast vs slow
    comparison_data = [
        ("Napoleonic Empire", 8, 11, "#8b1a1a"),
        ("Macedonian Empire", 13, 35, "#8b1a1a"),
        ("Qin Dynasty", 6, 15, "#8b1a1a"),
        ("Ottoman Empire", 150, 623, "#c9a84c"),
        ("Byzantine Empire", 225, 1123, "#c9a84c"),
        ("Kingdom of Kush", 400, 1420, "#c9a84c"),
    ]

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("""
        <p style="font-family:Cinzel,serif;font-size:0.75rem;letter-spacing:0.15em;
                  color:var(--text-dim);margin-bottom:1rem;">RISE vs LIFESPAN</p>
        """, unsafe_allow_html=True)
        for name, rise, life, colour in comparison_data:
            pct = min(rise / 500 * 100, 100)
            life_pct = min(life / 1420 * 100, 100)
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="font-family:Cinzel,serif;font-size:0.75rem;
                            color:{colour};margin-bottom:0.2rem;">{name}</div>
                <div style="display:flex;gap:0.4rem;align-items:center;">
                    <div style="width:{pct:.0f}%;height:6px;background:{colour};
                                opacity:0.5;border-radius:1px;"></div>
                    <span style="font-family:EB Garamond,serif;font-size:0.8rem;
                                 color:var(--text-dim);">{rise}yr rise → {life}yr life</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        plot_df = df.merge(
            survival_df[['name', 'survival_probability']], on='name', how='left'
        )
        fig = go.Figure()
        for region, colour in REGION_COLOURS.items():
            sub = plot_df[plot_df['region'] == region]
            fig.add_trace(go.Scatter(
                x=sub['rise_duration_years'],
                y=sub['lifespan_years'],
                mode='markers',
                name=region,
                marker=dict(color=colour, size=8, opacity=0.8,
                            line=dict(color='rgba(0,0,0,0.3)', width=0.5)),
                hovertemplate='<b>%{text}</b><br>Rise: %{x} yrs<br>Lifespan: %{y} yrs<extra></extra>',
                text=sub['name']
            ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8d5b0', height=320,
            margin=dict(t=10, b=30, l=10, r=10),
            xaxis=dict(title='Rise Duration (years)',
                       gridcolor='rgba(255,255,255,0.04)', color='#9a8060'),
            yaxis=dict(title='Total Lifespan (years)',
                       gridcolor='rgba(255,255,255,0.04)', color='#9a8060'),
            legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("◆  Show the model and SHAP analysis"):
        with st.container():
            st.markdown(f"""
            **Logistic Regression — Leave-One-Out Cross Validation (n={STATS['n_civs']})**
            - Accuracy: **{STATS['loo_acc_pct']}** (vs 50.0% majority-class baseline)
            - Train AUC: **0.997**
            - Correctly classified: **{int(STATS['loo_acc'] * STATS['n_civs'])}/{STATS['n_civs']}** civilisations
            - Misclassified: Mughal Empire, Mali Empire, Kushan Empire, Mutapa Empire, Kievan Rus
            """)
            label_map = {
                'rise_duration_years': 'Rise Duration',
                'peak_plateau_years': 'Peak Plateau',
                'n_contemporary_rivals': 'Number of Rivals',
                'founding_era_numeric': 'Founding Era',
                'n_stressors_during_collapse': 'Stressors During Collapse',
                'has_any_catastrophic': 'Faced Catastrophic Stressor',
                'n_pressuring_empires': 'Empires Pressuring',
                'log_peak_population': 'Peak Population (log)',
                'was_conquered': 'Was Directly Conquered',
                'trigger_fragmentation': 'Trigger: Fragmentation',
                'territory_per_capita': 'Territory Per Capita',
                'log_peak_territory': 'Peak Territory (log)',
                'trigger_climate': 'Trigger: Climate',
                'trigger_conquest': 'Trigger: Conquest',
                'trigger_overextension': 'Trigger: Overextension'
            }
            plot_shap = shap_imp.copy()
            plot_shap['label'] = plot_shap['feature'].map(label_map).fillna(plot_shap['feature'])
            plot_shap = plot_shap.sort_values('mean_abs_shap')
            colours = ['#c9a84c' if v > 0.5 else '#7a5f2a' if v > 0.2
                       else '#3a2e1a' for v in plot_shap['mean_abs_shap']]
            fig = go.Figure(go.Bar(
                x=plot_shap['mean_abs_shap'],
                y=plot_shap['label'],
                orientation='h',
                marker_color=colours,
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e8d5b0', height=420,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(title='Mean |SHAP value| — higher = stronger predictor',
                           gridcolor='rgba(255,255,255,0.04)', color='#9a8060'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#9a8060')
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="ornament">—</div>', unsafe_allow_html=True)

    # ── CHAPTER 6 — WHAT IF ───────────────────────────────────────────────────
    st.markdown('<p class="chapter-label">Chapter VI</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="chapter-title">What if history had been different?</h2>', unsafe_allow_html=True)
    _n = STATS['n_civs']; _t = STATS['time_span']
    st.markdown(f"""
    <p class="chapter-body">
        History is full of what-ifs. We built a model that can actually attempt
        to answer some of them — not with certainty, but with data. These are
        not guesses. They are the model's predictions, based on the patterns of
        {_n} civilisations across {_t:,} years.
    </p>
    """, unsafe_allow_html=True)

    # Featured scenario cards
    scenarios = [
        {
            "title": "The Qin Question",
            "body": "The Qin Dynasty unified China in 221 BCE — one of history's greatest achievements. It collapsed 15 years later. If it had built as patiently as the Han Dynasty that replaced it, the model predicts its survival probability would have jumped from",
            "from_pct": "4.6%", "to_pct": "86.5%", "change": "+81.8%", "positive": True,
            "footnote": "The Han did exactly this — and lasted 400 years."
        },
        {
            "title": "The Napoleon Problem",
            "body": "Napoleon's empire lasted 11 years. Give it the institutional depth of the Byzantine Empire — which lasted 1,100 years — and the model predicts survival probability rises from",
            "from_pct": "6.3%", "to_pct": "99.3%", "change": "+93.0%", "positive": True,
            "footnote": "The difference was not military genius. It was patience."
        },
        {
            "title": "The Aztec Surprise",
            "body": "The Aztec Empire was destroyed by Spanish conquest aided by smallpox. Remove the pandemic entirely — and the model barely moves. Survival probability shifts from",
            "from_pct": "22.8%", "to_pct": "21.1%", "change": "-1.7%", "positive": False,
            "footnote": "The structural vulnerability was already there before Cortés arrived."
        },
    ]

    for s in scenarios:
        change_class = "prob-change-pos" if s["positive"] else "prob-change-neg"
        st.markdown(f"""
        <div class="scenario-card">
            <p class="scenario-title">{s['title']}</p>
            <p class="scenario-body">
                {s['body']}
                <strong style="color:var(--text-dim);">{s['from_pct']}</strong>
                to
                <strong style="color:var(--parchment);">{s['to_pct']}</strong>
                — a <span class="{change_class}">{s['change']}</span> shift.
            </p>
            <p style="font-family:EB Garamond,serif;font-size:0.95rem;
                      color:var(--text-dim);font-style:italic;margin-top:0.7rem;">
                {s['footnote']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── INTERACTIVE EXPLORER ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:Cinzel,serif;font-size:0.9rem;color:var(--gold);letter-spacing:0.1em;">TRY YOUR OWN SCENARIO</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:EB Garamond,serif;font-size:1.05rem;color:var(--text-dim);">Pick a civilisation and adjust its characteristics. See how the model responds.</p>', unsafe_allow_html=True)

    civ_names = sorted(df['name'].tolist())
    selected_civ = st.selectbox("Choose a civilisation", civ_names,
                                 index=civ_names.index("Roman Empire"))

    civ_row = df[df['name'] == selected_civ].iloc[0]
    base_prob = survival_df[survival_df['name'] == selected_civ]['survival_probability'].values
    base_prob = float(base_prob[0]) if len(base_prob) > 0 else 0.5

    st.markdown(f"""
    <div style="background:rgba(201,168,76,0.05);border:1px solid var(--border);
                padding:1rem 1.5rem;margin:1rem 0;border-radius:2px;">
        <span style="font-family:Cinzel,serif;font-size:0.8rem;color:var(--text-dim);">
            BASE SURVIVAL PROBABILITY
        </span><br>
        <span style="font-family:Cinzel,serif;font-size:2rem;color:var(--gold);">
            {base_prob:.1%}
        </span>
        <span style="font-family:EB Garamond,serif;font-size:0.95rem;color:var(--text-dim);margin-left:1rem;">
            Founded: {abs(int(civ_row['founded_year']))} {'BCE' if civ_row['founded_year'] < 0 else 'CE'} ·
            Lifespan: {int(civ_row['lifespan_years'])} years ·
            Trigger: {civ_row['primary_collapse_trigger']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        new_rise = st.slider(
            "Rise duration (years)",
            min_value=5, max_value=600,
            value=int(civ_row['rise_duration_years']),
            help="How many years did it take to reach peak power?"
        )
        if new_rise > 500:
            st.caption("⚠ Outside historical range — extrapolation territory")
    with col2:
        new_plateau = st.slider(
            "Peak plateau (years)",
            min_value=0, max_value=1000,
            value=int(civ_row['peak_plateau_years']),
            help="How many years did it remain at peak before declining?"
        )
        if new_plateau > 874:
            st.caption("⚠ Outside historical range — extrapolation territory")
    with col3:
        new_rivals = st.slider(
            "Number of rival empires",
            min_value=0, max_value=6,
            value=int(civ_row['n_contemporary_rivals']),
            help="How many major rivals existed simultaneously?"
        )
        if new_rivals > 4:
            st.caption("⚠ Outside historical range — extrapolation territory")

    # Compute modified prediction
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    base_vec = X[df['name'].tolist().index(selected_civ)].copy()
    mod_vec = base_vec.copy()
    mod_vec[feat_idx['rise_duration_years']] = new_rise
    mod_vec[feat_idx['peak_plateau_years']] = new_plateau
    mod_vec[feat_idx['n_contemporary_rivals']] = new_rivals

    mod_prob = float(pipeline.predict_proba(mod_vec.reshape(1, -1))[0, 1])
    delta = mod_prob - base_prob

    delta_colour = "#4caf7d" if delta > 0.02 else "#c0392b" if delta < -0.02 else "#c9a84c"
    delta_arrow = "▲" if delta > 0.02 else "▼" if delta < -0.02 else "►"

    st.markdown(f"""
    <div style="background:rgba(201,168,76,0.08);border:1px solid var(--gold-dim);
                padding:1.2rem 1.5rem;margin-top:0.8rem;border-radius:2px;
                display:flex;align-items:center;gap:2rem;">
        <div>
            <span style="font-family:Cinzel,serif;font-size:0.75rem;
                         color:var(--text-dim);">MODIFIED PROBABILITY</span><br>
            <span style="font-family:Cinzel,serif;font-size:2.2rem;
                         color:var(--parchment);">{mod_prob:.1%}</span>
        </div>
        <div>
            <span style="font-family:Cinzel,serif;font-size:1.8rem;
                         color:{delta_colour};">{delta_arrow} {abs(delta):.1%}</span><br>
            <span style="font-family:EB Garamond,serif;font-size:0.9rem;
                         color:var(--text-dim);">
                {'increase' if delta > 0 else 'decrease' if delta < 0 else 'no change'}
                in predicted survival
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plain-English explanation
    explanations = []
    if new_rise != int(civ_row['rise_duration_years']):
        diff = new_rise - int(civ_row['rise_duration_years'])
        direction = "slower" if diff > 0 else "faster"
        explanations.append(f"Building {abs(diff)} years {direction} {'increases' if diff > 0 else 'decreases'} the predicted chance of long-term survival — rise speed is the strongest predictor in the model.")
    if new_plateau != int(civ_row['peak_plateau_years']):
        diff = new_plateau - int(civ_row['peak_plateau_years'])
        direction = "longer" if diff > 0 else "shorter"
        explanations.append(f"Staying at peak {abs(diff)} years {direction} {'strengthens' if diff > 0 else 'weakens'} the empire's predicted resilience.")
    if new_rivals != int(civ_row['n_contemporary_rivals']):
        diff = new_rivals - int(civ_row['n_contemporary_rivals'])
        explanations.append(f"{'More' if diff > 0 else 'Fewer'} rivals {'increases competitive pressure — historically associated with faster collapse' if diff > 0 else 'reduces pressure, but also reduces the institutional sharpening that rivals can force'}.")

    if explanations:
        st.markdown("<br>", unsafe_allow_html=True)
        for exp in explanations:
            st.markdown(f'<p style="font-family:EB Garamond,serif;font-size:1.05rem;color:var(--text-dim);font-style:italic;">— {exp}</p>', unsafe_allow_html=True)

    st.markdown(f'<p style="font-family:EB Garamond,serif;font-size:0.85rem;color:var(--text-dim);margin-top:1rem;font-style:italic;">Note: This model is trained on {STATS["n_civs"]} civilisations. Predictions are directional indicators based on historical patterns — not precise forecasts. Values beyond historical range are extrapolations and should be interpreted cautiously.</p>', unsafe_allow_html=True)

    # ── CLOSING ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="closing-quote">
        Nearly Five thousand years of history.<br>
        One Hundred and Twelve civilisations.<br>
        The same pattern, over and over.<br><br>
        <span style="font-size:1.2rem;color:var(--text-dim);font-style:italic;">
            Build slowly. Survive longer.<br>
            The collapses were not random.<br>
            They were almost predictable.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:1rem 0 3rem 0;">
        <p style="font-family:Cinzel,serif;font-size:0.7rem;letter-spacing:0.25em;
                  color:var(--text-dim);">
            DATA SOURCES
        </p>
        <p class="source-tag">
            CLIO-INFRA Historical Dataset · Maddison Project (University of Groningen) ·
            Our World in Data · Wikipedia-derived civilisation timelines ·
            Curated historical sources cited in methodology
        </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — THE MAP
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'map':
    col_back, col_fwd, col_sp = st.columns([1, 1, 3])
    with col_back:
        if st.button("⚔  ← The Story"):
            st.session_state.page = 'story'
            st.rerun()
    with col_fwd:
        if st.button("📜  The Proof →"):
            st.session_state.page = 'proof'
            st.rerun()

    st.markdown("""
    <div style="padding:2rem 0 1rem 0;border-bottom:1px solid var(--border);">
        <p style="font-family:Cinzel,serif;font-size:0.7rem;letter-spacing:0.3em;
                  color:var(--gold-dim);">HISTORY IN MOTION</p>
        <h1 style="font-family:Cinzel,serif;font-size:2.4rem;font-weight:900;
                   color:var(--parchment);margin:0.3rem 0;">
            3,000 years of empire — watch it unfold
        </h1>
        <p style="font-family:'EB Garamond',serif;font-size:1.1rem;
                  color:var(--text-dim);">
            Press play and watch civilisations appear, expand to their peak,
            and fade as they collapse. Every bubble is a real empire.
            Every movement is data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Era selector — shared between both tabs
    era_options = {
        "Ancient World (3000 BCE – 500 CE)":  "ancient",
        "Medieval World (500 – 1500 CE)":      "medieval",
        "Early Modern (1500 – 1800 CE)":       "early_modern",
        "Modern Era (1800 – 2010 CE)":         "modern",
    }
    selected_era = st.selectbox(
        "Choose an era — applies to both the animation and the explorer",
        options=list(era_options.keys()),
        index=1,
        key="era_selector"
    )
    era_key = era_options[selected_era]

    map_tab1, map_tab2 = st.tabs(["▶  Animation — Watch History Play", "🔍  Explorer — Any Moment in Detail"])

    with map_tab1:
        # GIF
        gif_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "geospatial", "gifs", f"collapse_{era_key}.gif"
        )

        if os.path.exists(gif_path):
            import base64 as _b64
            with open(gif_path, 'rb') as _f:
                _gif_b64 = _b64.b64encode(_f.read()).decode()
            st.markdown(
                f'<img src="data:image/gif;base64,{_gif_b64}" '
                f'style="width:100%;border-radius:2px;" alt="Empire animation">',
                unsafe_allow_html=True
            )
            st.markdown("""
            <p style="font-family:'EB Garamond',serif;font-size:0.8rem;
                      color:var(--text-dim);font-style:italic;text-align:center;
                      margin-top:0.4rem;">
                Real territorial boundaries · Historical Basemaps (Aourednik 2022) ·
                76 of 77 mapped civilisations have real boundary polygons · 112 civilisations in the analytical dataset
            </p>
            """, unsafe_allow_html=True)
        else:
            st.info(
                f"GIF for **{selected_era}** not yet generated. "
                f"Run notebook `11_map_gif.ipynb` to create it.",
                icon="🗺"
            )

    with map_tab2:
        # Inherit era from tab 1
        era_options_2 = {
            "Ancient World (3000 BCE – 500 CE)":  "ancient",
            "Medieval World (500 – 1500 CE)":      "medieval",
            "Early Modern (1500 – 1800 CE)":       "early_modern",
            "Modern Era (1800 – 2010 CE)":         "modern",
        }

        # Get exact snapshot years for this era
        _era_snap_ranges = {
            "ancient":      (-3000, 500),
            "medieval":     (500,   1500),
            "early_modern": (1500,  1800),
            "modern":       (1800,  2010),
        }
        _e_min, _e_max = _era_snap_ranges[era_key]
        _era_snaps = [y for y in geo_years if _e_min <= y <= _e_max]

        st.markdown(f"""
        <p style="font-family:'EB Garamond',serif;font-size:1rem;
                  color:var(--text-dim);margin-bottom:0.8rem;">
            Step through the exact boundary snapshots for
            <strong style="color:var(--gold);">
            {st.session_state.get('era_selector','Medieval World')}
            </strong>.
            Use ← → to move between frames.
        </p>
        """, unsafe_allow_html=True)

        # Initialise snapshot index in session state
        snap_key = f"snap_idx_{era_key}"
        if snap_key not in st.session_state:
            st.session_state[snap_key] = 0
        # Clamp index
        st.session_state[snap_key] = max(
            0, min(st.session_state[snap_key], len(_era_snaps) - 1))
        _idx = st.session_state[snap_key]

        # Prev / Next buttons + year display
        col_prev, col_year, col_next, col_reg, col_str = st.columns(
            [1, 2, 1, 2, 1])

        with col_prev:
            if st.button("← Previous", key="snap_prev",
                         disabled=(_idx == 0)):
                st.session_state[snap_key] -= 1
                st.rerun()

        with col_year:
            _snap_yr = _era_snaps[_idx]
            _snap_lbl = (f"{abs(_snap_yr)} BCE"
                         if _snap_yr < 0 else f"{_snap_yr} CE")
            st.markdown(f"""
            <div style="text-align:center;font-family:Cinzel,serif;
                        font-size:1.3rem;color:var(--gold);
                        padding:0.2rem 0;">
                {_snap_lbl}
            </div>
            <div style="text-align:center;font-family:'EB Garamond',serif;
                        font-size:0.8rem;color:var(--text-dim);">
                {_idx + 1} of {len(_era_snaps)} snapshots
            </div>
            """, unsafe_allow_html=True)

        with col_next:
            if st.button("Next →", key="snap_next",
                         disabled=(_idx == len(_era_snaps) - 1)):
                st.session_state[snap_key] += 1
                st.rerun()

        with col_reg:
            all_regions_e = sorted(geo_civs_df['region'].unique().tolist())
            sel_regions = st.multiselect(
                "Filter regions (empty = all)",
                options=all_regions_e,
                default=[],
                key="explore_region_filter"
            )

        with col_str:
            show_str = st.checkbox("Stressors", value=True,
                                   key="explore_stressors")

        with st.spinner("Rendering boundaries..."):
            folium_map = build_empire_map(
                target_year=_snap_yr,
                civs_df=geo_civs_df,
                mapping=geo_mapping,
                colour_map=geo_colours,
                available_years=geo_years,
                show_stressors=show_str,
                stressors_df=geo_stressors,
                filter_regions=sel_regions if sel_regions else None
            )
            st_folium(folium_map, height=540, use_container_width=True)

        st.markdown("""
        <p style="font-family:'EB Garamond',serif;font-size:0.8rem;
                  color:var(--text-dim);font-style:italic;margin-top:0.3rem;">
            Boundary data: Historical Basemaps (Aourednik, 2022) ·
            Lydian Kingdom shown as circle (no polygon available)
        </p>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — THE PROOF
# ══════════════════════════════════════════════════════════════════════════════
else:  # proof page
    col_back, col_map, col_sp = st.columns([1, 1, 3])
    with col_back:
        if st.button("⚔  ← The Story"):
            st.session_state.page = 'story'
            st.rerun()
    with col_map:
        if st.button("🌍  History in Motion"):
            st.session_state.page = 'map'
            st.rerun()

    st.markdown("""
    <div style="padding:2rem 0 1.5rem 0;border-bottom:1px solid var(--border);">
        <p style="font-family:Cinzel,serif;font-size:0.7rem;letter-spacing:0.3em;
                  color:var(--gold-dim);">THE PROOF</p>
        <h1 style="font-family:Cinzel,serif;font-size:2.2rem;font-weight:900;
                   color:var(--parchment);margin:0.3rem 0;">
            Behind the Data
        </h1>
        <p style="font-family:EB Garamond,serif;font-size:1.1rem;color:var(--text-dim);">
            Full analytical results for each of the five pillars.
            Every number that appears in the narrative is sourced here.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Pillar 1 — Clustering",
        "Pillar 2 — Geography",
        "Pillar 3 — Stressors",
        "Pillar 4 — Network",
        "Pillar 5 — Resilience",
        "The Data"
    ])

    # ── TAB 1: TEMPORAL CLUSTERING ────────────────────────────────────────────
    with tab1:
        st.markdown("### Temporal Clustering — Pillar 1")
        st.markdown(f"""
        **Method**: Kernel Density Estimation (bandwidth = 120 years) on collapse
        midpoints, followed by 10,000-iteration permutation test.

        | Metric | Value |
        |---|---|
        | Observed clustering score | **2.029** |
        | Null mean | 1.169 |
        | Null std | 0.129 |
        | Z-score | **6.65** |
        | P-value | **< 0.0001** |
        | Clusters identified | **5** |
        | Civilisations within clusters | **{int(STATS['n_civs']*0.60)} / {STATS['n_civs']} (60%)** |
        """)

        years = ((df['collapse_start_year'] + df['collapse_end_year']) / 2).values
        bw = 120 / years.std()
        kde = gaussian_kde(years, bw_method=bw)
        x_range = np.linspace(-3000, 2010, 4000)
        kde_vals = kde(x_range)

        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                            shared_xaxes=True, vertical_spacing=0.04)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_vals, fill='tozeroy',
            fillcolor='rgba(201,168,76,0.15)',
            line=dict(color='#c9a84c', width=2), name='Collapse density'
        ), row=1, col=1)

        cluster_centres = [-2140, -1110, -263, 793, 1915]
        cluster_labels = ['~2140 BCE', '~1110 BCE', '~263 BCE', '~793 CE', '~1915 CE']
        for cy, cl in zip(cluster_centres, cluster_labels):
            fig.add_vline(x=cy, line_dash='dash', line_color='#8b1a1a',
                          line_width=1.5, row=1, col=1)
            fig.add_annotation(x=cy, y=kde(np.array([cy]))[0] * 1.2,
                                text=cl, font=dict(size=8, color='#c9a84c'),
                                showarrow=False, row=1, col=1)

        for _, row_s in stressors[stressors['severity'] == 'catastrophic'].iterrows():
            fig.add_vrect(x0=row_s['start_year'], x1=row_s['end_year'],
                          fillcolor='rgba(139,26,26,0.08)',
                          layer='below', line_width=0, row=1, col=1)

        for _, row_d in df.iterrows():
            mid = (row_d['collapse_start_year'] + row_d['collapse_end_year']) / 2
            colour = REGION_COLOURS.get(row_d['region'], '#999')
            fig.add_trace(go.Scatter(
                x=[mid, mid], y=[0, 0.3], mode='lines',
                line=dict(color=colour, width=1.5), showlegend=False,
                hovertemplate=f"{row_d['name']}<extra></extra>"
            ), row=2, col=1)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8d5b0', height=450, showlegend=False,
            margin=dict(t=10, b=10)
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', color='#9a8060')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', color='#9a8060')
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: GEOGRAPHY ──────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Geographic Clustering — Pillar 2")
        st.markdown(f"""
        **Method**: Moran's I spatial autocorrelation on collapse timing vs geographic position.

        | Metric | Value | Interpretation |
        |---|---|---|
        | Global Moran's I | **{results['pillar2_global_morans_i']:.3f}** | Positive — overall spatial pattern exists |
        | Global p-value | **{results['pillar2_global_p_value']:.3f}** | Statistically significant |
        | Per-cluster Moran's I | **Negative (all 5)** | Collapses *dispersed* within clusters |
        | Conclusion | **Systemic, not contagion** | Top-down shared pressure, not neighbour spread |

        **Interpretation of negative per-cluster I**: Within each collapse cluster,
        the civilisations falling were more geographically spread than random
        expectation. This is the opposite of a regional contagion pattern.
        """)

        m = folium.Map(location=[25, 15], zoom_start=1,
                       tiles='CartoDB dark_matter')
        for _, row_m in df.iterrows():
            colour = REGION_COLOURS.get(row_m['region'], '#999')
            c_start = int(row_m['collapse_start_year'])
            c_end = int(row_m['collapse_end_year'])
            popup = f"""
            <b>{row_m['name']}</b><br>
            Collapse: {abs(c_start)} {'BCE' if c_start < 0 else 'CE'} –
            {abs(c_end)} {'BCE' if c_end < 0 else 'CE'}<br>
            Trigger: {row_m['primary_collapse_trigger']}<br>
            Region: {row_m['region']}
            """
            folium.CircleMarker(
                location=[row_m['latitude'], row_m['longitude']],
                radius=max(5, min(18, row_m['peak_territory_km2'] / 2000000 * 3)),
                color=colour, fill=True, fill_color=colour, fill_opacity=0.7,
                popup=folium.Popup(popup, max_width=200),
                tooltip=row_m['name']
            ).add_to(m)
        st_folium(m, height=420, use_container_width=True)

    # ── TAB 3: STRESSORS ──────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Contextual Overlay — Pillar 3")
        st.markdown(f"""
        **Method**: For each civilisation, documented stressor events (climate, pandemic,
        migration, war) were matched to collapse windows. Alignment scores computed
        per cluster. Statistical tests on stressor vs no-stressor groups.

        **Overall coverage**: {int(STATS['n_civs']*0.69)} / {STATS['n_civs']} civilisations ({int(69)}%) had at least one documented
        stressor during their collapse window.

        **Statistical tests** (all non-significant at p < 0.05):
        - Collapse duration with vs without stressor: p = 0.173
        - Conquest likelihood during catastrophic stressor: p = 0.212
        - Lifespan with vs without catastrophic stressor: p = 0.493
        - Stressor count vs collapse duration correlation: r = +0.41

        **Key interpretation**: Stressors predict the *timing* of collapses but not
        their *speed* or *type*. They are triggers on pre-existing vulnerabilities.
        """)

        st.markdown(f"""
            <div class="finding-card">
                <p class="finding-headline">Overall Alignment</p>
                <p class="finding-body">
                    Mean stressor alignment across all five collapse clusters:
                    <strong style="color:var(--gold);">{STATS['alignment_pct']}</strong>
                    — stressors were active during collapse in
                    {STATS['alignment_pct']} of cluster windows.
                    Stressors predict the <em>timing</em> of collapse,
                    not its speed or type.
                </p>
            </div>
            """, unsafe_allow_html=True)

        fig = go.Figure()
        colours_align = ['#c9a84c' if s == 1.0 else '#7a5f2a'
                         for s in alignment_df['alignment_score']]
        fig.add_trace(go.Bar(
            x=alignment_df['cluster_name'],
            y=alignment_df['alignment_score'] * 100,
            marker_color=colours_align,
            text=[f"{s:.0%}" for s in alignment_df['alignment_score']],
            textposition='outside',
            textfont=dict(color='#e8d5b0')
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8d5b0', height=380,
            yaxis=dict(title='Stressor alignment score (%)',
                       gridcolor='rgba(255,255,255,0.04)', color='#9a8060',
                       range=[0, 115]),
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#9a8060'),
            title='Cluster Stressor Alignment Scores'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Stressor events dataset**")
        st.dataframe(
            stressors[['event_name', 'event_type', 'start_year',
                        'end_year', 'severity', 'geographic_scope']].rename(columns={
                'event_name': 'Event', 'event_type': 'Type',
                'start_year': 'Start', 'end_year': 'End',
                'severity': 'Severity', 'geographic_scope': 'Scope'
            }),
            use_container_width=True, hide_index=True
        )

    # ── TAB 4: NETWORK ────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Collapse Interaction Network — Pillar 4")
        st.markdown(f"""
        **Method**: Directed graph (NetworkX) encoding conquest, pressure, and
        succession relationships between all {STATS['n_civs']} civilisations.

        | Metric | Value |
        |---|---|
        | Nodes | **{STATS['n_civs']}** |
        | Edges | **{results['pillar4_n_edges']}** |
        | Conquest edges | **{results['pillar4_conquest_edges']}** |
        | Pressure edges | **{results['pillar4_pressure_edges']}** |
        | Succession edges | **{results['pillar4_succession_edges']}** |
        | Longest causal chain | **7 civilisations · 1,884 years** |
        | Chains of ≥ 4 civilisations | **36** |

        **Longest chain**:
        Neo-Babylonian → Achaemenid → Seleucid → Parthian → Sassanid → Umayyad → Abbasid
        """)

        # ── Interactive network chart ──────────────────────────────────────
        st.markdown("**Interactive Collapse Interaction Network**")
        st.caption("Node size = PageRank influence · Colour = region · Hover for details")

        # Build node set — all {STATS['n_civs']} civilisation names exactly
        all_names = set(df['name'].tolist())

        G_vis = nx.DiGraph()
        for _, row_n in centrality_df.iterrows():
            G_vis.add_node(
                row_n['name'],
                pagerank=row_n['pagerank'],
                betweenness=row_n['betweenness'],
                region=row_n['region'],
                out_degree=row_n['out_degree'],
                in_degree=row_n['in_degree']
            )

        # Add edges — exact name match only, longest match wins to avoid substrings
        for _, row_e in df.iterrows():
            src = row_e['name']
            for field, etype in [('conquered_by', 'conquest'),
                                  ('pressured_by', 'pressure'),
                                  ('succeeded_by', 'succession')]:
                val = row_e[field] if field in row_e.index else ''
                if not pd.notna(val) or not str(val).strip():
                    continue
                for raw_target in str(val).split('|'):
                    raw_target = raw_target.strip()
                    if not raw_target:
                        continue
                    # Exact match first
                    if raw_target in all_names:
                        best = raw_target
                    else:
                        # Longest substring match as fallback
                        candidates = [n for n in all_names
                                      if raw_target.lower() == n.lower()]
                        if not candidates:
                            candidates = [n for n in all_names
                                          if raw_target.lower() in n.lower()
                                          and len(raw_target) >= len(n) * 0.6]
                        best = max(candidates, key=len) if candidates else None

                    if best and best in G_vis.nodes():
                        if etype == 'succession':
                            G_vis.add_edge(src, best, edge_type=etype)
                        else:
                            G_vis.add_edge(best, src, edge_type=etype)

        # Layout
        pos = nx.spring_layout(G_vis, seed=42, k=2.8)

        # Edge traces by type
        EDGE_COLS = {'conquest': '#c0392b', 'pressure': '#F28E2B', 'succession': '#59A14F'}
        edge_traces = []
        for etype, ecol in EDGE_COLS.items():
            ex, ey = [], []
            for u, v, d in G_vis.edges(data=True):
                if d.get('edge_type') == etype and u in pos and v in pos:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    ex += [x0, x1, None]
                    ey += [y0, y1, None]
            if ex:
                edge_traces.append(go.Scatter(
                    x=ex, y=ey, mode='lines',
                    line=dict(width=0.8, color=ecol),
                    opacity=0.5, name=etype,
                    hoverinfo='none'
                ))

        # Node trace
        node_x, node_y, node_text, node_hover, node_size, node_colour = [], [], [], [], [], []
        for node in G_vis.nodes():
            if node not in pos:
                continue
            x, y = pos[node]
            nd = G_vis.nodes[node]
            region = nd.get('region', 'Unknown')
            pr = nd.get('pagerank', 0.02)
            node_x.append(x)
            node_y.append(y)
            short = node.replace(' Empire', '').replace(' Dynasty', '').replace(' Caliphate', '')
            node_text.append(short if pr > 0.03 else '')
            node_hover.append(
                f"<b>{node}</b><br>"
                f"Region: {region}<br>"
                f"PageRank: {pr:.4f}<br>"
                f"Betweenness: {nd.get('betweenness', 0):.4f}<br>"
                f"Out-degree: {nd.get('out_degree', 0)}<br>"
                f"In-degree: {nd.get('in_degree', 0)}"
            )
            node_size.append(max(8, pr * 600))
            node_colour.append(REGION_COLOURS.get(region, '#999999'))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            textfont=dict(size=7, color='#e8d5b0', family='Cinzel'),
            hovertext=node_hover,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_colour,
                opacity=0.9,
                line=dict(color='rgba(0,0,0,0.4)', width=0.5)
            ),
            showlegend=False
        )

        fig_net = go.Figure(data=edge_traces + [node_trace])
        fig_net.update_layout(
            paper_bgcolor='rgba(14,11,7,0.95)',
            plot_bgcolor='rgba(14,11,7,0.95)',
            font_color='#e8d5b0',
            height=580,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                title=dict(text='Edge type', font=dict(color='#c9a84c')),
                bgcolor='rgba(22,16,9,0.9)',
                bordercolor='#3a2e1a', borderwidth=1,
                font=dict(size=9, color='#e8d5b0')
            )
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Tables below the chart
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top 10 — Historical Bridges (Betweenness)**")
            display_cent = centrality_df.nlargest(10, 'betweenness')[
                ['name', 'betweenness', 'out_degree', 'region']
            ].copy()
            display_cent['betweenness'] = display_cent['betweenness'].round(4)
            display_cent.columns = ['Civilisation', 'Betweenness', 'Out-degree', 'Region']
            st.dataframe(display_cent.reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**Top 10 — Most Predatory (Out-degree)**")
            display_out = centrality_df.nlargest(10, 'out_degree')[
                ['name', 'out_degree', 'trigger', 'region']
            ].copy()
            display_out.columns = ['Civilisation', 'Out-degree', 'Trigger', 'Region']
            st.dataframe(display_out.reset_index(drop=True),
                         use_container_width=True, hide_index=True)

    # ── TAB 5: RESILIENCE ─────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Resilience Paradox — Pillar 5")
        st.markdown(f"""
        **Method**: Logistic regression predicting whether a civilisation would exceed
        median lifespan (319 years), using 15 pre-collapse features.
        Leave-One-Out cross-validation appropriate for n={STATS['n_civs']}.

        | Metric | Value |
        |---|---|
        | LOO Accuracy | **{results['pillar5_loo_accuracy']:.1%}** |
        | Majority-class baseline | 50.0% |
        | Train AUC | 0.997 |
        | Correctly classified | **{int(STATS['loo_acc'] * STATS['n_civs'])} / {STATS['n_civs']}** |
        | Top predictive feature | **Rise duration (SHAP = 1.37)** |
        """)

        # SHAP chart
        label_map = {
            'rise_duration_years': 'Rise Duration', 'peak_plateau_years': 'Peak Plateau',
            'n_contemporary_rivals': 'Number of Rivals',
            'founding_era_numeric': 'Founding Era',
            'n_stressors_during_collapse': 'Stressors During Collapse',
            'has_any_catastrophic': 'Faced Catastrophic Stressor',
            'n_pressuring_empires': 'Empires Pressuring',
            'log_peak_population': 'Peak Population (log)',
            'was_conquered': 'Was Directly Conquered',
            'trigger_fragmentation': 'Trigger: Fragmentation',
            'territory_per_capita': 'Territory Per Capita',
            'log_peak_territory': 'Peak Territory (log)',
            'trigger_climate': 'Trigger: Climate',
            'trigger_conquest': 'Trigger: Conquest',
            'trigger_overextension': 'Trigger: Overextension'
        }
        plot_shap = shap_imp.copy()
        plot_shap['label'] = plot_shap['feature'].map(label_map).fillna(plot_shap['feature'])
        plot_shap = plot_shap.sort_values('mean_abs_shap')
        fig = go.Figure(go.Bar(
            x=plot_shap['mean_abs_shap'], y=plot_shap['label'],
            orientation='h',
            marker_color=['#c9a84c' if v > 0.5 else '#7a5f2a' if v > 0.2
                          else '#3a2e1a' for v in plot_shap['mean_abs_shap']]
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8d5b0', height=440,
            xaxis=dict(title='Mean |SHAP value|',
                       gridcolor='rgba(255,255,255,0.04)', color='#9a8060'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#9a8060'),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**All 9 Counterfactual Scenarios**")
        scenario_rows = []
        for s in results['pillar5_all_scenarios']:
            delta = s['change']
            scenario_rows.append({
                'Scenario': s['name'],
                'Base': f"{s['base']:.1%}",
                'Modified': f"{s['modified']:.1%}",
                'Change': f"{'+' if delta > 0 else ''}{delta:.1%}"
            })
        st.dataframe(pd.DataFrame(scenario_rows),
                     use_container_width=True, hide_index=True)

    # ── TAB 6: THE DATA ───────────────────────────────────────────────────────
    with tab6:
        st.markdown("### The Complete Dataset")

        # Region filter
        regions = ['All'] + sorted(df['region'].unique().tolist())
        region_filter = st.selectbox("Filter by region", regions)

        display_df = df.copy() if region_filter == 'All' else df[df['region'] == region_filter].copy()

        display_cols = {
            'name': 'Civilisation', 'region': 'Region',
            'founded_year': 'Founded', 'collapse_start_year': 'Collapse Start',
            'collapse_end_year': 'Collapse End', 'lifespan_years': 'Lifespan (yrs)',
            'collapse_duration_years': 'Collapse Duration (yrs)',
            'primary_collapse_trigger': 'Primary Trigger',
            'peak_population_m': 'Peak Pop (M)',
            'peak_territory_km2': 'Peak Territory (km²)',
        }

        st.dataframe(
            display_df[list(display_cols.keys())].rename(columns=display_cols)
                .sort_values('Founded').reset_index(drop=True),
            use_container_width=True, hide_index=True
        )

        st.markdown("---")
        st.markdown(f"""
        **Methodology notes**

        - Collapse start year = when sustained decline began, not a single battle date
        - BCE years stored as negative integers (e.g. -500 = 500 BCE)
        - Population estimates from Maddison Project, CLIO-INFRA, and historical scholarship
        - Territorial estimates from historical atlas sources and peer-reviewed estimates
        - Collapse triggers are the author's classification based on primary historical consensus;
          secondary triggers exist for most civilisations
        - Stressor events sourced from peer-reviewed climate science and historical epidemiology

        **Data limitations**

        - n={STATS['n_civs']} is meaningful for pattern detection but small for machine learning; 
                        LOO-CV mitigates but does not eliminate overfitting risk
        - Pre-1000 BCE data is significantly sparser and less reliable
        - Pre-Columbian Americas civilisations (Olmec, Chimu, Mixtec, Toltec) have limited network connections — 
           no trans-oceanic contact is encoded before 1492
        """)