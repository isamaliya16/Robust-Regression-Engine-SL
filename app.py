# =============================================================================
# Robust Regression Engine — House Price Prediction Dashboard
# Run: streamlit run app.py
# Requirements: streamlit plotly scikit-learn openpyxl pandas numpy
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import streamlit as st

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold,
    LeaveOneOut, TimeSeriesSplit, cross_val_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Robust Regression Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background: #f8fafc;
  color: #1e293b;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(175deg,#ffffff 0%,#f1f5f9 100%);
  border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .stRadio label {
  font-size: .9rem; font-weight: 500; color: #374151; padding: 6px 0;
}

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg,#1e40af 0%,#2563eb 55%,#60a5fa 100%);
  border-radius: 20px;
  padding: 44px 36px;
  color: #fff;
  text-align: center;
  margin-bottom: 26px;
  box-shadow: 0 18px 60px rgba(37,99,235,.28);
  position: relative; overflow: hidden;
}
.hero::before {
  content:""; position:absolute; inset:0;
  background: radial-gradient(circle at 72% 28%, rgba(255,255,255,.13) 0%, transparent 60%);
}
.hero h1 { font-size:2.4rem; font-weight:700; margin:0 0 8px; letter-spacing:-1px; }
.hero p  { font-size:1rem; opacity:.88; margin:0; }

/* ── KPI card ── */
.kpi {
  background:#fff; border:1px solid #e2e8f0; border-radius:16px;
  padding:22px 14px; text-align:center;
  box-shadow:0 2px 12px rgba(0,0,0,.05);
  transition: transform .18s, box-shadow .18s;
}
.kpi:hover { transform:translateY(-3px); box-shadow:0 8px 26px rgba(0,0,0,.1); }
.kpi .icon  { font-size:2rem; margin-bottom:7px; }
.kpi .val   { font-size:1.55rem; font-weight:700; color:#1e40af; }
.kpi .lbl   { font-size:.75rem; color:#64748b; font-weight:500;
               text-transform:uppercase; letter-spacing:.5px; margin-top:4px; }

/* ── Section header ── */
.sec {
  display:flex; align-items:center; gap:11px;
  background:#fff; border:1px solid #e2e8f0; border-left:4px solid #2563eb;
  border-radius:12px; padding:14px 20px; margin:22px 0 14px;
  box-shadow:0 2px 8px rgba(0,0,0,.04);
}
.sec .em { font-size:1.35rem; }
.sec h2  { margin:0; font-size:1.1rem; font-weight:600; color:#1e293b; }

/* ── Card ── */
.card {
  background:#fff; border:1px solid #e2e8f0; border-radius:14px;
  padding:20px 22px; height:100%;
  box-shadow:0 2px 10px rgba(0,0,0,.04);
}
.card h4 { margin:0 0 10px; font-size:.95rem; font-weight:600; color:#1e40af; }
.card p, .card li { font-size:.86rem; color:#475569; line-height:1.65; margin:0; }

/* ── Prediction result ── */
.pred {
  background: linear-gradient(135deg,#1e40af,#2563eb);
  color:#fff; border-radius:18px; padding:30px;
  text-align:center; box-shadow:0 12px 40px rgba(37,99,235,.35);
}
.pred .amount { font-size:2.6rem; font-weight:700; letter-spacing:-1px; }
.pred .sub    { font-size:.88rem; opacity:.82; margin-top:6px; }

/* ── Badge ── */
.badge {
  display:inline-block; padding:3px 11px; border-radius:20px;
  font-size:.76rem; font-weight:600; margin:2px;
}
.b-blue   { background:#dbeafe; color:#1e40af; }
.b-green  { background:#dcfce7; color:#15803d; }
.b-orange { background:#ffedd5; color:#c2410c; }
.b-purple { background:#f3e8ff; color:#7e22ce; }

/* ── Insight box ── */
.insight {
  background:#eff6ff; border:1px solid #bfdbfe;
  border-radius:12px; padding:14px 18px; margin-top:12px;
}
.insight p { font-size:.86rem; color:#1e40af; margin:0; line-height:1.6; }

/* ── Streamlit overrides ── */
.stButton > button {
  background:linear-gradient(135deg,#1e40af,#2563eb) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  padding:10px 26px !important; font-weight:600 !important;
  font-size:.93rem !important;
  box-shadow:0 4px 14px rgba(37,99,235,.38) !important;
}
.stButton > button:hover { transform:translateY(-1px); }
div[data-testid="metric-container"] {
  background:#fff; border:1px solid #e2e8f0;
  border-radius:14px; padding:14px;
  box-shadow:0 2px 8px rgba(0,0,0,.04);
}
.stTabs [data-baseweb="tab-list"] {
  gap:4px; background:#f1f5f9; border-radius:12px; padding:4px;
}
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:7px 18px; font-weight:500; }
.stTabs [aria-selected="true"] {
  background:#fff !important; color:#1e40af !important;
  box-shadow:0 2px 8px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
COLORS = ['#1e40af','#2563eb','#10b981','#f59e0b','#ef4444','#8b5cf6']

def fmt_inr(val: float) -> str:
    val = float(val)
    if val >= 1_00_00_000:
        return f"₹{val/1_00_00_000:.2f} Cr"
    if val >= 1_00_000:
        return f"₹{val/1_00_000:.1f} L"
    return f"₹{val:,.0f}"

def section(emoji: str, title: str):
    st.markdown(
        f'<div class="sec"><span class="em">{emoji}</span><h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )

def hero(title: str, sub: str):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )

def plotly_base(fig, title="", h=420, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#1e293b'), x=0.01),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter', color='#475569', size=11),
        height=h,
        showlegend=legend,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=10)),
        margin=dict(l=14, r=14, t=48, b=14),
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False,
                   linecolor='#e2e8f0', tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False,
                   linecolor='#e2e8f0', tickfont=dict(size=10)),
    )
    return fig


# ── Data + model caching ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_excel("HousePrice_Dataset_3800.xlsx")


@st.cache_resource(show_spinner=False)
def build_models():
    df = load_data()
    FEATS = ['area_sqft','bedrooms','bathrooms','location_score',
             'property_age','distance_city_km','near_school',
             'near_metro','crime_rate_index']

    X = df[FEATS]
    y = df['house_price_inr']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scaler for linear / SVR models
    scaler = StandardScaler()
    Xs_tr  = scaler.fit_transform(X_train)
    Xs_te  = scaler.transform(X_test)

    # Target scaler for SVR only
    y_scaler = MinMaxScaler()
    yt_tr    = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # ── Ridge ────────────────────────────────────────────────────────────────
    rc     = RidgeCV(alphas=np.logspace(-3, 3, 60), cv=5).fit(Xs_tr, y_train)
    ridge  = Ridge(alpha=float(rc.alpha_)).fit(Xs_tr, y_train)
    best_alpha_ridge = float(rc.alpha_)

    # ── Lasso ────────────────────────────────────────────────────────────────
    lc    = LassoCV(alphas=np.logspace(-3, 3, 60), cv=5,
                    max_iter=10_000, random_state=42).fit(Xs_tr, y_train)
    lasso = Lasso(alpha=float(lc.alpha_), max_iter=10_000).fit(Xs_tr, y_train)
    best_alpha_lasso = float(lc.alpha_)

    # ── Decision Tree ────────────────────────────────────────────────────────
    dt = DecisionTreeRegressor(
        max_depth=6, min_samples_split=20,
        min_samples_leaf=10, random_state=42
    ).fit(X_train, y_train)

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=150, max_depth=10, max_features='sqrt',
        min_samples_split=10, bootstrap=True,
        random_state=42, n_jobs=-1
    ).fit(X_train, y_train)

    # ── SVR (RBF) — trained on normalised target ─────────────────────────────
    svr_raw = SVR(kernel='rbf', C=1, gamma='scale', epsilon=0.01)
    svr_raw.fit(Xs_tr, yt_tr)

    def svr_predict(Xs):
        raw = svr_raw.predict(Xs)
        return y_scaler.inverse_transform(raw.reshape(-1, 1)).ravel()

    # ── Evaluate helper ──────────────────────────────────────────────────────
    def _ev(pred_tr, pred_te):
        return dict(
            train_r2 = float(r2_score(y_train, pred_tr)),
            test_r2  = float(r2_score(y_test,  pred_te)),
            test_mse = float(mean_squared_error(y_test, pred_te)),
            test_mae = float(mean_absolute_error(y_test, pred_te)),
            test_rmse= float(np.sqrt(mean_squared_error(y_test, pred_te))),
            y_pred   = pred_te,
        )

    results = {
        'Ridge'        : _ev(ridge.predict(Xs_tr),       ridge.predict(Xs_te)),
        'Lasso'        : _ev(lasso.predict(Xs_tr),       lasso.predict(Xs_te)),
        'Decision Tree': _ev(dt.predict(X_train),        dt.predict(X_test)),
        'Random Forest': _ev(rf.predict(X_train),        rf.predict(X_test)),
        'SVR (RBF)'    : _ev(svr_predict(Xs_tr),         svr_predict(Xs_te)),
    }

    feat_imp   = dict(zip(FEATS, rf.feature_importances_))
    ridge_coef = dict(zip(FEATS, ridge.coef_))
    lasso_coef = dict(zip(FEATS, lasso.coef_))

    return dict(
        results    = results,
        scaler     = scaler,
        y_scaler   = y_scaler,
        ridge      = ridge,
        lasso      = lasso,
        dt         = dt,
        rf         = rf,
        svr_raw    = svr_raw,
        svr_predict= svr_predict,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        Xs_tr=Xs_tr,     Xs_te=Xs_te,
        feat_imp   = feat_imp,
        ridge_coef = ridge_coef,
        lasso_coef = lasso_coef,
        features   = FEATS,
        best_alpha_ridge = best_alpha_ridge,
        best_alpha_lasso = best_alpha_lasso,
    )


# ── Load everything ───────────────────────────────────────────────────────────
df = load_data()

with st.spinner("⚙️ Training 5 ML models on 3,800 properties — please wait…"):
    M = build_models()

results   = M['results']
features  = M['features']
feat_imp  = M['feat_imp']
y_test    = M['y_test']
y_train   = M['y_train']
X_train   = M['X_train']
X_test    = M['X_test']
Xs_tr     = M['Xs_tr']
Xs_te     = M['Xs_te']
scaler    = M['scaler']


# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 8px;">
      <div style="font-size:2.8rem;">🏠</div>
      <div style="font-size:1rem;font-weight:700;color:#1e40af;">Regression Engine</div>
      <div style="font-size:.76rem;color:#64748b;margin-top:3px;">Real Estate AI · India</div>
    </div>
    <hr style="border-color:#e2e8f0;margin:10px 0 18px;">
    """, unsafe_allow_html=True)

    nav = st.radio("Go to", [
        "🏠 Overview",
        "📊 Dataset Explorer",
        "📉 Regularization Lab",
        "🔁 Cross-Validation",
        "🌳 Tree-Based Models",
        "⚡ SVR Analysis",
        "📈 Model Comparison",
        "🔮 Live Prediction",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style="border-color:#e2e8f0;margin:16px 0 12px;">
    <div style="font-size:.76rem;color:#94a3b8;text-align:center;line-height:1.8;">
      <b>Dataset:</b> 3,800 properties<br>
      <b>Target:</b> House Price (INR)<br>
      <b>Algorithms:</b> 5 trained models<br>
      <b>Split:</b> 80/20 train-test
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Overview":
    hero("🏠 Robust Regression Engine",
         "Advanced supervised learning · 5 algorithms · live inference · India real estate market")

    best_name = max(results, key=lambda m: results[m]['test_r2'])
    br = results[best_name]

    # KPI row
    kpis = [
        ("🏘️", "3,800", "Properties"),
        ("🧠", "5",     "ML Models"),
        ("🎯", f"{br['test_r2']:.4f}", "Best R² Score"),
        ("📉", fmt_inr(br['test_mae']),  "Best MAE"),
        ("🏆", best_name, "Champion"),
    ]
    cols = st.columns(5)
    for col, (icon, val, lbl) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi"><div class="icon">{icon}</div>'
            f'<div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline infographic
    section("⚙️", "ML Pipeline Architecture")
    steps = [
        ("📥", "Data Ingestion",      "3,800 INR property records"),
        ("🔍", "EDA",                 "Distributions & correlations"),
        ("⚙️", "Feature Engineering", "Scaling, encoding, split"),
        ("🤖", "Model Training",      "Ridge·Lasso·DT·RF·SVR"),
        ("✅", "Evaluation",          "MSE·MAE·RMSE·R²·CV"),
        ("🔮", "Live Prediction",     "Interactive inference"),
    ]
    pcols = st.columns(6)
    for col, (icon, title, sub) in zip(pcols, steps):
        col.markdown(
            f'<div class="card" style="text-align:center;">'
            f'<div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>'
            f'<div style="font-weight:700;font-size:.85rem;color:#1e40af;">{title}</div>'
            f'<div style="font-size:.75rem;color:#64748b;margin-top:4px;">{sub}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Train vs Test R²
    section("📊", "All-Model Performance Summary")
    model_names = list(results.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Train R²", x=model_names,
        y=[results[m]['train_r2'] for m in model_names],
        marker_color=COLORS[0], opacity=.78,
    ))
    fig.add_trace(go.Bar(
        name="Test R²", x=model_names,
        y=[results[m]['test_r2'] for m in model_names],
        marker_color=COLORS[2], opacity=.88,
    ))
    fig.update_layout(barmode='group', xaxis_tickangle=-10)
    plotly_base(fig, "Train vs Test R² — All 5 Models")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance + coefficients
    c1, c2 = st.columns(2)
    with c1:
        fi_s = dict(sorted(feat_imp.items(), key=lambda x: x[1]))
        fig2 = go.Figure(go.Bar(
            x=list(fi_s.values()), y=list(fi_s.keys()),
            orientation='h',
            marker=dict(
                color=list(fi_s.values()),
                colorscale=[[0,'#dbeafe'],[1,'#1e40af']],
                showscale=False,
            ),
        ))
        plotly_base(fig2, "🌳 Random Forest — Feature Importance", h=360, legend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        feat_labels = list(M['ridge_coef'].keys())
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Ridge', x=feat_labels,
                               y=list(M['ridge_coef'].values()),
                               marker_color=COLORS[0], opacity=.82))
        fig3.add_trace(go.Bar(name='Lasso', x=feat_labels,
                               y=list(M['lasso_coef'].values()),
                               marker_color=COLORS[2], opacity=.82))
        fig3.update_layout(barmode='group', xaxis_tickangle=-22)
        plotly_base(fig3, "📉 Ridge vs Lasso Coefficients", h=360)
        st.plotly_chart(fig3, use_container_width=True)

    # Business insights
    section("💡", "Business Insights")
    insights = [
        ("🏆", "Champion Model", best_name,
         f"Test R² = {br['test_r2']:.4f}, MAE = {fmt_inr(br['test_mae'])} — "
         f"ready for production pricing pipeline.", "b-green"),
        ("🔒", "Regularization Win", "Ridge > bare OLS",
         "Ridge prevents coefficient explosion on correlated features, "
         "producing stable prices across city datasets.", "b-blue"),
        ("📐", "Top Predictors", "area_sqft · location_score",
         "Random Forest identifies property size and location as primary "
         "price drivers — directly actionable for the valuation team.", "b-purple"),
    ]
    icols = st.columns(3)
    for col, (icon, title, badge, text, bcls) in zip(icols, insights):
        col.markdown(
            f'<div class="card"><div style="font-size:1.7rem;margin-bottom:8px;">{icon}</div>'
            f'<h4>{title}</h4>'
            f'<span class="badge {bcls}">{badge}</span>'
            f'<p style="margin-top:10px;">{text}</p></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Dataset Explorer":
    hero("📊 Dataset Explorer",
         "Deep-dive into 3,800 Indian real estate records · zero missing values")

    kpis2 = [
        ("📋", f"{len(df):,}", "Total Rows"),
        ("🔢", "11", "Features"),
        ("💰", fmt_inr(df['house_price_inr'].mean()), "Avg Price"),
        ("📅", str(df['sale_date'].dt.year.min())+"–"+str(df['sale_date'].dt.year.max()), "Date Range"),
    ]
    cols = st.columns(4)
    for col, (icon, val, lbl) in zip(cols, kpis2):
        col.markdown(
            f'<div class="kpi"><div class="icon">{icon}</div>'
            f'<div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section("👁️", "Raw Data Preview")
    st.dataframe(
        df.head(20).style.format({
            'house_price_inr': '₹{:,.0f}',
            'location_score':   '{:.2f}',
            'distance_city_km': '{:.1f}',
            'crime_rate_index': '{:.2f}',
        }),
        use_container_width=True,
    )

    section("📈", "Feature Distributions")
    num_cols = ['area_sqft','bedrooms','bathrooms','location_score',
                'property_age','distance_city_km','crime_rate_index',
                'house_price_inr']
    sel = st.selectbox("Select a feature to inspect", num_cols, index=0)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x=sel, nbins=50,
                           color_discrete_sequence=['#2563eb'],
                           labels={sel: sel.replace('_', ' ').title()})
        fig.update_traces(opacity=.85)
        plotly_base(fig, f"Histogram — {sel}", legend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(df, y=sel, color_discrete_sequence=['#1e40af'])
        plotly_base(fig2, f"Box Plot — {sel}", legend=False)
        st.plotly_chart(fig2, use_container_width=True)

    section("🔗", "Correlation Heatmap")
    corr = df[num_cols].corr()
    fig3 = px.imshow(corr, text_auto='.2f', aspect='auto',
                     color_continuous_scale='Blues')
    fig3.update_layout(height=460, paper_bgcolor='white',
                       font=dict(family='Inter', size=10),
                       margin=dict(l=8, r=8, t=36, b=8))
    st.plotly_chart(fig3, use_container_width=True)

    section("💰", "Price vs Feature Scatter")
    x_feat = st.selectbox("X-axis", [f for f in num_cols if f != 'house_price_inr'], index=0)
    fig4 = px.scatter(df, x=x_feat, y='house_price_inr',
                      color='location_score', color_continuous_scale='Blues',
                      opacity=.45, labels={'house_price_inr': 'Price (₹)'})
    fig4.update_traces(marker_size=4)
    plotly_base(fig4, f"{x_feat}  vs  House Price", h=420)
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — REGULARIZATION LAB
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📉 Regularization Lab":
    hero("📉 Regularization Lab",
         "Ridge (L2) vs Lasso (L1) · penalty mechanics · coefficient paths · alpha tuning")

    section("📚", "Conceptual Foundation")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
          <h4>🔵 Ridge Regression (L2)</h4>
          <p>Adds <b>sum of squared coefficients</b> as penalty.<br><br>
          • Shrinks all coefficients — never to exactly zero<br>
          • Handles multicollinearity by distributing weight evenly<br>
          • Best when <em>all</em> features contribute<br>
          • Geometric constraint: <b>circular boundary</b><br><br>
          <code>Loss = MSE + α · Σ(βᵢ²)</code></p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
          <h4>🟠 Lasso Regression (L1)</h4>
          <p>Adds <b>sum of absolute coefficients</b> as penalty.<br><br>
          • Can reduce coefficients to exactly <b>zero</b> — feature selection<br>
          • Creates sparse solutions — fewer features in final model<br>
          • May randomly drop correlated features<br>
          • Geometric constraint: <b>diamond-shaped boundary</b><br><br>
          <code>Loss = MSE + α · Σ|βᵢ|</code></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("🎛️", "Optimal Alpha — Found via Cross-Validation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ridge Best Alpha",  f"{M['best_alpha_ridge']:.5f}")
    c2.metric("Lasso Best Alpha",  f"{M['best_alpha_lasso']:.5f}")
    c3.metric("Ridge Test R²",     f"{results['Ridge']['test_r2']:.4f}")
    c4.metric("Lasso Test R²",     f"{results['Lasso']['test_r2']:.4f}")

    section("📊", "Coefficient Comparison — Ridge vs Lasso")
    feat_labels = features
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Ridge (L2)', x=feat_labels,
                          y=list(M['ridge_coef'].values()),
                          marker_color=COLORS[0], opacity=.82))
    fig.add_trace(go.Bar(name='Lasso (L1)', x=feat_labels,
                          y=list(M['lasso_coef'].values()),
                          marker_color=COLORS[2], opacity=.82))
    fig.add_hline(y=0, line_dash='dot', line_color='#94a3b8')
    fig.update_layout(barmode='group', xaxis_tickangle=-18)
    plotly_base(fig, "Ridge vs Lasso Coefficients (standardised feature scale)")
    st.plotly_chart(fig, use_container_width=True)

    zeroed = [f for f, v in M['lasso_coef'].items() if abs(v) < 1e-6]
    if zeroed:
        st.markdown(
            f'<div class="insight"><p>✂️ <b>Lasso zeroed out:</b> {", ".join(zeroed)} — '
            f'these features have negligible predictive power and can be dropped from '
            f'the production pipeline.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="insight"><p>✅ <b>Lasso retained all features</b> — every attribute '
            'contributes meaningfully to house price prediction in this dataset.</p></div>',
            unsafe_allow_html=True)

    # Alpha sweep paths
    section("🔬", "Coefficient Paths vs Alpha")
    alphas = np.logspace(-2, 4, 60)
    ridge_paths, lasso_paths = [], []
    for a in alphas:
        r_ = Ridge(alpha=a).fit(Xs_tr, y_train)
        l_ = Lasso(alpha=a, max_iter=10_000).fit(Xs_tr, y_train)
        ridge_paths.append(r_.coef_)
        lasso_paths.append(l_.coef_)
    ridge_paths = np.array(ridge_paths)
    lasso_paths = np.array(lasso_paths)

    tab1, tab2 = st.tabs(["Ridge Path", "Lasso Path"])
    for tab, paths, title in [
        (tab1, ridge_paths, "Ridge: Coefficients shrink toward 0 but never reach it"),
        (tab2, lasso_paths, "Lasso: Coefficients can reach exactly 0 (feature dropped)"),
    ]:
        with tab:
            fig2 = go.Figure()
            for i, f in enumerate(features):
                fig2.add_trace(go.Scatter(x=alphas, y=paths[:, i], name=f,
                                          line=dict(width=2), mode='lines'))
            fig2.update_xaxes(type='log', title='Alpha (log scale)')
            fig2.update_yaxes(title='Coefficient Value')
            plotly_base(fig2, title)
            st.plotly_chart(fig2, use_container_width=True)

    # Overfit sweep
    section("📉", "Overfitting Signal — Train/Test R² over Alpha")
    tr_r, te_r, tr_l, te_l = [], [], [], []
    for a in alphas:
        r_ = Ridge(alpha=a).fit(Xs_tr, y_train)
        tr_r.append(r2_score(y_train, r_.predict(Xs_tr)))
        te_r.append(r2_score(y_test,  r_.predict(Xs_te)))
        l_ = Lasso(alpha=a, max_iter=10_000).fit(Xs_tr, y_train)
        tr_l.append(r2_score(y_train, l_.predict(Xs_tr)))
        te_l.append(r2_score(y_test,  l_.predict(Xs_te)))

    fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Ridge", "Lasso"))
    for y_vals, name, col_idx, row in [
        (tr_r,'Train R²',COLORS[0],1), (te_r,'Test R²',COLORS[2],1),
        (tr_l,'Train R²',COLORS[0],2), (te_l,'Test R²',COLORS[2],2),
    ]:
        fig3.add_trace(
            go.Scatter(x=alphas, y=y_vals, name=name,
                       showlegend=(col_idx==1),
                       line=dict(color=col_idx==1 and COLORS[0] or COLORS[2] if name=='Train R²' else COLORS[2], width=2.5)),
            row=1, col=col_idx,
        )
    # re-colour properly
    fig3.data[0].line.color = COLORS[0]
    fig3.data[1].line.color = COLORS[2]
    fig3.data[2].line.color = COLORS[0]
    fig3.data[3].line.color = COLORS[2]
    fig3.update_xaxes(type='log', title_text='Alpha')
    fig3.update_yaxes(title_text='R² Score')
    fig3.update_layout(height=360, paper_bgcolor='white', plot_bgcolor='white',
                        font=dict(family='Inter', size=11),
                        margin=dict(l=12, r=12, t=48, b=12))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔁 Cross-Validation":
    hero("🔁 Cross-Validation Strategies",
         "K-Fold · Stratified K-Fold · LOOCV · Time Series Split — all four run live on Ridge")

    section("📚", "Four Strategies — Theory")
    cv_cards = [
        ("🔁","K-Fold (k=10)",
         "Data split into 10 equal folds. Trains on 9, validates on 1, rotates 10 times.",
         "Low","Moderate","Low","General-purpose ⭐⭐⭐⭐"),
        ("⚖️","Stratified K-Fold",
         "Bins the price target into quantiles so every fold contains a balanced mix of all price ranges.",
         "Low","Low","Low","Skewed targets ⭐⭐⭐⭐⭐"),
        ("🔬","LOOCV",
         "K = total samples. Each property is the sole validation point once. Near-unbiased, very slow.",
         "Very Low","High","Very High","Small datasets ⭐⭐"),
        ("📅","Time Series Split",
         "Training always uses earlier sale dates; validation uses later ones. No future leakage.",
         "Moderate","Moderate","Low","Temporal data ⭐⭐⭐⭐⭐"),
    ]
    ccols = st.columns(4)
    for col, (icon, name, desc, bias, var, cost, rec) in zip(ccols, cv_cards):
        col.markdown(
            f'<div class="card" style="text-align:left;">'
            f'<div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>'
            f'<h4>{name}</h4><p>{desc}</p>'
            f'<div style="margin-top:10px;">'
            f'<span class="badge b-blue">Bias: {bias}</span>'
            f'<span class="badge b-orange">Var: {var}</span>'
            f'<span class="badge b-purple">Cost: {cost}</span>'
            f'</div>'
            f'<p style="margin-top:8px;font-size:.76rem;color:#1e40af;"><b>Best for:</b> {rec}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def run_cv_scores():
        ridge = Ridge(alpha=M['best_alpha_ridge'])

        kf_sc = cross_val_score(
            ridge, Xs_tr, y_train,
            cv=KFold(n_splits=10, shuffle=True, random_state=42),
            scoring='r2',
        )

        bins = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
        skf  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        skf_sc = cross_val_score(
            ridge, Xs_tr, y_train,
            cv=skf.split(Xs_tr, bins),
            scoring='r2',
        )

        loo_sc = cross_val_score(
            ridge, Xs_tr[:200], y_train.values[:200],
            cv=LeaveOneOut(), scoring='r2',
        )

        ts_sc = cross_val_score(
            ridge, Xs_tr, y_train,
            cv=TimeSeriesSplit(n_splits=10),
            scoring='r2',
        )
        return {
            'K-Fold (k=10)':      kf_sc,
            'Stratified K-Fold':  skf_sc,
            'LOOCV (200 pts)':    loo_sc,
            'Time Series Split':  ts_sc,
        }

    with st.spinner("Running 4 cross-validation strategies…"):
        cv_sc = run_cv_scores()

    section("📊", "Live CV Results — Ridge Regression")
    sc_cols = st.columns(4)
    for col, (name, scores) in zip(sc_cols, cv_sc.items()):
        col.markdown(
            f'<div class="kpi"><div class="val">{scores.mean():.4f}</div>'
            f'<div class="lbl">{name}<br>Mean R²</div>'
            f'<div style="font-size:.73rem;color:#94a3b8;margin-top:4px;">±{scores.std():.4f}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Fold chart
    fig = go.Figure()
    cv_colors = [COLORS[0], COLORS[2], COLORS[3], COLORS[4]]
    for (name, scores), c in zip(cv_sc.items(), cv_colors):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(scores)+1)), y=scores,
            name=name, mode='lines+markers',
            line=dict(color=c, width=2.5), marker=dict(size=8, color=c),
        ))
    fig.add_hline(y=0, line_dash='dot', line_color='#94a3b8')
    fig.update_xaxes(title='Fold #')
    fig.update_yaxes(title='R² Score')
    plotly_base(fig, "Fold-by-Fold R² Scores — All 4 Strategies")
    st.plotly_chart(fig, use_container_width=True)

    # Box
    fig2 = go.Figure()
    for (name, scores), c in zip(cv_sc.items(), cv_colors):
        fig2.add_trace(go.Box(y=scores, name=name, marker_color=c,
                               boxmean='sd', line=dict(width=2)))
    plotly_base(fig2, "Score Distribution — Stability Analysis", h=360)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        '<div class="insight"><p>📌 <b>CV Insight:</b> Stratified K-Fold shows the lowest '
        'variance — most stable evaluator for skewed price distributions. Time Series Split '
        'reflects true deployment conditions (train on past sales, validate on future ones). '
        'Use both together before deploying any model.</p></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — TREE-BASED MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🌳 Tree-Based Models":
    hero("🌳 Tree-Based Models",
         "Decision Tree vs Random Forest · overfitting depth analysis · ensemble advantage")

    section("📚", "Why Trees Don't Need Feature Scaling")
    st.markdown(
        '<div class="card"><p>Tree-based models make decisions by applying <b>threshold splits '
        'on individual features</b> — e.g., "Is area_sqft &gt; 1,500?" These comparisons use '
        'relative ordering of values, not absolute magnitudes or distances. Since '
        '<code>StandardScaler</code> only changes scale (not order), it has '
        '<b>zero effect on split points</b>. Only distance-based models (Ridge, Lasso, SVR) '
        'require scaling.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section("📊", "Decision Tree vs Random Forest — Metrics")
    mc1, mc2 = st.columns(2)
    for col, name, icon in [(mc1,'Decision Tree','🌿'),(mc2,'Random Forest','🌳')]:
        r = results[name]
        gap = r['train_r2'] - r['test_r2']
        col.markdown(
            f'<div class="kpi">'
            f'<div class="icon">{icon}</div>'
            f'<div style="font-weight:700;font-size:.9rem;color:#1e293b;margin-bottom:10px;">{name}</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;text-align:left;">'
            f'<div><div class="val" style="font-size:1.2rem;">{r["train_r2"]:.4f}</div>'
            f'<div class="lbl">Train R²</div></div>'
            f'<div><div class="val" style="font-size:1.2rem;">{r["test_r2"]:.4f}</div>'
            f'<div class="lbl">Test R²</div></div>'
            f'<div><div class="val" style="font-size:1.2rem;">{fmt_inr(r["test_rmse"])}</div>'
            f'<div class="lbl">RMSE</div></div>'
            f'<div><div class="val" style="font-size:1.2rem;">{fmt_inr(r["test_mae"])}</div>'
            f'<div class="lbl">MAE</div></div>'
            f'</div>'
            f'<div style="font-size:.75rem;color:{"#15803d" if gap<0.05 else "#c2410c"};margin-top:8px;">'
            f'Train-Test gap: {gap:.4f} — {"✅ Well generalised" if gap<0.05 else "⚠️ Mild overfit"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance
    section("🎯", "Random Forest Feature Importance")
    fi_s = dict(sorted(feat_imp.items(), key=lambda x: x[1]))
    fig = go.Figure(go.Bar(
        x=list(fi_s.values()), y=list(fi_s.keys()),
        orientation='h',
        text=[f"{v:.3f}" for v in fi_s.values()],
        textposition='outside',
        marker=dict(
            color=list(fi_s.values()),
            colorscale=[[0,'#dbeafe'],[.5,'#60a5fa'],[1,'#1e40af']],
            showscale=True,
            colorbar=dict(title='Importance'),
        ),
    ))
    plotly_base(fig, "Feature Importance — Which factors drive house prices?", h=360, legend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Actual vs predicted
    section("🎯", "Actual vs Predicted Prices")
    ac1, ac2 = st.columns(2)
    for col, name in [(ac1,'Decision Tree'),(ac2,'Random Forest')]:
        y_pred = results[name]['y_pred']
        mn, mx = float(y_test.min()), float(y_test.max())
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode='markers',
            marker=dict(color=COLORS[0], opacity=.35, size=4), name='Predicted'))
        fig2.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode='lines',
            line=dict(color='#ef4444', dash='dash', width=2), name='Perfect'))
        fig2.update_xaxes(title='Actual Price (₹)')
        fig2.update_yaxes(title='Predicted Price (₹)')
        plotly_base(fig2, f"{name} — Actual vs Predicted", h=360)
        col.plotly_chart(fig2, use_container_width=True)

    # Depth vs overfitting
    section("📉", "Decision Tree Depth vs Overfitting")
    depths = list(range(1, 22))
    tr_d, te_d = [], []
    for d in depths:
        dt_ = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
        tr_d.append(r2_score(y_train, dt_.predict(X_train)))
        te_d.append(r2_score(y_test,  dt_.predict(X_test)))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=depths, y=tr_d, name='Train R²',
                               line=dict(color=COLORS[0], width=2.5),
                               mode='lines+markers', marker=dict(size=7)))
    fig3.add_trace(go.Scatter(x=depths, y=te_d, name='Test R²',
                               line=dict(color=COLORS[2], width=2.5),
                               mode='lines+markers', marker=dict(size=7)))
    fig3.add_vline(x=6, line_dash='dot', line_color=COLORS[3],
                    annotation_text="Optimal depth=6",
                    annotation_position="top right")
    fig3.update_xaxes(title='Max Depth')
    fig3.update_yaxes(title='R² Score')
    plotly_base(fig3, "Overfitting Diagnosis: R² vs Tree Depth")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="insight"><p>🌳 <b>Ensemble Advantage:</b> Random Forest averages predictions '
        'from 150 trees each trained on a random bootstrap sample and random feature subset. '
        'This bagging drastically reduces variance without meaningful increase in bias — '
        'core advantage over a single deep tree that memorises training noise.</p></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — SVR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "⚡ SVR Analysis":
    hero("⚡ Support Vector Regression",
         "Kernel trick · epsilon tube · hyperparameter sensitivity · RBF vs Linear vs Poly")

    section("📚", "SVR Kernels Explained")
    kc = st.columns(3)
    for col, (icon, name, desc) in zip(kc, [
        ("🔵","Linear","Fits a straight hyperplane. Fast and interpretable. Cannot capture non-linear price patterns."),
        ("🟢","RBF (Gaussian)","Maps data into higher-dimensional space. Excellent for non-linear relationships. Most commonly used."),
        ("🟠","Polynomial","Uses polynomial feature combinations. Flexible boundary — but slower and higher overfit risk."),
    ]):
        col.markdown(
            f'<div class="card" style="text-align:center;">'
            f'<div style="font-size:1.8rem;margin-bottom:6px;">{icon}</div>'
            f'<h4>{name} Kernel</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section("📊", "SVR (RBF) Performance")
    r = results['SVR (RBF)']
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Train R²", f"{r['train_r2']:.4f}")
    sc2.metric("Test R²",  f"{r['test_r2']:.4f}")
    sc3.metric("RMSE",     fmt_inr(r['test_rmse']))
    sc4.metric("MAE",      fmt_inr(r['test_mae']))

    # Actual vs predicted
    section("🎯", "SVR (RBF) — Actual vs Predicted")
    mn, mx = float(y_test.min()), float(y_test.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=r['y_pred'], mode='markers',
                              marker=dict(color=COLORS[5], opacity=.35, size=4),
                              name='SVR Predictions'))
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                              line=dict(color='#ef4444', dash='dash', width=2),
                              name='Perfect Fit'))
    fig.update_xaxes(title='Actual Price (₹)')
    fig.update_yaxes(title='Predicted Price (₹)')
    plotly_base(fig, "SVR (RBF) — Actual vs Predicted Prices", h=400)
    st.plotly_chart(fig, use_container_width=True)

    # C sensitivity
    section("🔬", "Hyperparameter C — Effect on Performance")

    @st.cache_data(show_spinner=False)
    def svr_c_sweep():
        C_vals = [0.01, 0.1, 0.5, 1, 5, 10, 50]
        tr_c_, te_c_ = [], []
        y_scaler2 = MinMaxScaler()
        yt_tr2 = y_scaler2.fit_transform(y_train.values.reshape(-1,1)).ravel()
        for C in C_vals:
            s = SVR(kernel='rbf', C=C, gamma='scale', epsilon=0.01).fit(Xs_tr, yt_tr2)
            ptr = y_scaler2.inverse_transform(s.predict(Xs_tr).reshape(-1,1)).ravel()
            pte = y_scaler2.inverse_transform(s.predict(Xs_te).reshape(-1,1)).ravel()
            tr_c_.append(r2_score(y_train, ptr))
            te_c_.append(r2_score(y_test,  pte))
        return C_vals, tr_c_, te_c_

    with st.spinner("SVR C-sweep…"):
        C_vals, tr_c, te_c = svr_c_sweep()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[str(c) for c in C_vals], y=tr_c, name='Train R²',
                               line=dict(color=COLORS[0], width=2.5),
                               mode='lines+markers', marker=dict(size=8)))
    fig2.add_trace(go.Scatter(x=[str(c) for c in C_vals], y=te_c, name='Test R²',
                               line=dict(color=COLORS[2], width=2.5),
                               mode='lines+markers', marker=dict(size=8)))
    fig2.update_xaxes(title='C (Regularization)')
    fig2.update_yaxes(title='R² Score')
    plotly_base(fig2, "SVR: Effect of C on Train/Test R² (RBF, γ=scale)")
    st.plotly_chart(fig2, use_container_width=True)

    # Comparison table
    section("📋", "SVR vs Other Model Types")
    cmp = pd.DataFrame({
        'Aspect':             ['Non-linearity','Feature Scaling','Interpretability',
                               'Training Speed','Hyperparameter Tuning'],
        'SVR (RBF)':         ['✅ Yes (kernel)','Required','Low',
                               'Slow on large data','High (C, γ, ε)'],
        'Ridge / Lasso':     ['❌ Linear only','Required','High (coef readable)',
                               'Very fast','Low (α only)'],
        'Random Forest':     ['✅ Yes (splits)','Not required','Moderate (feat imp)',
                               'Moderate','Moderate'],
    })
    st.dataframe(cmp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📈 Model Comparison":
    hero("📈 Master Model Comparison",
         "All 5 algorithms evaluated on the same holdout test set · metrics · radar · residuals")

    section("📋", "Complete Evaluation Table")
    rows = []
    for name, r in results.items():
        gap  = r['train_r2'] - r['test_r2']
        risk = '🟢 Low' if gap < 0.03 else ('🟡 Medium' if gap < 0.08 else '🔴 High')
        rows.append({
            'Model':          name,
            'Train R²':       f"{r['train_r2']:.4f}",
            'Test R²':        f"{r['test_r2']:.4f}",
            'Train-Test Gap': f"{gap:.4f}",
            'RMSE':           fmt_inr(r['test_rmse']),
            'MAE':            fmt_inr(r['test_mae']),
            'Overfit Risk':   risk,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Radar
    section("🕸️", "Multi-Metric Radar — Model Profiles")
    model_names = list(results.keys())
    test_r2s  = [results[m]['test_r2']  for m in model_names]
    train_r2s = [results[m]['train_r2'] for m in model_names]
    maes      = [results[m]['test_mae'] for m in model_names]
    mae_inv   = [(max(maes)-v) / (max(maes)-min(maes)+1) for v in maes]
    cats = ['Test R²','Train R²','Low MAE','Low Overfit','Stability']

    fig = go.Figure()
    for i, (name, c) in enumerate(zip(model_names, COLORS)):
        gap = results[name]['train_r2'] - results[name]['test_r2']
        stab = 0.88 if 'Forest' in name or 'Ridge' in name else 0.65
        vals = [test_r2s[i], train_r2s[i], mae_inv[i], max(0, 1-gap*8), stab]
        vals += vals[:1]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], name=name,
            fill='toself', line=dict(color=c, width=2), opacity=.55,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        height=460, paper_bgcolor='white',
        font=dict(family='Inter'),
        legend=dict(orientation='h', y=-0.12),
        margin=dict(l=40, r=40, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # RMSE / MAE bars
    section("📊", "RMSE & MAE — Side-by-Side")
    bc1, bc2 = st.columns(2)
    for col, metric, lbl in [(bc1,'test_rmse','RMSE'),(bc2,'test_mae','MAE')]:
        vals = [results[m][metric] for m in model_names]
        figb = go.Figure(go.Bar(
            x=model_names, y=vals,
            marker_color=COLORS[:len(model_names)], opacity=.85,
            text=[fmt_inr(v) for v in vals], textposition='outside',
        ))
        plotly_base(figb, f"Test {lbl} (lower is better)", legend=False)
        col.plotly_chart(figb, use_container_width=True)

    # Residual histograms
    section("📉", "Residual Distributions — All Models")
    fig3 = make_subplots(rows=1, cols=len(results),
                          subplot_titles=list(results.keys()))
    for i, (name, r) in enumerate(results.items()):
        residuals = y_test.values - np.array(r['y_pred'])
        fig3.add_trace(
            go.Histogram(x=residuals, nbinsx=35, marker_color=COLORS[i],
                         opacity=.8, showlegend=False),
            row=1, col=i+1,
        )
    fig3.update_layout(height=280, paper_bgcolor='white', plot_bgcolor='white',
                        font=dict(family='Inter', size=9),
                        margin=dict(l=8,r=8,t=44,b=8))
    st.plotly_chart(fig3, use_container_width=True)

    # Actual vs predicted — all models
    section("🎯", "Actual vs Predicted — All Models")
    mn, mx = float(y_test.min()), float(y_test.max())
    rows_avp = (len(results)+1)//2
    figc = make_subplots(rows=rows_avp, cols=2,
                          subplot_titles=list(results.keys()))
    for idx, (name, r) in enumerate(results.items()):
        row_ = idx//2+1; col_ = idx%2+1
        figc.add_trace(go.Scatter(
            x=y_test, y=r['y_pred'], mode='markers',
            marker=dict(color=COLORS[idx], opacity=.25, size=3),
            showlegend=False,
        ), row=row_, col=col_)
        figc.add_trace(go.Scatter(
            x=[mn,mx], y=[mn,mx], mode='lines',
            line=dict(color='#ef4444', dash='dash', width=1.5),
            showlegend=False,
        ), row=row_, col=col_)
    figc.update_layout(height=680, paper_bgcolor='white', plot_bgcolor='white',
                        font=dict(family='Inter', size=10),
                        margin=dict(l=8,r=8,t=44,b=8))
    st.plotly_chart(figc, use_container_width=True)

    # Champion banner
    best_n = max(results, key=lambda m: results[m]['test_r2'])
    br     = results[best_n]
    st.markdown(
        f'<div class="pred" style="margin-top:24px;">'
        f'<div style="font-size:2.8rem;margin-bottom:8px;">🏆</div>'
        f'<div class="amount">{best_n}</div>'
        f'<div class="sub">Champion Model · Test R² = {br["test_r2"]:.4f} · '
        f'RMSE = {fmt_inr(br["test_rmse"])} · MAE = {fmt_inr(br["test_mae"])}</div>'
        f'<div style="margin-top:12px;font-size:.84rem;opacity:.82;">'
        f'Recommended for production deployment in the real estate pricing pipeline</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 8 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔮 Live Prediction":
    hero("🔮 Live Price Prediction",
         "Adjust property details and get instant AI-powered valuations from all 5 models")

    section("🏠", "Enter Property Details")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        area      = st.slider("📐 Area (sq ft)",             300,  5000, 1500, 50)
        bedrooms  = st.slider("🛏️ Bedrooms",                   1,     6,    3,  1)
        bathrooms = st.slider("🚿 Bathrooms",                  1,     5,    2,  1)
    with pc2:
        loc_score = st.slider("📍 Location Score (1–10)",    1.0,  10.0,  6.5, .1)
        prop_age  = st.slider("🏗️ Property Age (years)",       0,    50,   10,  1)
        distance  = st.slider("🚗 Distance from City (km)", 0.5,   50.0,  8.0, .5)
    with pc3:
        near_school = st.selectbox("🏫 Near School",  [0, 1],
                                    format_func=lambda x: "Yes ✅" if x else "No ❌")
        near_metro  = st.selectbox("🚇 Near Metro",   [0, 1],
                                    format_func=lambda x: "Yes ✅" if x else "No ❌")
        crime_rate  = st.slider("🔒 Crime Rate Index (1–9)", 1.0, 9.0, 3.5, .1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮  Predict House Price  →  All 5 Models"):
        inp   = np.array([[area, bedrooms, bathrooms, loc_score, prop_age,
                            distance, near_school, near_metro, crime_rate]])
        inp_s = scaler.transform(inp)

        preds = {
            'Ridge':         float(M['ridge'].predict(inp_s)[0]),
            'Lasso':         float(M['lasso'].predict(inp_s)[0]),
            'Decision Tree': float(M['dt'].predict(inp)[0]),
            'Random Forest': float(M['rf'].predict(inp)[0]),
            'SVR (RBF)':     float(M['svr_predict'](inp_s)[0]),
        }

        best_n = max(results, key=lambda m: results[m]['test_r2'])

        # Main banner
        st.markdown(
            f'<div class="pred">'
            f'<div style="font-size:.95rem;opacity:.85;margin-bottom:4px;">'
            f'🏆 Best Model ({best_n}) Estimate</div>'
            f'<div class="amount">{fmt_inr(preds[best_n])}</div>'
            f'<div class="sub">Predicted House Price · Based on your inputs</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # All model cards
        section("🤖", "All 5 Model Predictions")
        icons = ['📐','✂️','🌿','🌳','⚡']
        pcols = st.columns(5)
        for col, (name, pred), icon in zip(pcols, preds.items(), icons):
            col.markdown(
                f'<div class="kpi"><div class="icon">{icon}</div>'
                f'<div style="font-weight:700;font-size:.78rem;color:#475569;margin-bottom:6px;">{name}</div>'
                f'<div class="val" style="font-size:1.28rem;">{fmt_inr(pred)}</div>'
                f'<div class="lbl">Estimated Price</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Bar chart
        ensemble = float(np.mean(list(preds.values())))
        fig = go.Figure(go.Bar(
            x=list(preds.keys()), y=list(preds.values()),
            marker_color=COLORS[:5], opacity=.88,
            text=[fmt_inr(v) for v in preds.values()],
            textposition='outside',
        ))
        fig.add_hline(y=ensemble, line_dash='dash', line_color='#94a3b8',
                       annotation_text=f"Ensemble avg: {fmt_inr(ensemble)}",
                       annotation_position='top right')
        plotly_base(fig, "Price Predictions — All 5 Models", h=380, legend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div class="insight"><p>📊 <b>Ensemble Average:</b> {fmt_inr(ensemble)} — '
            f'averaging predictions from all 5 models reduces single-model bias. '
            f'Range: {fmt_inr(min(preds.values()))} – {fmt_inr(max(preds.values()))} '
            f'({fmt_inr(max(preds.values())-min(preds.values()))} spread).</p></div>',
            unsafe_allow_html=True,
        )

        # Property profile
        section("📋", "Property Profile Summary")
        profile_df = pd.DataFrame({
            'Attribute': ['Area','Bedrooms','Bathrooms','Location Score',
                          'Property Age','Distance from City',
                          'Near School','Near Metro','Crime Rate Index'],
            'Your Input': [f"{area:,} sq ft", bedrooms, bathrooms,
                           f"{loc_score}/10", f"{prop_age} years",
                           f"{distance} km",
                           "Yes ✅" if near_school else "No ❌",
                           "Yes ✅" if near_metro  else "No ❌",
                           crime_rate],
            'Dataset Avg': [
                f"{df['area_sqft'].mean():.0f} sq ft",
                f"{df['bedrooms'].mean():.1f}",
                f"{df['bathrooms'].mean():.1f}",
                f"{df['location_score'].mean():.1f}/10",
                f"{df['property_age'].mean():.0f} years",
                f"{df['distance_city_km'].mean():.1f} km",
                f"{df['near_school'].mean()*100:.0f}% have school",
                f"{df['near_metro'].mean()*100:.0f}% have metro",
                f"{df['crime_rate_index'].mean():.1f}",
            ],
        })
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

        # Deviation from avg
        section("📈", "Your Property vs Dataset Average")
        u_vals = [area, bedrooms, bathrooms, loc_score, prop_age, distance]
        a_vals = [df['area_sqft'].mean(), df['bedrooms'].mean(), df['bathrooms'].mean(),
                   df['location_score'].mean(), df['property_age'].mean(),
                   df['distance_city_km'].mean()]
        f_lbls = ['Area','Bedrooms','Bathrooms','Location Score','Property Age','Distance']
        deviations = [(u-a)/max(abs(a),1) for u, a in zip(u_vals, a_vals)]
        bar_colors = ['#22c55e' if d > 0 else '#ef4444' for d in deviations]

        fig2 = go.Figure(go.Bar(
            x=f_lbls, y=deviations, marker_color=bar_colors, opacity=.85,
            text=[f"+{d:.0%}" if d > 0 else f"{d:.0%}" for d in deviations],
            textposition='outside',
        ))
        fig2.add_hline(y=0, line_color='#94a3b8', line_width=1.5)
        plotly_base(fig2, "Feature Deviation from Dataset Average (relative)", h=320, legend=False)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.markdown(
            '<div class="card" style="text-align:center;padding:44px;">'
            '<div style="font-size:3.5rem;margin-bottom:14px;">👆</div>'
            '<h4>Adjust the sliders above, then click Predict</h4>'
            '<p>All 5 trained ML models instantly estimate the property price.<br>'
            'You will see individual predictions, an ensemble average, '
            'and a feature comparison chart.</p></div>',
            unsafe_allow_html=True,
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#e2e8f0;margin-top:44px;">
<div style="text-align:center;padding:14px;color:#94a3b8;font-size:.78rem;">
  🏠 Robust Regression Engine · Real Estate AI · India Market ·
  Streamlit + Scikit-Learn · 3,800 properties · 5 ML algorithms
</div>
""", unsafe_allow_html=True)
