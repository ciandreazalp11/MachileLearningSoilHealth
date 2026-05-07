import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap, FastMarkerCluster
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from io import BytesIO
from contextlib import nullcontext
import joblib
import base64
from PIL import Image
import warnings
import re
from advanced_processing import (
    validate_dataset,
    generate_data_profile_report,
    advanced_missing_value_imputation,
    detect_and_handle_outliers,
    feature_engineering_suggestions,
    create_suggested_features,
    get_processing_recommendations,
)

warnings.filterwarnings("ignore", category=UserWarning)


def inject_tuqlas_chatbot() -> None:
    components.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
            if (doc.getElementById("tuqlas-chatbot-script")) return;

            const script = doc.createElement("script");
            script.id = "tuqlas-chatbot-script";
            script.src = "https://www.tuqlas.com/chatbot.js";
            script.dataset.key = "tq_live_e2ea6c743da14cd7f2eb52609629b42647f8a449";
            script.dataset.api = "https://www.tuqlas.com";
            script.defer = true;
            doc.body.appendChild(script);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# =========================
# Land-only filter (Mindanao) for Folium map
# Uses Natural Earth 10m land polygons (downloaded once + cached).
# Falls back to a conservative bounding-box filter if geopandas/shapely isn't available.
# =========================

import os
import io
import zipfile
from pathlib import Path

# Optional geo stack: used ONLY to hide offshore points on maps (no effect on modeling).
try:
    from shapely.geometry import Point, box, shape as _shape
    from shapely.ops import unary_union
    from shapely.prepared import prep
except Exception:
    Point = box = _shape = unary_union = prep = None

try:
    import geopandas as gpd
except Exception:
    gpd = None

@st.cache_resource
def _get_mindanao_land_polygon():
    """Return a prepared polygon approximating Mindanao LAND area only.

    Strategy (in order):
    1) Use GeoPandas + Shapely (fast + robust) if available.
    2) If GeoPandas is unavailable but Shapely is available, read the Natural Earth shapefile via `shapefile` (pyshp).
    3) If geo libs aren't available, return None (caller falls back to a conservative bbox filter).

    The Natural Earth 10m land dataset is downloaded once and cached locally.
    """
    if box is None or prep is None:
        return None

    cache_dir = Path.home() / ".cache" / "soil_health_app"
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "ne_10m_land.zip"
    shp_path = cache_dir / "ne_10m_land.shp"

    if not shp_path.exists():
        # Download Natural Earth 10m land (public domain)
        import requests  # local import to avoid hard dependency at import time
        url = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(cache_dir)

    # Mindanao broad bbox (includes some sea; removed by land polygon)
    mindanao_bbox = box(121.0, 4.0, 128.5, 11.5)

    geoms = []

    # --- Preferred path: GeoPandas if available ---
    if gpd is not None:
        land = gpd.read_file(str(shp_path))
        clipped = land[land.intersects(mindanao_bbox)].copy()
        if not clipped.empty:
            geoms = list(clipped.geometry)

    # --- Fallback: pyshp (no GeoPandas needed) ---
    if (not geoms) and (_shape is not None):
        try:
            import shapefile as pyshp  # pyshp package (import name: shapefile)
            sf = pyshp.Reader(str(shp_path))
            minx, miny, maxx, maxy = mindanao_bbox.bounds

            for sr in sf.shapeRecords():
                # quick bbox reject first (fast)
                bx0, by0, bx1, by1 = sr.shape.bbox
                if (bx1 < minx) or (bx0 > maxx) or (by1 < miny) or (by0 > maxy):
                    continue
                try:
                    geom = _shape(sr.shape.__geo_interface__)
                except Exception:
                    continue
                if geom.intersects(mindanao_bbox):
                    geoms.append(geom)
        except Exception:
            geoms = []

    if not geoms:
        return None

    geom = unary_union(geoms).intersection(mindanao_bbox)

    # Keep largest polygon piece (main landmass in this bbox)
    if getattr(geom, "geom_type", None) == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    return prep(geom)

def _filter_mindanao_land_only(df_in: pd.DataFrame) -> pd.DataFrame:
    """Hide points that are NOT on Mindanao land (automatic sea removal).
    Requires Latitude/Longitude columns.
    Falls back to conservative bbox if land polygon isn't available.
    """
    df0 = df_in.dropna(subset=["Latitude", "Longitude"]).copy()

    # Fast pre-filter (keeps runtime low even on big datasets)
    df0 = df0[
        (df0["Latitude"].between(4.0, 11.5)) &
        (df0["Longitude"].between(121.0, 128.5))
    ]
    if df0.empty:
        return df0

    prepared_land = _get_mindanao_land_polygon()
    if prepared_land is None or Point is None:
        # Fallback: tight "land-safe" bbox (may remove some coastal land points)
        return df0[
            (df0["Latitude"].between(5.0, 10.5)) &
            (df0["Longitude"].between(121.5, 127.3))
        ]

    # Strict point-in-land test
    keep_mask = []
    for lat, lon in zip(df0["Latitude"].astype(float), df0["Longitude"].astype(float)):
        keep_mask.append(prepared_land.contains(Point(lon, lat)))

    return df0.loc[keep_mask]


@st.cache_data(show_spinner=False)
def _cached_filter_mindanao_land_only(df_in: pd.DataFrame) -> pd.DataFrame:
    return _filter_mindanao_land_only(df_in)

st.set_page_config(
    page_title="Machine Learning-Driven Soil Analysis for Sustainable Agriculture System",
    layout="wide",
    page_icon="🌿",
)

inject_tuqlas_chatbot()


def silent_spinner(*args, **kwargs):
    return nullcontext()


st.spinner = silent_spinner

# === THEMES (UNCHANGED) ===
theme_classification = {
    "background_main": "linear-gradient(135deg, #07130d 0%, #102418 48%, #1e2b18 100%)",
    "sidebar_bg": "linear-gradient(180deg, #07130d 0%, #102418 58%, #0b1b12 100%)",
    "primary_color": "#7bbf4d",
    "secondary_color": "#d4a954",
    "button_gradient": "linear-gradient(90deg, #8bc34a, #5f9f36)",
    "button_text": "#07130d",
    "header_glow_color_1": "#8bc34a",
    "header_glow_color_2": "#d4a954",
    "menu_icon_color": "#9ed56a",
    "nav_link_color": "#eef7df",
    "nav_link_selected_bg": "#2f6b3f",
    "info_bg": "#13291c",
    "info_border": "#7bbf4d",
    "success_bg": "#183621",
    "success_border": "#8bc34a",
    "warning_bg": "#3a2f17",
    "warning_border": "#d4a954",
    "error_bg": "#3a1d18",
    "error_border": "#d9826b",
    "text_color": "#eef7df",
    "title_color": "#f5f1df",
}

theme_sakura = {
    "background_main": "linear-gradient(135deg, #08120d 0%, #182718 50%, #302719 100%)",
    "sidebar_bg": "linear-gradient(180deg, #08120d 0%, #142116 58%, #241d12 100%)",
    "primary_color": "#c7a04a",
    "secondary_color": "#86c65a",
    "button_gradient": "linear-gradient(90deg, #d4a954, #8f6f2a)",
    "button_text": "#08120d",
    "header_glow_color_1": "#d4a954",
    "header_glow_color_2": "#86c65a",
    "menu_icon_color": "#d4bd74",
    "nav_link_color": "#f7efd8",
    "nav_link_selected_bg": "#7a612a",
    "info_bg": "#202817",
    "info_border": "#c7a04a",
    "success_bg": "#183621",
    "success_border": "#86c65a",
    "warning_bg": "#3a2f17",
    "warning_border": "#d4a954",
    "error_bg": "#3a1d18",
    "error_border": "#d9826b",
    "text_color": "#f7efd8",
    "title_color": "#fff7df",
}

# === SESSION STATE ===
if "current_theme" not in st.session_state:
    st.session_state["current_theme"] = theme_classification
if "df" not in st.session_state:
    st.session_state["df"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "task_mode" not in st.session_state:
    st.session_state["task_mode"] = "Classification"
if "trained_on_features" not in st.session_state:
    st.session_state["trained_on_features"] = None
if "profile_andre" not in st.session_state:
    st.session_state["profile_andre"] = None
if "profile_rica" not in st.session_state:
    st.session_state["profile_rica"] = None
if "page_override" not in st.session_state:
    st.session_state["page_override"] = None
if "last_sidebar_selected" not in st.session_state:
    st.session_state["last_sidebar_selected"] = None
if "location_tag" not in st.session_state:
    st.session_state["location_tag"] = ""
if "uploaded_files_signature" not in st.session_state:
    st.session_state["uploaded_files_signature"] = None


if "nutrient_models" not in st.session_state:
    st.session_state["nutrient_models"] = {}   # e.g. {"Nitrogen": model, "Phosphorus": model, "Potassium": model}
if "nutrient_scalers" not in st.session_state:
    st.session_state["nutrient_scalers"] = {}
if "nutrient_features" not in st.session_state:
    st.session_state["nutrient_features"] = {}

# =========================
# WORKFLOW (Objectives 1–5) helper utilities
# Goal: preserve your original pages/design, but add a clear "flow" feeling
# without removing any existing features.
# =========================








def render_page_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    """Compact professional page heading used across the main app pages."""
    eyebrow_html = f"<div class='page-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <section class="page-header">
            {eyebrow_html}
            <h1>{title}</h1>
            {subtitle_html}
        </section>
        """,
        unsafe_allow_html=True,
    )



# === APPLY THEME (FIXED, CLEAN, WORKING) ===

# === STYLE INJECTION HELPER ===
def inject_style(css_html: str) -> None:
    import streamlit as st
    # Why: without this flag, <style> prints as text
    st.markdown(css_html, unsafe_allow_html=True)

def apply_theme(theme: dict) -> None:
    """Core theme shell. Behavior stays in the page code below."""
    import streamlit as st

    base_bg   = theme.get("background_main", "")
    sidebar   = theme.get("sidebar_bg", "")
    title_col = theme.get("title_color", "#ffffff")
    text_col  = theme.get("text_color", "#ffffff")
    btn_grad  = theme.get("button_gradient", "linear-gradient(90deg,#66bb6a,#4caf50)")
    btn_text  = theme.get("button_text", "#0c1d1d")

    css = f"""
<style>
.stApp {{
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif!important;
  color:{text_col};
  min-height:100vh;
  background:
    linear-gradient(180deg, rgba(255,255,255,.025), rgba(255,255,255,0) 240px),
    {base_bg};
  background-attachment:fixed;
  position:relative;
  overflow-x:hidden;
}}
h1,h2,h3,h4,h5,h6 {{
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif!important;
  color:{title_col}; font-weight:700!important;
}}
section[data-testid="stSidebar"] {{
  background:{sidebar}!important; height:100vh!important;
  border-right:1px solid rgba(255,255,255,.14);
  z-index:1!important;
}}
[data-testid="stAppViewContainer"], .main {{ position:relative!important; z-index:2!important; }}
[data-testid="stJson"], [data-testid="stDataFrame"], .stMetric, .element-container .stAlert {{
  background:rgba(8,18,13,.70)!important; border-radius:8px!important;
  border:1px solid rgba(255,255,255,.12)!important;
  box-shadow:0 12px 30px rgba(0,0,0,.22)!important;
}}
.stButton>button, .stDownloadButton>button {{
  background:{btn_grad}!important; color:{btn_text}!important;
  border-radius:8px!important; padding:.6rem 1.2rem!important;
  transition:.15s; box-shadow:0 10px 22px rgba(0,0,0,.22);
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
  transform:translateY(-1px); box-shadow:0 10px 28px rgba(0,0,0,.22);
}}
</style>
"""
    inject_style(css)
    inject_style('<div class="bg-decor" style="display:none"></div>')

apply_theme(st.session_state["current_theme"])

def apply_professional_overrides(theme: dict) -> None:
    """Final UI layer: cleaner dashboard styling without changing app behavior."""
    primary = theme.get("primary_color", "#4caf50")
    secondary = theme.get("secondary_color", "#a5d6a7")
    sidebar = theme.get("sidebar_bg", "rgba(15, 30, 30, 0.95)")
    text = theme.get("text_color", "#e0ffe0")
    title = theme.get("title_color", "#a5d6a7")
    button_text = theme.get("button_text", "#0c1d1d")

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --soil-primary: {primary};
  --soil-secondary: {secondary};
  --soil-text: {text};
  --soil-title: {title};
  --soil-surface: rgba(10, 24, 24, .72);
  --soil-surface-soft: rgba(255, 255, 255, .08);
  --soil-border: rgba(255, 255, 255, .14);
}}

.stApp {{
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  color: var(--soil-text);
}}

.stApp::before,
.stApp::after {{
  opacity: .16 !important;
  filter: blur(22px) !important;
}}

h1, h2, h3, h4, h5, h6 {{
  font-family: 'Inter', system-ui, sans-serif !important;
  letter-spacing: 0 !important;
  text-shadow: none !important;
  animation: none !important;
}}

[data-testid="stAppViewContainer"] > .main .block-container {{
  max-width: 1280px;
  padding: 2rem 2.4rem 3rem;
}}

section[data-testid="stSidebar"] {{
  background: {sidebar} !important;
  border-right: 1px solid var(--soil-border) !important;
  box-shadow: 10px 0 30px rgba(0,0,0,.18);
}}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] small {{
  color: rgba(255,255,255,.78);
}}

.sidebar-header {{
  padding: 8px 2px 4px;
}}

.sidebar-title {{
  font-size: 1.22rem !important;
  line-height: 1.2 !important;
  margin: 0 !important;
  color: var(--soil-title) !important;
}}

.sidebar-sub {{
  margin-top: 4px;
  color: rgba(255,255,255,.68);
  font-size: .82rem;
  font-weight: 500;
}}

.sidebar-status {{
  border: 1px solid var(--soil-border);
  border-radius: 8px;
  padding: 12px;
  background: rgba(255,255,255,.06);
  margin: 12px 0 16px;
}}

.sidebar-status-title {{
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--soil-secondary);
  margin-bottom: 8px;
}}

.sidebar-status-row {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: .82rem;
  padding: 5px 0;
  border-top: 1px solid rgba(255,255,255,.08);
}}

.sidebar-status-row:first-of-type {{
  border-top: 0;
}}

.sidebar-status-row span:first-child {{
  color: rgba(255,255,255,.62);
}}

.sidebar-status-row span:last-child {{
  color: #fff;
  font-weight: 700;
  text-align: right;
}}

.page-header {{
  margin: 0 0 1.2rem;
  padding: 1.15rem 1.25rem;
  border: 1px solid var(--soil-border);
  border-left: 4px solid var(--soil-primary);
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(255,255,255,.12), rgba(255,255,255,.045));
  box-shadow: 0 14px 36px rgba(0,0,0,.18);
}}

.page-header h1 {{
  margin: 0 !important;
  color: var(--soil-title) !important;
  font-size: clamp(1.55rem, 2vw, 2.25rem) !important;
  line-height: 1.16 !important;
  font-weight: 800 !important;
}}

.page-header p {{
  margin: .55rem 0 0;
  color: rgba(255,255,255,.78);
  max-width: 860px;
  font-size: .96rem;
}}

.page-eyebrow {{
  margin-bottom: .45rem;
  color: var(--soil-secondary);
  font-size: .72rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  font-weight: 800;
}}

[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
.element-container .stAlert,
div[data-testid="stExpander"],
div[data-testid="stForm"] {{
  border-radius: 8px !important;
  border: 1px solid var(--soil-border) !important;
  background: var(--soil-surface-soft) !important;
  box-shadow: 0 10px 28px rgba(0,0,0,.14) !important;
}}

[data-testid="stMetric"] {{
  padding: 12px 14px;
}}

[data-testid="stMetricLabel"] p {{
  color: rgba(255,255,255,.68) !important;
  font-weight: 700 !important;
}}

[data-testid="stMetricValue"] {{
  color: #fff !important;
  font-size: 1.45rem !important;
}}

.stButton>button,
.stDownloadButton>button {{
  min-height: 40px;
  border: 0 !important;
  border-radius: 8px !important;
  font-weight: 800 !important;
  color: {button_text} !important;
  box-shadow: 0 8px 18px rgba(0,0,0,.18) !important;
}}

.stButton>button:hover,
.stDownloadButton>button:hover {{
  transform: translateY(-1px);
  filter: brightness(1.04);
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 8px;
  border-bottom: 1px solid var(--soil-border);
}}

.stTabs [data-baseweb="tab"] {{
  border-radius: 8px 8px 0 0;
  background: rgba(255,255,255,.06);
  padding: 10px 16px;
  font-weight: 700;
}}

.stTabs [aria-selected="true"] {{
  background: rgba(255,255,255,.16) !important;
  color: #fff !important;
}}

hr {{
  border-color: rgba(255,255,255,.12) !important;
  margin: 1.2rem 0 !important;
}}

label, .stMarkdown, [data-testid="stCaptionContainer"] {{
  color: rgba(255,255,255,.82) !important;
}}

.stApp {{
  background:
    linear-gradient(120deg, rgba(123,191,77,.08), transparent 34%),
    linear-gradient(180deg, rgba(212,169,84,.055), transparent 32%),
    repeating-linear-gradient(90deg, rgba(255,255,255,.018) 0 1px, transparent 1px 96px),
    linear-gradient(135deg, #07130d 0%, #102418 52%, #1b2515 100%) !important;
}}

.stApp::before,
.stApp::after {{
  content: none !important;
  display: none !important;
}}

[data-testid="stHeader"] {{
  background: rgba(7,19,13,.76) !important;
  border-bottom: 1px solid rgba(255,255,255,.08);
  backdrop-filter: blur(10px);
}}

[data-testid="stAppViewContainer"] > .main .block-container {{
  padding-top: 1.35rem;
}}

.page-header {{
  position: relative;
  overflow: hidden;
  margin-bottom: 1.4rem;
  padding: 1.35rem 1.45rem 1.28rem;
  border-radius: 10px;
  border: 1px solid rgba(238,247,223,.12);
  border-left: 5px solid var(--soil-primary);
  background:
    linear-gradient(135deg, rgba(18,47,29,.94), rgba(11,27,18,.88)),
    linear-gradient(90deg, rgba(212,169,84,.16), transparent);
  box-shadow: 0 18px 44px rgba(0,0,0,.28);
}}

.page-header::after {{
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--soil-primary), var(--soil-secondary), transparent);
  opacity: .85;
}}

.page-header h1 {{
  color: #f5f1df !important;
  font-size: clamp(1.7rem, 2.3vw, 2.45rem) !important;
  letter-spacing: 0 !important;
}}

.page-header p {{
  color: rgba(238,247,223,.78) !important;
  font-size: .98rem;
  line-height: 1.55;
}}

.page-eyebrow {{
  color: #d4a954 !important;
  letter-spacing: .12em;
}}

section[data-testid="stSidebar"] {{
  background:
    linear-gradient(180deg, rgba(7,19,13,.98), rgba(16,36,24,.98) 56%, rgba(7,19,13,.98)) !important;
}}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
  gap: .45rem;
}}

.nav-link {{
  border-radius: 8px !important;
  margin: 3px 0 !important;
  color: rgba(238,247,223,.82) !important;
  transition: all .14s ease;
}}

.nav-link:hover {{
  background: rgba(255,255,255,.075) !important;
  color: #fff !important;
}}

.nav-link-selected {{
  background: linear-gradient(90deg, #2f6b3f, #5f9f36) !important;
  color: #fff !important;
  box-shadow: 0 10px 24px rgba(0,0,0,.2);
}}

.sidebar-status {{
  background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.045));
  border-color: rgba(238,247,223,.13);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 12px 30px rgba(0,0,0,.18);
}}

[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
.element-container .stAlert,
div[data-testid="stExpander"],
div[data-testid="stForm"] {{
  background: linear-gradient(180deg, rgba(13,31,21,.82), rgba(8,20,13,.74)) !important;
  border-color: rgba(238,247,223,.12) !important;
  box-shadow: 0 16px 38px rgba(0,0,0,.24) !important;
}}

[data-testid="stMetric"] {{
  border-left: 3px solid var(--soil-primary) !important;
}}

.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"],
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {{
  background: rgba(6,16,10,.76) !important;
  border-color: rgba(238,247,223,.16) !important;
  color: #f5f1df !important;
  border-radius: 8px !important;
}}

.stSlider [data-baseweb="slider"] div {{
  color: var(--soil-primary) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  background: rgba(6,16,10,.34);
  padding: 6px;
  border-radius: 10px;
  border: 1px solid rgba(238,247,223,.1);
}}

.stTabs [data-baseweb="tab"] {{
  border-radius: 8px;
  background: transparent;
  color: rgba(238,247,223,.7);
}}

.stTabs [aria-selected="true"] {{
  background: linear-gradient(90deg, rgba(47,107,63,.95), rgba(95,159,54,.9)) !important;
  color: #fff !important;
  box-shadow: 0 8px 18px rgba(0,0,0,.18);
}}

.stButton>button,
.stDownloadButton>button {{
  background: linear-gradient(90deg, #8bc34a, #5f9f36) !important;
  border: 1px solid rgba(255,255,255,.16) !important;
  text-transform: none;
}}

.stButton>button:focus,
.stDownloadButton>button:focus {{
  box-shadow: 0 0 0 3px rgba(139,195,74,.24), 0 8px 18px rgba(0,0,0,.18) !important;
}}

[data-testid="stFileUploader"] section {{
  background: linear-gradient(180deg, rgba(13,31,21,.72), rgba(8,20,13,.66)) !important;
  border: 1px dashed rgba(139,195,74,.45) !important;
  border-radius: 10px !important;
}}

div[data-testid="stExpander"] summary {{
  color: #f5f1df !important;
  font-weight: 800;
}}

.js-plotly-plot,
iframe {{
  border-radius: 8px;
  overflow: hidden;
}}

@media (max-width: 768px) {{
  [data-testid="stAppViewContainer"] > .main .block-container {{
    padding: 1rem 1rem 2rem;
  }}
  .page-header {{
    padding: 1rem;
  }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )

apply_professional_overrides(st.session_state["current_theme"])

# === SIDEBAR (UNCHANGED LAYOUT) ===
with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-header">
          <h2 class="sidebar-title">🌱 Soil Health System</h2>
          <div class="sidebar-sub">ML-Driven Soil Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("---")
    selected = option_menu(
        None,
        ["🏠 Home", "🤖 Modeling", "📊 Visualization", "📈 Results", "🌿 Insights", "👤 About"],
        icons=["house", "robot", "bar-chart", "graph-up", "lightbulb", "person-circle"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {
                "color": st.session_state["current_theme"]["menu_icon_color"],
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "3px 0",
                "padding": "11px 12px",
                "border-radius": "8px",
                "color": st.session_state["current_theme"]["nav_link_color"],
                "font-weight": "650",
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #2f6b3f, #5f9f36)",
                "color": "#ffffff",
                "font-weight": "800",
            },
        },
    )
    st.write("---")
    df_status = "Loaded" if st.session_state.get("df") is not None else "No dataset"
    model_status = "Trained" if st.session_state.get("model") is not None else "Not trained"
    result_status = "Available" if st.session_state.get("results") is not None else "Pending"
    st.markdown(
        f"""
        <div class="sidebar-status">
            <div class="sidebar-status-title">System Status</div>
            <div class="sidebar-status-row"><span>Dataset</span><span>{df_status}</span></div>
            <div class="sidebar-status-row"><span>Mode</span><span>{st.session_state.get("task_mode", "Classification")}</span></div>
            <div class="sidebar-status-row"><span>Model</span><span>{model_status}</span></div>
            <div class="sidebar-status-row"><span>Results</span><span>{result_status}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:12px;color:{st.session_state['current_theme']['text_color']};opacity:0.85'>Developed for sustainable agriculture</div>",
        unsafe_allow_html=True,
    )
    if st.session_state["last_sidebar_selected"] != selected:
        st.session_state["page_override"] = None
        st.session_state["last_sidebar_selected"] = selected

page = (
    st.session_state["page_override"]
    if st.session_state["page_override"]
    else selected
)

# === COLUMN MAPPING & BASE HELPERS ===
column_mapping = {
    "pH": [
        "pH",
        "ph",
        "soil_pH",
        "soil ph",
        "soilph",
    ],
    "Nitrogen": [
        "Nitrogen",
        "nitrogen",
        "n",
        "nitrogen_level",
        "total_nitrogen",
    ],
    "Phosphorus": [
        "Phosphorus",
        "phosphorus",
        "p",
        "p2o5",
    ],
    "Potassium": [
        "Potassium",
        "potassium",
        "k",
        "k2o",
    ],
    "Moisture": [
        "Moisture",
        "moisture",
        "soil_moisture",
        "moisture_index",
        "moisture content",
        "moisturecontent",
    ],
    "Organic Matter": [
        "Organic Matter",
        "organic matter",
        "organic_matter",
        "organicmatter",
        "om",
        "oc",
        "orgmatter",
        "organic carbon",
        "organic_carbon",
        "organiccarbon",
    ],
    "Latitude": [
        "Latitude",
        "latitude",
        "lat",
    ],
    "Longitude": [
        "Longitude",
        "longitude",
        "lon",
        "lng",
        "longitude_1",
        "longitude_2",
    ],
    "Fertility_Level": [
        "Fertility_Level",
        "fertility_level",
        "fertility class",
        "fertility_class",
        "fertilityclass",
    ],
    "Province": [
        "Province",
        "province",
        "prov",
    ],
}

required_columns = [
    "pH",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Moisture",
    "Organic Matter",
]


def normalize_col_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    s = re.sub(r"\d+$", "", s)
    return s


def safe_to_numeric_columns(df, cols):
    numeric_found = []
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_found.append(c)
    return numeric_found


def download_df_button(
    df,
    filename="final_preprocessed_soil_dataset.csv",
    label="⬇️ Download Cleaned & Preprocessed Data",
):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(label=label, data=buf, file_name=filename, mime="text/csv")


def create_fertility_label(df, col="Nitrogen", q=3):
    labels = ["Low", "Moderate", "High"]
    try:
        fert = pd.qcut(df[col], q=q, labels=labels, duplicates="drop")
        if fert.nunique() < 3:
            fert = pd.cut(df[col], bins=3, labels=labels)
    except Exception:
        fert = pd.cut(df[col], bins=3, labels=labels, include_lowest=True)
    return fert.astype(str)


def interpret_label(label):
    l = str(label).lower()
    if l in ["high", "good", "healthy", "3", "2.0"]:
        return ("Good", "green", "✅ Nutrients are balanced. Ideal for most crops.")
    if l in ["moderate", "medium", "2", "1.0"]:
        return (
            "Moderate",
            "orange",
            "⚠️ Some nutrient imbalance. Consider minor adjustments.",
        )
    return ("Poor", "red", "🚫 Deficient or problematic — take corrective action.")


# === CROP PROFILES ===
CROP_PROFILES = {
    "Rice": {
        "pH": (5.5, 6.5),
        "Nitrogen": (0.25, 0.8),
        "Phosphorus": (15, 40),
        "Potassium": (80, 200),
        "Moisture": (40, 80),
        "Organic Matter": (2.0, 6.0),
    },
    "Corn (Maize)": {
        "pH": (5.8, 7.0),
        "Nitrogen": (0.3, 1.2),
        "Phosphorus": (10, 40),
        "Potassium": (100, 250),
        "Moisture": (20, 60),
        "Organic Matter": (1.5, 4.0),
    },
    "Cassava": {
        "pH": (5.0, 7.0),
        "Nitrogen": (0.1, 0.5),
        "Phosphorus": (5, 25),
        "Potassium": (100, 300),
        "Moisture": (20, 60),
        "Organic Matter": (1.0, 3.5),
    },
    "Vegetables (general)": {
        "pH": (6.0, 7.5),
        "Nitrogen": (0.3, 1.5),
        "Phosphorus": (15, 50),
        "Potassium": (120, 300),
        "Moisture": (30, 70),
        "Organic Matter": (2.0, 5.0),
    },
    "Banana": {
        "pH": (5.5, 7.0),
        "Nitrogen": (0.2, 0.8),
        "Phosphorus": (10, 30),
        "Potassium": (200, 500),
        "Moisture": (40, 80),
        "Organic Matter": (2.0, 6.0),
    },
    "Coconut": {
        "pH": (5.5, 7.5),
        "Nitrogen": (0.1, 0.6),
        "Phosphorus": (5, 25),
        "Potassium": (80, 250),
        "Moisture": (30, 70),
        "Organic Matter": (1.0, 4.0),
    },
}


def crop_match_score(sample: dict, crop_profile: dict):
    scores = []
    for k, rng in crop_profile.items():
        if k not in sample or pd.isna(sample[k]):
            continue
        val = float(sample[k])
        low, high = rng
        if low <= val <= high:
            scores.append(1.0)
        else:
            width = max(1e-6, high - low)
            if val < low:
                dist = (low - val) / width
            else:
                dist = (val - high) / width
            s = max(0.0, np.exp(-dist))
            scores.append(s)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def recommend_crops_for_sample(sample_series: pd.Series, top_n=3):
    sample = sample_series.to_dict()
    scored = []
    for crop, profile in CROP_PROFILES.items():
        s = crop_match_score(sample, profile)
        scored.append((crop, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def evaluate_crop_nutrient_gaps(sample: dict, crop_profile: dict):
    issues = []
    for param, (low, high) in crop_profile.items():
        val = sample.get(param, np.nan)
        if pd.isna(val):
            continue
        v = float(val)
        if v < low:
            issues.append(f"{param} too low")
        elif v > high:
            issues.append(f"{param} too high")
    return issues


def build_crop_evaluation_table(sample_series: pd.Series, top_n: int = 6) -> pd.DataFrame:
    sample = sample_series.to_dict()
    rows = []
    for crop, profile in CROP_PROFILES.items():
        score = crop_match_score(sample, profile)
        issues = evaluate_crop_nutrient_gaps(sample, profile)
        if score >= 0.66:
            suitability = "Good"
        elif score >= 0.33:
            suitability = "Moderate"
        else:
            suitability = "Poor"

        if not issues:
            recommendation = "Soil meets most nutrient requirements for this crop."
        else:
            recommendation = (
                "Improve soil for this crop by addressing: " + ", ".join(issues)
            )

        rows.append(
            {
                "Crop": crop,
                "Suitability": suitability,
                "MatchScore": round(score, 3),
                "LimitingFactors": ", ".join(issues) if issues else "None",
                "Recommendation": recommendation,
            }
        )
    if not rows:
        return pd.DataFrame()
    df_eval = pd.DataFrame(rows)
    df_eval = df_eval.sort_values("MatchScore", ascending=False)
    if top_n and top_n > 0:
        df_eval = df_eval.head(top_n)
    return df_eval


def compute_suitability_score(row, features=None):
    if features is None:
        features = [
            "pH",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Moisture",
            "Organic Matter",
        ]
    vals = []
    for f in features:
        if f not in row or pd.isna(row[f]):
            continue
        vals.append(row[f])
    if not vals:
        return 0.0

    df = st.session_state.get("df")
    if df is not None:
        score_components = []
        for f in features:
            if f not in df.columns or f not in row or pd.isna(row[f]):
                continue
            low = df[f].quantile(0.05)
            high = df[f].quantile(0.95)
            if high - low <= 0:
                norm = 0.5
            else:
                norm = (row[f] - low) / (high - low)
            norm = float(np.clip(norm, 0, 1))
            score_components.append(norm)
        if not score_components:
            return 0.0
        base_score = float(np.mean(score_components))
    else:
        base_score = float(np.mean(vals)) / (
            np.max(vals) if np.max(vals) != 0 else 1.0
        )

    fi = None
    feat = None
    if st.session_state.get("results"):
        fi = st.session_state["results"].get("feature_importances")
        feat = st.session_state["results"].get("X_columns")
    if fi and feat:
        weights = {f_name: w for f_name, w in zip(feat, fi)}
        weighted = []
        for f, w in weights.items():
            if f in row and f in (df.columns if df is not None else []) and not pd.isna(
                row[f]
            ):
                low = df[f].quantile(0.05)
                high = df[f].quantile(0.95)
                if high - low <= 0:
                    norm = 0.5
                else:
                    norm = (row[f] - low) / (high - low)
                norm = float(np.clip(norm, 0, 1))
                weighted.append(norm * w)
        if weighted:
            wsum = float(np.sum(list(weights.values())))
            if wsum > 0:
                return float(np.sum(weighted) / wsum)
    return base_score


def suitability_color(score):
    if score >= 0.66:
        return ("Green", "#2ecc71")
    if score >= 0.33:
        return ("Orange", "#f39c12")
    return ("Red", "#e74c3c")


def clip_soil_ranges(df: pd.DataFrame) -> pd.DataFrame:
    if "pH" in df.columns:
        df["pH"] = df["pH"].clip(3.5, 9.0)
    if "Moisture" in df.columns:
        df["Moisture"] = df["Moisture"].clip(0, 100)
    if "Organic Matter" in df.columns:
        df["Organic Matter"] = df["Organic Matter"].clip(lower=0)
    for ncol in ["Nitrogen", "Phosphorus", "Potassium"]:
        if ncol in df.columns:
            q_low = df[ncol].quantile(0.01)
            q_high = df[ncol].quantile(0.99)
            if pd.notna(q_low) and pd.notna(q_high) and q_high > q_low:
                df[ncol] = df[ncol].clip(q_low, q_high)
    return df


def run_kmeans_on_df(
    df: pd.DataFrame, features: list, n_clusters: int = 3
):
    sub = df[features].dropna().copy()
    if sub.shape[0] < n_clusters or n_clusters < 1:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    sub["cluster"] = labels
    return sub, model


def upload_and_preprocess_widget():
    st.markdown("### 📂 Upload Soil Data")
    st.markdown(
        "Upload one or more soil analysis files (.csv or .xlsx). The app will attempt "
        "to standardize column names and auto-preprocess numeric columns."
    )
    uploaded_files = st.file_uploader(
        "Select datasets", type=["csv", "xlsx"], accept_multiple_files=True
    )
    uploaded_signature = (
        tuple((file.name, getattr(file, "size", None)) for file in uploaded_files)
        if uploaded_files
        else None
    )

    if st.session_state["df"] is not None and not uploaded_files:
        st.success(
            f"✅ Loaded preprocessed dataset "
            f"({st.session_state['df'].shape[0]} rows, "
            f"{st.session_state['df'].shape[1]} cols)."
        )
        st.dataframe(st.session_state["df"].head())
        st.caption("**Dataset preview (first rows):** Use this to confirm the file loaded correctly (columns, units, and any missing values) before moving to modeling/visuals.")
        if st.button("🔁 Clear current dataset"):
            st.session_state["df"] = None
            st.session_state["results"] = None
            st.session_state["model"] = None
            st.session_state["scaler"] = None
            st.session_state["uploaded_files_signature"] = None
            st.experimental_rerun()

    if (
        uploaded_files
        and st.session_state["df"] is not None
        and st.session_state.get("uploaded_files_signature") == uploaded_signature
    ):
        st.success(
            f"✅ Using cached preprocessed dataset "
            f"({st.session_state['df'].shape[0]} rows, "
            f"{st.session_state['df'].shape[1]} cols)."
        )
        st.dataframe(st.session_state["df"].head())
        st.caption("Dataset is already preprocessed. Clear it only when you need to reload the uploaded files.")
        if st.button("🔁 Clear current dataset", key="clear_cached_dataset"):
            st.session_state["df"] = None
            st.session_state["results"] = None
            st.session_state["model"] = None
            st.session_state["scaler"] = None
            st.session_state["uploaded_files_signature"] = None
            st.experimental_rerun()
        return

    cleaned_dfs = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                if hasattr(file, "size") and file.size > 200 * 1024 * 1024:
                    st.warning(f"{file.name} is too large! Must be <200MB.")
                    continue
                if not (file.name.endswith(".csv") or file.name.endswith(".xlsx")):
                    st.warning(f"{file.name}: Unsupported extension.")
                    continue

                if file.name.endswith(".csv"):
                    with st.spinner(f"📂 Loading {file.name}..."):
                        df_file = pd.read_csv(file)
                else:
                    with st.spinner(f"📂 Loading {file.name}..."):
                        df_file = pd.read_excel(file)

                if df_file.empty:
                    st.warning(f"{file.name} is empty!")
                    continue
                if len(df_file.columns) < 2:
                    st.warning(f"{file.name} has too few columns for analysis.")
                    continue

                # --- robust column standardization (FIRST match wins) ---
                with st.spinner(f"🔄 Standardizing columns..."):
                    col_norm_map = {}
                    for c in df_file.columns:
                        key = normalize_col_name(c)
                        if key not in col_norm_map:
                            col_norm_map[key] = c  # keep first encountered column

                renamed = {}
                for std_col, alt_names in column_mapping.items():
                    candidates = [std_col] + alt_names
                    for cand in candidates:
                        norm_cand = normalize_col_name(cand)
                        if norm_cand in col_norm_map:
                            src_col = col_norm_map[norm_cand]
                            renamed[src_col] = std_col
                            break

                if renamed:
                    df_file.rename(columns=renamed, inplace=True)

                numeric_core_cols = [
                    col
                    for col in required_columns + ["Latitude", "Longitude"]
                    if col in df_file.columns
                ]
                safe_to_numeric_columns(df_file, numeric_core_cols)

                df_file.drop_duplicates(inplace=True)
                cleaned_dfs.append(df_file)

                recognized = [
                    c
                    for c in required_columns
                    + ["Latitude", "Longitude", "Fertility_Level", "Province"]
                    if c in df_file.columns
                ]
                recog_text = (
                    ", ".join(recognized) if recognized else "no core soil features"
                )
                st.success(
                    f"✅ Cleaned {file.name} — recognized: {recog_text} "
                    f"({df_file.shape[0]} rows)"
                )
            except Exception as e:
                st.warning(f"⚠️ Skipped {file.name}: {e}")

        if cleaned_dfs and any(len(x) > 0 for x in cleaned_dfs):
            with st.spinner("🔗 Concatenating and finalizing dataset..."):
                df = pd.concat(cleaned_dfs, ignore_index=True, sort=False)
                if df.empty:
                    st.error("All loaded files are empty after concatenation.")
                    return

                safe_to_numeric_columns(df, required_columns + ["Latitude", "Longitude"])

            with st.spinner("🧹 Cleaning missing values..."):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    medians = df[numeric_cols].median()
                    df[numeric_cols] = df[numeric_cols].fillna(medians)

            with st.spinner("📊 Processing categorical data..."):
                cat_cols = df.select_dtypes(exclude=[np.number]).columns
                for c in cat_cols:
                    try:
                        if df[c].isnull().sum() > 0:
                            df[c].fillna(df[c].mode().iloc[0], inplace=True)
                    except Exception:
                        df[c].fillna(method="ffill", inplace=True)

                df.dropna(how="all", inplace=True)

                df = clip_soil_ranges(df)

            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                st.error(f"Some required columns missing: {missing_required}")
                return

            if "Nitrogen" in df.columns:
                if "Fertility_Level" not in df.columns or df["Fertility_Level"].nunique() < 2:
                    df["Fertility_Level"] = create_fertility_label(
                        df, col="Nitrogen", q=3
                    )

            st.session_state["df"] = df
            st.session_state["uploaded_files_signature"] = uploaded_signature
            st.success("✨ Dataset preprocessed and stored in session.")
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
            st.dataframe(df.head())
            download_df_button(df)
            st.caption("**Preprocessed preview:** This is the cleaned/standardized version the models will use. If something looks wrong here (e.g., columns missing), adjust preprocessing or your input file.")
            
            # === Advanced Processing Options ===
            with st.expander("🔧 Advanced Processing Options", expanded=False):
                st.markdown("**Enhance your dataset with advanced processing techniques**")
                
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    if st.checkbox("🔍 Detect & Handle Outliers", key="outlier_check"):
                        outlier_method = st.radio("Method:", ["IQR", "Z-Score"], key="outlier_method")
                        outlier_action = st.radio("Action:", ["Flag", "Remove", "Winsorize"], key="outlier_action")
                        
                        if st.button("Apply Outlier Detection", key="apply_outliers"):
                            method_map = {"IQR": "iqr", "Z-Score": "zscore"}
                            action_map = {"Flag": "flag", "Remove": "remove", "Winsorize": "winsorize"}
                            
                            with st.spinner(f"🔍 Detecting outliers using {outlier_method} method..."):
                                df_processed, outlier_stats = detect_and_handle_outliers(
                                    df,
                                    method=method_map[outlier_method],
                                    action=action_map[outlier_action]
                                )
                            
                            st.session_state["df"] = df_processed
                            st.success(f"✅ Outliers processed. Rows: {len(df_processed)}")
                            st.json(outlier_stats)
                
                with adv_col2:
                    if st.checkbox("💧 Advanced Missing Value Imputation", key="impute_check"):
                        impute_method = st.selectbox(
                            "Imputation Method:",
                            ["auto", "median", "knn", "mean", "forward_fill"],
                            key="impute_method"
                        )
                        
                        if st.button("Apply Imputation", key="apply_impute"):
                            with st.spinner(f"💧 Applying {impute_method} imputation..."):
                                df_imputed = advanced_missing_value_imputation(df, method=impute_method)
                            st.session_state["df"] = df_imputed
                            st.success(f"✅ Imputation applied using '{impute_method}' method")
                            st.dataframe(df_imputed.head())
                
                # Feature Engineering
                if st.checkbox("⚙️ Auto Feature Engineering", key="feature_eng_check"):
                    suggestions = feature_engineering_suggestions(df)
                    st.markdown("**Suggested Features:**")
                    for feat_type, feats in suggestions.items():
                        if feats:
                            st.markdown(f"- **{feat_type.replace('_', ' ').title()}:**")
                            for feat in feats:
                                st.markdown(f"  - {feat}")
                    
                    if st.button("Create Suggested Features", key="create_features"):
                        with st.spinner("⚙️ Generating feature engineering..."):
                            df_enhanced = create_suggested_features(df)
                        st.session_state["df"] = df_enhanced
                        st.success(f"✅ Features created. New shape: {df_enhanced.shape}")
                        st.dataframe(df_enhanced.head())

        else:
            st.error(
                "No valid sheets processed. Check file formats and column headers."
            )


def render_profile(name, asset_filename):
    st.markdown(
        """
    <style>
    .avatar-card {
        display: flex; flex-direction: column; align-items: center; 
        margin-bottom: 22px;
    }
    .avatar-holo {
        width: 170px; height: 170px;
        border-radius: 50%;
        background: conic-gradient(from 180deg at 50% 50%, #8bc34a, #d4a954, #2f6b3f, #8bc34a 100%);
        padding: 6px;
        box-shadow: 0 12px 28px rgba(0,0,0,.25), 0 0 0 1px rgba(238,247,223,.18);
        position: relative;
        margin-bottom: 18px;
        transition: box-shadow 0.3s ease;
    }
    .avatar-holo img {
        width: 100%; height: 100%; object-fit: cover; border-radius: 50%;
        box-shadow: 0 3px 15px #0004;
        background: #fff;
    }
    .avatar-name {
      font-size: 22px; font-weight: 700;
      color: #f5f1df; 
      margin-bottom: 6px; margin-top: -4px;
      letter-spacing: 1px;
    }
    .avatar-role {
      font-size: 14px;
      color: rgba(238,247,223,.72); font-style: italic;
      padding-bottom: 2px;
    }
    .bsis-label {
      margin-top: 7px; margin-bottom: 7px;
      padding: 5px 18px; font-size: 16.5px; font-weight: 700;
      color: #07130d; background: linear-gradient(to right, #8bc34a, #d4a954);
      border-radius: 18px; border: none;
      box-shadow: 0 8px 18px rgba(0,0,0,.22);
      text-align: center; display: inline-block; letter-spacing: 1.3px;
      outline: none;
      transition: background 0.2s;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    asset_path = f"assets/{asset_filename}"
    try:
        image = Image.open(asset_path)
        buf = BytesIO()
        image.save(buf, format="PNG", unsafe_allow_html=True)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_html = f'<img src="data:image/png;base64,{img_b64}" alt="profile" />'
    except Exception:
        img_html = (
            '<div style="width:170px;height:170px;background:#eee;border-radius:50%;'
            "display:flex;align-items:center;justify-content:center;color:#aaa;"
            '">No Image</div>'
        )

    role_line = ""
    if "andre" in name.lower():
        role_line = "Developer | Machine Learning, Full Stack, Soil Science"
    elif "rica" in name.lower():
        role_line = "Developer | Data Analysis, Visualization, Soil Science"

    st.markdown(f"""
    <div class="avatar-card">
        <div class="avatar-holo">{img_html}</div>
        <div class="avatar-name">{name}</div>
        <div class="avatar-role">{role_line}</div>
        <div class="bsis-label">BSIS-4A</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ---------------- MAIN PAGE LOGIC -----------------
if page == "🏠 Home":
    render_page_header(
        "Soil Health Analysis Dashboard",
        "Upload, clean, analyze, model, and interpret soil data for sustainable agriculture decisions.",
        "Capstone Project",
    )
    st.write("")
    upload_and_preprocess_widget()

elif page == "🤖 Modeling":
    render_page_header(
        "Model Training",
        "Train Random Forest models for soil health classification, fertility regression, and nutrient prediction.",
        "Machine Learning",
    )

    st.write("")
    if st.session_state["df"] is None:
        st.info("Please upload a dataset first in 'Home'.")
    else:
        df = st.session_state["df"].copy()

        st.markdown("#### Model Mode")
        default_checkbox = (
            True if st.session_state.get("task_mode") == "Regression" else False
        )
        chk = st.checkbox(
            "Switch to Regression mode", value=default_checkbox, key="model_mode_checkbox"
        )
        if chk:
            st.session_state["task_mode"] = "Regression"
            st.session_state["current_theme"] = theme_sakura
        else:
            st.session_state["task_mode"] = "Classification"
            st.session_state["current_theme"] = theme_classification

        apply_theme(st.session_state["current_theme"])
        apply_professional_overrides(st.session_state["current_theme"])

        switch_color = (
            "#d4a954"
            if st.session_state["task_mode"] == "Regression"
            else "#8bc34a"
        )
        st.markdown(
            f"""
        <style>
        .fake-switch {{
            width:70px;
            height:36px;
            border-radius:20px;
            background:{switch_color};
            display:inline-block;
            position:relative;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }}
        .fake-knob {{
            width:28px;height:28px;border-radius:50%;
            background:rgba(255,255,255,0.95); position:absolute; top:4px;
            transition: all .18s ease;
        }}
        .knob-left {{ left:4px; }}
        .knob-right {{ right:4px; }}
        .switch-label {{ font-weight:600; margin-left:10px; color:{st.session_state['current_theme']['text_color']}; }}
        </style>
        <div style="display:flex;align-items:center;margin-bottom:10px;">
          <div class="fake-switch">
            <div class="fake-knob {'knob-right' if st.session_state['task_mode']=='Regression' else 'knob-left'}"></div>
          </div>
          <div class="switch-label">{'Regression' if st.session_state['task_mode']=='Regression' else 'Classification'}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("---", unsafe_allow_html=True)

        fertility_derived_from_n = False

        if st.session_state["task_mode"] == "Classification":
            # Prefer an existing fertility label (from dataset). Only derive it from Nitrogen if missing.
            if "Fertility_Level" not in df.columns and "Nitrogen" in df.columns:
                df["Fertility_Level"] = create_fertility_label(df, col="Nitrogen", q=3)
                fertility_derived_from_n = True
            y = df["Fertility_Level"] if "Fertility_Level" in df.columns else None
        else:
            # Default regression target is Nitrogen
            y = df["Nitrogen"] if "Nitrogen" in df.columns else None

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove location columns from feature list
        for loccol in ["Latitude", "Longitude", "latitude", "longitude", "longitude_1", "longitude_2"]:
            if loccol in numeric_features:
                numeric_features.remove(loccol)

        # Remove Nitrogen only when it is the target (Regression) or when Fertility_Level was derived from Nitrogen
        # (to avoid label leakage).
        if st.session_state["task_mode"] == "Regression" or fertility_derived_from_n:
            if "Nitrogen" in numeric_features:
                numeric_features.remove("Nitrogen")

        st.subheader("Feature Selection")
        st.markdown("Select numeric features to include in the model.")
        selected_features = st.multiselect(
            "Features", options=numeric_features, default=numeric_features
        )

        if not selected_features:
            st.warning("Select at least one feature.")
        else:
            X = df[selected_features]

            st.subheader("Hyperparameters")
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("n_estimators", 50, 500, 150, step=50)
            with col2:
                max_depth = st.slider("max_depth", 2, 50, 12)

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)

            test_size = st.slider("Test set fraction (%)", 10, 40, 20, step=5)
            deep_evaluation = st.checkbox(
                "Run cross-validation + permutation importance (slower)",
                value=False,
                help="Turn this on only when you need the more detailed evaluation. Leaving it off makes training much faster.",
            )

            # Keep class proportions in train/test for better evaluation on imbalanced data
            stratify_y = y if st.session_state["task_mode"] == "Classification" else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled_df,
                    y,
                    test_size=test_size / 100,
                    random_state=42,
                    stratify=stratify_y,
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled_df, y, test_size=test_size / 100, random_state=42
                )

            # High-class threshold (helps prevent "High" metrics from collapsing to zero on imbalanced data)
            high_threshold = None
            if st.session_state["task_mode"] == "Classification" and y is not None:
                y_counts = pd.Series(y).value_counts(dropna=False)
                try:
                    if ("High" in y_counts.index.astype(str).tolist()) and (y_counts.min() / max(1, y_counts.max()) < 0.2):
                        high_threshold = st.slider(
                            "High-class threshold",
                            min_value=0.05,
                            max_value=0.50,
                            value=0.20,
                            step=0.05,
                            help="Lower = more sensitive to High (higher recall). Higher = fewer High predictions (higher precision).",
                        )
                except Exception:
                    high_threshold = None

            if st.button("🚀 Train Random Forest"):
                if n_estimators > 300:
                    st.info(
                        "High n_estimators may take a while to train! "
                        "Consider lowering for faster results."
                    )
                with st.spinner("Training Random Forest..."):
                    if st.session_state["task_mode"] == "Classification":
                        # Handle class imbalance automatically (helps minority classes like "High")
                        class_weight = None
                        try:
                            y_counts = pd.Series(y_train).value_counts()
                            if y_counts.min() / max(1, y_counts.max()) < 0.2:
                                class_weight = "balanced_subsample"
                        except Exception:
                            class_weight = None

                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42,
                            n_jobs=-1,
                            class_weight=class_weight,
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42,
                            n_jobs=-1,
                        )

                    with st.spinner("Training Random Forest model..."):
                        model.fit(X_train, y_train)
                        # Default prediction
                        y_pred = model.predict(X_test)

                    # Adjust decision boundary for the minority "High" class (cost-sensitive thresholding)
                    if (
                        st.session_state["task_mode"] == "Classification"
                        and high_threshold is not None
                        and hasattr(model, "predict_proba")
                    ):
                        try:
                            proba = model.predict_proba(X_test)
                            classes = list(model.classes_)
                            if "High" in classes:
                                hi = classes.index("High")
                                adjusted = []
                                for p in proba:
                                    if p[hi] >= float(high_threshold):
                                        adjusted.append("High")
                                    else:
                                        p2 = p.copy()
                                        p2[hi] = -1.0
                                        adjusted.append(classes[int(np.argmax(p2))])
                                y_pred = np.array(adjusted)
                        except Exception:
                            pass

                    if deep_evaluation:
                        try:
                            if st.session_state["task_mode"] == "Classification":
                                cv_scores = cross_val_score(
                                    model,
                                    X_scaled_df,
                                    y,
                                    cv=5,
                                    scoring="accuracy",
                                )
                                cv_summary = {
                                    "mean_cv": float(np.mean(cv_scores)),
                                    "std_cv": float(np.std(cv_scores)),
                                }
                            else:
                                cv_res = cross_validate(
                                    model,
                                    X_scaled_df,
                                    y,
                                    cv=5,
                                    scoring={
                                        "r2": "r2",
                                        "rmse": "neg_root_mean_squared_error",
                                        "mae": "neg_mean_absolute_error",
                                    },
                                    return_train_score=False,
                                )
                                r2_scores = np.array(cv_res.get("test_r2", []), dtype=float)
                                rmse_scores = -np.array(cv_res.get("test_rmse", []), dtype=float)
                                mae_scores = -np.array(cv_res.get("test_mae", []), dtype=float)

                                cv_summary = {
                                    "mean_cv": float(np.mean(r2_scores)) if len(r2_scores) else None,
                                    "std_cv": float(np.std(r2_scores)) if len(r2_scores) else None,
                                    "r2_mean": float(np.mean(r2_scores)) if len(r2_scores) else None,
                                    "r2_std": float(np.std(r2_scores)) if len(r2_scores) else None,
                                    "rmse_mean": float(np.mean(rmse_scores)) if len(rmse_scores) else None,
                                    "rmse_std": float(np.std(rmse_scores)) if len(rmse_scores) else None,
                                    "mae_mean": float(np.mean(mae_scores)) if len(mae_scores) else None,
                                    "mae_std": float(np.std(mae_scores)) if len(mae_scores) else None,
                                }
                        except Exception:
                            cv_summary = None
                    else:
                        cv_summary = None

                    if deep_evaluation:
                        try:
                            perm_imp = permutation_importance(
                                model,
                                X_test,
                                y_test,
                                n_repeats=5,
                                random_state=42,
                                n_jobs=-1,
                            )
                            perm_df = pd.DataFrame(
                                {
                                    "feature": selected_features,
                                    "importance": perm_imp.importances_mean,
                                }
                            )
                            perm_df = perm_df.sort_values("importance", ascending=False)
                            perm_data = perm_df.to_dict("records")
                        except Exception:
                            perm_data = None
                    else:
                        perm_data = None

                    st.session_state["model"] = model
                    st.session_state["scaler"] = scaler
                    st.session_state["results"] = {
                        "task": st.session_state["task_mode"],
                        "y_test": y_test.tolist(),
                        "y_pred": np.array(y_pred).tolist(),
                        "model_name": f"Random Forest {st.session_state['task_mode']} Model",
                        "X_columns": selected_features,
                        "X_test_rows": X_test.head(500).to_dict("records"),
                        "feature_importances": model.feature_importances_.tolist(),
                        "cv_summary": cv_summary,
                        "permutation_importance": perm_data,
                    }
                    st.session_state["trained_on_features"] = selected_features
                    st.success(
                        "✅ Training completed. Go to 'Results' to inspect performance and explanations."
                    )


# =====================
        # Nutrient deficiency prediction
        # =====================
        st.markdown("---")
        st.subheader("🧪 Nutrient Deficiency Prediction")
        st.info(
            "Train additional Random Forest regressors to **predict nutrient levels** "
            "(Nitrogen / Phosphorus / Potassium) and help flag potential deficiencies.",
            icon="🧪",
        )

        with st.expander("Train nutrient-level predictors (N / P / K)", expanded=False):
            available_targets = [c for c in ["Nitrogen", "Phosphorus", "Potassium"] if c in df.columns]
            if not available_targets:
                st.info("N/P/K columns not found in your dataset. Upload data that contains nutrient columns to use this feature.")
            else:
                targets = st.multiselect(
                    "Select nutrient targets to predict",
                    options=available_targets,
                    default=available_targets[:1],
                    key="nutrient_targets_select",
                )

                base_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
                for loccol in ["Latitude", "Longitude"]:
                    if loccol in base_numeric:
                        base_numeric.remove(loccol)

                # Features exclude the selected target(s) to avoid leakage.
                candidate_features = [c for c in base_numeric if c not in set(targets)]
                if not candidate_features:
                    st.warning("No valid numeric features left after excluding the target(s).")
                else:
                    feat_nut = st.multiselect(
                        "Features used to predict the selected nutrient(s)",
                        options=candidate_features,
                        default=candidate_features,
                        key="nutrient_feature_select",
                        help="Tip: keep pH, Moisture, Organic Matter, and other soil indicators.",
                    )

                    colN1, colN2 = st.columns(2)
                    with colN1:
                        n_estimators_n = st.slider("n_estimators (nutrient models)", 50, 500, 200, 50, key="nut_n_estimators")
                    with colN2:
                        max_depth_n = st.slider("max_depth (nutrient models)", 2, 50, 14, key="nut_max_depth")

                    if st.button("🧠 Train nutrient predictors", key="train_nutrient_models_btn"):
                        if not targets or not feat_nut:
                            st.warning("Choose at least one target and at least one feature.")
                        else:
                            with st.spinner("Training nutrient prediction models..."):
                                # Scale features (consistent with the main modeling flow)
                                Xn = df[feat_nut].copy()
                                Xn = Xn.apply(pd.to_numeric, errors="coerce").fillna(Xn.median(numeric_only=True))

                                scaler_n = MinMaxScaler()
                                Xn_scaled = scaler_n.fit_transform(Xn)
                                Xn_scaled_df = pd.DataFrame(Xn_scaled, columns=feat_nut)

                                # Train one model per nutrient target
                                trained = {}
                                metrics = {}
                                for tgt in targets:
                                    with st.spinner(f"🔬 Training {tgt} predictor..."):
                                        yn = pd.to_numeric(df[tgt], errors="coerce")
                                        yn = yn.fillna(yn.median())

                                        X_train, X_test, y_train, y_test = train_test_split(
                                            Xn_scaled_df, yn, test_size=0.2, random_state=42
                                        )

                                        m_n = RandomForestRegressor(
                                            n_estimators=n_estimators_n,
                                            max_depth=max_depth_n,
                                            random_state=42,
                                            n_jobs=-1,
                                        )
                                        m_n.fit(X_train, y_train)
                                        pred_n = m_n.predict(X_test)

                                        rmse_n = float(np.sqrt(mean_squared_error(y_test, pred_n)))
                                        r2_n = float(r2_score(y_test, pred_n))

                                        trained[tgt] = m_n
                                        metrics[tgt] = {"RMSE": rmse_n, "R2": r2_n}

                                # Store in session
                                st.session_state["nutrient_models"].update(trained)
                                # Use one scaler per run for all nutrient models (same feature set)
                                for tgt in targets:
                                    st.session_state["nutrient_scalers"][tgt] = scaler_n
                                    st.session_state["nutrient_features"][tgt] = feat_nut


                                st.success("✅ Nutrient predictors trained and saved in session.")
                                st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
                                st.caption("**Nutrient model metrics:** Higher **R²** and lower **RMSE/MAE** indicate better predictions for that nutrient. Use this to compare which nutrient is easiest/hardest to predict from your features.")

elif page == "📊 Visualization":
    render_page_header(
        "Visual Analytics",
        "Explore distributions, correlations, spatial patterns, and soil clusters in one organized workspace.",
        "Exploration",
    )

    st.write("")

    if st.session_state["df"] is None:
        st.info("Please upload data first in 'Home' (Upload Data is integrated there).")
    else:
        df = st.session_state["df"].copy()

        # Ensure a fertility label exists (used by several visuals)
        if "Nitrogen" in df.columns and "Fertility_Level" not in df.columns:
            df["Fertility_Level"] = create_fertility_label(df, col="Nitrogen", q=3)

        # ---- Dataset status header ----
        st.markdown("### Dataset Status")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        has_coords = ("Latitude" in df.columns) and ("Longitude" in df.columns)
        if has_coords:
            try:
                land_count = len(_cached_filter_mindanao_land_only(df))
            except Exception:
                land_count = 0
            c4.metric("Mindanao land points", f"{land_count:,}")
        else:
            c4.metric("Mindanao land points", "—")

        performance_mode = st.toggle(
            "⚡ Performance mode (recommended for large datasets)",
            value=True,
            help="Automatically samples rows for heavy plots (pairplots, cluster pairplots) to keep the app responsive.",
        )

        st.markdown("---")

        # ---- Organized tabs ----
        tab_validation, tab_eda, tab_spatial, tab_clusters = st.tabs(["📋 Data Quality", "🔎 EDA", "🗺️ Spatial", "🧩 Clusters"])

        # =====================
        # 📋 DATA QUALITY & VALIDATION
        # =====================
        with tab_validation:
            st.subheader("Data Quality Assessment")
            st.caption("Comprehensive validation, profiling, and quality metrics for your dataset.")
            
            # Get validation results
            with st.spinner("📊 Analyzing dataset quality..."):
                validation_results = validate_dataset(df)
            
            # Display key metrics
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Total Rows", f"{validation_results['total_rows']:,}")
            q2.metric("Total Columns", f"{validation_results['total_cols']}")
            q3.metric("Duplicates", f"{validation_results['duplicates']}")
            q4.metric("Memory (MB)", f"{validation_results['memory_mb']:.2f}")
            
            st.markdown("---")
            
            # Missing values analysis
            if validation_results['missing_values']:
                st.subheader("Missing Values Analysis")
                missing_df = pd.DataFrame([
                    {
                        "Column": col,
                        "Count": v["count"],
                        "Percentage": f"{v['percentage']}%"
                    }
                    for col, v in validation_results['missing_values'].items()
                ])
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values detected!")
            
            st.markdown("---")
            
            # Outlier detection
            if validation_results['outliers_detected']:
                st.subheader("Outliers Detected (IQR Method)")
                outlier_df = pd.DataFrame([
                    {
                        "Column": col,
                        "Count": v["count"],
                        "Percentage": f"{v['percentage']}%"
                    }
                    for col, v in validation_results['outliers_detected'].items()
                ])
                st.dataframe(outlier_df, use_container_width=True)
            else:
                st.info("ℹ️ No outliers detected using IQR method.")
            
            st.markdown("---")
            
            # Detailed column profiling
            st.subheader("Column Profile Report")
            with st.spinner("📋 Generating column profile report..."):
                profile_df = generate_data_profile_report(df)
            st.dataframe(profile_df, use_container_width=True)
            
            st.markdown("---")
            
            # Processing recommendations
            st.subheader("Processing Recommendations")
            recommendations = get_processing_recommendations(df)
            for rec in recommendations:
                st.info(rec)

        # =====================
        # 🔎 EDA
        # =====================
        with tab_eda:
            st.subheader("Parameter Overview")
            st.caption("Distributions of key soil parameters. Use this to spot skew, outliers, and typical ranges.")

            param_cols = [
                c
                for c in ["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture", "Organic Matter"]
                if c in df.columns
            ]

            if not param_cols:
                st.warning(
                    "No recognized parameter columns found. Example columns: pH, Nitrogen, Phosphorus, Potassium, Moisture, Organic Matter"
                )
            else:
                for col in param_cols:
                    fig = px.histogram(
                        df,
                        x=col,
                        nbins=30,
                        marginal="box",
                        title=f"Distribution: {col}",
                        color_discrete_sequence=[st.session_state["current_theme"]["primary_color"]],
                    )
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Histogram + box summary for **{col}**.")
                    st.markdown("---")

            st.subheader("Correlation Matrix")
            st.caption("How numeric features move together. Strong correlations can indicate redundancy or meaningful relationships.")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                exclude_like = {"latitude", "lat", "longitude", "lon", "lng", "long", "longitude_2"}
                default_cols = [c for c in numeric_cols if c.strip().lower() not in exclude_like][:12]
                selected_cols = st.multiselect(
                    "Select numeric columns (fewer columns = clearer heatmap)",
                    options=numeric_cols,
                    default=default_cols if len(default_cols) >= 2 else numeric_cols[: min(12, len(numeric_cols))],
                )

                if selected_cols is None or len(selected_cols) < 2:
                    st.info("Select at least 2 numeric columns to view the correlation matrix.")
                else:
                    show_values = st.checkbox(
                        "Show correlation values on cells (can get cluttered)",
                        value=False,
                    )

                    corr = df[selected_cols].corr()
                    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                    fig_corr = px.imshow(
                        corr,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        zmin=-1,
                        zmax=1,
                        aspect="auto",
                        title="Correlation Heatmap",
                    )
                    fig_corr.update_layout(
                        template="plotly_dark",
                        height=650,
                        margin=dict(l=40, r=40, t=70, b=40),
                    )
                    fig_corr.update_xaxes(tickangle=45, tickfont=dict(size=12))
                    fig_corr.update_yaxes(tickfont=dict(size=12))

                    if show_values and len(selected_cols) <= 15:
                        txt = corr.round(2).astype(str).values
                        fig_corr.update_traces(
                            text=txt,
                            texttemplate="%{text}",
                            textfont=dict(color="white", size=10),
                        )
                    else:
                        fig_corr.update_traces(text=None)

                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption("Correlation heatmap for the selected numeric features.")
            else:
                st.info("No numeric columns available for correlation matrix.")

            # ---- Notebook-style EDA (in expanders) ----
            st.markdown("---")
            st.subheader("📒 Notebook EDA (Colab-style)")
            st.caption("These visuals mirror the notebook’s EDA without cluttering the main flow.")

            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
                _has_sns = True
            except Exception:
                sns = None
                _has_sns = False

            def _pick_col(_df, candidates):
                for c in candidates:
                    if c in _df.columns:
                        return c
                return None

            fert_col = _pick_col(df, ["fertility_class", "Fertility_Level"])

            features = []
            for c in ["pH", "ph", "PH"]:
                if c in df.columns:
                    features.append(c)
                    break
            for c in ["Nitrogen", "Phosphorus", "Potassium", "Moisture", "Organic Matter", "Organic_Matter"]:
                if c in df.columns and c not in features:
                    features.append(c)

            for c in features:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            with st.expander("Pairplot (slow)", expanded=False):
                st.markdown("**What it shows:** relationships between every pair of key soil features.")
                if not _has_sns:
                    st.info("Seaborn is not installed, so pairplots are unavailable.")
                elif len(features) < 2:
                    st.info("Not enough numeric columns available for a pairplot.")
                elif not st.toggle(
                    "Render pairplot now",
                    value=False,
                    key="render_nb_pairplot",
                    help="Pairplots are expensive, so they stay off until you need them.",
                ):
                    st.info("Turn this on only when you want to generate the pairplot.")
                else:
                    max_n = 1500 if performance_mode else 6000
                    sample_n = st.slider(
                        "Sample size",
                        min_value=200,
                        max_value=max_n,
                        value=min(1500, max_n),
                        step=100,
                        key="nb_pairplot_sample",
                    )
                    d = df[features].dropna()
                    if len(d) > sample_n:
                        d = d.sample(sample_n, random_state=42)
                    g = sns.pairplot(d, diag_kind="kde")
                    st.pyplot(g.figure, clear_figure=True)
                    st.caption("Pairplot: diagonals show KDE distributions; off-diagonals show pairwise scatter.")

            with st.expander("Histograms + KDE (per feature)", expanded=False):
                st.markdown("**What it shows:** each feature’s distribution + density curve.")
                if not _has_sns:
                    st.info("Seaborn is not installed, so KDE histograms are unavailable.")
                elif not features:
                    st.info("No notebook numeric features found in this dataset.")
                elif not st.toggle(
                    "Render KDE histograms now",
                    value=False,
                    key="render_nb_histograms",
                    help="Keeps the page responsive by skipping these matplotlib charts until requested.",
                ):
                    st.info("Turn this on to generate the KDE histogram charts.")
                else:
                    chosen = st.multiselect("Choose features", features, default=features, key="nb_hist_features")
                    for f in chosen:
                        x = df[f].dropna()
                        if x.empty:
                            continue
                        fig, ax = plt.subplots(figsize=(9, 4))
                        sns.histplot(data=df, x=f, kde=True, ax=ax)
                        ax.set_title(f"Distribution of {f}")
                        ax.set_xlabel(f)
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig, clear_figure=True)
                        st.caption(f"Histogram + KDE for **{f}**.")

            with st.expander("Boxplots by Fertility (per feature)", expanded=False):
                st.markdown("**What it shows:** differences in feature distributions across fertility groups.")
                if not _has_sns:
                    st.info("Seaborn is not installed, so boxplots are unavailable.")
                elif fert_col is None:
                    st.info("Fertility label not found. Expected `fertility_class` or `Fertility_Level`.")
                elif not features:
                    st.info("No notebook numeric features found in this dataset.")
                elif not st.toggle(
                    "Render fertility boxplots now",
                    value=False,
                    key="render_nb_boxplots",
                    help="Keeps these charts from being rebuilt on every Streamlit rerun.",
                ):
                    st.info("Turn this on to generate the boxplots.")
                else:
                    chosen = st.multiselect(
                        "Choose features",
                        features,
                        default=features,
                        key="nb_box_features",
                    )
                    for f in chosen:
                        dtmp = df[[fert_col, f]].dropna()
                        if dtmp.empty:
                            continue
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.boxplot(data=dtmp, x=fert_col, y=f, ax=ax)
                        ax.set_title(f"{f} by {fert_col}")
                        ax.set_xlabel(fert_col)
                        ax.set_ylabel(f)
                        st.pyplot(fig, clear_figure=True)
                        st.caption(f"Boxplot of **{f}** grouped by **{fert_col}**.")

            with st.expander("Correlated Pairs (auto-selected)", expanded=False):
                st.markdown("**What it shows:** the strongest correlated feature pairs as scatterplots.")
                if not _has_sns:
                    st.info("Seaborn is not installed, so scatterplots are unavailable.")
                elif len(features) < 2:
                    st.info("Need at least 2 numeric features to compute correlations.")
                else:
                    threshold = st.slider("Correlation threshold (abs)", 0.05, 0.95, 0.10, 0.05, key="nb_corr_thr")
                    topk = st.slider("Max pairs to plot", 1, 25, 10, 1, key="nb_corr_topk")

                    corr2 = df[features].corr(numeric_only=True)
                    pairs = (
                        corr2.where(np.triu(np.ones(corr2.shape), k=1).astype(bool))
                        .stack()
                        .reset_index()
                    )
                    pairs.columns = ["Feature1", "Feature2", "Correlation"]
                    pairs["AbsCorr"] = pairs["Correlation"].abs()
                    pairs = pairs[pairs["AbsCorr"] >= threshold].sort_values("AbsCorr", ascending=False).head(topk)

                    if pairs.empty:
                        st.info("No correlated pairs meet the threshold.")
                    else:
                        st.dataframe(pairs[["Feature1", "Feature2", "Correlation"]], use_container_width=True)
                        st.caption("**Highly correlated pairs:** Values near **+1/-1** mean the two features move together strongly. Very high absolute correlation can suggest redundancy (you may keep one) or a meaningful relationship worth discussing.")
                        render_corr_plots = st.toggle(
                            "Render correlated-pair scatterplots now",
                            value=False,
                            key="render_nb_corr_plots",
                            help="The table is quick; the scatterplots are generated only when requested.",
                        )
                        if not render_corr_plots:
                            st.info("Turn this on to generate the scatterplots.")
                            pairs = pairs.iloc[0:0]
                        for _, r in pairs.iterrows():
                            f1, f2, cval = r["Feature1"], r["Feature2"], r["Correlation"]
                            dtmp = df[[f1, f2]].dropna()
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.scatterplot(data=dtmp, x=f1, y=f2, ax=ax, alpha=0.7)
                            ax.set_title(f"{f2} vs {f1} (corr={cval:.2f})")
                            st.pyplot(fig, clear_figure=True)
                            st.caption(f"Scatterplot for correlated pair **{f1} ↔ {f2}**.")

        # =====================
        # 🗺️ Spatial
        # =====================
        with tab_spatial:
            # =====================
            # Folium Fertility & Nutrient Maps
            # =====================

            # ---- Notebook spatial visuals ----
            st.markdown("---")
            st.subheader("📒 Notebook Spatial Visuals")
            st.caption("These replicate the notebook’s geo scatter and fertility folium maps, to hide offshore/outside points.")

            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
                _has_sns2 = True
            except Exception:
                sns = None
                _has_sns2 = False

            def _pick_col(_df, candidates):
                for c in candidates:
                    if c in _df.columns:
                        return c
                return None

            fert_col = _pick_col(df, ["fertility_class", "Fertility_Level"])
            lat_col = _pick_col(df, ["latitude", "Latitude"])
            lon_col = _pick_col(df, ["longitude_1", "Longitude", "longitude", "lon", "lng"])

            # Geo scatter (N/P/K)
            with st.expander("Geographical Distribution (N/P/K)", expanded=False):
                st.markdown("**What it shows:** where nutrient concentrations are higher/lower across locations.")
                if not _has_sns2:
                    st.info("Seaborn is not installed, so geo scatterplots are unavailable.")
                elif lat_col is None or lon_col is None:
                    st.info("Latitude/Longitude not found for geo scatterplots.")
                else:
                    nutrients = [c for c in ["Nitrogen", "Phosphorus", "Potassium"] if c in df.columns]
                    if not nutrients:
                        st.info("N/P/K columns not found.")
                    elif not st.toggle(
                        "Render geo scatterplots now",
                        value=False,
                        key="render_geo_scatterplots",
                        help="Geo filtering and matplotlib rendering can be slow on larger datasets.",
                    ):
                        st.info("Turn this on to generate the geographical nutrient scatterplots.")
                    else:
                        for n in nutrients:
                            dtmp = df[[lat_col, lon_col, n]].dropna().copy()

                            # Land-only filtering (hides offshore points without changing the main dataset)
                            dstd = dtmp.rename(columns={lat_col: "Latitude", lon_col: "Longitude"})
                            dstd = _cached_filter_mindanao_land_only(dstd)
                            dtmp = dstd.rename(columns={"Latitude": lat_col, "Longitude": lon_col})

                            if dtmp.empty:
                                continue
                            fig, ax = plt.subplots(figsize=(10, 6))
                            # Bin values into 3 levels so colors are meaningful and consistent
                            try:
                                q1, q2 = dtmp[n].quantile([0.33, 0.66]).values

                                def _bin(v):
                                    if pd.isna(v):
                                        return np.nan
                                    if v <= q1:
                                        return "Low"
                                    if v <= q2:
                                        return "Moderate"
                                    return "High"

                                dtmp["Level"] = dtmp[n].apply(_bin)
                            except Exception:
                                dtmp["Level"] = "Moderate"

                            palette = {"Low": "red", "Moderate": "orange", "High": "green"}

                            sns.scatterplot(
                                data=dtmp,
                                x=lon_col,
                                y=lat_col,
                                hue="Level",
                                palette=palette,
                                alpha=0.75,
                                ax=ax,
                                legend=True,
                            )
                            ax.set_title(f"Geographical Distribution of {n}")
                            ax.set_xlabel("Longitude")
                            ax.set_ylabel("Latitude")
                            ax.grid(True)
                            st.pyplot(fig, clear_figure=True)
                            st.caption(f"Geo scatter for **{n}**; marker size reflects concentration.")

            # Fertility Folium maps (overall + by class) with land filtering
            with st.expander("Fertility Folium Maps", expanded=False):
                st.markdown("**What it shows:** interactive fertility maps using your dataset’s fertility labels (Green/Orange/Red).")

                if fert_col is None:
                    st.info("Fertility label not found (expected `fertility_class` or `Fertility_Level`).")
                elif lat_col is None or lon_col is None:
                    st.info("Latitude/Longitude not found for folium mapping.")
                elif not st.toggle(
                    "Render interactive Folium maps now",
                    value=False,
                    key="render_folium_maps",
                    help="Interactive maps are one of the slowest parts of this app, so they are generated on demand.",
                ):
                    st.info("Turn this on to build the interactive fertility maps.")
                else:
                    color_map = {"Low": "red", "Moderate": "orange", "High": "green"}

                    base = df.dropna(subset=[lat_col, lon_col, fert_col]).copy()

                    # Keep the view clean by removing offshore points (does not change your main dataset)
                    base_std = base.rename(columns={lat_col: "Latitude", lon_col: "Longitude"}).copy()
                    base_std = _cached_filter_mindanao_land_only(base_std)

                    base_std[lat_col] = base_std["Latitude"]
                    base_std[lon_col] = base_std["Longitude"]

                    PH_BOUNDS = [[4.5, 116.8], [21.4, 127.1]]  # Philippines bounding box

                    def _render_folium_map(sub_df, title, description):
                        st.markdown(f"### {title}")
                        st.caption(description)

                        if sub_df.empty:
                            st.warning("No rows available for this map.")
                            return
                        if performance_mode and len(sub_df) > 1500:
                            sub_df = sub_df.sample(1500, random_state=42)
                            st.caption("Performance mode: showing a representative 1,500-point sample on this map.")

                        center_lat = float(sub_df[lat_col].mean())
                        center_lon = float(sub_df[lon_col].mean())

                        m = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=7,
                            tiles="OpenStreetMap",
                            control_scale=True,
                            min_zoom=5,
                            max_zoom=14,
                        )

                        # Lock map interaction to the Philippines (prevents zooming/panning far away)
                        m.options["maxBounds"] = PH_BOUNDS
                        m.options["maxBoundsViscosity"] = 1.0

                        for _, row in sub_df.iterrows():
                            fert = str(row[fert_col]).strip().title()
                            col = color_map.get(fert, "blue")
                            popup = (
                                f"<b>Fertility:</b> {fert}<br>"
                                f"<b>N:</b> {row.get('Nitrogen', np.nan)}<br>"
                                f"<b>P:</b> {row.get('Phosphorus', np.nan)}<br>"
                                f"<b>K:</b> {row.get('Potassium', np.nan)}"
                            )
                            folium.CircleMarker(
                                location=[float(row[lat_col]), float(row[lon_col])],
                                radius=5,
                                color=col,
                                fill=True,
                                fill_color=col,
                                fill_opacity=0.75,
                                popup=folium.Popup(popup, max_width=300),
                            ).add_to(m)

                        # Legend
                        legend_html = """
                        <div style="
                            position: fixed;
                            bottom: 30px;
                            left: 30px;
                            z-index: 9999;
                            background: rgba(255,255,255,0.92);
                            padding: 10px 12px;
                            border-radius: 10px;
                            color: #111;
                            font-size: 13px;
                            line-height: 18px;
                            border: 1px solid rgba(0,0,0,0.15);
                            ">
                            <div style="font-weight:700; margin-bottom:6px;">Fertility Level</div>
                            <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:green;margin-right:8px;"></span>High</div>
                            <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:orange;margin-right:8px;"></span>Moderate</div>
                            <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:red;margin-right:8px;"></span>Low</div>
                        </div>
                        """
                        m.get_root().html.add_child(folium.Element(legend_html))

                        with st.spinner(f"🗺️ Displaying {title}..."):
                            st_folium(m, width=1024, height=520)

                    # 3 maps: High, Moderate, Low (stacked one-by-one)
                    _render_folium_map(
                        base_std[base_std[fert_col].astype(str).str.title() == "High"],
                        "Map 1: High Fertility (Green)",
                        "Green points indicate high fertility areas—generally suitable for nutrient-demanding crops.",
                    )
                    _render_folium_map(
                        base_std[base_std[fert_col].astype(str).str.title() == "Moderate"],
                        "Map 2: Moderate Fertility (Orange)",
                        "Orange points indicate moderate fertility—productive with balanced fertilization and soil management.",
                    )
                    _render_folium_map(
                        base_std[base_std[fert_col].astype(str).str.title() == "Low"],
                        "Map 3: Low Fertility (Red)",
                        "Red points indicate low fertility—apply soil amendments or choose hardy crops.",
                    )

                    st.markdown("---")
                    st.subheader("Crop Guide by Map Color")

                    crop_guide = pd.DataFrame(
                        [
                            {
                                "Color": "Green",
                                "Fertility Level": "High",
                                "Suggested Crops": "Rice, corn, vegetables, banana",
                                "Why": "High nutrient availability supports high-demand crops and intensive production.",
                            },
                            {
                                "Color": "Orange",
                                "Fertility Level": "Moderate",
                                "Suggested Crops": "Corn, legumes (mungbean/peanut), coconut, cassava",
                                "Why": "Moderate nutrients—works well with balanced fertilization and organic matter improvement.",
                            },
                            {
                                "Color": "Red",
                                "Fertility Level": "Low",
                                "Suggested Crops": "Cassava, sweet potato, peanut, coconut (with amendments)",
                                "Why": "Low nutrients—choose hardy crops and improve soil using compost/lime/fertilizer as needed.",
                            },
                        ]
                    )
                    st.dataframe(crop_guide, use_container_width=True)
                    st.caption("**Crop guide by color:** A quick reference mapping each suitability color (Green/Orange/Red) to example crops. Use this table to interpret the map colors in practical agricultural terms.")


        # =====================
        # 🧩 Clusters
        # =====================
        with tab_clusters:
            st.subheader("KMeans Clustering (Notebook-style)")
            st.caption("Clusters reveal natural groupings in your soil profiles.")

            try:
                import seaborn as sns
                _has_sns3 = True
            except Exception:
                sns = None
                _has_sns3 = False

            features = []
            for c in ["pH", "ph", "PH"]:
                if c in df.columns:
                    features.append(c)
                    break
            for c in ["Nitrogen", "Phosphorus", "Potassium", "Moisture", "Organic Matter", "Organic_Matter"]:
                if c in df.columns and c not in features:
                    features.append(c)

            if not _has_sns3:
                st.info("Seaborn is not installed, so cluster pairplot is unavailable.")
            elif len(features) < 2:
                st.info("Not enough features for KMeans + pairplot.")
            elif not st.toggle(
                "Run KMeans cluster pairplot now",
                value=False,
                key="render_kmeans_pairplot",
                help="KMeans plus seaborn pairplots can be slow, so this runs only when requested.",
            ):
                st.info("Turn this on to run KMeans and generate the cluster pairplot.")
            else:
                k = st.slider("KMeans clusters (K)", 2, 8, 3, 1, key="kmeans_k")
                max_n = 1500 if performance_mode else 6000
                sample_n = st.slider(
                    "Sample size (cluster pairplot)",
                    min_value=200,
                    max_value=max_n,
                    value=min(1500, max_n),
                    step=100,
                    key="kmeans_sample",
                )

                X = df[features].copy()
                X = X.apply(pd.to_numeric, errors="coerce")
                X = X.fillna(X.median(numeric_only=True))
                Xs = StandardScaler().fit_transform(X)

                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(Xs)

                d = df[features].copy()
                d["Cluster"] = labels
                d = d.dropna()
                if len(d) > sample_n:
                    d = d.sample(sample_n, random_state=42)

                g = sns.pairplot(d, hue="Cluster", diag_kind="kde")
                st.pyplot(g.figure, clear_figure=True)
                st.caption("Pairplot colored by KMeans cluster. Use it to see which variables separate clusters.")


elif page == "📈 Results":
    render_page_header(
        "Model Results",
        "Review performance metrics, prediction quality, feature importance, and saved model outputs.",
        "Evaluation",
    )

    st.write("")
    if not st.session_state.get("results"):
        st.info(
            "No trained model in session. Train a model first (Modeling or Quick Model)."
        )
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])

        st.subheader("Model Summary")
        colA, colB = st.columns([3, 2])
        with colA:
            st.write(f"**Model:** {results.get('model_name', 'Random Forest')}")
            st.write(f"**Features:** {', '.join(results.get('X_columns', []))}")
            if results.get("cv_summary"):
                cv = results["cv_summary"]
                # Classification: one headline CV metric (accuracy)
                if task == "Classification":
                    if cv.get("mean_cv") is not None and cv.get("std_cv") is not None:
                        st.write(
                            f"Cross-val accuracy: **{cv['mean_cv']:.3f}** "
                            f"(std: {cv['std_cv']:.3f})"
                        )
                # Regression: report CV R² + RMSE + MAE (mean ± std)
                else:
                    if cv.get("r2_mean") is not None and cv.get("r2_std") is not None:
                        st.write(
                            f"Cross-val R²: **{cv['r2_mean']:.3f}** "
                            f"(std: {cv['r2_std']:.3f})"
                        )
                    if cv.get("rmse_mean") is not None and cv.get("rmse_std") is not None:
                        st.write(
                            f"Cross-val RMSE: **{cv['rmse_mean']:.3f}** "
                            f"(std: {cv['rmse_std']:.3f})"
                        )
                    if cv.get("mae_mean") is not None and cv.get("mae_std") is not None:
                        st.write(
                            f"Cross-val MAE: **{cv['mae_mean']:.3f}** "
                            f"(std: {cv['mae_std']:.3f})"
                        )
        with colB:
            if st.button("💾 Save Model"):
                if st.session_state.get("model"):
                    joblib.dump(st.session_state["model"], "rf_model.joblib")
                    st.success("Model saved as rf_model.joblib")
                else:
                    st.warning("No model in session to save.")
            if st.button("💾 Save Scaler"):
                if st.session_state.get("scaler"):
                    joblib.dump(st.session_state["scaler"], "scaler.joblib")
                    st.success("Scaler saved as scaler.joblib")
                else:
                    st.warning("No scaler in session to save.")
        st.markdown("---")

        metrics_col, explain_col = st.columns([2, 1])
        with metrics_col:
            st.subheader("Performance Metrics")
            st.caption("These scores summarize how well the model performs on the **test set**. Use them to justify model reliability.")
            if task == "Classification":
                try:
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{acc:.3f}")
                except Exception:
                    st.write("Accuracy N/A")

                st.markdown("**Confusion Matrix**")
                try:
                    cm = confusion_matrix(
                        y_test, y_pred, labels=["Low", "Moderate", "High"]
                    )
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title="Confusion Matrix (Low / Moderate / High)",
                    )
                    fig_cm.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    st.caption("**Confusion Matrix:** Rows are actual classes and columns are predicted classes. Values on the **diagonal** are correct predictions; off-diagonals show where the model confuses classes (e.g., High mislabeled as Moderate).")
                except Exception:
                    st.write("Confusion matrix not available")

                st.markdown("#### 📊 Classification Report (Detailed)")
                try:
                    rep = classification_report(
                        y_test, y_pred, output_dict=True
                    )
                    rep_df = pd.DataFrame(rep).transpose().reset_index()
                    rep_df.rename(columns={"index": "Class"}, inplace=True)
                    cols_order = [
                        "Class",
                        "precision",
                        "recall",
                        "f1-score",
                        "support",
                    ]
                    rep_df = rep_df[
                        [c for c in cols_order if c in rep_df.columns]
                    ]
                    st.dataframe(rep_df[cols_order], use_container_width=True)
                    st.caption("**How to read:** **Precision** = correctness when predicting a class; **Recall** = how many real cases were found; **F1** balances both. If **High** recall is low/zero, the model is missing High samples (often due to imbalance).")
                    # --- NEW: Classification metrics chart (per-class) ---
                    try:
                        plot_df = rep_df.copy()
                        plot_df["Class"] = plot_df["Class"].astype(str)
                        plot_df = plot_df[~plot_df["Class"].isin(["accuracy", "macro avg", "weighted avg"])]
                        metric_cols = [c for c in ["precision", "recall", "f1-score"] if c in plot_df.columns]
                        if len(plot_df) > 0 and len(metric_cols) > 0:
                            long_df = plot_df[["Class"] + metric_cols].melt(
                                id_vars="Class", var_name="Metric", value_name="Score"
                            )
                            fig_rep = px.bar(
                                long_df,
                                x="Class",
                                y="Score",
                                color="Metric",
                                barmode="group",
                                title="Classification Metrics by Class",
                            )
                            fig_rep.update_yaxes(range=[0, 1])
                            fig_rep.update_layout(xaxis_title="", yaxis_title="Score (0–1)")
                            st.plotly_chart(fig_rep, use_container_width=True)
                        st.caption("**Per-class metrics:** Compare Precision/Recall/F1 across Low/Moderate/High. A low **High** bar usually indicates class imbalance—use it to justify balancing/threshold tuning in your methodology.")
                    except Exception:
                        pass
                    # --- END NEW ---

                except Exception:
                    st.text(classification_report(y_test, y_pred))
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.metric("RMSE", f"{rmse:.3f}")
                st.metric("MAE", f"{mae:.3f}")
                st.metric("R²", f"{r2:.3f}")
                # Cross-validation summary (mean ± std)
                cv = results.get("cv_summary") or {}
                if cv.get("rmse_mean") is not None:
                    st.markdown("**Cross-validation (5-fold)**")
                    cv_tbl = pd.DataFrame([
                        {"Metric": "R²", "Mean": cv.get("r2_mean"), "Std": cv.get("r2_std")},
                        {"Metric": "RMSE", "Mean": cv.get("rmse_mean"), "Std": cv.get("rmse_std")},
                        {"Metric": "MAE", "Mean": cv.get("mae_mean"), "Std": cv.get("mae_std")},
                    ])
                    st.dataframe(cv_tbl, use_container_width=True)
                    st.caption("**Cross-validation (mean ± std):** Shows average performance across multiple splits. Lower **RMSE/MAE** and higher **R²** mean better generalization; a large **std** suggests unstable performance across folds.")


                df_res = pd.DataFrame(
                    {"Actual_Nitrogen": y_test, "Predicted_Nitrogen": y_pred}
                )
                st.markdown("**Sample predictions**")
                st.dataframe(df_res.head(10), use_container_width=True)
                st.caption("**Sample predictions:** Compare Actual vs Predicted values. Large gaps indicate where the model struggles; these can often be linked to unusual soil conditions or outlier inputs.")

                st.markdown("**Actual vs Predicted**")
                try:
                    fig1 = px.scatter(
                        df_res,
                        x="Actual_Nitrogen",
                        y="Predicted_Nitrogen",
                        trendline="ols",
                        title="Actual vs Predicted Nitrogen (Model Predictions)",
                    )
                    fig1.update_layout(template="plotly_dark")
                    st.plotly_chart(fig1, use_container_width=True)
                except Exception:
                    fig1 = px.scatter(
                        df_res,
                        x="Actual_Nitrogen",
                        y="Predicted_Nitrogen",
                        title="Actual vs Predicted Nitrogen (no trendline available)",
                    )
                    fig1.update_layout(template="plotly_dark")
                    st.plotly_chart(fig1, use_container_width=True)
                st.caption("**Actual vs Predicted:** Points close to the diagonal line mean accurate predictions. Systematic offsets (points consistently above/below) indicate bias; wide scatter indicates higher error.")

                df_res["residual"] = (
                    df_res["Actual_Nitrogen"] - df_res["Predicted_Nitrogen"]
                )
                fig_res = px.histogram(
                    df_res,
                    x="residual",
                    nbins=30,
                    title="Residual Distribution",
                )
                fig_res.update_layout(template="plotly_dark")
                st.plotly_chart(fig_res, use_container_width=True)
                st.caption("**Residuals (errors):** Centered around 0 is ideal. A wide spread means larger errors; strong skew may indicate the model struggles more for low or high nitrogen ranges.")

        with explain_col:
            st.subheader("What the metrics mean")
            if task == "Classification":
                st.markdown("- **Accuracy:** Overall fraction of correct predictions.")
                st.markdown(
                    "- **Confusion Matrix:** Rows = true classes, Columns = predicted classes."
                )
                st.markdown(
                    "- **Precision:** Of all predicted positive, how many were actually positive."
                )
                st.markdown(
                    "- **Recall:** Of all actual positive samples, how many were found."
                )
                st.markdown(
                    "- **F1-score:** Harmonic mean of precision and recall; balanced measure."
                )
            else:
                st.markdown(
                    "- **RMSE:** Root Mean Squared Error — lower is better; same units as target."
                )
                st.markdown(
                    "- **MAE:** Mean Absolute Error — average magnitude of errors."
                )
                st.markdown(
                    "- **R²:** Proportion of variance explained by the model (1 is perfect)."
                )

        st.markdown("---")

        st.subheader("🌳 Random Forest Feature Importance")
        st.caption("**Interpretation:** This explains which inputs the trained forest relied on most. Use it to sanity-check that the model focuses on meaningful soil factors.")
        feat_names = results.get("X_columns", [])
        fi_list = results.get("feature_importances", [])

        if fi_list and feat_names:
            fi_df = pd.DataFrame(
                {"feature": feat_names, "importance": fi_list}
            ).sort_values("importance", ascending=False)
            fig_fi = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Mean Decrease in Impurity (Feature Importance)",
            )
            fig_fi.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("**Feature importance (MDI):** Features higher on the chart contribute more to the forest’s decisions. Treat this as a *ranking*, not a causal claim (correlated features can share importance).")
            st.dataframe(fi_df, use_container_width=True)
            st.caption("**Importance table:** Same ranking as the chart in numeric form. Use this for reporting (e.g., top 5 drivers) in your thesis discussion.")
        else:
            st.info("Feature importances not available for this run.")

        perm_data = results.get("permutation_importance")
        if perm_data:
            st.subheader("🔁 Permutation Importance (robust importance)")
            st.caption("**Interpretation:** A feature is important if shuffling it causes a noticeable performance drop. This is often a more defensible importance measure for a thesis.")
            perm_df = pd.DataFrame(perm_data)
            perm_df = perm_df.sort_values("importance", ascending=False)
            fig_perm = px.bar(
                perm_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Permutation Importance",
            )
            fig_perm.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_perm, use_container_width=True)
            st.caption("**Permutation importance:** Measures how much performance drops when a feature is shuffled. This is usually more reliable than MDI when features are correlated.")
            st.dataframe(perm_df, use_container_width=True)
            st.caption("**Permutation table:** Bigger values mean the model depends more on that feature. Near-zero importance suggests the feature adds little predictive value in this model run.")
        else:
            st.info("Permutation importance not computed or unavailable.")

        # =====================
        # 🧠 SHAP Explanation (advanced) — Regression only
        # =====================
        if task != "Classification":
            try:
                import shap  # optional dependency
                import matplotlib.pyplot as plt
                _has_shap = True
            except Exception:
                shap = None
                plt = None
                _has_shap = False

            if not _has_shap:
                st.info("SHAP explanations are available if you install `shap` (add it to requirements.txt).")
            else:
                m = st.session_state.get("model")
                x_rows = results.get("X_test_rows") or []
                if m is not None and len(x_rows) > 0:
                    try:
                        X_explain = pd.DataFrame(x_rows)
                        # Keep it fast in Streamlit
                        if len(X_explain) > 300:
                            X_explain = X_explain.sample(300, random_state=42)

                        st.subheader("🧠 SHAP Explanation (advanced)")
                        st.caption("SHAP shows how each feature contributes to the model prediction.")

                        explainer = shap.TreeExplainer(m)
                        shap_vals = explainer.shap_values(X_explain)

                        plt.figure()
                        shap.summary_plot(shap_vals, X_explain, show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)
                        st.caption("**SHAP summary (beeswarm):** Shows how features push predictions higher/lower across many samples. Wider spread means the feature has stronger influence.")

                        plt.figure()
                        shap.summary_plot(shap_vals, X_explain, plot_type="bar", show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)
                        st.caption("**SHAP importance (bar):** Average absolute impact of each feature. Use this as an interpretable ranking to identify major influencing parameters.")
                    except Exception as e:
                        st.info(f"SHAP explanation could not be generated: {e}")


        if fi_list and feat_names:
            fi_pairs = list(zip(feat_names, fi_list))
            fi_pairs.sort(key=lambda x: x[1], reverse=True)
            top_feats = [name for name, _ in fi_pairs[:3]]
            st.markdown(
                f"_The model relies most on: **{', '.join(top_feats)}** when making predictions._"
            )

        st.markdown("---")

        st.subheader("🔍 Prediction Explorer — Custom Sample")
        model = st.session_state.get("model")
        scaler = st.session_state.get("scaler")
        df_full = st.session_state.get("df")

        if not model or not scaler:
            st.info(
                "Train a model and keep it in memory to use the prediction explorer."
            )
        elif not feat_names:
            st.info("Feature list is unavailable; retrain the model to populate it.")
        elif df_full is None:
            st.info("Original dataset not found; cannot derive input ranges.")
        else:
            with st.form("prediction_form"):
                if task == "Classification":
                    st.markdown("Provide a hypothetical soil sample to predict fertility class.")
                    submit_label = "Predict Fertility"
                else:
                    st.markdown("Provide a hypothetical soil sample to predict Nitrogen.")
                    submit_label = "Predict Nitrogen"

                input_values = {}
                for f in feat_names:
                    if f in df_full.columns and pd.api.types.is_numeric_dtype(
                        df_full[f]
                    ):
                        col_min = float(df_full[f].min())
                        col_max = float(df_full[f].max())
                        if col_min == col_max:
                            col_min -= 0.1
                            col_max += 0.1
                        default_val = float(df_full[f].median())
                        step = (col_max - col_min) / 100.0
                        if step <= 0:
                            step = 0.01
                        input_values[f] = st.slider(
                            f,
                            min_value=float(col_min),
                            max_value=float(col_max),
                            value=float(default_val),
                            step=float(step),
                        )
                    else:
                        input_values[f] = st.number_input(f, value=0.0)

                submitted = st.form_submit_button(submit_label)
            if submitted:
                sample_df = pd.DataFrame([input_values])
                try:
                    sample_scaled = scaler.transform(sample_df[feat_names])
                    pred = model.predict(sample_scaled)[0]

                    if task == "Classification":
                        pred_label = str(pred)
                        health_label, color, message = interpret_label(pred_label)
                        st.markdown(
                            f"**Predicted Fertility Class:** `{pred_label}`<br>"
                            f"**Soil Health Interpretation:** **{health_label}** — {message}",
                            unsafe_allow_html=True,
                        )
                    else:
                        pred_nitrogen = float(pred)
                        cv = results.get("cv_summary") or {}
                        expected_rmse = None
                        try:
                            if cv.get("rmse_mean") is not None:
                                expected_rmse = float(cv.get("rmse_mean"))
                            else:
                                expected_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                        except Exception:
                            expected_rmse = None

                        if expected_rmse is not None:
                            st.markdown(
                                f"**Predicted Nitrogen:** `{pred_nitrogen:.3f}` ± `{expected_rmse:.3f}` (expected error, RMSE)"
                            )
                        else:
                            st.markdown(
                                f"**Predicted Nitrogen:** `{pred_nitrogen:.3f}` (same units as your dataset)"
                            )
                        if (
                            df_full is not None
                            and "Nitrogen" in df_full.columns
                            and df_full["Nitrogen"].notna().sum() > 5
                        ):
                            q = df_full["Nitrogen"].quantile([0.33, 0.66])
                            low_th, high_th = q.iloc[0], q.iloc[1]
                            if pred_nitrogen <= low_th:
                                fert = "Low"
                            elif pred_nitrogen <= high_th:
                                fert = "Moderate"
                            else:
                                fert = "High"
                            health_label, color, message = interpret_label(fert)
                            st.markdown(
                                f"**Derived Fertility Category:** `{fert}`<br>"
                                f"**Soil Health Interpretation:** **{health_label}** — {message}",
                                unsafe_allow_html=True,
                            )

                    if fi_list and feat_names:
                        fi_pairs = list(zip(feat_names, fi_list))
                        fi_pairs.sort(key=lambda x: x[1], reverse=True)
                        top_reasons = ", ".join(
                            [name for name, _ in fi_pairs[:3]]
                        )
                        st.markdown(
                            f"_Model explanation:_ This model is globally most sensitive "
                            f"to **{top_reasons}** when predicting soil health."
                        )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# =====================
        # Objective 4 (Extension): Nutrient deficiency explorer (based on trained nutrient models)
        # =====================
        st.markdown("---")
        st.subheader("🧪 Nutrient Deficiency Explorer")
        st.caption(
            "If you trained the optional nutrient predictors in 🤖 Modeling, you can use them here to estimate "
            "N/P/K levels and flag potential nutrient gaps."
        )

        nutrient_models = st.session_state.get("nutrient_models", {})
        nutrient_features_map = st.session_state.get("nutrient_features", {})
        nutrient_scalers_map = st.session_state.get("nutrient_scalers", {})
        df_full = st.session_state.get("df")

        if not nutrient_models:
            st.info("No nutrient predictors found. Train them in 🤖 Modeling → 'Nutrient Deficiency Prediction'.")
        elif df_full is None:
            st.info("Dataset not found in session; cannot derive input ranges for nutrient explorer.")
        else:
            available_tgts = [t for t in ["Nitrogen", "Phosphorus", "Potassium"] if t in nutrient_models]
            if not available_tgts:
                st.info("Nutrient predictors are present, but none match Nitrogen/Phosphorus/Potassium.")
            else:
                # Union of features across available targets (so one form works for all)
                union_feats = []
                for t in available_tgts:
                    feats = nutrient_features_map.get(t, [])
                    for f in feats:
                        if f not in union_feats:
                            union_feats.append(f)

                with st.form("nutrient_deficiency_form"):
                    st.markdown("Provide a hypothetical soil sample, then predict nutrient levels.")
                    input_vals = {}
                    for f in union_feats:
                        if f in df_full.columns and pd.api.types.is_numeric_dtype(df_full[f]):
                            col_min = float(df_full[f].min())
                            col_max = float(df_full[f].max())
                            if col_min == col_max:
                                col_min -= 0.1
                                col_max += 0.1
                            default_val = float(df_full[f].median())
                            step = (col_max - col_min) / 100.0
                            if step <= 0:
                                step = 0.01
                            input_vals[f] = st.slider(
                                f,
                                min_value=float(col_min),
                                max_value=float(col_max),
                                value=float(default_val),
                                step=float(step),
                                key=f"nut_in_{f}",
                            )
                        else:
                            input_vals[f] = st.number_input(f, value=0.0, key=f"nut_in_num_{f}")

                    run_pred = st.form_submit_button("Predict N/P/K (Deficiency Check)")

                if run_pred:
                    sample = pd.DataFrame([input_vals])

                    def _bucket_by_quantiles(series: pd.Series, val: float):
                        q1, q2 = series.quantile([0.33, 0.66]).values
                        if val <= q1:
                            return "Low (Possible deficiency)"
                        if val <= q2:
                            return "Moderate"
                        return "High"

                    rows = []
                    for tgt in available_tgts:
                        model_t = nutrient_models.get(tgt)
                        scaler_t = nutrient_scalers_map.get(tgt)
                        feats_t = nutrient_features_map.get(tgt, [])

                        if (model_t is None) or (scaler_t is None) or (not feats_t):
                            continue
                        if any(f not in sample.columns for f in feats_t):
                            continue

                        try:
                            Xs = sample[feats_t]
                            Xs_scaled = scaler_t.transform(Xs)
                            pred_val = float(model_t.predict(Xs_scaled)[0])

                            if tgt in df_full.columns:
                                status = _bucket_by_quantiles(df_full[tgt].dropna(), pred_val)
                            else:
                                status = "—"

                            rows.append({"Nutrient": tgt, "Predicted": round(pred_val, 4), "Status": status})
                        except Exception as e:
                            rows.append({"Nutrient": tgt, "Predicted": "Error", "Status": str(e)})

                    if rows:
                        out_df = pd.DataFrame(rows)
                        st.dataframe(out_df, use_container_width=True)
                        st.caption("**Nutrient status table:** Predicted nutrient levels are flagged against your dataset-based thresholds (Low/Moderate/High). Use this to quickly spot which nutrients may be limiting for a chosen sample.")

                        st.markdown("**General guidance (non-prescriptive):**")
                        st.markdown("- **Low Nitrogen** → consider organic matter, legume rotation, or N fertilization (crop-dependent).")
                        st.markdown("- **Low Phosphorus** → consider phosphate amendments and pH management (P availability depends on pH).")
                        st.markdown("- **Low Potassium** → consider potash sources and residue/compost management.")
                        st.caption("Always validate with lab results and local agronomy recommendations for your crop and soil type.")
                    else:
                        st.info("No predictions were produced. Ensure your nutrient models and feature columns match this dataset.")

elif page == "🌿 Insights":
    render_page_header(
        "Soil Health Insights",
        "Translate soil conditions and model outputs into crop suitability and management guidance.",
        "Recommendations",
    )
    if st.session_state["df"] is None:
        st.info("Upload and preprocess a dataset first (Home).")
    else:
        df = st.session_state["df"].copy()
        features = [
            "pH",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Moisture",
            "Organic Matter",
        ]

        st.session_state["location_tag"] = st.text_input(
            "Location / farm name (for context in reports)",
            value=st.session_state.get("location_tag", ""),
        )
        context_label = (
            st.session_state["location_tag"].strip()
            if st.session_state["location_tag"]
            else "this dataset"
        )

        if all(f in df.columns for f in features):
            median_row = df[features].median()
            overall_score = compute_suitability_score(median_row, features=features)
            label, color_hex = suitability_color(overall_score)
            crops = recommend_crops_for_sample(median_row, top_n=3)
            crop_list = ", ".join([c[0] for c in crops])

            fertility_map = {
                "Green": (
                    "Good",
                    "🟢",
                    "Fertility Level: Good",
                    "Soil health is optimal for cropping and sustainability.",
                ),
                "Orange": (
                    "Moderate",
                    "🟠",
                    "Fertility Level: Moderate",
                    "Soil is moderately fertile. Soil improvement can boost yield.",
                ),
                "Red": (
                    "Poor",
                    "🔴",
                    "Fertility Level: Poor",
                    "Soil has low fertility; significant amendments are needed.",
                ),
            }
            level_text, circle, level_label, description = fertility_map[label]

            st.markdown(
                f"""
            <div style="
                border:2.5px solid {color_hex};
                border-radius:18px;
                background: linear-gradient(100deg, {color_hex}22 0%, #f4fff4 100%);
                padding:28px 22px 16px 22px;
                margin-bottom:34px;
                box-shadow:0 0px 32px 0px {color_hex}33;
                text-align:left;">
                <h3 style='margin-top:0;margin-bottom:0.2em;'>
                    {circle}
                    <span style="
                        color:{color_hex};
                        font-size:33px;
                        font-weight:bold;
                        vertical-align:middle;">
                        {level_label}
                    </span>
                </h3>
                <div style='font-size:18px;font-weight:600;padding-top:2px;'>
                    Soil health summary for <b>{context_label}</b>.
                </div>
                <div style='font-size:20px;font-weight:600;padding-top:6px;'>
                    {description}
                </div>
                <div style='font-size:16px;padding-top:8px;'>
                    <b>Recommended crops (overall suitability):</b> 
                    <span style='color:{color_hex};font-weight:700'>{crop_list}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.subheader("Dataset overview")
        st.write(f"Samples: {df.shape[0]}  — Columns: {df.shape[1]}")
        st.markdown("---")

        st.subheader("Sample-level suitability & agriculture validation")

        display_cols = [
            c
            for c in [
                "Province",
                "pH",
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Moisture",
                "Organic Matter",
                "Latitude",
                "Longitude",
                "Fertility_Level",
            ]
            if c in df.columns
        ]

        preview = df[display_cols].copy()
        preview["suitability_score"] = preview.apply(
            lambda r: compute_suitability_score(
                r,
                features=[
                    "pH",
                    "Nitrogen",
                    "Phosphorus",
                    "Potassium",
                    "Moisture",
                    "Organic Matter",
                ],
            ),
            axis=1,
        )
        preview["suitability_label"] = preview["suitability_score"].apply(
            lambda s: suitability_color(s)[0]
        )
        preview["suitability_hex"] = preview["suitability_score"].apply(
            lambda s: suitability_color(s)[1]
        )

        def _rec_small(row):
            s = recommend_crops_for_sample(row, top_n=3)
            return ", ".join([f"{c} ({score:.2f})" for c, score in s])

        preview["top_crops"] = preview.apply(lambda r: _rec_small(r), axis=1)

        def agriculture_verdict(row):
            score = row["suitability_score"]
            label, hex_color = suitability_color(score)
            crops = recommend_crops_for_sample(row, top_n=3)
            crop_names = ", ".join([c[0] for c in crops])
            if label == "Green":
                verdict = (
                    f"🟢 Soil is sustainable for cropping. "
                    f"Ideal crops: {crop_names}."
                )
            elif label == "Orange":
                verdict = (
                    f"🟠 Soil is moderately suitable. "
                    f"Consider minor fertilizer or pH adjustment. "
                    f"Crops that may grow: {crop_names}."
                )
            else:
                verdict = (
                    f"🔴 Soil is NOT recommended for cropping without amendments. "
                    f"Major improvement needed. Possible crops (with treatment): "
                    f"{crop_names}."
                )
            return verdict

        preview["verdict"] = preview.apply(agriculture_verdict, axis=1)

        col_g, col_o, col_r = st.columns(3)
        for label_name, col_obj in zip(
            ["Green", "Orange", "Red"], [col_g, col_o, col_r]
        ):
            cnt = int((preview["suitability_label"] == label_name).sum())
            col_obj.metric(f"{label_name} samples", cnt)

        st.dataframe(
            preview[
                [
                    col
                    for col in [
                        "Province",
                        "pH",
                        "Nitrogen",
                        "Phosphorus",
                        "Potassium",
                        "Moisture",
                        "Organic Matter",
                        "Fertility_Level",
                        "suitability_score",
                        "suitability_label",
                        "top_crops",
                        "verdict",
                    ]
                    if col in preview.columns
                ]
            ],
            use_container_width=True,
        )
        st.caption("**Soil health preview:** Each row shows the computed suitability score/label (Green/Orange/Red) and suggested crop groupings. Use this to verify the scoring makes sense before summarizing by province.")
        st.markdown("---")

        if "Province" in preview.columns:
            st.subheader("Per-province soil health summary")
            prov_summary = (
                preview.groupby("Province")
                .agg(
                    samples=("suitability_score", "count"),
                    avg_suitability=("suitability_score", "mean"),
                    green_samples=(
                        "suitability_label",
                        lambda x: (x == "Green").sum(),
                    ),
                    orange_samples=(
                        "suitability_label",
                        lambda x: (x == "Orange").sum(),
                    ),
                    red_samples=(
                        "suitability_label",
                        lambda x: (x == "Red").sum(),
                    ),
                )
                .reset_index()
            )
            prov_summary["avg_suitability"] = prov_summary["avg_suitability"].round(3)
            st.dataframe(prov_summary, use_container_width=True)
            st.caption("**Province summary:** Aggregates average suitability and counts per color. This helps compare areas and identify where soil improvement programs may be prioritized.")
            st.markdown("---")

        st.markdown("### Soil Suitability Color Legend")
        st.markdown(
            """
        <style>
        .legend-table {
            width: 97%;
            margin: 0 auto;
            background: rgba(255,255,255,0.06);
            border-radius: 11px;
            border: 1.4px solid #eee;
            box-shadow: 0 4px 16px #0001;
            font-size:17px;
        }
        .legend-table td {
            padding:10px 16px;
        }
        </style>
        <table class="legend-table">
          <tr>
            <td><span style="color:#2ecc71;font-weight:900;font-size:20px;">🟢 Green</span></td>
            <td><b>Good/Sustainable</b>. Soil is ideal for cropping.<br>
            <b>Recommended crops:</b> Rice, Corn, Cassava, Vegetables, Banana, Coconut.</td>
          </tr>
          <tr>
            <td><span style="color:#f39c12;font-weight:900;font-size:20px;">🟠 Orange</span></td>
            <td><b>Moderate</b>. Soil is OK but may require improvement.<br>
            <b>Actions:</b> Nutrient/fertilizer adjustment and checking pH.<br>
            <b>Crops:</b> Corn, Cassava, selected vegetables.</td>
          </tr>
          <tr>
            <td><span style="color:#e74c3c;font-weight:900;font-size:20px;">🔴 Red</span></td>
            <td><b>Poor/Unsuitable</b>. Not recommended for cropping.<br>
            <b>Actions:</b> Major improvement with organic matter, fertilizers, and pH correction.<br>
            <b>Crops:</b> Only hardy types after soil amendment.</td>
          </tr>
        </table>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---", unsafe_allow_html=True)

        st.subheader("Detailed crop evaluation for a specific soil sample")
        if df.shape[0] > 0:
            idx = st.number_input(
                "Select sample index (0-based)",
                min_value=0,
                max_value=int(df.shape[0] - 1),
                value=0,
                step=1,
            )
            sample_row = df.iloc[int(idx)]
            eval_df = build_crop_evaluation_table(sample_row, top_n=6)
            if not eval_df.empty:
                st.dataframe(eval_df, use_container_width=True)
                st.caption("**Crop suitability table:** Ranks crops for the selected sample based on your scoring logic. Higher scores indicate better match to the soil conditions represented by that row.")
            else:
                st.info(
                    "Unable to compute crop evaluation for this sample (missing values?)."
                )
        else:
            st.info("No samples available for detailed crop evaluation.")

        st.markdown("---")

        st.subheader("Soil pattern clustering (K-Means)")
        cluster_features = [f for f in features if f in df.columns]
        if len(cluster_features) < 2:
            st.info(
                "Need at least two numeric soil parameters (e.g., pH and Nitrogen) "
                "to run clustering."
            )
        else:
            n_clusters = st.slider("Number of clusters", 2, 5, 3, step=1)
            with st.spinner("🔗 Running KMeans clustering..."):
                clustered_df, kmeans_model = run_kmeans_on_df(
                    df, cluster_features, n_clusters=n_clusters
                )
            if clustered_df is None:
                st.info(
                    "Not enough valid rows to run clustering with the selected number of clusters."
                )
            else:
                counts = clustered_df["cluster"].value_counts().sort_index()
                st.write("Cluster sizes:")
                st.write(counts)

                x_feat = cluster_features[0]
                y_feat = cluster_features[1]
                fig_cluster = px.scatter(
                    clustered_df,
                    x=x_feat,
                    y=y_feat,
                    color="cluster",
                    title=f"K-Means clusters using {x_feat} vs {y_feat}",
                )
                fig_cluster.update_layout(template="plotly_dark")
                st.plotly_chart(fig_cluster, use_container_width=True)
                st.caption("**Clusters:** Each color is a group of samples with similar soil/environment patterns based on the selected features. Use this to discuss natural groupings or zones in your study area.")

elif page == "👤 About":
    render_page_header(
        "About the Makers",
        "Project contributors and development background.",
        "Team",
    )
    st.markdown("<div style='font-size:19px;'>Developed by:</div>", unsafe_allow_html=True)
    st.write("")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        render_profile("Andre Oneal A. Plaza", "andre.png")
    with col_b:
        render_profile("Rica Baliling", "rica.png")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:15px;color:#d4a954;font-weight:700;'>All glory to God.</div>",
        unsafe_allow_html=True,
    )
    st.write("Developed for a capstone project.")
