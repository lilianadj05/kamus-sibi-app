"""
Styling aplikasi — palet warna & font mengikuti tampilan Kamus SIBI
(sidebar navy gelap, aksen biru, header abu-abu, font Calibri).
"""

import datetime
import streamlit as st

# ── Palet warna (diambil dari referensi tampilan Kamus SIBI) ───────
NAVY_DARK   = "#2b3542"
NAVY_DARKER = "#232b36"
ACCENT_BLUE = "#1e88e5"
ACCENT_BLUE_LIGHT = "#2f9bff"
GRAY_HEADER = "#6c7680"
BG_LIGHT    = "#f4f6f9"
BLACK_TAG   = "#111111"
TEXT_DARK   = "#2b3542"
WHITE       = "#ffffff"

SIDEBAR_STYLES = {
    "container": {"padding": "0", "background-color": NAVY_DARK},
    "icon": {"color": "#c7ccd4", "font-size": "16px"},
    "nav-link": {
        "font-family": "Calibri, 'Carlito', 'Segoe UI', sans-serif",
        "font-size": "15px",
        "color": "#e4e7eb",
        "text-align": "left",
        "margin": "2px 0",
        "padding": "10px 16px",
        "border-radius": "0px",
        "--hover-color": NAVY_DARKER,
    },
    "nav-link-selected": {
        "background-color": ACCENT_BLUE,
        "color": WHITE,
        "font-weight": "600",
    },
    "menu-title": {
        "font-family": "Calibri, 'Carlito', 'Segoe UI', sans-serif",
        "color": WHITE,
    },
}


def inject_global_css():
    st.markdown(f"""
        <style>
        html, body, [class*="css"], .stMarkdown, .stText, p, span, div, button {{
            font-family: Calibri, 'Carlito', 'Segoe UI', sans-serif !important;
        }}

        .stApp {{
            background-color: {BG_LIGHT};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {NAVY_DARK};
        }}
        section[data-testid="stSidebar"] > div {{
            padding-top: 0rem;
        }}

        .sibi-topbar {{
            background: linear-gradient(90deg, {ACCENT_BLUE_LIGHT}, {ACCENT_BLUE});
            padding: 14px 24px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: white;
            margin-bottom: 18px;
        }}
        .sibi-topbar .title {{
            font-size: 18px;
            font-weight: 600;
        }}
        .sibi-topbar .meta {{
            font-size: 13px;
            opacity: 0.95;
        }}

        .sibi-section-header {{
            background-color: {GRAY_HEADER};
            color: white;
            padding: 10px 16px;
            border-radius: 4px;
            font-size: 15px;
            font-weight: 600;
            margin: 10px 0 14px 0;
        }}

        .sibi-card {{
            background-color: {WHITE};
            border-radius: 6px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin-bottom: 16px;
        }}

        .sibi-result-badge {{
            display: inline-block;
            background-color: {BLACK_TAG};
            color: white;
            font-size: 22px;
            font-weight: 700;
            padding: 10px 22px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }}

        .sibi-caption {{
            color: {GRAY_HEADER};
            font-size: 13px;
        }}

        .sibi-sidebar-title {{
            color: white;
            font-size: 19px;
            font-weight: 600;
            padding: 18px 16px;
            border-bottom: 1px solid {NAVY_DARKER};
            margin-bottom: 4px;
        }}

        div[data-testid="stMetricValue"] {{
            color: {ACCENT_BLUE};
        }}
        </style>
    """, unsafe_allow_html=True)


def render_topbar(page_title: str, breadcrumb: str):
    now = datetime.datetime.now()
    hari_map = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    hari = hari_map[now.weekday()]
    tanggal = now.strftime(f"{hari}, %d %B %Y")
    jam = now.strftime("%H:%M:%S")
    st.markdown(f"""
        <div class="sibi-topbar">
            <div class="title">📖 {page_title}</div>
            <div class="meta">{breadcrumb} &nbsp;|&nbsp; 🗓️ {tanggal} &nbsp; 🕒 {jam}</div>
        </div>
    """, unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="sibi-section-header">{text}</div>', unsafe_allow_html=True)
