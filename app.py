# app.py
from __future__ import annotations

from io import BytesIO
from datetime import datetime
import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

from src.models import ProjectData, CapexOpex, FinancialParameters, Revenues, MunicipalityFees
from src.engine import run_financial_model
from src.kpis import calc_kpis


# ----------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ----------------------------
st.set_page_config(
    page_title="BESS - Model App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# PASSWORD PROTECTION (AFTER set_page_config)
# ==================================================
_raw = os.getenv("APP_PASSWORDS", "")
_allowed = {p.strip() for p in _raw.split(",") if p.strip()}

if _allowed:
    pw = st.sidebar.text_input("🔒 Password", type="password")
    if pw not in _allowed:
        st.stop()


# ----------------------------
# PREMIUM STYLE
# ----------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2.2rem; max-width: 1250px; }
    h1, h2, h3 { letter-spacing: -0.2px; }
    .kpi-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 14px 14px;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    .kpi-title { font-size: 0.85rem; color: rgba(0,0,0,0.55); margin-bottom: 6px; }
    .kpi-value { font-size: 1.65rem; font-weight: 750; margin: 0; }
    .kpi-sub { font-size: 0.82rem; color: rgba(0,0,0,0.55); margin-top: 6px; }
    .section-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 18px;
        padding: 16px 16px;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        margin-bottom: 14px;
    }
    .section-title { font-size: 1.05rem; font-weight: 800; margin-bottom: 10px; }
    .note-footer {
        margin-top: 14px;
        padding-top: 8px;
        border-top: 1px solid rgba(0,0,0,0.08);
        font-size: 0.82rem;
        color: rgba(0,0,0,0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# CONSTANTS: FOOTER
# ----------------------------
FOOTER_IT = "(c) Copyright ILIOS S.r.l. – Strumento informativo non vincolante. Nessuna consulenza (legale/fiscale/finanziaria)."
FOOTER_EN = "(c) Copyright ILIOS S.r.l. – Non-binding informational tool. No legal/tax/financial advice."


def page_note():
    st.markdown(
        f"""
        <div class="note-footer">
            <div>{FOOTER_IT}</div>
            <div>{FOOTER_EN}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# FORMATTING HELPERS
# ----------------------------
def fmt_eur(x: float | int | None, decimals: int) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    s = f"{v:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def fmt_num(x: float | int | None, decimals: int) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    s = f"{v:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def fmt_pct_from_decimal(x: float | None, decimals: int) -> str:
    if x is None:
        return "—"
    try:
        v = float(x) * 100.0
    except Exception:
        return "—"
    s = f"{v:,.{decimals}f}%"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# PDF REPORT (PORTRAIT COVER + LANDSCAPE REST)
# ----------------------------
DARK_BLUE = colors.HexColor("#0B2A4A")  # barra header blu scuro


def _fmt_pct(x, d=2):
    return "—" if x is None else f"{float(x)*100:.{d}f}%"


def _fmt_eur(x, d=0):
    if x is None:
        return "—"
    return f"{float(x):,.{d}f} €"


def _plotly_png(fig, width=1600, height=900):
    try:
        return pio.to_image(fig, format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def build_pdf_report_investor(
    df: pd.DataFrame,
    kpi: dict,
    pj: ProjectData,
    cx: CapexOpex,
    fp: FinancialParameters,
    rv: Revenues,
    mf: MunicipalityFees,
    figs: dict,
    footer_it: str,
    footer_en: str,
) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    def set_portrait():
        c.setPageSize(A4)
        return A4

    def set_landscape():
        c.setPageSize(landscape(A4))
        return landscape(A4)

    def header_bar(title: str, subtitle: str, w: float, h: float):
        bar_h = 1.55 * cm
        c.setFillColor(DARK_BLUE)
        c.rect(0, h - bar_h, w, bar_h, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, h - 1.05 * cm, title)

        c.setFont("Helvetica", 8.8)
        c.drawRightString(w - 2 * cm, h - 1.08 * cm, subtitle)
        c.setFillColor(colors.black)

    def footer(w: float):
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(2 * cm, 1.2 * cm, footer_it)
        c.drawString(2 * cm, 0.7 * cm, footer_en)
        c.setFillColor(colors.black)

    def draw_kv_table(x: float, y: float, rows: list[tuple[str, str]], colw1: float, colw2: float, row_h: float):
        c.setFont("Helvetica", 9)
        for k, v in rows:
            c.setFillColor(colors.HexColor("#222222"))
            c.drawString(x, y, str(k))
            c.setFillColor(colors.black)
            c.drawRightString(x + colw1 + colw2, y, str(v))
            y -= row_h
        return y

    def draw_image_fit(img_bytes: bytes, x: float, y_top: float, w_box: float, h_box: float):
        img = ImageReader(BytesIO(img_bytes))
        iw, ih = img.getSize()
        aspect = iw / ih
        box_aspect = w_box / h_box
        if aspect >= box_aspect:
            ww = w_box
            hh = ww / aspect
        else:
            hh = h_box
            ww = hh * aspect
        c.drawImage(img, x + (w_box - ww) / 2, y_top - hh - (h_box - hh) / 2, width=ww, height=hh, mask="auto")

    def draw_data_preview(title: str, series: pd.DataFrame, x: float, y_top: float):
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.black)
        c.drawString(x, y_top - 0.8 * cm, title)

        c.setFont("Helvetica", 8)
        y = y_top - 1.4 * cm
        for i in range(min(len(series), 6)):
            row = series.iloc[i]
            c.drawString(x, y, f"Y{int(row['Year'])}: {float(row['Value']):.0f}")
            y -= 0.45 * cm

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # =========================
    # PAGE 1: PORTRAIT COVER/SUMMARY
    # =========================
    w, h = set_portrait()
    header_bar("BESS – Model App | Report", f"Generated: {generated}", w, h)

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.HexColor("#111111"))
    c.drawString(2 * cm, h - 3.0 * cm, "BESS Financial Model")

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#444444"))
    c.drawString(2 * cm, h - 3.65 * cm, f"Floor: {rv.floor_type} | Municipality Fees: {'YES' if mf.enabled else 'NO'}")
    c.setFillColor(colors.black)

    top_y = h - 4.7 * cm
    col_gap = 1.0 * cm
    left_x = 2 * cm
    right_x = w / 2 + col_gap / 2
    colw1 = 6.0 * cm
    colw2 = (w / 2 - 2 * cm - col_gap / 2)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_x, top_y, "Key KPIs")
    y_left = top_y - 0.65 * cm
    rows_kpi = [
        ("Equity IRR", _fmt_pct(kpi.get("equity_irr"))),
        ("Project IRR", _fmt_pct(kpi.get("project_irr"))),
        ("Equity NPV", _fmt_eur(kpi.get("equity_npv"), 0)),
        ("Project NPV", _fmt_eur(kpi.get("project_npv"), 0)),
        ("Min DSCR", "—" if kpi.get("min_dscr") is None else f"{float(kpi.get('min_dscr')):.3f}"),
        ("Discount rate (Equity)", _fmt_pct(kpi.get("discount_rate_equity"))),
        ("Discount rate (Project)", _fmt_pct(kpi.get("discount_rate_project"))),
    ]
    draw_kv_table(left_x, y_left, rows_kpi, colw1=colw1, colw2=colw2, row_h=0.55 * cm)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(right_x, top_y, "Inputs snapshot")
    y_right = top_y - 0.65 * cm
    rows_in = [
        ("Project life (y)", str(pj.project_life)),
        ("Nominal Power (MW)", f"{pj.nominal_power_mw:.2f}"),
        ("Nominal Energy (MWh)", f"{pj.nominal_energy_mwh:.2f}"),
        ("Degradation rate", _fmt_pct(pj.degradation_rate)),
        ("Cycles/day", f"{pj.cycles_per_day:.2f}"),
        ("SOC min / max", f"{pj.soc_min:.2f} / {pj.soc_max:.2f}"),
        ("Unavailability", _fmt_pct(pj.grid_system_unavailability)),
        ("CAPEX €/MW", f"{getattr(cx, 'initial_capex_per_mw', 0.0):,.0f}"),
        ("Debt %", _fmt_pct(fp.debt_pct_on_capex)),
        ("Interest rate", _fmt_pct(fp.interest_rate)),
        ("IRES / IRAP", f"{fp.ires*100:.1f}% / {fp.irap*100:.1f}%"),
        ("Municipality Fees %", _fmt_pct(mf.royalty_pct) if mf.enabled else "—"),
        ("Fees paid upfront", "YES" if (mf.enabled and mf.discounted_upfront) else "NO"),
    ]
    draw_kv_table(right_x, y_right, rows_in, colw1=colw1, colw2=colw2, row_h=0.48 * cm)

    footer(w)
    c.showPage()

    # =========================
    # PAGE 2: LANDSCAPE CHARTS
    # =========================
    w, h = set_landscape()
    header_bar("Charts", "Revenues and Cash Flows", w, h)

    box_w = (w - 4 * cm - 1.0 * cm) / 2
    box_h = h - 4.6 * cm - 2.2 * cm
    x1 = 2 * cm
    x2 = 2 * cm + box_w + 1.0 * cm
    y_top = h - 2.7 * cm

    # LEFT: rev_ebitda
    if "rev_ebitda" in figs:
        img = _plotly_png(figs["rev_ebitda"])
        if img:
            draw_image_fit(img, x1, y_top, box_w, box_h)
        else:
            tmp = df.loc[df["Year"] <= 6, ["Year", "Revenue_Total"]].copy()
            tmp.columns = ["Year", "Value"]
            draw_data_preview("Revenue_Total (preview)", tmp, x1, y_top)
            tmp2 = df.loc[df["Year"] <= 6, ["Year", "EBITDA"]].copy()
            tmp2.columns = ["Year", "Value"]
            draw_data_preview("EBITDA (preview)", tmp2, x1 + 7.0 * cm, y_top)

    # RIGHT: cf
    if "cf" in figs:
        img = _plotly_png(figs["cf"])
        if img:
            draw_image_fit(img, x2, y_top, box_w, box_h)
        else:
            tmp = df.loc[df["Year"] <= 6, ["Year", "Project_FCF"]].copy()
            tmp.columns = ["Year", "Value"]
            draw_data_preview("Project_FCF (preview)", tmp, x2, y_top)
            tmp2 = df.loc[df["Year"] <= 6, ["Year", "Equity_CF"]].copy()
            tmp2.columns = ["Year", "Value"]
            draw_data_preview("Equity_CF (preview)", tmp2, x2 + 7.0 * cm, y_top)

    footer(w)
    c.showPage()

    # =========================
    # PAGE 3: LANDSCAPE CHARTS
    # =========================
    w, h = set_landscape()
    header_bar("Charts", "Debt Service and DSCR", w, h)

    if "debt" in figs:
        img = _plotly_png(figs["debt"])
        if img:
            draw_image_fit(img, x1, y_top, box_w, box_h)
        else:
            tmp = df.loc[df["Year"] <= 6, ["Year", "Debt_Service"]].copy()
            tmp.columns = ["Year", "Value"]
            draw_data_preview("Debt_Service (preview)", tmp, x1, y_top)

    if "dscr" in figs:
        img = _plotly_png(figs["dscr"])
        if img:
            draw_image_fit(img, x2, y_top, box_w, box_h)
        else:
            tmp = df.loc[df["Year"] <= 6, ["Year", "DSCR"]].copy()
            tmp["DSCR"] = pd.to_numeric(tmp["DSCR"], errors="coerce").fillna(0.0)
            tmp.columns = ["Year", "Value"]
            draw_data_preview("DSCR (preview)", tmp, x2, y_top)

    footer(w)
    c.showPage()

    # =========================
    # RESULTS TABLE: LANDSCAPE (paginated) — FIXED FIT
    # =========================
    w, h = set_landscape()
    header_bar("Results table", "Main yearly outputs", w, h)

    cols = [
        "Year", "Revenue_Total", "Revenue_Floor", "Revenue_Tolling", "Revenue_Merchant",
        "OPEX", "Municipality_Royalty", "EBITDA",
        "Depreciation", "Interest", "EBT", "Taxes",
        "CAPEX", "Augmentation", "Cash Reserve",
        "Debt_Service", "DSCR", "Project_FCF", "Equity_CF"
    ]
    d = df.copy()
    for col in cols:
        if col not in d.columns:
            d[col] = 0.0
    d = d[cols].copy()

    c.setFont("Helvetica-Bold", 6.5)
    x0 = 1.2 * cm
    y = h - 2.9 * cm

    col_w = [
        1.0 * cm, 2.0 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm,
        1.4 * cm, 1.7 * cm, 1.4 * cm,
        1.4 * cm, 1.4 * cm, 1.4 * cm, 1.2 * cm,
        1.4 * cm, 1.4 * cm, 1.4 * cm,
        1.4 * cm, 1.0 * cm, 1.6 * cm, 1.6 * cm
    ]  # len = 19

    def draw_table_header(ypos):
        x = x0
        for i, col in enumerate(cols):
            c.drawString(x, ypos, col[:10])
            x += col_w[i]
        return ypos - 0.40 * cm

    y = draw_table_header(y)
    c.setFont("Helvetica", 6.0)

    max_y = 1.8 * cm
    for i in range(len(d)):
        if y < max_y:
            footer(w)
            c.showPage()
            w, h = set_landscape()
            header_bar("Results table (cont.)", "", w, h)
            c.setFont("Helvetica-Bold", 6.5)
            y = h - 2.9 * cm
            y = draw_table_header(y)
            c.setFont("Helvetica", 6.0)

        row = d.iloc[i]
        x = x0
        for j, col in enumerate(cols):
            v = row[col]
            if col == "Year":
                txt = str(int(v))
            elif col == "DSCR":
                txt = "" if pd.isna(v) else f"{float(v):.3f}"
            else:
                txt = "" if pd.isna(v) else f"{float(v):.0f}"
            c.drawString(x, y, txt[:10])
            x += col_w[j]
        y -= 0.35 * cm

    footer(w)
    c.save()
    return buf.getvalue()


# ----------------------------
# INPUT HELPERS
# ----------------------------
def num_input(label: str, key: str, default: float, min_value: float = 0.0, max_value: float = 1e12, step: float = 1.0):
    inp_dec = int(st.session_state.get("input_decimals", 2))
    fmt = f"%.{inp_dec}f"
    if key not in st.session_state:
        st.session_state[key] = default
    return st.number_input(label, min_value=min_value, max_value=max_value, value=float(st.session_state[key]), step=step, format=fmt, key=key)


def pct_input(label: str, key_decimal: str, key_ui: str, default_pct: float, min_pct: float = 0.0, max_pct: float = 100.0, step: float = 0.1):
    inp_dec = int(st.session_state.get("input_decimals", 2))
    fmt = f"%.{inp_dec}f"

    if key_ui not in st.session_state:
        if key_decimal in st.session_state:
            st.session_state[key_ui] = float(st.session_state[key_decimal]) * 100.0
        else:
            st.session_state[key_ui] = float(default_pct)

    val_pct = st.number_input(label, min_value=min_pct, max_value=max_pct, value=float(st.session_state[key_ui]), step=step, format=fmt, key=key_ui)
    st.session_state[key_decimal] = float(val_pct) / 100.0
    return st.session_state[key_decimal]


# ----------------------------
# BUILD OBJECTS FROM INPUTS (APPLY)
# ----------------------------
def build_draft_objects():
    project_life = int(st.session_state.get("d_project_life", 20))

    pj = ProjectData(
        project_life=project_life,
        nominal_power_mw=float(st.session_state.get("d_nominal_power", 50.0)),
        nominal_energy_mwh=float(st.session_state.get("d_nominal_energy", 200.0)),
        degradation_rate=float(st.session_state.get("d_deg_rate", 0.0)),
        cycles_per_day=float(st.session_state.get("d_cycles_per_day", 1.0)),
        soc_min=float(st.session_state.get("d_soc_min", 0.10)),
        soc_max=float(st.session_state.get("d_soc_max", 0.90)),
        grid_system_unavailability=float(st.session_state.get("d_grid_unavail", 0.02)),
    )

    cx = CapexOpex(
        initial_capex_per_mw=float(st.session_state.get("d_capex_per_mw", 250000.0)),
        battery_share_of_capex=float(st.session_state.get("d_batt_share", 0.60)),
        augmentation_cost_pct_of_batt_capex=float(st.session_state.get("d_aug_pct", 0.25)),
        augmentation_year_1=int(st.session_state.get("d_aug_y1", 0)),
        augmentation_year_2=int(st.session_state.get("d_aug_y2", 0)),
        fixed_om_per_mw_year=float(st.session_state.get("d_om_fixed", 8000.0)),
        insurance_grid_per_mw_year=float(st.session_state.get("d_om_ins", 5000.0)),
        decommissioning_per_mw=float(st.session_state.get("d_decom_per_mw", 15000.0)),
        land_cost_eur=float(st.session_state.get("d_land_cost", 0.0)),
    )

    fp = FinancialParameters(
        debt_tenor_years=int(st.session_state.get("d_debt_tenor", 10)),
        debt_pct_on_capex=float(st.session_state.get("d_debt_pct", 0.60)),
        interest_rate=float(st.session_state.get("d_interest", 0.055)),
        amortization_type=str(st.session_state.get("d_amort_type", "annuity")),
        debt_upfront_fees_pct=float(st.session_state.get("d_debt_fees", 0.01)),
        ires=float(st.session_state.get("d_ires", 0.24)),
        irap=float(st.session_state.get("d_irap", 0.039)),
        depreciation_life_years=int(st.session_state.get("d_depr_life", 15)),
        discount_rate_equity=float(st.session_state.get("d_disc_eq", 0.10)),
    )

    mf = MunicipalityFees(
        enabled=bool(st.session_state.get("d_muni_on", False)),
        royalty_pct=float(st.session_state.get("d_muni_pct", 0.03)),
        discounted_upfront=bool(st.session_state.get("d_muni_upfront", False)),
        discount_rate_wacc=float(st.session_state.get("d_muni_wacc", 0.08)),
    )

    floor_type = st.session_state.get("d_floor_type", "CM")

    tolling_base = float(st.session_state.get("d_toll_base", 60000.0))
    toll_years = int(st.session_state.get("d_toll_years", 0))

    merchant_allowed = (project_life > toll_years)
    merchant_enabled = bool(st.session_state.get("d_merch_on", False)) and merchant_allowed

    tolling_end_year = min(project_life, toll_years) if toll_years > 0 and tolling_base > 0 else 0
    tolling_start_year = 1 if tolling_end_year > 0 else 0

    rv = Revenues(
        floor_type=floor_type,

        cm_price_per_mw_year=float(st.session_state.get("d_cm_price", 60000.0)),
        cm_share_of_mw=float(st.session_state.get("d_cm_share", 0.50)),
        cm_duration_years=int(st.session_state.get("d_cm_years", 5)),
        cm_escalation=float(st.session_state.get("d_cm_esc", 0.0)),

        macse_price_per_mwh=float(st.session_state.get("d_macse_price", 90.0)),
        macse_share_of_nom_energy=float(st.session_state.get("d_macse_share", 0.50)),
        macse_duration_years=int(st.session_state.get("d_macse_years", 10)),
        macse_escalation=float(st.session_state.get("d_macse_esc", 0.0)),

        tolling_base_1_per_mw_year=tolling_base if tolling_end_year > 0 else 0.0,
        tolling_1_start_year=tolling_start_year,
        tolling_1_end_year=tolling_end_year,
        tolling_1_booked_cycles=float(st.session_state.get("d_cycles_per_day", 1.0)) if tolling_end_year > 0 else 0.0,

        tolling_base_2_per_mw_year=0.0,
        tolling_2_start_year=0,
        tolling_2_end_year=0,
        tolling_2_booked_cycles=0.0,

        tolling_escalation=float(st.session_state.get("d_toll_esc", 0.0)),
        tolling_profit_sharing_pct=float(st.session_state.get("d_toll_extra_pct", 0.0)),

        merchant_enabled=merchant_enabled,
        merchant_selling_price_per_mwh=float(st.session_state.get("d_merch_price", 120.0)),
        merchant_price_escalation=float(st.session_state.get("d_merch_esc", 0.02)),

        terminal_value_enabled=bool(st.session_state.get("d_tv_on", False)),
        terminal_value_per_mw=float(st.session_state.get("d_tv_per_mw", 0.0)),
    )

    apply_degradation = bool(st.session_state.get("d_degrade_energy", True))

    df = run_financial_model(pj, cx, fp, rv, mf, apply_degradation=apply_degradation)
    return pj, cx, fp, rv, mf, df


# ----------------------------
# TITLE
# ----------------------------
st.title("BESS - Model App")

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("Controls")

    if "input_decimals" not in st.session_state:
        st.session_state["input_decimals"] = 2
    if "display_decimals" not in st.session_state:
        st.session_state["display_decimals"] = 0
    if "d_degrade_energy" not in st.session_state:
        st.session_state["d_degrade_energy"] = True

    apply = st.button("Apply changes", type="primary")

    st.selectbox("Input decimals", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(st.session_state["input_decimals"]), key="input_decimals")
    st.selectbox("Display decimals", [0, 1, 2], index=[0, 1, 2].index(st.session_state["display_decimals"]), key="display_decimals")
    st.toggle("Apply degradation", key="d_degrade_energy")

    st.divider()
    st.caption("Percent inputs: type the % number (e.g., 5.5 = 5.5%). The app stores decimals internally.")

    if apply:
        pj, cx, fp, rv, mf, df = build_draft_objects()
        st.session_state.active_pj = pj
        st.session_state.active_cx = cx
        st.session_state.active_fp = fp
        st.session_state.active_rv = rv
        st.session_state.active_mf = mf
        st.session_state.active_df = df
        st.success("Changes applied")


# Ensure initial active model
if "active_df" not in st.session_state:
    try:
        pj, cx, fp, rv, mf, df = build_draft_objects()
        st.session_state.active_pj = pj
        st.session_state.active_cx = cx
        st.session_state.active_fp = fp
        st.session_state.active_rv = rv
        st.session_state.active_mf = mf
        st.session_state.active_df = df
    except Exception:
        st.session_state.active_df = None


# ----------------------------
# TABS
# ----------------------------
tabs = st.tabs(
    [
        "PROJECT DATA",
        "CAPEX & OPEX",
        "FINANCIAL PARAMETERS",
        "MUNICIPALITY FEES / ROYALTIES",
        "REVENUES",
        "TERMINAL VALUE",
        "RESULTS",
    ]
)

disp_dec = int(st.session_state.get("display_decimals", 0))


# ----------------------------
# TAB 1: PROJECT DATA
# ----------------------------
with tabs[0]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">PROJECT DATA</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Project Lifetime (years)", 1, 60, 20, 1, key="d_project_life")
        num_input("Nominal Power (MW)", "d_nominal_power", 50.0, 0.0, 1e6, 1.0)
        num_input("Nominal Energy (MWh)", "d_nominal_energy", 200.0, 0.0, 1e9, 1.0)
    with c2:
        pct_input("Degradation rate (annual) (%)", "d_deg_rate", "d_deg_rate_ui", 1.0, 0.0, 10.0, 0.1)
        num_input("Cycles per day", "d_cycles_per_day", 1.0, 0.0, 10.0, 0.1)
        pct_input("Grid & System unavailability (%)", "d_grid_unavail", "d_grid_unavail_ui", 2.0, 0.0, 30.0, 0.1)
    with c3:
        pct_input("SOC min (%)", "d_soc_min", "d_soc_min_ui", 10.0, 0.0, 100.0, 1.0)
        pct_input("SOC max (%)", "d_soc_max", "d_soc_max_ui", 90.0, 0.0, 100.0, 1.0)

    pj_tmp = ProjectData(
        project_life=int(st.session_state.get("d_project_life", 20)),
        nominal_power_mw=float(st.session_state.get("d_nominal_power", 50.0)),
        nominal_energy_mwh=float(st.session_state.get("d_nominal_energy", 200.0)),
        degradation_rate=float(st.session_state.get("d_deg_rate", 0.0)),
        cycles_per_day=float(st.session_state.get("d_cycles_per_day", 1.0)),
        soc_min=float(st.session_state.get("d_soc_min", 0.10)),
        soc_max=float(st.session_state.get("d_soc_max", 0.90)),
        grid_system_unavailability=float(st.session_state.get("d_grid_unavail", 0.02)),
    )

    operating_days = 365.0 * (1.0 - float(pj_tmp.grid_system_unavailability))
    cycled_energy = float(pj_tmp.nominal_energy_mwh) * (float(pj_tmp.soc_max) - float(pj_tmp.soc_min))
    annual_cycled_energy = float(pj_tmp.cycles_per_day) * cycled_energy * operating_days

    st.divider()
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Operating days/year", fmt_num(operating_days, 2), "365 × (1 − unavailability)")
    with k2:
        kpi_card("Cycled Energy (MWh)", fmt_num(cycled_energy, 2), "Nom. Energy × (SOCmax − SOCmin)")
    with k3:
        kpi_card("Annual Cycled Energy (MWh)", fmt_num(annual_cycled_energy, 0), "cycles/day × cycled × operating days")

    st.markdown("</div>", unsafe_allow_html=True)
    page_note()


# ----------------------------
# TAB 2: CAPEX & OPEX
# ----------------------------
with tabs[1]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">CAPEX & OPEX</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        num_input("Initial Capex per MWh (€/MWh)", "d_capex_per_mw", 250000.0, 0.0, 1e9, 1000.0)
        pct_input("Battery share of CAPEX (%)", "d_batt_share", "d_batt_share_ui", 60.0, 0.0, 100.0, 1.0)
        pct_input(
            "Augmentation cost (% of initial battery CAPEX per event) (%)",
            "d_aug_pct",
            "d_aug_pct_ui",
            25.0,
            0.0,
            200.0,
            1.0,
        )

    with c2:
        st.number_input("Augmentation year #1", 0, 60, 0, 1, key="d_aug_y1")
        st.number_input("Augmentation year #2", 0, 60, 0, 1, key="d_aug_y2")

        num_input("Decommissioning/end-of-life (€/MW)", "d_decom_per_mw", 15000.0, 0.0, 1e9, 1000.0)
        st.caption(
            "If any decommissioning cost, the model will automatically calculate annual cash reserves "
            "- these will not impact on EBITDA/EBIT."
        )

    with c3:
        num_input("Fixed O&M (€/MW·year)", "d_om_fixed", 8000.0, 0.0, 1e9, 100.0)
        num_input("Insurance + grid (€/MW·year)", "d_om_ins", 5000.0, 0.0, 1e9, 100.0)

        num_input("Land Cost (€)", "d_land_cost", 0.0, 0.0, 1e12, 1000.0)
        st.caption("Land cost including taxes - purchase or upfront lease at year-0")

    nominal_energy = float(st.session_state.get("d_nominal_energy", 200.0))
    nominal_power = float(st.session_state.get("d_nominal_power", 50.0))

    capex_per_mw = float(st.session_state.get("d_capex_per_mw", 250000.0))
    total_capex = nominal_energy * capex_per_mw

    om_fixed = float(st.session_state.get("d_om_fixed", 8000.0))
    om_ins = float(st.session_state.get("d_om_ins", 5000.0))
    total_om_per_mw = om_fixed + om_ins
    opex = total_om_per_mw * nominal_power

    batt_share = float(st.session_state.get("d_batt_share", 0.60))
    aug_pct = float(st.session_state.get("d_aug_pct", 0.25))
    aug_cost_each = total_capex * batt_share * aug_pct

    aug_y1 = int(st.session_state.get("d_aug_y1", 0))
    aug_y2 = int(st.session_state.get("d_aug_y2", 0))
    aug_events = len({y for y in [aug_y1, aug_y2] if y > 0})

    decom_per_mw = float(st.session_state.get("d_decom_per_mw", 15000.0))
    decom_cost = decom_per_mw * nominal_power

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Initial Total CAPEX (€)", fmt_eur(total_capex, 0), "Nominal Energy × CAPEX €/MWh")
    with k2:
        kpi_card("OPEX (€/year)", fmt_eur(opex, 0), "Total O&M/MW × Nominal Power")
    with k3:
        kpi_card("Augmentation (€)", fmt_eur(aug_cost_each, 0), f"Per event | Events: {aug_events}")
    with k4:
        kpi_card("Decommissioning (€)", fmt_eur(decom_cost, 0), "€ / MW × Nominal Power")

    st.markdown("</div>", unsafe_allow_html=True)
    page_note()


# ----------------------------
# TAB 3: FINANCIAL PARAMETERS
# ----------------------------
with tabs[2]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">FINANCIAL PARAMETERS</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Debt Tenor (years)", 0, 60, 10, 1, key="d_debt_tenor")
        pct_input("Debt % on Capex (%)", "d_debt_pct", "d_debt_pct_ui", 60.0, 0.0, 100.0, 1.0)
        pct_input("Interest rate (%)", "d_interest", "d_interest_ui", 5.5, 0.0, 30.0, 0.1)

    with c2:
        st.selectbox("Debt amortization type", ["annuity", "equal_principal"], index=0, key="d_amort_type")
        pct_input("Debt Up-front fees (%)", "d_debt_fees", "d_debt_fees_ui", 1.0, 0.0, 10.0, 0.1)
        st.number_input("Depreciation Life (years)", 1, 60, 15, 1, key="d_depr_life")

    with c3:
        pct_input("Corporate Taxes (IRES) (%)", "d_ires", "d_ires_ui", 24.0, 0.0, 60.0, 0.1)
        pct_input("IRAP (%)", "d_irap", "d_irap_ui", 3.9, 0.0, 20.0, 0.1)
        pct_input("Discount rate - Equity NPV (%)", "d_disc_eq", "d_disc_eq_ui", 10.0, 0.0, 40.0, 0.1)

    nominal_energy = float(st.session_state.get("d_nominal_energy", 200.0))
    capex_per_mw = float(st.session_state.get("d_capex_per_mw", 250000.0))
    total_capex = nominal_energy * capex_per_mw

    debt_pct = float(st.session_state.get("d_debt_pct", 0.60))
    debt_amt = total_capex * debt_pct
    eq_amt = total_capex - debt_amt

    ires = float(st.session_state.get("d_ires", 0.24))
    irap = float(st.session_state.get("d_irap", 0.039))
    total_tax = ires + irap

    re = float(st.session_state.get("d_disc_eq", 0.10))
    rd = float(st.session_state.get("d_interest", 0.055))
    D = debt_pct
    E = 1.0 - debt_pct
    wacc_like = E * re + D * rd * (1.0 - ires)

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Corporate Taxes", fmt_pct_from_decimal(total_tax, 2), "IRES + IRAP")
    with k2:
        kpi_card("Debt amount (€)", fmt_eur(debt_amt, 0), "Debt% × Total CAPEX")
    with k3:
        kpi_card("Equity amount (€)", fmt_eur(eq_amt, 0), "CAPEX − Debt")
    with k4:
        kpi_card("WACC-like (Project)", fmt_pct_from_decimal(wacc_like, 2), "E·Re + D·Rd·(1−IRES)")

    st.markdown("</div>", unsafe_allow_html=True)
    page_note()


# ----------------------------
# TAB 4: MUNICIPALITY FEES / ROYALTIES
# ----------------------------
with tabs[3]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">MUNICIPALITY FEES</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.toggle("Apply municipality fees?", key="d_muni_on")
    with c2:
        pct_input("Fee rate (%)", "d_muni_pct", "d_muni_pct_ui", 3.0, 0.0, 20.0, 0.1)
    with c3:
        st.toggle("Discount fees and pay upfront at Year 0?", key="d_muni_upfront")
        pct_input("Discount rate (WACC) (%)", "d_muni_wacc", "d_muni_wacc_ui", 8.0, 0.0, 40.0, 0.1)

    st.divider()
    df = st.session_state.get("active_df", None)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Run 'Apply changes' to compute royalties.")
    else:
        muni_on = bool(st.session_state.get("d_muni_on", False))
        upfront = bool(st.session_state.get("d_muni_upfront", False))

        if not muni_on:
            st.caption("Fees disabled.")
        else:
            if upfront and "Municipality_Royalty_Upfront" in df.columns:
                pv_val = float(df["Municipality_Royalty_Upfront"].iloc[0])
                kpi_card("Municipality Fees PV (paid upfront @ Year 0)", fmt_eur(pv_val, 0), "PV of future Municipality Fees (paid at Year 0)")
                tbl = df.loc[df["Year"] >= 0, ["Year", "Revenue_Total", "Municipality_Royalty", "Municipality_Royalty_Upfront"]].copy()
                st.dataframe(tbl, use_container_width=True)
            else:
                tbl = df.loc[df["Year"] >= 0, ["Year", "Revenue_Total", "Municipality_Royalty"]].copy()
                st.dataframe(tbl, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    page_note()


# ----------------------------
# TAB 5: REVENUES
# ----------------------------
with tabs[4]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">REVENUES</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.radio("Floor type", ["CM", "MACSE"], horizontal=True, key="d_floor_type")

        if st.session_state.get("d_floor_type", "CM") == "CM":
            num_input("CM price (€/MW·year)", "d_cm_price", 60000.0, 0.0, 1e9, 1000.0)
            pct_input("CM share of MW (%)", "d_cm_share", "d_cm_share_ui", 50.0, 0.0, 100.0, 1.0)
            st.number_input("CM duration (years)", 0, 60, 5, 1, key="d_cm_years")
            pct_input("CM escalation (%/year)", "d_cm_esc", "d_cm_esc_ui", 0.0, 0.0, 20.0, 0.1)
        else:
            num_input("MACSE price (€/MWh)", "d_macse_price", 90.0, 0.0, 1e6, 1.0)
            pct_input("MACSE share of Nominal Energy (%)", "d_macse_share", "d_macse_share_ui", 50.0, 0.0, 100.0, 1.0)
            st.number_input("MACSE duration (years)", 0, 60, 10, 1, key="d_macse_years")
            pct_input("MACSE escalation (%/year)", "d_macse_esc", "d_macse_esc_ui", 0.0, 0.0, 20.0, 0.1)

    with c2:
        st.markdown("**Tolling**")
        num_input("Tolling base (€/MW·year)", "d_toll_base", 60000.0, 0.0, 1e9, 1000.0)
        pct_input("Tolling escalation (%/year)", "d_toll_esc", "d_toll_esc_ui", 2.0, 0.0, 50.0, 0.1)
        pct_input("Tolling extra income (% on Tolling Revenues)", "d_toll_extra_pct", "d_toll_extra_pct_ui", 0.0, 0.0, 100.0, 0.5)
        st.number_input("Tolling duration (years)", 0, 60, 0, 1, key="d_toll_years")

        st.divider()

        project_life = int(st.session_state.get("d_project_life", 0))
        toll_years = int(st.session_state.get("d_toll_years", 0))
        can_enable_merchant = (project_life > toll_years)

        st.markdown("**Merchant (after Tolling)**")
        st.toggle("Enable Merchant after Tolling", key="d_merch_on", disabled=not can_enable_merchant)
        if not can_enable_merchant and bool(st.session_state.get("d_merch_on", False)):
            st.session_state["d_merch_on"] = False

        if bool(st.session_state.get("d_merch_on", False)) and can_enable_merchant:
            num_input("Merchant selling price (€/MWh)", "d_merch_price", 120.0, 0.0, 1e6, 1.0)
            pct_input("Merchant escalation (%/year)", "d_merch_esc", "d_merch_esc_ui", 2.0, 0.0, 50.0, 0.1)

    st.markdown("</div>", unsafe_allow_html=True)
    page_note()


# ----------------------------
# TAB 6: TERMINAL VALUE
# ----------------------------
with tabs[5]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">TERMINAL VALUE</div>', unsafe_allow_html=True)

    st.toggle("Enable Terminal Value?", key="d_tv_on")
    num_input("Terminal Value (€/MW)", "d_tv_per_mw", 0.0, 0.0, 1e12, 1000.0)
    st.caption("If enabled, Terminal Value will be added to the last year's total revenues (no dedicated column in RESULTS).")

    st.markdown("</div>", unsafe_allow_html=True)
    page_note()


# ----------------------------
# TAB 7: RESULTS
# ----------------------------
with tabs[6]:
    df = st.session_state.get("active_df", None)
    fp_active = st.session_state.get("active_fp", None)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">RESULTS</div>', unsafe_allow_html=True)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty or fp_active is None:
        st.info("Run 'Apply changes' to compute results.")
        st.markdown("</div>", unsafe_allow_html=True)
        page_note()
    else:
        kpi = calc_kpis(df, fp_active)

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            kpi_card("Equity IRR", fmt_pct_from_decimal(kpi.get("equity_irr"), 2), "Levered")
        with r2:
            kpi_card("Project IRR", fmt_pct_from_decimal(kpi.get("project_irr"), 2), "Unlevered")
        with r3:
            kpi_card("Equity NPV", fmt_eur(kpi.get("equity_npv"), 0), f"@ {fmt_pct_from_decimal(kpi.get('discount_rate_equity'), 2)}")
        with r4:
            kpi_card("Project NPV", fmt_eur(kpi.get("project_npv"), 0), "Project rate")
        with r5:
            md = kpi.get("min_dscr")
            kpi_card("Min DSCR", "—" if md is None else f"{float(md):.3f}", "CFADS / Debt Service")

        st.divider()

        cols_rev = ["Year", "Revenue_Floor", "Revenue_Tolling", "Revenue_Merchant", "Revenue_Total"]
        cols_pl = ["OPEX", "Municipality_Royalty", "EBITDA", "Depreciation", "Interest", "EBT", "Taxes"]
        cols_cf = ["CAPEX", "Augmentation", "Cash Reserve", "Debt_Service", "DSCR", "Project_FCF", "Equity_CF"]

        for col in cols_rev + cols_pl + cols_cf:
            if col not in df.columns:
                df[col] = 0.0

        show = df[cols_rev + cols_pl + cols_cf].copy()

        def format_df(d: pd.DataFrame) -> pd.DataFrame:
            out = d.copy()
            for ccol in out.columns:
                if ccol == "Year":
                    out[ccol] = out[ccol].astype(int)
                elif ccol == "DSCR":
                    out[ccol] = out[ccol].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
                else:
                    out[ccol] = out[ccol].apply(lambda x: "" if pd.isna(x) else fmt_num(float(x), int(st.session_state.get("display_decimals", 0))))
            return out

        st.dataframe(format_df(show), use_container_width=True)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(df, x="Year", y=["Revenue_Total", "EBITDA"], markers=False)
            st.plotly_chart(fig1, use_container_width=True, key="plot_results_rev_ebitda")
        with c2:
            fig2 = px.line(df, x="Year", y=["Project_FCF", "Equity_CF"], markers=False)
            st.plotly_chart(fig2, use_container_width=True, key="plot_results_cf")

        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.line(df, x="Year", y=["Debt_Service"], markers=False)
            st.plotly_chart(fig3, use_container_width=True, key="plot_results_debtservice")
        with c4:
            ds = pd.to_numeric(df["DSCR"], errors="coerce")
            dds = pd.DataFrame({"Year": df["Year"], "DSCR": ds})
            fig4 = px.line(dds, x="Year", y="DSCR", markers=False)
            st.plotly_chart(fig4, use_container_width=True, key="plot_results_dscr")

        st.divider()

        figs = {"rev_ebitda": fig1, "cf": fig2, "debt": fig3, "dscr": fig4}
        pdf_bytes = build_pdf_report_investor(
            df=df,
            kpi=kpi,
            pj=st.session_state.active_pj,
            cx=st.session_state.active_cx,
            fp=st.session_state.active_fp,
            rv=st.session_state.active_rv,
            mf=st.session_state.active_mf,
            figs=figs,
            footer_it=FOOTER_IT,
            footer_en=FOOTER_EN,
        )

        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="BESS_ModelApp_Report.pdf",
            mime="application/pdf",
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV (full model)",
            data=csv_bytes,
            file_name="BESS_Model_FullOutput.csv",
            mime="text/csv",
        )

        st.markdown("</div>", unsafe_allow_html=True)
        page_note()