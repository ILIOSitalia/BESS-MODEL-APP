# app.py
from __future__ import annotations

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.models import ProjectData, CapexOpex, FinancialParameters, Revenues, MunicipalityFees
from src.engine import run_financial_model
from src.kpis import calc_kpis


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="BESS Financial Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# OPTIONAL PASSWORD (SAFE FOR DEPLOY)
# --------------------------------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if APP_PASSWORD:
    pw = st.sidebar.text_input("Password", type="password")
    if pw != APP_PASSWORD:
        st.stop()

# --------------------------------------------------
# INPUT HELPERS (NO MIXED TYPES)
# --------------------------------------------------
def int_input(label: str, key: str, value: int, min_value: int = 0, max_value: int = 10_000, step: int = 1) -> int:
    return int(st.number_input(label, key=key, value=int(value), min_value=min_value, max_value=max_value, step=int(step)))

def float_input(label: str, key: str, value: float, min_value: float = 0.0, max_value: float = 1e12, step: float = 0.1) -> float:
    return float(st.number_input(label, key=key, value=float(value), min_value=float(min_value), max_value=float(max_value), step=float(step)))

def pct_input(label: str, key_dec: str, key_ui: str, default_dec: float, min_pct: float = 0.0, max_pct: float = 100.0, step: float = 0.1) -> float:
    # store as decimal in key_dec; UI shows percent in key_ui
    if key_dec not in st.session_state:
        st.session_state[key_dec] = float(default_dec)
    if key_ui not in st.session_state:
        st.session_state[key_ui] = float(st.session_state[key_dec]) * 100.0

    v_pct = float(st.number_input(label, key=key_ui, value=float(st.session_state[key_ui]), min_value=min_pct, max_value=max_pct, step=step))
    st.session_state[key_dec] = v_pct / 100.0
    return float(st.session_state[key_dec])

def eur0(x: float | int | None) -> str:
    if x is None:
        return "—"
    return f"{float(x):,.0f} €"


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("Controls")
    run_model_btn = st.button("Run model", type="primary")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tabs = st.tabs(["PROJECT", "CAPEX & OPEX", "FINANCIAL", "REVENUES", "RESULTS"])


# --------------------------------------------------
# TAB 1 — PROJECT
# --------------------------------------------------
with tabs[0]:
    st.header("Project data")

    pj_life = int_input("Project life (years)", "d_project_life", 20, 1, 60, 1)
    pj_power = float_input("Nominal power (MW)", "d_nominal_power", 50.0, 0.0, 1e6, 1.0)
    pj_energy = float_input("Nominal energy (MWh)", "d_nominal_energy", 200.0, 0.0, 1e9, 1.0)

    pj_cycles = float_input("Cycles per day", "d_cycles", 1.0, 0.0, 10.0, 0.1)
    pj_deg = pct_input("Annual degradation (%)", "d_deg", "d_deg_ui", 0.01, 0.0, 20.0, 0.1)

    pj_soc_min = pct_input("SOC min (%)", "d_soc_min", "d_soc_min_ui", 0.10, 0.0, 100.0, 1.0)
    pj_soc_max = pct_input("SOC max (%)", "d_soc_max", "d_soc_max_ui", 0.90, 0.0, 100.0, 1.0)
    pj_unav = pct_input("Grid & System unavailability (%)", "d_unav", "d_unav_ui", 0.02, 0.0, 50.0, 0.1)

    operating_days = 365.0 * (1.0 - pj_unav)
    cycled_energy = pj_energy * (pj_soc_max - pj_soc_min)
    annual_cycled_energy = pj_cycles * cycled_energy * operating_days

    c1, c2, c3 = st.columns(3)
    c1.metric("Operating days/year", f"{operating_days:,.1f}")
    c2.metric("Cycled Energy (MWh)", f"{cycled_energy:,.2f}")
    c3.metric("Annual Cycled Energy (MWh)", f"{annual_cycled_energy:,.0f}")


# --------------------------------------------------
# TAB 2 — CAPEX & OPEX
# --------------------------------------------------
with tabs[1]:
    st.header("CAPEX & OPEX")

    # NOTE: we keep the underlying key name the same as your model expects (per MWh)
    # If your model is power-based, we can swap to per MW in one line.
    capex_per_mwh = float_input("Initial CAPEX per MWh (€/MWh)", "d_capex_per_mwh", 250_000.0, 0.0, 1e9, 1_000.0)

    batt_share = pct_input("Battery share of CAPEX (%)", "d_batt_share", "d_batt_share_ui", 0.60, 0.0, 100.0, 1.0)
    aug_pct = pct_input("Augmentation cost (% of initial battery CAPEX per event) (%)", "d_aug_pct", "d_aug_pct_ui", 0.25, 0.0, 200.0, 1.0)
    aug_y1 = int_input("Augmentation year #1", "d_aug_y1", 0, 0, 60, 1)
    aug_y2 = int_input("Augmentation year #2", "d_aug_y2", 0, 0, 60, 1)

    om_fixed = float_input("Fixed O&M (€/MW·year)", "d_om_fixed", 8_000.0, 0.0, 1e9, 100.0)
    om_ins = float_input("Insurance + grid (€/MW·year)", "d_om_ins", 5_000.0, 0.0, 1e9, 100.0)
    decom_per_mw = float_input("Decommissioning (€/MW)", "d_decom", 15_000.0, 0.0, 1e9, 1_000.0)

    total_capex = pj_energy * capex_per_mwh
    total_opex = (om_fixed + om_ins) * pj_power

    aug_cost_each = total_capex * batt_share * aug_pct
    aug_events = len({y for y in [aug_y1, aug_y2] if y > 0})
    decom_cost = decom_per_mw * pj_power

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total CAPEX", eur0(total_capex))
    c2.metric("Annual OPEX", eur0(total_opex))
    c3.metric("Augmentation (per event)", eur0(aug_cost_each), f"Events: {aug_events}")
    c4.metric("Decommissioning", eur0(decom_cost))


# --------------------------------------------------
# TAB 3 — FINANCIAL
# --------------------------------------------------
with tabs[2]:
    st.header("Financial parameters")

    debt_pct = pct_input("Debt % on CAPEX (%)", "d_debt_pct", "d_debt_pct_ui", 0.60, 0.0, 100.0, 1.0)
    debt_tenor = int_input("Debt tenor (years)", "d_debt_tenor", 10, 0, 60, 1)
    interest = pct_input("Interest rate (%)", "d_interest", "d_interest_ui", 0.055, 0.0, 30.0, 0.1)
    amort_type = st.selectbox("Amortization type", ["annuity", "equal_principal"], index=0, key="d_amort_type")
    fees = pct_input("Debt up-front fees (%)", "d_debt_fees", "d_debt_fees_ui", 0.01, 0.0, 10.0, 0.1)

    ires = pct_input("IRES (%)", "d_ires", "d_ires_ui", 0.24, 0.0, 60.0, 0.1)
    irap = pct_input("IRAP (%)", "d_irap", "d_irap_ui", 0.039, 0.0, 20.0, 0.1)
    disc_eq = pct_input("Equity discount rate (%)", "d_disc_eq", "d_disc_eq_ui", 0.10, 0.0, 40.0, 0.1)

    total_tax = ires + irap
    debt_amt = total_capex * debt_pct
    eq_amt = total_capex - debt_amt

    # WACC-like: E*Re + D*Rd*(1 - IRES)
    E = 1.0 - debt_pct
    D = debt_pct
    wacc_like = E * disc_eq + D * interest * (1.0 - ires)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tax rate", f"{total_tax*100:.2f}%")
    c2.metric("Debt amount", eur0(debt_amt))
    c3.metric("Equity amount", eur0(eq_amt))
    c4.metric("WACC-like (Project)", f"{wacc_like*100:.2f}%")


# --------------------------------------------------
# TAB 4 — REVENUES (CM/MACSE + Tolling can coexist)
# --------------------------------------------------
with tabs[3]:
    st.header("Revenues")

    st.subheader("Floor")
    floor_type = st.radio("Floor type", ["CM", "MACSE"], key="d_floor_type", horizontal=True)

    if floor_type == "CM":
        cm_price = float_input("CM price (€/MW·year)", "d_cm_price", 60_000.0, 0.0, 1e9, 1_000.0)
        cm_share = pct_input("CM share of MW (%)", "d_cm_share", "d_cm_share_ui", 0.50, 0.0, 100.0, 1.0)
        cm_years = int_input("CM duration (years)", "d_cm_years", 5, 0, 60, 1)
        cm_esc = pct_input("CM escalation (%/year)", "d_cm_esc", "d_cm_esc_ui", 0.0, 0.0, 30.0, 0.1)
        macse_price = 0.0
        macse_share = 0.0
        macse_years = 0
        macse_esc = 0.0
    else:
        macse_price = float_input("MACSE price (€/MWh)", "d_macse_price", 90.0, 0.0, 1e6, 1.0)
        macse_share = pct_input("MACSE share of Nominal Energy (%)", "d_macse_share", "d_macse_share_ui", 0.50, 0.0, 100.0, 1.0)
        macse_years = int_input("MACSE duration (years)", "d_macse_years", 10, 0, 60, 1)
        macse_esc = pct_input("MACSE escalation (%/year)", "d_macse_esc", "d_macse_esc_ui", 0.0, 0.0, 30.0, 0.1)
        cm_price = 0.0
        cm_share = 0.0
        cm_years = 0
        cm_esc = 0.0

    st.subheader("Revenue mode (Tolling vs Merchant)")
    mode = st.radio("Mode", ["TOLLING", "MERCHANT"], key="d_mode", horizontal=True)

    # TOLLING (can coexist with CM/MACSE)
    if mode == "TOLLING":
        toll_base = float_input("Tolling base (€/MW·year)", "d_toll_base", 60_000.0, 0.0, 1e9, 1_000.0)
        toll_esc = pct_input("Tolling escalation (%/year)", "d_toll_esc", "d_toll_esc_ui", 0.02, 0.0, 50.0, 0.1)
        toll_extra = pct_input("Tolling extra income (% on Tolling Revenues)", "d_toll_extra", "d_toll_extra_ui", 0.0, 0.0, 100.0, 0.5)

        merchant_enabled = False
        merch_price = 0.0
        merch_esc = 0.0
    else:
        # MERCHANT (exclusive with Tolling)
        toll_base = 0.0
        toll_esc = 0.0
        toll_extra = 0.0

        merchant_enabled = True
        merch_price = float_input("Merchant price (€/MWh)", "d_merch_price", 120.0, 0.0, 1e6, 1.0)
        merch_esc = pct_input("Merchant escalation (%/year)", "d_merch_esc", "d_merch_esc_ui", 0.02, 0.0, 50.0, 0.1)

    st.subheader("Municipality royalties (optional)")
    muni_on = st.toggle("Enable municipality royalties?", key="d_muni_on")
    muni_pct = pct_input("Royalty rate (%)", "d_muni_pct", "d_muni_pct_ui", 0.03, 0.0, 20.0, 0.1)
    muni_upfront = st.toggle("Discount royalties and pay upfront at Year 0?", key="d_muni_upfront")
    muni_wacc = pct_input("Discount rate (WACC) (%)", "d_muni_wacc", "d_muni_wacc_ui", 0.08, 0.0, 40.0, 0.1)


# --------------------------------------------------
# TAB 5 — RESULTS
# --------------------------------------------------
with tabs[4]:
    st.header("Results")

    if not run_model_btn:
        st.info("Click **Run model** in the sidebar.")
    else:
        pj = ProjectData(
            project_life=pj_life,
            nominal_power_mw=pj_power,
            nominal_energy_mwh=pj_energy,
            degradation_rate=pj_deg,
            cycles_per_day=pj_cycles,
            soc_min=pj_soc_min,
            soc_max=pj_soc_max,
            grid_system_unavailability=pj_unav,
        )

        cx = CapexOpex(
            initial_capex_per_mwh=capex_per_mwh,
            battery_share_of_capex=batt_share,
            augmentation_cost_pct_of_batt_capex=aug_pct,
            augmentation_year_1=aug_y1,
            augmentation_year_2=aug_y2,
            fixed_om_per_mw_year=om_fixed,
            insurance_grid_per_mw_year=om_ins,
            decommissioning_per_mw=decom_per_mw,
        )

        fp = FinancialParameters(
            debt_tenor_years=debt_tenor,
            debt_pct_on_capex=debt_pct,
            interest_rate=interest,
            amortization_type=amort_type,
            debt_upfront_fees_pct=fees,
            ires=ires,
            irap=irap,
            depreciation_life_years=int_input("Depreciation life (years)", "d_depr_life", 15, 1, 60, 1),
            discount_rate_equity=disc_eq,
        )

        rv = Revenues(
            floor_type=floor_type,
            cm_price_per_mw_year=cm_price,
            cm_share_of_mw=cm_share,
            cm_duration_years=cm_years,
            cm_escalation=cm_esc,
            macse_price_per_mwh=macse_price,
            macse_share_of_nom_energy=macse_share,
            macse_duration_years=macse_years,
            macse_escalation=macse_esc,

            tolling_base_1_per_mw_year=toll_base,
            tolling_1_start_year=1 if toll_base > 0 else 0,
            tolling_1_end_year=pj_life if toll_base > 0 else 0,
            tolling_1_booked_cycles=pj_cycles if toll_base > 0 else 0.0,

            tolling_base_2_per_mw_year=0.0,
            tolling_2_start_year=0,
            tolling_2_end_year=0,
            tolling_2_booked_cycles=0.0,

            tolling_escalation=toll_esc,
            tolling_profit_sharing_pct=toll_extra,

            merchant_enabled=merchant_enabled,
            merchant_selling_price_per_mwh=merch_price,
            merchant_price_escalation=merch_esc,
        )

        mf = MunicipalityFees(
            enabled=bool(st.session_state.get("d_muni_on", False)),
            royalty_pct=float(st.session_state.get("d_muni_pct", 0.03)),
            discounted_upfront=bool(st.session_state.get("d_muni_upfront", False)),
            discount_rate_wacc=float(st.session_state.get("d_muni_wacc", 0.08)),
        )

        df = run_financial_model(pj, cx, fp, rv, mf, apply_degradation=True)
        kpi = calc_kpis(df, fp)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity IRR", "—" if kpi["equity_irr"] is None else f"{kpi['equity_irr']*100:.2f}%")
        c2.metric("Project IRR", "—" if kpi["project_irr"] is None else f"{kpi['project_irr']*100:.2f}%")
        c3.metric("Equity NPV", eur0(kpi["equity_npv"]))
        c4.metric("Min DSCR", "—" if kpi["min_dscr"] is None else f"{float(kpi['min_dscr']):.3f}")

        st.divider()
        st.subheader("Revenues breakdown")
        fig = px.line(df, x="Year", y=["Revenue_Floor", "Revenue_Tolling", "Revenue_Merchant", "Revenue_Total"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cash flows")
        fig2 = px.line(df, x="Year", y=["Project_FCF", "Equity_CF"])
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Results table")
        st.dataframe(df, use_container_width=True)