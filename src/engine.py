from __future__ import annotations

import pandas as pd

from .models import ProjectData, CapexOpex, FinancialParameters, Revenues, MunicipalityFees
from .derived import derive_project, derive_capex_opex, derive_financial


def _degrade_factor(rate: float, year: int) -> float:
    # year 1 => factor 1.0
    return (1.0 - rate) ** max(0, year - 1)


def _annuity_payment(P: float, r: float, n: int) -> float:
    if n <= 0:
        return 0.0
    if r <= 0:
        return P / n
    return P * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def _debt_schedule(debt_amount: float, fp: FinancialParameters, project_life: int) -> pd.DataFrame:
    n = int(fp.debt_tenor_years)
    r = float(fp.interest_rate)

    years = list(range(0, project_life + 1))
    bal = debt_amount
    rows = []

    if n <= 0 or debt_amount <= 0:
        for y in years:
            rows.append(
                {
                    "Year": y,
                    "Debt_Open": 0.0,
                    "Interest": 0.0,
                    "Principal": 0.0,
                    "Debt_Service": 0.0,
                    "Debt_Close": 0.0,
                }
            )
        return pd.DataFrame(rows)

    if fp.amortization_type == "annuity":
        pay = _annuity_payment(debt_amount, r, n)
        for y in years:
            if y == 0:
                rows.append(
                    {
                        "Year": 0,
                        "Debt_Open": 0.0,
                        "Interest": 0.0,
                        "Principal": 0.0,
                        "Debt_Service": 0.0,
                        "Debt_Close": debt_amount,
                    }
                )
            elif 1 <= y <= n:
                interest = bal * r
                principal = max(0.0, pay - interest)
                principal = min(principal, bal)
                bal2 = bal - principal
                rows.append(
                    {
                        "Year": y,
                        "Debt_Open": bal,
                        "Interest": interest,
                        "Principal": principal,
                        "Debt_Service": interest + principal,
                        "Debt_Close": bal2,
                    }
                )
                bal = bal2
            else:
                rows.append(
                    {
                        "Year": y,
                        "Debt_Open": 0.0,
                        "Interest": 0.0,
                        "Principal": 0.0,
                        "Debt_Service": 0.0,
                        "Debt_Close": 0.0,
                    }
                )
    else:  # equal_principal
        principal_fixed = debt_amount / n
        for y in years:
            if y == 0:
                rows.append(
                    {
                        "Year": 0,
                        "Debt_Open": 0.0,
                        "Interest": 0.0,
                        "Principal": 0.0,
                        "Debt_Service": 0.0,
                        "Debt_Close": debt_amount,
                    }
                )
            elif 1 <= y <= n:
                interest = bal * r
                principal = min(principal_fixed, bal)
                bal2 = bal - principal
                rows.append(
                    {
                        "Year": y,
                        "Debt_Open": bal,
                        "Interest": interest,
                        "Principal": principal,
                        "Debt_Service": interest + principal,
                        "Debt_Close": bal2,
                    }
                )
                bal = bal2
            else:
                rows.append(
                    {
                        "Year": y,
                        "Debt_Open": 0.0,
                        "Interest": 0.0,
                        "Principal": 0.0,
                        "Debt_Service": 0.0,
                        "Debt_Close": 0.0,
                    }
                )

    return pd.DataFrame(rows)


def run_financial_model(
    pj: ProjectData,
    cx: CapexOpex,
    fp: FinancialParameters,
    rv: Revenues,
    mf: MunicipalityFees,
    apply_degradation: bool = True,
) -> pd.DataFrame:
    # Derived blocks
    dp = derive_project(pj)
    dc = derive_capex_opex(pj, cx)
    dfp = derive_financial(fp, dc.total_capex_eur)

    years = list(range(0, pj.project_life + 1))

    # ---------------------------
    # CAPEX / AUG / DECOM
    # ---------------------------
    capex0 = dc.total_capex_eur

    aug_years: list[int] = []
    if cx.augmentation_year_1 and cx.augmentation_year_1 <= pj.project_life:
        aug_years.append(cx.augmentation_year_1)
    if cx.augmentation_year_2 and cx.augmentation_year_2 <= pj.project_life and cx.augmentation_year_2 != cx.augmentation_year_1:
        aug_years.append(cx.augmentation_year_2)

    aug_cost_each = capex0 * cx.battery_share_of_capex * cx.augmentation_cost_pct_of_batt_capex

    capex_by_year = {0: capex0}
    for y in aug_years:
        capex_by_year[y] = capex_by_year.get(y, 0.0) + aug_cost_each

    decom_year = pj.project_life
    decom_cost = cx.decommissioning_per_mw * pj.nominal_power_mw

    # ---------------------------
    # DEBT
    # ---------------------------
    debt_amount = dfp.debt_amount_eur
    debt_fees = debt_amount * fp.debt_upfront_fees_pct
    debt_df = _debt_schedule(debt_amount, fp, pj.project_life).set_index("Year")

    # ---------------------------
    # DEPRECIATION (per tranche)
    # ---------------------------
    dep_life = int(min(fp.depreciation_life_years, pj.project_life))
    dep_by_year = {y: 0.0 for y in years}
    for cap_year, cap_amt in capex_by_year.items():
        if cap_amt <= 0:
            continue
        start = 1 if cap_year == 0 else cap_year
        for y in range(start, pj.project_life + 1):
            if y - start < dep_life:
                dep_by_year[y] += cap_amt / dep_life

    # ---------------------------
    # FLOOR SHARE by YEAR
    # ---------------------------
    def floor_share_y(y: int) -> float:
        if y <= 0:
            return 0.0
        if rv.floor_type == "CM":
            return float(rv.cm_share_of_mw) if y <= int(rv.cm_duration_years) else 0.0
        # MACSE
        return float(rv.macse_share_of_nom_energy) if y <= int(rv.macse_duration_years) else 0.0

    # ---------------------------
    # REVENUES
    # ---------------------------
    rev_floor: dict[int, float] = {0: 0.0}
    rev_toll: dict[int, float] = {0: 0.0}
    rev_merch: dict[int, float] = {0: 0.0}

    def _active(y: int, s: int, e: int) -> bool:
        return (s > 0 and e > 0 and s <= y <= e)

    for y in years:
        if y == 0:
            continue

        f_deg = _degrade_factor(pj.degradation_rate, y) if apply_degradation else 1.0
        power_y = pj.nominal_power_mw * f_deg
        energy_y = pj.nominal_energy_mwh * f_deg

        share = floor_share_y(y)
        avail_share = max(0.0, 1.0 - share)

        # ---- FLOOR revenue ----
        if rv.floor_type == "CM":
            if y <= int(rv.cm_duration_years):
                cm_price_y = float(rv.cm_price_per_mw_year) * ((1.0 + float(rv.cm_escalation)) ** (y - 1))
                rev_floor[y] = cm_price_y * power_y * float(rv.cm_share_of_mw)
            else:
                rev_floor[y] = 0.0
        else:
            # MACSE on Nominal Energy (as you requested)
            if y <= int(rv.macse_duration_years):
                macse_price_y = float(rv.macse_price_per_mwh) * ((1.0 + float(rv.macse_escalation)) ** (y - 1))
                rev_floor[y] = macse_price_y * energy_y * float(rv.macse_share_of_nom_energy)
            else:
                rev_floor[y] = 0.0

        # ---- Mutually exclusive: if merchant_enabled => tolling OFF ----
        if bool(rv.merchant_enabled):
            rev_toll[y] = 0.0
        else:
            toll = 0.0
            # Tolling base applies on "available MW" net-of-floor share for years where floor share>0
            # and on full MW after duration (share=0)
            if _active(y, int(rv.tolling_1_start_year), int(rv.tolling_1_end_year)):
                base_y = float(rv.tolling_base_1_per_mw_year) * ((1.0 + float(rv.tolling_escalation)) ** (y - 1))
                toll += base_y * power_y * avail_share
            if _active(y, int(rv.tolling_2_start_year), int(rv.tolling_2_end_year)):
                base2_y = float(rv.tolling_base_2_per_mw_year) * ((1.0 + float(rv.tolling_escalation)) ** (y - 1))
                toll += base2_y * power_y * avail_share

            # Extra income (% on tolling revenues)
            toll *= (1.0 + float(getattr(rv, "tolling_profit_sharing_pct", 0.0)))

            rev_toll[y] = toll

        # ---- MERCHANT ----
        if bool(rv.merchant_enabled):
            # Merchant on energy available net-of-floor share
            # and considering cycles/day (we do NOT use tolling booked cycles because tolling excluded in merchant mode)
            cycled_energy_y = energy_y * (pj.soc_max - pj.soc_min)
            annual_energy_y = cycled_energy_y * float(pj.cycles_per_day) * dp.operating_days_per_year

            annual_energy_y *= avail_share  # net-of-floor

            price_y = float(rv.merchant_selling_price_per_mwh) * ((1.0 + float(rv.merchant_price_escalation)) ** (y - 1))
            rev_merch[y] = annual_energy_y * price_y
        else:
            rev_merch[y] = 0.0

    # ---------------------------
    # ROYALTIES: on TOTAL REVENUES
    # ---------------------------
    # PV upfront (if discounted), paid at Year 0
    upfront_pv = 0.0
    if mf.enabled and mf.discounted_upfront:
        pv = 0.0
        for yy in range(1, pj.project_life + 1):
            revenue_total_yy = float(rev_floor.get(yy, 0.0)) + float(rev_toll.get(yy, 0.0)) + float(rev_merch.get(yy, 0.0))
            rt = revenue_total_yy * float(mf.royalty_pct)
            # Use (yy-1) so Year 1 is not additionally discounted (consistent “period 1” convention)
            pv += rt / ((1.0 + float(mf.discount_rate_wacc)) ** (yy - 1))
        upfront_pv = pv

    # ---------------------------
    # TAX LOSS CARRYFORWARD (IRES only, 80% cap)
    # ---------------------------
    loss_cf = 0.0

    rows = []
    for y in years:
        Revenue_Floor = float(rev_floor.get(y, 0.0))
        Revenue_Tolling = float(rev_toll.get(y, 0.0))
        Revenue_Merchant = float(rev_merch.get(y, 0.0))
        Revenue_Total = Revenue_Floor + Revenue_Tolling + Revenue_Merchant

        OPEX = 0.0 if y == 0 else float(dc.opex_eur_year)

        # Royalties on revenues
        Royalty_yearly = 0.0
        if mf.enabled and y >= 1 and (not mf.discounted_upfront):
            Royalty_yearly = Revenue_Total * float(mf.royalty_pct)

        Royalty_Upfront_y0 = upfront_pv if (mf.enabled and mf.discounted_upfront and y == 0) else 0.0

        EBITDA = Revenue_Total - OPEX - Royalty_yearly

        Depreciation = float(dep_by_year.get(y, 0.0))
        Interest = float(debt_df.loc[y, "Interest"]) if y in debt_df.index else 0.0
        Principal = float(debt_df.loc[y, "Principal"]) if y in debt_df.index else 0.0
        Debt_Service = float(debt_df.loc[y, "Debt_Service"]) if y in debt_df.index else 0.0
        Debt_Close = float(debt_df.loc[y, "Debt_Close"]) if y in debt_df.index else 0.0

        # EBT (accounting)
        EBT = EBITDA - Depreciation - Interest

        # IRES with loss CF and 80% cap
        ires_taxable = 0.0
        loss_used = 0.0

        if y >= 1:
            if EBT < 0:
                loss_cf += (-EBT)
                ires_taxable = 0.0
                loss_used = 0.0
            else:
                max_offset = 0.80 * EBT
                loss_used = min(loss_cf, max_offset)
                ires_taxable = max(0.0, EBT - loss_used)
                loss_cf = max(0.0, loss_cf - loss_used)

        IRES = ires_taxable * float(fp.ires)

        # IRAP approximation: max(0, EBITDA - Depreciation) (no loss CF)
        irap_base = max(0.0, EBITDA - Depreciation) if y >= 1 else 0.0
        IRAP = irap_base * float(fp.irap)

        Taxes = IRES + IRAP

        CAPEX = float(capex_by_year.get(y, 0.0))
        Decommissioning = float(decom_cost) if (y == decom_year and y != 0) else 0.0

        # CFADS (simplified) and DSCR
        CFADS = EBITDA - Taxes
        DSCR = None
        if Debt_Service > 0:
            DSCR = CFADS / Debt_Service

        # Project FCF: EBITDA - Taxes - CAPEX - Decom - upfront PV at y0
        Project_FCF = EBITDA - Taxes - CAPEX - Decommissioning - Royalty_Upfront_y0

        # Equity CF
        if y == 0:
            Equity_CF = -(CAPEX - debt_amount) - float(debt_fees) - Royalty_Upfront_y0
        else:
            Equity_CF = Project_FCF - Debt_Service

        rows.append(
            {
                "Year": y,
                "Revenue_Floor": Revenue_Floor,
                "Revenue_Tolling": Revenue_Tolling,
                "Revenue_Merchant": Revenue_Merchant,
                "Revenue_Total": Revenue_Total,

                "Municipality_Royalty": float(Royalty_yearly + Royalty_Upfront_y0),
                "Municipality_Royalty_Upfront": float(Royalty_Upfront_y0),

                "OPEX": OPEX,
                "EBITDA": EBITDA,

                "Depreciation": Depreciation,
                "Interest": Interest,
                "EBT": EBT,

                "Taxable_IRES": ires_taxable,
                "Loss_CF_End": loss_cf,
                "Loss_Used": loss_used,
                "IRES": IRES,
                "IRAP_Base": irap_base,
                "IRAP": IRAP,
                "Taxes": Taxes,

                "CAPEX": CAPEX,
                "Decommissioning": Decommissioning,

                "Debt_Close": Debt_Close,
                "Debt_Service": Debt_Service,
                "DSCR": DSCR,

                "Project_FCF": Project_FCF,
                "Equity_CF": Equity_CF,
            }
        )

    df = pd.DataFrame(rows)

    # helpful meta outputs for KPIs
    df["Debt_Amount"] = float(debt_amount)
    df["Debt_Fees"] = float(debt_fees)
    df["Discount_Rate_Equity"] = float(fp.discount_rate_equity)
    df["Discount_Rate_Project"] = float(dfp.discount_rate_project)
    df["Total_Tax_Rate"] = float(fp.ires + fp.irap)

    return df