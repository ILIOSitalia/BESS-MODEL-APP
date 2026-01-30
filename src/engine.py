# src/engine.py
from __future__ import annotations

import pandas as pd

from .models import ProjectData, CapexOpex, FinancialParameters, Revenues, MunicipalityFees
from .derived import derive_project, derive_capex_opex, derive_financial


def _degrade_factor(rate: float, year: int) -> float:
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
                principal = min(max(0.0, pay - interest), bal)
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

    dp = derive_project(pj)
    dc = derive_capex_opex(pj, cx)
    dfp = derive_financial(fp, dc.total_capex_eur)

    years = list(range(0, pj.project_life + 1))

    # ---------------------------
    # CAPEX / AUGMENTATION / DECOMMISSIONING
    # ---------------------------
    capex0 = dc.total_capex_eur

    # augmentation events
    aug_years: list[int] = []
    if cx.augmentation_year_1 and cx.augmentation_year_1 <= pj.project_life:
        aug_years.append(int(cx.augmentation_year_1))
    if cx.augmentation_year_2 and cx.augmentation_year_2 <= pj.project_life and cx.augmentation_year_2 != cx.augmentation_year_1:
        aug_years.append(int(cx.augmentation_year_2))

    aug_cost_each = capex0 * float(cx.battery_share_of_capex) * float(cx.augmentation_cost_pct_of_batt_capex)

    # CAPEX (initial only) and Augmentation (separate column)
    capex_by_year = {0: capex0}
    augmentation_by_year = {y: 0.0 for y in years}
    for y in aug_years:
        augmentation_by_year[y] += aug_cost_each

    decom_year = pj.project_life
    decom_cost = float(cx.decommissioning_per_mw) * float(pj.nominal_power_mw)

    # ---------------------------
    # DEBT
    # ---------------------------
    debt_amount = dfp.debt_amount_eur
    debt_fees = debt_amount * fp.debt_upfront_fees_pct
    debt_df = _debt_schedule(debt_amount, fp, pj.project_life).set_index("Year")

    # ---------------------------
    # DEPRECIATION (per tranche: initial + each augmentation)
    # ---------------------------
    dep_life = int(min(fp.depreciation_life_years, pj.project_life))
    dep_by_year = {y: 0.0 for y in years}

    # initial capex depreciates from year 1
    for y in range(1, pj.project_life + 1):
        if (y - 1) < dep_life:
            dep_by_year[y] += capex0 / dep_life

    # each augmentation depreciates from its year
    for ay in aug_years:
        for y in range(ay, pj.project_life + 1):
            if (y - ay) < dep_life:
                dep_by_year[y] += aug_cost_each / dep_life

    # ---------------------------
    # REVENUES (CM/MACSE + TOLLING/MERCHANT coexisting; no netting)
    # ---------------------------
    rev_floor, rev_toll, rev_merch = {}, {}, {}

    def _active(y, s, e):
        return s > 0 and e > 0 and s <= y <= e

    for y in years:
        if y == 0:
            rev_floor[y] = rev_toll[y] = rev_merch[y] = 0.0
            continue

        f_deg = _degrade_factor(pj.degradation_rate, y) if apply_degradation else 1.0
        power_y = pj.nominal_power_mw * f_deg
        energy_y = pj.nominal_energy_mwh * f_deg

        # FLOOR
        if rv.floor_type == "CM" and y <= rv.cm_duration_years:
            price = rv.cm_price_per_mw_year * ((1 + rv.cm_escalation) ** (y - 1))
            rev_floor[y] = price * power_y * rv.cm_share_of_mw
        elif rv.floor_type == "MACSE" and y <= rv.macse_duration_years:
            price = rv.macse_price_per_mwh * ((1 + rv.macse_escalation) ** (y - 1))
            rev_floor[y] = price * energy_y * rv.macse_share_of_nom_energy
        else:
            rev_floor[y] = 0.0

        # TOLLING (full MW)
        if (not rv.merchant_enabled) and _active(y, rv.tolling_1_start_year, rv.tolling_1_end_year):
            base = rv.tolling_base_1_per_mw_year * ((1 + rv.tolling_escalation) ** (y - 1))
            rev_toll[y] = base * power_y * (1.0 + rv.tolling_profit_sharing_pct)
        else:
            rev_toll[y] = 0.0

        # MERCHANT (full energy)
        if rv.merchant_enabled:
            cycled = energy_y * (pj.soc_max - pj.soc_min)
            annual_energy = cycled * pj.cycles_per_day * dp.operating_days_per_year
            price = rv.merchant_selling_price_per_mwh * ((1 + rv.merchant_price_escalation) ** (y - 1))
            rev_merch[y] = annual_energy * price
        else:
            rev_merch[y] = 0.0

    # ---------------------------
    # ROYALTIES (on total revenues)
    # ---------------------------
    upfront_pv = 0.0
    if mf.enabled and mf.discounted_upfront:
        for yy in range(1, pj.project_life + 1):
            total_yy = rev_floor[yy] + rev_toll[yy] + rev_merch[yy]
            upfront_pv += total_yy * mf.royalty_pct / ((1 + mf.discount_rate_wacc) ** (yy - 1))

    # ---------------------------
    # TAX & CASH FLOWS
    # ---------------------------
    loss_cf = 0.0
    rows = []

    for y in years:
        Revenue_Floor = rev_floor[y]
        Revenue_Tolling = rev_toll[y]
        Revenue_Merchant = rev_merch[y]
        Revenue_Total = Revenue_Floor + Revenue_Tolling + Revenue_Merchant

        OPEX = 0.0 if y == 0 else dc.opex_eur_year

        Royalty_yearly = 0.0
        if mf.enabled and y >= 1 and not mf.discounted_upfront:
            Royalty_yearly = Revenue_Total * mf.royalty_pct

        Royalty_Upfront_y0 = upfront_pv if (mf.enabled and mf.discounted_upfront and y == 0) else 0.0

        EBITDA = Revenue_Total - OPEX - Royalty_yearly

        Depreciation = dep_by_year[y]
        Interest = debt_df.loc[y, "Interest"] if y in debt_df.index else 0.0
        Principal = debt_df.loc[y, "Principal"] if y in debt_df.index else 0.0
        Debt_Service = debt_df.loc[y, "Debt_Service"] if y in debt_df.index else 0.0
        Debt_Close = debt_df.loc[y, "Debt_Close"] if y in debt_df.index else 0.0

        EBT = EBITDA - Depreciation - Interest

        ires_taxable = 0.0
        loss_used = 0.0
        if y >= 1:
            if EBT < 0:
                loss_cf += -EBT
            else:
                max_offset = 0.8 * EBT
                loss_used = min(loss_cf, max_offset)
                ires_taxable = EBT - loss_used
                loss_cf -= loss_used

        IRES = ires_taxable * fp.ires
        irap_base = max(0.0, EBITDA - Depreciation) if y >= 1 else 0.0
        IRAP = irap_base * fp.irap
        Taxes = IRES + IRAP

        CAPEX = capex_by_year.get(y, 0.0)
        Augmentation = augmentation_by_year.get(y, 0.0)
        Decommissioning = decom_cost if (y == decom_year and y != 0) else 0.0

        CFADS = EBITDA - Taxes
        DSCR = CFADS / Debt_Service if Debt_Service > 0 else None

        Project_FCF = EBITDA - Taxes - CAPEX - Augmentation - Decommissioning - Royalty_Upfront_y0

        Equity_CF = (
            -(CAPEX - debt_amount) - debt_fees - Royalty_Upfront_y0
            if y == 0
            else Project_FCF - Debt_Service
        )

        rows.append(
            {
                "Year": y,
                "Revenue_Floor": Revenue_Floor,
                "Revenue_Tolling": Revenue_Tolling,
                "Revenue_Merchant": Revenue_Merchant,
                "Revenue_Total": Revenue_Total,

                "Municipality_Royalty": Royalty_yearly + Royalty_Upfront_y0,
                "Municipality_Royalty_Upfront": Royalty_Upfront_y0,

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
                "Augmentation": Augmentation,
                "Decommissioning": Decommissioning,

                "Debt_Close": Debt_Close,
                "Debt_Service": Debt_Service,
                "DSCR": DSCR,

                "Project_FCF": Project_FCF,
                "Equity_CF": Equity_CF,
            }
        )

    df = pd.DataFrame(rows)

    df["Debt_Amount"] = debt_amount
    df["Debt_Fees"] = debt_fees
    df["Discount_Rate_Equity"] = fp.discount_rate_equity
    df["Discount_Rate_Project"] = dfp.discount_rate_project
    df["Total_Tax_Rate"] = fp.ires + fp.irap

    return df