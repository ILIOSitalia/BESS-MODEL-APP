from __future__ import annotations

import math
import pandas as pd
from .models import FinancialParameters


def _npv(rate: float | None, cashflows: list[float]) -> float:
    if rate is None:
        return float("nan")
    r = float(rate)
    out = 0.0
    for t, cf in enumerate(cashflows):
        out += cf / ((1 + r) ** t)
    return out


def _irr(cashflows: list[float]) -> float | None:
    """
    Robust-ish IRR via bisection on [-0.99, 5.0]
    Requires at least one sign change.
    """
    if not cashflows or all(cf == 0 for cf in cashflows):
        return None
    has_pos = any(cf > 0 for cf in cashflows)
    has_neg = any(cf < 0 for cf in cashflows)
    if not (has_pos and has_neg):
        return None

    def f(r: float) -> float:
        return _npv(r, cashflows)

    lo, hi = -0.99, 5.0
    f_lo, f_hi = f(lo), f(hi)

    # If no bracket, try expanding hi
    if f_lo * f_hi > 0:
        for hi2 in [10.0, 20.0, 50.0]:
            f_hi2 = f(hi2)
            if f_lo * f_hi2 <= 0:
                hi, f_hi = hi2, f_hi2
                break
        else:
            return None

    for _ in range(100):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return (lo + hi) / 2


def calc_kpis(df: pd.DataFrame, fp: FinancialParameters) -> dict:
    df = df.copy()

    equity_cf = df.get("Equity_CF", pd.Series([], dtype=float)).fillna(0.0).tolist()
    project_cf = df.get("Project_FCF", pd.Series([], dtype=float)).fillna(0.0).tolist()

    equity_irr = _irr(equity_cf)
    project_irr = _irr(project_cf)

    # Discount rates
    dr_eq = float(fp.discount_rate_equity)

    dr_prj = None
    if "Discount_Rate_Project" in df.columns and len(df) > 0:
        try:
            dr_prj = float(df["Discount_Rate_Project"].iloc[0])
        except Exception:
            dr_prj = None

    equity_npv = _npv(dr_eq, equity_cf)
    project_npv = _npv(dr_prj, project_cf) if dr_prj is not None else float("nan")

    # DSCR min (ignore NaN/None/inf)
    dscr = df.get("DSCR", pd.Series([], dtype=float))
    dscr_num = pd.to_numeric(dscr, errors="coerce")
    dscr_num = dscr_num.replace([math.inf, -math.inf], math.nan).dropna()
    min_dscr = float(dscr_num.min()) if len(dscr_num) else None

    return {
        "equity_irr": equity_irr,
        "project_irr": project_irr,
        "equity_npv": equity_npv,
        "project_npv": project_npv,
        "min_dscr": min_dscr,
        "discount_rate_equity": dr_eq,
        "discount_rate_project": dr_prj,
    }