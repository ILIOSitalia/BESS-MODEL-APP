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


def _find_brackets_for_irr(cashflows: list[float]) -> list[tuple[float, float]]:
    """
    Scan a grid of rates and return intervals (r_lo, r_hi) where NPV changes sign.
    This handles multiple-IRR cases (multiple sign changes) much better than a fixed bracket.
    """
    if not cashflows:
        return []

    # Rate grid: dense around [-0.9..1], then wider up to 50 (5000%)
    rates = []
    # negative to 0
    rates += [-0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]
    # 0 to 1 in steps
    rates += [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
    # wider range
    rates += [1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    # compute NPV values, skip invalid points
    npvs: list[tuple[float, float]] = []
    for r in rates:
        try:
            v = _npv(r, cashflows)
            if math.isnan(v) or math.isinf(v):
                continue
            npvs.append((r, v))
        except Exception:
            continue

    brackets: list[tuple[float, float]] = []
    for i in range(1, len(npvs)):
        r0, v0 = npvs[i - 1]
        r1, v1 = npvs[i]
        if v0 == 0.0:
            # exact root at r0
            brackets.append((r0, r0))
        elif v0 * v1 < 0:
            brackets.append((r0, r1))

    return brackets


def _irr_first_root(cashflows: list[float]) -> tuple[float | None, str | None]:
    """
    Returns (irr, note). If multiple brackets exist, returns the first root found and note.
    If no bracket exists, returns (None, note).
    """
    if not cashflows or all(cf == 0 for cf in cashflows):
        return None, "no_cashflows"

    has_pos = any(cf > 0 for cf in cashflows)
    has_neg = any(cf < 0 for cf in cashflows)
    if not (has_pos and has_neg):
        return None, "no_sign_change"

    brackets = _find_brackets_for_irr(cashflows)
    if not brackets:
        return None, "no_bracket_found"

    note = None
    if len(brackets) > 1:
        note = "multiple_irr_possible"

    r_lo, r_hi = brackets[0]

    # exact root
    if r_lo == r_hi:
        return r_lo, note

    # bisection
    def f(r: float) -> float:
        return _npv(r, cashflows)

    lo, hi = r_lo, r_hi
    f_lo, f_hi = f(lo), f(hi)

    # safety: if sign isn't opposite (numerical oddity), bail
    if f_lo == 0.0:
        return lo, note
    if f_hi == 0.0:
        return hi, note
    if f_lo * f_hi > 0:
        return None, "bracket_invalid"

    for _ in range(120):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < 1e-8:
            return mid, note
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return (lo + hi) / 2, note


def calc_kpis(df: pd.DataFrame, fp: FinancialParameters) -> dict:
    df = df.copy()

    equity_cf = df.get("Equity_CF", pd.Series([], dtype=float)).fillna(0.0).tolist()
    project_cf = df.get("Project_FCF", pd.Series([], dtype=float)).fillna(0.0).tolist()

    equity_irr, equity_note = _irr_first_root(equity_cf)
    project_irr, project_note = _irr_first_root(project_cf)

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

    out = {
        "equity_irr": equity_irr,
        "project_irr": project_irr,
        "equity_npv": equity_npv,
        "project_npv": project_npv,
        "min_dscr": min_dscr,
        "discount_rate_equity": dr_eq,
        "discount_rate_project": dr_prj,
    }

    # Optional notes (won't break your UI)
    if equity_note:
        out["equity_irr_note"] = equity_note
    if project_note:
        out["project_irr_note"] = project_note

    return out