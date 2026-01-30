from __future__ import annotations

from dataclasses import dataclass
from .models import ProjectData, CapexOpex, FinancialParameters


@dataclass
class DerivedProject:
    operating_days_per_year: float
    cycled_energy_mwh: float
    annual_cycled_energy_mwh: float


@dataclass
class DerivedCapexOpex:
    total_capex_eur: float
    total_om_per_mw_year: float
    opex_eur_year: float


@dataclass
class DerivedFinancial:
    total_corporate_tax_rate: float
    discount_rate_project: float
    debt_amount_eur: float
    equity_amount_eur: float


def derive_project(pj: ProjectData) -> DerivedProject:
    operating_days = 365.0 * (1.0 - float(pj.grid_system_unavailability))
    cycled_energy = float(pj.nominal_energy_mwh) * (float(pj.soc_max) - float(pj.soc_min))
    annual_cycled_energy = cycled_energy * float(pj.cycles_per_day) * operating_days
    return DerivedProject(
        operating_days_per_year=operating_days,
        cycled_energy_mwh=cycled_energy,
        annual_cycled_energy_mwh=annual_cycled_energy,
    )


def derive_capex_opex(pj: ProjectData, cx: CapexOpex) -> DerivedCapexOpex:
    # Total CAPEX is on MWh (as per your app)
    total_capex = float(pj.nominal_energy_mwh) * float(cx.initial_capex_per_mw)

    total_om_per_mw_year = float(cx.fixed_om_per_mw_year) + float(cx.insurance_grid_per_mw_year)
    opex_year = total_om_per_mw_year * float(pj.nominal_power_mw)

    return DerivedCapexOpex(
        total_capex_eur=total_capex,
        total_om_per_mw_year=total_om_per_mw_year,
        opex_eur_year=opex_year,
    )


def derive_financial(fp: FinancialParameters, total_capex_eur: float) -> DerivedFinancial:
    total_tax = float(fp.ires) + float(fp.irap)

    debt_amount = float(fp.debt_pct_on_capex) * float(total_capex_eur)
    equity_amount = float(total_capex_eur) - debt_amount

    # Project discount rate (WACC-style):
    # E/V * Re + D/V * Rd * (1 - IRES)
    E = 1.0 - float(fp.debt_pct_on_capex)
    D = float(fp.debt_pct_on_capex)
    discount_project = E * float(fp.discount_rate_equity) + D * float(fp.interest_rate) * (1.0 - float(fp.ires))

    return DerivedFinancial(
        total_corporate_tax_rate=total_tax,
        discount_rate_project=discount_project,
        debt_amount_eur=debt_amount,
        equity_amount_eur=equity_amount,
    )