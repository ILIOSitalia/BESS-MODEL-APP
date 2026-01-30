from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


# =========================================================
# PROJECT DATA
# =========================================================
class ProjectData(BaseModel):
    project_life: int = Field(default=20, ge=1, le=60)

    nominal_power_mw: float = Field(default=50.0, gt=0)
    nominal_energy_mwh: float = Field(default=200.0, gt=0)

    degradation_rate: float = Field(default=0.01, ge=0.0, le=0.20)  # decimal
    cycles_per_day: float = Field(default=1.0, gt=0.0, le=10.0)

    soc_min: float = Field(default=0.10, ge=0.0, le=1.0)
    soc_max: float = Field(default=0.90, ge=0.0, le=1.0)

    grid_system_unavailability: float = Field(default=0.02, ge=0.0, le=0.50)  # decimal

    @field_validator("soc_max")
    @classmethod
    def _soc_max_gt_soc_min(cls, v, info):
        soc_min = info.data.get("soc_min", 0.0)
        if v <= soc_min:
            raise ValueError("SOC max must be > SOC min")
        return v


# =========================================================
# CAPEX & OPEX
# =========================================================
class CapexOpex(BaseModel):
    # Capex is on MWh in your app
    initial_capex_per_mw: float = Field(default=250000.0, ge=0.0)

    battery_share_of_capex: float = Field(default=0.60, ge=0.0, le=1.0)
    augmentation_cost_pct_of_batt_capex: float = Field(default=0.25, ge=0.0, le=2.0)

    augmentation_year_1: int = Field(default=0, ge=0, le=60)
    augmentation_year_2: int = Field(default=0, ge=0, le=60)

    fixed_om_per_mw_year: float = Field(default=8000.0, ge=0.0)
    insurance_grid_per_mw_year: float = Field(default=5000.0, ge=0.0)

    decommissioning_per_mw: float = Field(default=15000.0, ge=0.0)


# =========================================================
# FINANCIAL PARAMETERS
# =========================================================
class FinancialParameters(BaseModel):
    debt_tenor_years: int = Field(default=10, ge=0, le=60)
    debt_pct_on_capex: float = Field(default=0.60, ge=0.0, le=1.0)
    interest_rate: float = Field(default=0.055, ge=0.0, le=0.30)

    amortization_type: Literal["annuity", "equal_principal"] = "annuity"

    debt_upfront_fees_pct: float = Field(default=0.01, ge=0.0, le=0.10)

    ires: float = Field(default=0.24, ge=0.0, le=0.50)
    irap: float = Field(default=0.039, ge=0.0, le=0.20)

    depreciation_life_years: int = Field(default=15, ge=1, le=60)

    discount_rate_equity: float = Field(default=0.10, ge=0.0, le=0.50)


# =========================================================
# MUNICIPALITY FEES / ROYALTIES
# =========================================================
class MunicipalityFees(BaseModel):
    enabled: bool = False
    royalty_pct: float = Field(default=0.03, ge=0.0, le=0.20)

    discounted_upfront: bool = False
    discount_rate_wacc: float = Field(default=0.08, ge=0.0, le=0.40)


# =========================================================
# REVENUES
# =========================================================
class Revenues(BaseModel):
    floor_type: Literal["CM", "MACSE"] = "CM"

    # ---- Capacity Market (€/MW·year) ----
    cm_price_per_mw_year: float = Field(default=60000.0, ge=0.0)
    cm_share_of_mw: float = Field(default=0.50, ge=0.0, le=1.0)
    cm_duration_years: int = Field(default=5, ge=0, le=60)
    cm_escalation: float = Field(default=0.0, ge=0.0, le=0.50)

    # ---- MACSE (€/MWh) on Nominal Energy ----
    macse_price_per_mwh: float = Field(default=90.0, ge=0.0)
    macse_share_of_nom_energy: float = Field(default=0.50, ge=0.0, le=1.0)
    macse_duration_years: int = Field(default=10, ge=0, le=60)
    macse_escalation: float = Field(default=0.0, ge=0.0, le=0.50)

    # ---- Tolling contracts (€/MW·year) ----
    tolling_base_1_per_mw_year: float = Field(default=60000.0, ge=0.0)
    tolling_1_start_year: int = Field(default=1, ge=0, le=60)
    tolling_1_end_year: int = Field(default=5, ge=0, le=60)
    tolling_1_booked_cycles: float = Field(default=1.0, ge=0.0, le=10.0)

    tolling_base_2_per_mw_year: float = Field(default=0.0, ge=0.0)
    tolling_2_start_year: int = Field(default=0, ge=0, le=60)
    tolling_2_end_year: int = Field(default=0, ge=0, le=60)
    tolling_2_booked_cycles: float = Field(default=0.0, ge=0.0, le=10.0)

    # escalation and extra income (premium) on tolling revenues
    tolling_escalation: float = Field(default=0.0, ge=0.0, le=1.0)
    tolling_profit_sharing_pct: float = Field(default=0.0, ge=0.0, le=2.0)  # 0.10=10%

    # ---- Merchant ----
    merchant_enabled: bool = False
    merchant_selling_price_per_mwh: float = Field(default=120.0, ge=0.0)
    merchant_price_escalation: float = Field(default=0.02, ge=0.0, le=1.0)

    @field_validator("tolling_1_end_year")
    @classmethod
    def _toll1_end_after_start(cls, v, info):
        s = int(info.data.get("tolling_1_start_year", 0))
        if s > 0 and v > 0 and v < s:
            raise ValueError("Tolling 1 end must be >= Tolling 1 start")
        return v

    @field_validator("tolling_2_end_year")
    @classmethod
    def _toll2_end_after_start(cls, v, info):
        s = int(info.data.get("tolling_2_start_year", 0))
        if s > 0 and v > 0 and v < s:
            raise ValueError("Tolling 2 end must be >= Tolling 2 start")
        return v