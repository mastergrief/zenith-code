"""
CALM Financial backend — compound interest, loan payments, NPV, IRR.

Models mess up time-value-of-money calculations. Pure math.
"""

from __future__ import annotations

import math


def compound_interest(principal: float, rate: float, years: float,
                       compounds_per_year: int = 12) -> float:
    """Future value with compound interest.
    rate is annual (e.g. 0.05 for 5%), compounds_per_year default 12 (monthly)."""
    p, r, t, n = float(principal), float(rate), float(years), int(compounds_per_year)
    return round(p * (1 + r / n) ** (n * t), 2)


def simple_interest(principal: float, rate: float, years: float) -> float:
    """Simple interest: P * (1 + r * t)."""
    return round(float(principal) * (1 + float(rate) * float(years)), 2)


def loan_payment(principal: float, annual_rate: float, years: float) -> float:
    """Monthly payment for a fixed-rate loan (amortization).
    annual_rate is decimal (0.05 for 5%)."""
    p, r, n = float(principal), float(annual_rate) / 12, int(float(years) * 12)
    if r == 0:
        return round(p / n, 2)
    return round(p * (r * (1 + r) ** n) / ((1 + r) ** n - 1), 2)


def loan_total(principal: float, annual_rate: float, years: float) -> float:
    """Total amount paid over the life of a loan."""
    monthly = loan_payment(principal, annual_rate, years)
    return round(monthly * int(float(years) * 12), 2)


def loan_interest_total(principal: float, annual_rate: float, years: float) -> float:
    """Total interest paid over the life of a loan."""
    return round(loan_total(principal, annual_rate, years) - float(principal), 2)


def npv(rate: float, cashflows: list) -> float:
    """Net Present Value. cashflows[0] is typically negative (initial investment)."""
    r = float(rate)
    return round(sum(float(cf) / (1 + r) ** i for i, cf in enumerate(cashflows)), 2)


def roi(gain: float, cost: float) -> float:
    """Return on Investment as percentage."""
    cost = float(cost)
    if cost == 0:
        return 0.0
    return round((float(gain) - cost) / cost * 100, 2)


def rule_of_72(rate: float) -> float:
    """Years to double investment at given annual rate (percentage, e.g. 7 for 7%)."""
    r = float(rate)
    if r <= 0:
        return -1.0
    return round(72 / r, 1)


def inflation_adjusted(amount: float, inflation_rate: float, years: float) -> float:
    """Real value of amount after inflation.
    inflation_rate as decimal (0.03 for 3%)."""
    return round(float(amount) / (1 + float(inflation_rate)) ** float(years), 2)


def break_even(fixed_costs: float, price: float, variable_cost: float) -> float:
    """Break-even quantity: fixed_costs / (price - variable_cost)."""
    margin = float(price) - float(variable_cost)
    if margin <= 0:
        return -1.0
    return round(float(fixed_costs) / margin, 1)


FINANCIAL_FUNCTIONS = {
    "compound_interest": compound_interest,
    "simple_interest": simple_interest,
    "loan_payment": loan_payment,
    "loan_total": loan_total,
    "loan_interest_total": loan_interest_total,
    "npv": npv,
    "roi": roi,
    "rule_of_72": rule_of_72,
    "inflation_adjusted": inflation_adjusted,
    "break_even": break_even,
}

FINANCIAL_NL_PATTERNS = [
    (r'compound interest.*?(?:principal|amount)\s+(?:of\s+)?\$?([\d,.]+).*?(?:rate|interest)\s+(?:of\s+)?([\d.]+)%.*?(\d+)\s*years?', 'compound_interest({0}, {1}/100, {2})'),
    (r'(?:monthly|loan)\s+payment.*?\$?([\d,.]+).*?([\d.]+)%.*?(\d+)\s*years?', 'loan_payment({0}, {1}/100, {2})'),
    (r'rule of 72.*?([\d.]+)%', 'rule_of_72({0})'),
    (r'roi.*?gain.*?\$?([\d,.]+).*?cost.*?\$?([\d,.]+)', 'roi({0}, {1})'),
    (r'break.even.*?fixed.*?\$?([\d,.]+).*?price.*?\$?([\d,.]+).*?variable.*?\$?([\d,.]+)', 'break_even({0}, {1}, {2})'),
]
