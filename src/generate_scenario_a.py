"""Generate Scenario A synthetic weekly MMM data.

This script creates a minimal synthetic dataset compatible with the
PyMC-Marketing MMM workflow.

Outputs:
    data/scenario_a/scenario_a_full.csv
    data/scenario_a/scenario_a_train.csv
    data/scenario_a/scenario_a_test.csv
    data/scenario_a/scenario_a_truth.csv
    data/scenario_a/true_parameters.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 20260801

N_WEEKS = 156
TRAIN_WEEKS = 130
TEST_WEEKS = N_WEEKS - TRAIN_WEEKS

L_MAX = 8

START_DATE = "2023-01-02"


# ---------------------------------------------------------------------
# Synthetic transformation helpers
# ---------------------------------------------------------------------

def geometric_adstock(
    x: np.ndarray,
    alpha: float,
    l_max: int,
) -> np.ndarray:
    """Apply normalized geometric adstock.

    A_t = sum_{l=0}^{L-1} w_l x_{t-l}

    where

    w_l = alpha^l / sum_{j=0}^{L-1} alpha^j
    """
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    if not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")

    if l_max < 1:
        raise ValueError("l_max must be at least 1.")

    weights = alpha ** np.arange(l_max)
    weights = weights / weights.sum()

    return np.convolve(
        x,
        weights,
        mode="full",
    )[: len(x)]


def logistic_saturation(
    x: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Apply logistic saturation used by PyMC-Marketing.

    S(x) = (1 - exp(-lambda * x)) / (1 + exp(-lambda * x))
    """
    if lam <= 0:
        raise ValueError("lam must be positive.")

    exp_term = np.exp(-lam * x)

    return (1.0 - exp_term) / (1.0 + exp_term)


# ---------------------------------------------------------------------
# Main data-generation function
# ---------------------------------------------------------------------

def generate_scenario_a() -> None:
    """Generate and save the complete Scenario A synthetic dataset."""

    # Resolve paths relative to the repository root.
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "data" / "scenario_a"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # -----------------------------------------------------------------
    # Dates and time index
    # -----------------------------------------------------------------

    dates = pd.date_range(
        start=START_DATE,
        periods=N_WEEKS,
        freq="W-MON",
    )

    t = np.arange(N_WEEKS)

    time_index = np.linspace(
        0.0,
        1.0,
        N_WEEKS,
    )

    # -----------------------------------------------------------------
    # Promotion
    # -----------------------------------------------------------------

    promotion = rng.binomial(
        n=1,
        p=0.15,
        size=N_WEEKS,
    )

    promotion_contribution_true = 100.0 * promotion

    # -----------------------------------------------------------------
    # Raw weekly media spend
    # Units: thousands of dollars
    # -----------------------------------------------------------------

    # TV: campaign-style spending with inactive weeks.
    tv_spend = rng.gamma(
        shape=2.0,
        scale=25.0,
        size=N_WEEKS,
    )

    tv_active = rng.binomial(
        n=1,
        p=0.65,
        size=N_WEEKS,
    )

    tv_spend = np.clip(
        tv_spend * tv_active,
        0.0,
        120.0,
    )

    # Google Search: relatively stable, always-on spending.
    google_search_spend = rng.normal(
        loc=45.0,
        scale=10.0,
        size=N_WEEKS,
    )

    google_search_spend = np.clip(
        google_search_spend,
        15.0,
        75.0,
    )

    # TikTok: smaller campaign-style channel.
    tiktok_spend = rng.gamma(
        shape=2.0,
        scale=12.0,
        size=N_WEEKS,
    )

    tiktok_spend = np.clip(
        tiktok_spend,
        0.0,
        60.0,
    )

    # -----------------------------------------------------------------
    # Fixed business-scale normalization
    # -----------------------------------------------------------------

    tv_scaled = tv_spend / 120.0
    search_scaled = google_search_spend / 75.0
    tiktok_scaled = tiktok_spend / 60.0

    # -----------------------------------------------------------------
    # True adstock transformations
    # -----------------------------------------------------------------

    tv_adstock_true = geometric_adstock(
        tv_scaled,
        alpha=0.70,
        l_max=L_MAX,
    )

    google_search_adstock_true = geometric_adstock(
        search_scaled,
        alpha=0.20,
        l_max=L_MAX,
    )

    tiktok_adstock_true = geometric_adstock(
        tiktok_scaled,
        alpha=0.40,
        l_max=L_MAX,
    )

    # -----------------------------------------------------------------
    # True saturated media contributions
    # -----------------------------------------------------------------

    tv_contribution_true = (
        180.0
        * logistic_saturation(
            tv_adstock_true,
            lam=2.00,
        )
    )

    google_search_contribution_true = (
        140.0
        * logistic_saturation(
            google_search_adstock_true,
            lam=4.00,
        )
    )

    tiktok_contribution_true = (
        90.0
        * logistic_saturation(
            tiktok_adstock_true,
            lam=3.00,
        )
    )

    # -----------------------------------------------------------------
    # Baseline, trend, and annual seasonality
    # -----------------------------------------------------------------

    baseline = 1000.0

    baseline_true = np.full(
        N_WEEKS,
        baseline,
    )

    trend_true = 80.0 * time_index

    seasonality_true = (
        70.0 * np.sin(2.0 * np.pi * t / 52.0)
        + 25.0 * np.cos(2.0 * np.pi * t / 52.0)
    )

    # -----------------------------------------------------------------
    # Expected revenue and observation noise
    # -----------------------------------------------------------------

    mu_true = (
        baseline_true
        + trend_true
        + seasonality_true
        + promotion_contribution_true
        + tv_contribution_true
        + google_search_contribution_true
        + tiktok_contribution_true
    )

    noise_sd = 35.0

    noise_true = rng.normal(
        loc=0.0,
        scale=noise_sd,
        size=N_WEEKS,
    )

    revenue = mu_true + noise_true

    # -----------------------------------------------------------------
    # Observable data
    # -----------------------------------------------------------------

    full_df = pd.DataFrame(
        {
            "date_week": dates,
            "revenue": revenue,
            "tv_spend": tv_spend,
            "google_search_spend": google_search_spend,
            "tiktok_spend": tiktok_spend,
            "promotion": promotion,
            "time_index": time_index,
        }
    )

    # -----------------------------------------------------------------
    # Ground-truth data
    # -----------------------------------------------------------------

    truth_df = full_df.copy()

    truth_df["baseline_true"] = baseline_true
    truth_df["trend_true"] = trend_true
    truth_df["seasonality_true"] = seasonality_true

    truth_df["promotion_contribution_true"] = (
        promotion_contribution_true
    )

    truth_df["tv_adstock_true"] = tv_adstock_true
    truth_df["google_search_adstock_true"] = (
        google_search_adstock_true
    )
    truth_df["tiktok_adstock_true"] = tiktok_adstock_true

    truth_df["tv_contribution_true"] = tv_contribution_true
    truth_df["google_search_contribution_true"] = (
        google_search_contribution_true
    )
    truth_df["tiktok_contribution_true"] = (
        tiktok_contribution_true
    )

    truth_df["mu_true"] = mu_true
    truth_df["noise_true"] = noise_true

    # -----------------------------------------------------------------
    # Time-based train/test split
    # -----------------------------------------------------------------

    train_df = full_df.iloc[:TRAIN_WEEKS].copy()
    test_df = full_df.iloc[TRAIN_WEEKS:].copy()

    # -----------------------------------------------------------------
    # True parameter metadata
    # -----------------------------------------------------------------

    true_parameters = {
        "random_seed": RANDOM_SEED,
        "n_weeks": N_WEEKS,
        "train_weeks": TRAIN_WEEKS,
        "test_weeks": TEST_WEEKS,
        "start_date": START_DATE,
        "frequency": "W-MON",
        "l_max": L_MAX,
        "baseline": baseline,
        "trend_total": 80.0,
        "promotion_effect": 100.0,
        "noise_sd": noise_sd,
        "tv": {
            "alpha": 0.70,
            "lam": 2.00,
            "beta": 180.0,
            "scale": 120.0,
        },
        "google_search": {
            "alpha": 0.20,
            "lam": 4.00,
            "beta": 140.0,
            "scale": 75.0,
        },
        "tiktok": {
            "alpha": 0.40,
            "lam": 3.00,
            "beta": 90.0,
            "scale": 60.0,
        },
    }

    # -----------------------------------------------------------------
    # Minimal validation checks
    # -----------------------------------------------------------------

    assert len(full_df) == N_WEEKS
    assert len(train_df) == TRAIN_WEEKS
    assert len(test_df) == TEST_WEEKS

    assert not full_df.isna().any().any()

    assert (
        full_df[
            [
                "tv_spend",
                "google_search_spend",
                "tiktok_spend",
            ]
        ]
        >= 0
    ).all().all()

    assert (full_df["revenue"] > 0).all()

    assert np.allclose(
        truth_df["revenue"],
        truth_df["mu_true"] + truth_df["noise_true"],
    )

    assert (
        truth_df[
            [
                "tv_contribution_true",
                "google_search_contribution_true",
                "tiktok_contribution_true",
            ]
        ]
        >= 0
    ).all().all()

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------

    full_path = output_dir / "scenario_a_full.csv"
    train_path = output_dir / "scenario_a_train.csv"
    test_path = output_dir / "scenario_a_test.csv"
    truth_path = output_dir / "scenario_a_truth.csv"
    parameters_path = output_dir / "true_parameters.json"

    full_df.to_csv(
        full_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    train_df.to_csv(
        train_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    test_df.to_csv(
        test_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    truth_df.to_csv(
        truth_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    with parameters_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            true_parameters,
            file,
            indent=2,
        )

    print("Scenario A data generated successfully.")
    print(f"Full data:       {full_path}")
    print(f"Training data:   {train_path}")
    print(f"Test data:       {test_path}")
    print(f"Ground truth:    {truth_path}")
    print(f"True parameters: {parameters_path}")


if __name__ == "__main__":
    generate_scenario_a()