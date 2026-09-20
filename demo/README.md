# Bayesian Marketing Mix Modeling Demo with PyMC-Marketing

This demo is a compact end-to-end Bayesian marketing mix modeling workflow built with [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing). It uses a synthetic weekly dataset to demonstrate how an MMM can move from model fitting to business decisions about channel efficiency and budget allocation.

The dataset contains 365 weekly observations. I use the first 313 weeks for model development and reserve the final 52 weeks as a holdout period. The model includes eight media channels — Google Search, DV360, Facebook, AMS, TV, VOD, OOH, and Radio — together with Numeric Distribution, RSP, and Promotion as controls. Media effects are modeled with geometric adstock and logistic saturation so that the model can represent both carryover and diminishing returns.

For business interpretation in this demo, the reporting scale is:

* 1 spend unit = **$10K**
* 1 revenue unit = **$250K**

This scaling is used only to translate the synthetic data into intuitive business units.

## Workflow

1. Load and inspect the weekly data, define the 313-week development period and 52-week holdout period, build the Bayesian MMM, and fit the model with NUTS. ([Notebook 01](notebook/01_data_and_model.ipynb))
2. Evaluate model results, estimate channel contributions, historical ROAS, marginal ROAS, and response behavior across all eight channels. ([Notebook 02](notebook/02_model_results.ipynb))
3. Compare the current channel mix with an optimized allocation under the same fixed 13-week budget. The current mix is based on the most recent 13 training weeks, and each channel is allowed to vary between 50% and 150% of its current allocation. ([Notebook 03](notebook/03_budget_optimization.ipynb))

## Main results

The fitted model converged well, with a maximum reported R-hat of **1.004**.

The model separates media contribution from the non-media controls and provides posterior estimates of both historical ROAS and marginal ROAS for all eight channels. Historical ROAS summarizes average return over the observed period, while marginal ROAS measures the expected return from a small additional increase in spend at the current spend level.

Because the synthetic target and spend variables are expressed in different reporting units, dollar-based ROAS and marginal ROAS are obtained by applying the reporting-scale conversion:

```text
Dollar ROAS = native ROAS × ($250K / $10K) = native ROAS × 25
```

The budget optimization then moves from channel-level efficiency to a portfolio decision. Rather than increasing total spend, it reallocates the same 13-week budget across channels subject to the 75%–125% allocation bounds. The notebook reports the current allocation, optimized allocation, expected media contribution under each strategy, and the resulting uplift.

The important distinction is that the channel with the highest historical ROAS does not necessarily receive the largest additional allocation. Budget optimization depends on the current position on each channel's response curve, so marginal return changes as spend is reallocated.

## Demo structure

```text
demo/
├── data/
│   └── mock_cgp_data.csv
├── artifacts/
│   └── mmm_demo_8ch.nc
├── notebook/
│   ├── 01_data_and_model.ipynb
│   ├── 02_model_results.ipynb
│   └── 03_budget_optimization.ipynb
└── README.md
```

## Limitations

This is a synthetic-data demonstration designed to show the MMM workflow and decision logic. The reporting units are illustrative, and the results should not be interpreted as real commercial estimates. As with observational MMM more broadly, model-based attribution is not automatically causal. Experimental calibration or lift-test evidence would be a natural next step before using the model for real budget decisions.

For the full project, including the original three-channel synthetic study, see the [main project README](../README.md).
