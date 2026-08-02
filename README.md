# Bayesian Retail Media Mix Modeling with PyMC-Marketing

This project is a small end-to-end Bayesian media mix modeling study built with [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) and informed by Jin et al. (2017), [“Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects”](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/). 
I used synthetic weekly retail data so the workflow could be developed and checked in a controlled setting before applying the same ideas to real marketing data. 

The dataset contains TV, Google Search, and TikTok spend, together with promotion, trend, and yearly seasonality. The current analysis uses a 130-week development set. Media effects are modeled with channel-specific geometric adstock and logistic saturation, allowing the model to represent both carryover and diminishing returns.

## Workflow

1. Validate the weekly data and inspect spend patterns and correlations.
2. Build the MMM explicitly and review its variables, dimensions, and priors.
3. Run prior predictive checks.
4. Fit the model with NUTS and diagnose R-hat, effective sample size, BFMI, and divergences.
5. Run posterior predictive checks and residual diagnostics.
6. Estimate channel parameters, contributions, historical ROAS, and marginal ROAS.

## Main results

The initial model showed divergences caused by a strong nonlinear relationship between the Google Search saturation parameters. I addressed this with tighter regularizing priors rather than simply increasing the number of draws. The revised model had no post-warmup divergences, all reported R-hat values were at or below 1.002, and effective sample sizes were comfortably above 1,600.

Posterior predictive performance on the training period was:

- RMSE: **32.27**
- MAE: **25.48**
- 94% interval coverage: **96.9%**
- Lag-1 residual autocorrelation: **0.010**

Median historical ROAS estimates were **1.11 for TV**, **1.49 for Google Search**, and **1.25 for TikTok**. Median marginal ROAS values from a 1% spend increase were lower: **0.92**, **0.80**, and **0.97**, respectively. Google Search also had the widest uncertainty interval, so its point estimate should be interpreted cautiously.

## Limitations

This is a synthetic-data demonstration, not evidence that observational MMM estimates are automatically causal. The current ROAS calculation also excludes carryover beyond the final training week. Holdout validation, experiment calibration, and constrained budget optimization are natural next steps.
