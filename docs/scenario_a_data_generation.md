# Scenario A: Synthetic Data Generation

Scenario A is a reproducible weekly synthetic dataset designed for an end-to-end Bayesian media mix modeling workflow with PyMC-Marketing.

The dataset contains 156 weekly observations starting on January 2, 2023. The first 130 weeks are used for model development, and the final 26 weeks are reserved as a time-based holdout set.

## Data-Generating Process

Weekly revenue is generated as:

$$y_t = \mu_t + \epsilon_t,
\qquad
\epsilon_t \sim \mathcal{N}(0, 35^2).$$

The expected weekly revenue is:

$$\mu_t = 1000 + \mathrm{Trend}_t + \mathrm{Seasonality}_t + 100\,\mathrm{promotion}_t + C_{t,\mathrm{TV}} + C_{t,\mathrm{Search}} + C_{t,\mathrm{TikTok}}.$$

Here $$ C_{t,m} =  \beta_m  \frac{1-\exp(-\lambda_m A_{t,m})}  {1+\exp(-\lambda_m A_{t,m})}. $$
The components represent:

- a baseline weekly revenue of 1,000;
- a linear trend that increases by 80 over the full three-year period;
- annual seasonality with a 52-week period;
- a promotion effect of 100 during promotion weeks;
- channel-level media contributions;
- independent Gaussian observation noise with standard deviation 35.

The revenue and spend variables are measured in thousands of dollars.

## Promotion

The promotion indicator is independently generated each week:

$$
\text{promotion}_t \sim \operatorname{Bernoulli}(0.15).
$$

Each promotion week contributes an additional 100 units of expected revenue:

$$
C_{t,\text{promotion}}
=
100\,\text{promotion}_t.
$$

## Media Channels

The dataset contains three media channels:

- TV;
- Google Search;
- TikTok.

Each channel follows the same transformation sequence:

```text
Raw spend
→ Fixed-scale normalization
→ Geometric adstock
→ Logistic saturation
→ Channel contribution
```

The channels use the same transformation families but different parameters, allowing them to represent different carryover and saturation behaviors.

## Media-Spend Generation

### TV

TV follows a campaign-style spending pattern with inactive weeks.

$$
x_{t,\text{TV}}^{\text{raw}}
\sim
\operatorname{Gamma}(2,25).
$$

A weekly activity indicator is generated as:

$$
I_{t,\text{TV}}
\sim
\operatorname{Bernoulli}(0.65).
$$

Observed TV spend is:

$$
x_{t,\text{TV}}
=
\operatorname{clip}
\left(
x_{t,\text{TV}}^{\text{raw}}
I_{t,\text{TV}},
0,
120
\right).
$$

### Google Search

Google Search represents a relatively stable, always-on channel:

$$
x_{t,\text{Search}}^{\text{raw}}
\sim
\mathcal{N}(45,10^2).
$$

The generated spend is clipped to the interval from 15 to 75:

$$
x_{t,\text{Search}}
=
\operatorname{clip}
\left(
x_{t,\text{Search}}^{\text{raw}},
15,
75
\right).
$$

### TikTok

TikTok follows a smaller campaign-style spending pattern:

$$
x_{t,\text{TikTok}}^{\text{raw}}
\sim
\operatorname{Gamma}(2,12).
$$

The generated spend is clipped to the interval from 0 to 60:

$$
x_{t,\text{TikTok}}
=
\operatorname{clip}
\left(
x_{t,\text{TikTok}}^{\text{raw}},
0,
60
\right).
$$

## Spend Scaling

Before applying media transformations, raw spend is divided by a fixed channel-specific scaling denominator:

$$
\widetilde{x}_{t,\text{TV}}
=
\frac{x_{t,\text{TV}}}{120},
$$

$$
\widetilde{x}_{t,\text{Search}}
=
\frac{x_{t,\text{Search}}}{75},
$$

$$
\widetilde{x}_{t,\text{TikTok}}
=
\frac{x_{t,\text{TikTok}}}{60}.
$$

Fixed denominators are used instead of sample-dependent maximum values so that the transformation parameters remain interpretable and reproducible across runs.

The final observable CSV files contain the original spend values, not the scaled values.

## Geometric Adstock

Each scaled media series is transformed using normalized geometric adstock with a maximum lag of eight weeks:

$$
A_{t,m}
=
\sum_{\ell=0}^{7}
w_{\ell,m}\widetilde{x}_{t-\ell,m},
$$

where

$$
w_{\ell,m}
=
\frac{\alpha_m^\ell}
{\sum_{j=0}^{7}\alpha_m^j}.
$$

The retention parameter $\alpha_m$ controls how quickly the advertising effect decays over time.

A larger value of $\alpha_m$ implies stronger and more persistent carryover.

## Logistic Saturation

The adstocked media value is passed through a logistic saturation function:

$$
S(A_{t,m};\lambda_m)
=
\frac{1-\exp(-\lambda_m A_{t,m})}
{1+\exp(-\lambda_m A_{t,m})}.
$$

The parameter $\lambda_m$ controls the curvature of the response.

Larger values produce faster saturation, meaning that marginal returns decline more rapidly as media exposure increases.

## Channel Contribution

The true contribution of channel $m$ is:

$$
C_{t,m}
=
\beta_m S(A_{t,m};\lambda_m).
$$

The parameter $\beta_m$ controls the maximum scale of the channel contribution.

## True Media Parameters

| Channel | Spend distribution | Spend range | Adstock $\alpha$ | Saturation $\lambda$ | Contribution scale $\beta$ | Scaling denominator |
|---|---|---:|---:|---:|---:|---:|
| TV | Gamma(shape = 2, scale = 25), active in 65% of weeks | 0–120 | 0.70 | 2.00 | 180 | 120 |
| Google Search | Normal(mean = 45, SD = 10) | 15–75 | 0.20 | 4.00 | 140 | 75 |
| TikTok | Gamma(shape = 2, scale = 12) | 0–60 | 0.40 | 3.00 | 90 | 60 |

These parameters represent the following channel behaviors:

- TV has the strongest carryover and the largest contribution scale.
- Google Search has weak carryover and relatively fast saturation.
- TikTok has moderate carryover and a smaller contribution scale.

The expected contribution ordering is approximately:

```text
TV
> Google Search
> TikTok
```

This ordering is treated as a reasonable expectation rather than a strict constraint.

## Baseline, Trend, and Seasonality

The baseline contribution is constant:

$$
C_{t,\text{baseline}} = 1000.
$$

The normalized time index ranges from zero to one:

$$
\text{time\_index}_t
\in [0,1].
$$

The linear trend is:

$$
C_{t,\text{trend}}
=
80\,\text{time\_index}_t.
$$

The annual seasonal component is:

$$
C_{t,\text{seasonality}}
=
70\sin\left(\frac{2\pi t}{52}\right)
+
25\cos\left(\frac{2\pi t}{52}\right).
$$

The fitted PyMC-Marketing model will represent annual seasonality using:

```python
yearly_seasonality=2
```

The Fourier components used to generate the data are therefore not included as observable columns in the CSV files.

## Complete Revenue Equation

Combining all components, expected weekly revenue is:

$$
\mu_t = 1000 + 80\,\mathrm{timeIndex}_t + 70\sin\left(\frac{2\pi t}{52}\right) + 25\cos\left(\frac{2\pi t}{52}\right) + 100\,\mathrm{promotion}_t + C_{t,\mathrm{TV}} + C_{t,\mathrm{Search}} + C_{t,\mathrm{TikTok}}.
$$

Observed revenue is:

$$ y_t =  \mu_t +  \epsilon_t,  \qquad  \epsilon_t  \sim  \mathcal{N}(0,35^2). $$

## Dataset Configuration

| Parameter | Value |
|---|---:|
| Random seed | 20260801 |
| Start date | 2023-01-02 |
| Frequency | Weekly, Monday |
| Total observations | 156 |
| Development observations | 130 |
| Holdout observations | 26 |
| Maximum adstock lag | 8 weeks |
| Baseline | 1000 |
| Total trend increase | 80 |
| Promotion probability | 0.15 |
| Promotion effect | 100 |
| Observation-noise standard deviation | 35 |

## Observable Variables

The model-visible dataset contains:

| Column | Description |
|---|---|
| `date_week` | Weekly date |
| `revenue` | Observed weekly revenue |
| `tv_spend` | Weekly TV spend |
| `google_search_spend` | Weekly Google Search spend |
| `tiktok_spend` | Weekly TikTok spend |
| `promotion` | Promotion indicator |
| `time_index` | Normalized linear time trend |

The intended PyMC-Marketing configuration is:

```python
date_column = "date_week"

channel_columns = [
    "tv_spend",
    "google_search_spend",
    "tiktok_spend",
]

control_columns = [
    "promotion",
    "time_index",
]
```

## Ground-Truth Variables

The hidden truth dataset additionally contains:

| Column | Description |
|---|---|
| `baseline_true` | True baseline contribution |
| `trend_true` | True linear-trend contribution |
| `seasonality_true` | True seasonal contribution |
| `promotion_contribution_true` | True promotion contribution |
| `tv_adstock_true` | True adstocked TV value |
| `google_search_adstock_true` | True adstocked Search value |
| `tiktok_adstock_true` | True adstocked TikTok value |
| `tv_contribution_true` | True TV contribution |
| `google_search_contribution_true` | True Search contribution |
| `tiktok_contribution_true` | True TikTok contribution |
| `mu_true` | True expected revenue |
| `noise_true` | Generated observation noise |

These hidden variables are not used as model inputs. They are retained only for later validation of estimated contributions, response curves, and model behavior.

## Generated Files

Running:

```bash
python src/generate_scenario_a.py
```

generates:

```text
data/scenario_a/
├── scenario_a_full.csv
├── scenario_a_train.csv
├── scenario_a_test.csv
├── scenario_a_truth.csv
└── true_parameters.json
```

The files serve the following purposes:

- `scenario_a_full.csv`: all 156 observable weekly records;
- `scenario_a_train.csv`: the first 130 weeks;
- `scenario_a_test.csv`: the final 26-week holdout period;
- `scenario_a_truth.csv`: observable data plus hidden ground-truth components;
- `true_parameters.json`: the complete set of data-generation parameters.

## Modeling Scope

Scenario A intentionally excludes several sources of complexity:

- price effects;
- holiday effects;
- macroeconomic variables;
- channel interactions;
- unobserved confounding;
- residual autocorrelation;
- structural breaks;
- delayed adstock;
- mixed adstock families.

The objective is to create a small, interpretable, package-compatible benchmark that can be used to demonstrate the complete PyMC-Marketing workflow before introducing more realistic identification challenges.

## Intended PyMC-Marketing Model

The initial model uses the same transformation families as the synthetic data generator:

```python
from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
    MMM,
)

mmm = MMM(
    date_column="date_week",
    channel_columns=[
        "tv_spend",
        "google_search_spend",
        "tiktok_spend",
    ],
    control_columns=[
        "promotion",
        "time_index",
    ],
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)
```

The known synthetic ground truth makes it possible to evaluate whether the fitted model recovers reasonable carryover behavior, response curves, media contributions, and uncertainty estimates.

Synthetic ground-truth recovery validates the implementation, but it does not imply that an observational MMM fitted to real business data is automatically causally identified.