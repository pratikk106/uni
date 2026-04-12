# Predictive Forecasting of Care Load and Placement Demand  
## Unaccompanied Children (UAC) Program — Daily Operational Data

**Author:** [Your Name]  
**Affiliation:** [University / Course]  
**Date:** April 2026  

---

### Abstract

Federal care systems for unaccompanied children operate under uncertainty from border flows, policy changes, and placement capacity. This study uses publicly structured **daily time-series** fields—intake-related counts, CBP custody, transfers into HHS care, **children in HHS care**, and **discharges (sponsor placements)**—to (1) characterize historical dynamics through exploratory analysis, (2) produce **short-horizon forecasts** of care load and discharge demand with **statistical and machine-learning models**, and (3) translate findings into **operational recommendations**. Results support **proactive staffing and bed planning** when combined with qualitative scenario review; forecast intervals should be treated as **indicative**, not definitive, under structural breaks.

**Keywords:** time series, forecasting, HHS care load, placement demand, Streamlit, walk-forward validation

---

### 1. Introduction

Descriptive reporting explains what already occurred; decision-makers also need **forward-looking** views: expected care population, whether exits can offset inflows, and early signals of **capacity stress**. This project implements a reproducible pipeline—data preparation, feature engineering, model comparison, and a **Streamlit dashboard**—aligned with program analytics goals while acknowledging limits of extrapolation during crises.

---

### 2. Data Description

The analysis uses a **daily** table with (at minimum) the following concepts:

| Concept | Role in analysis |
|--------|-------------------|
| Date | Time index |
| Apprehensions / intake-related counts | Exogenous pressure |
| Children in CBP custody | Upstream constraint signal |
| Transfers out of CBP | Flow into HHS |
| **Children in HHS care** | Primary **stock** outcome |
| **Discharges** | **Placement throughput** / exit demand proxy |

Raw files may contain **irregular reporting dates**; the pipeline **reindexes to a continuous daily calendar** and applies **time-based interpolation** for missing days, then boundary fills—documented so reviewers can assess sensitivity.

---

### 3. Exploratory Data Analysis (EDA)

**3.1 Levels and trends**  
HHS care load is a **slow-moving stock** influenced by transfers, discharges, and other operational factors. Discharges exhibit **higher day-to-day variability**, reflecting placement and case-processing rhythms.

**3.2 Flow imbalance**  
Define **net pressure** as *transfers minus discharges* (same-day, descriptive). Sustained positive values suggest **net accumulation** if other levers are unchanged—a simple **leading indicator** for planners, not a causal claim.

**3.3 Seasonality**  
Weekly **seasonality** (e.g., reporting artifacts or operational cadence) is plausible. Decomposition (additive, 7-day period) helps separate **trend**, **seasonal**, and **residual** components for narrative and model design.

**3.4 Associations**  
Pairwise **correlations** among numeric program fields describe co-movement in-sample. High correlation does **not** imply causation, especially with shared reporting cycles or confounding policy phases.

**3.5 Dashboard EDA**  
The Streamlit app’s **“EDA & insights”** tab provides **live** summary statistics, standardized multi-series plots, a correlation heatmap, and **data-driven bullet insights** that refresh when the user loads or uploads a CSV.

---

### 4. Methodology

**4.1 Feature engineering (ML track)**  
Lags (e.g., 1, 7, 14 days), rolling means and variances (7 and 14 days), net-pressure features, calendar indicators (day-of-week, month), and lagged exogenous flow fields.

**4.2 Models**  
- **Baselines:** naïve (last value), 7-day moving average.  
- **Statistical:** SARIMA with **weekly seasonality**; **Holt–Winters**-style exponential smoothing with seasonal period 7.  
- **Machine learning:** Random Forest and Gradient Boosting regressors with **recursive multi-step** forecasts for horizons &gt; 1.

**4.3 Validation**  
Strict **time-ordered** evaluation: **walk-forward** (expanding training window), metrics at multiple horizons (**1, 7, 14 days**): **MAE**, **RMSE**, **MAPE**.

**4.4 Uncertainty**  
Where available, **approximate prediction intervals** (e.g., SARIMA confidence bands; scaled residual fan for ML) are shown for **communication of dispersion**, not as calibrated operational guarantees.

---

### 5. Results (Typical Patterns)

*Exact numbers depend on the CSV vintage; replace bracketed items with values from your run.*

- **Baseline vs. structured models:** Naïve and moving-average benchmarks anchor expectations; SARIMA/ETS often capture **smooth persistence and weekly structure** in HHS care load.  
- **ML models:** May improve short-horizon fit when **nonlinear interactions** among lags and flows matter; require **more data** and careful **leakage** review for production.  
- **Horizon effect:** Error generally **rises with horizon**; medium-term forecasts should be paired with **qualitative scenarios**.  
- **KPIs in the app:** Forecast accuracy proxy, surge lead-time heuristic (vs. historical high quantile), breach probability vs. a user threshold, and a stability index support **executive storytelling** with caveats.

---

### 6. Discussion and Limitations

- **Structural breaks:** Policy or surge events violate stationarity assumptions; models should be **refit** and **monitored**.  
- **Interpolation:** Filled missing days **smooth** true volatility; sensitivity analysis can compare “interpolated” vs. “missing masked” pipelines.  
- **Causality:** Correlations and forecasts are **associative**; resource decisions need **subject-matter** review.  
- **Ethics:** Analytics support **capacity and timeliness**; they must not replace **child welfare** judgment or legal protections.

---

### 7. Recommendations

1. **Institutionalize** daily ingestion and **versioned** datasets for reproducible forecasts.  
2. Use **multiple models** and **scenarios** (base, high-transfer, low-discharge) in the dashboard—not a single point forecast.  
3. Pair **quantitative** thresholds (e.g., care load vs. bed capacity) with **early-warning** playbooks (staff surge, medical surge, placement partnerships).  
4. Report **validation metrics by horizon** to procurement and leadership; refresh **quarterly** or after major policy shifts.  
5. Extend with **holiday indicators** and **external covariates** (where approved) only under governance and **privacy** review.

---

### 8. Conclusion

The project demonstrates an end-to-end **forecasting and visualization** capability for **HHS care load** and **discharge demand**, grounded in EDA and rigorous **time-based validation**. The accompanying **Streamlit** application makes analytics **interactive** for analysts and **presentable** for stakeholders when paired with this paper and the **executive summary**.

---

### References (illustrative — adapt to your citation style)

1. Hyndman, R.J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.). OTexts.  
2. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. *Time Series Analysis: Forecasting and Control*. Wiley.  
3. U.S. Department of Health and Human Services — public materials on the Unaccompanied Children program (verify current URLs for your bibliography).

---

### Appendix A: Reproducibility

- **Environment:** Python 3.11+ (see `requirements.txt`).  
- **Run dashboard:** `streamlit run app.py` from the project root.  
- **Data:** Place `HHS_Unaccompanied_Alien_Children_Program.csv` in the project root or upload via the app.
