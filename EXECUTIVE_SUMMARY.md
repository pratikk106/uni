# Executive Summary  
## Predictive Analytics for Care Load and Placement Demand (UAC Context)

**Prepared for:** Government and program stakeholders  
**Purpose:** Support proactive planning—not replace professional judgment  
**Date:** April 2026  

---

### What problem this addresses

Programs that shelter and place unaccompanied children must balance **how many children are in care today** with **how many can safely exit to sponsors** tomorrow. When decisions rely only on backward-looking reports, responses to rising census or placement bottlenecks can be **delayed**, increasing strain on facilities, staff, and—most importantly—**time to a safe, stable placement**.

This project adds **short-horizon forecasting** and a **live dashboard** so leaders can ask: *Where is care load likely headed?* *Does expected discharge throughput keep pace with inflows?* *When should we escalate staffing or surge capacity?*

---

### What we built

1. **Research and analytics foundation**  
   - Exploratory analysis of daily flows: transfers into HHS care, discharges (placements), and related intake signals.  
   - Clear treatment of **missing reporting days** so trends are visible on a continuous timeline (with documented tradeoffs).  
   - Simple **imbalance indicators** (e.g., transfers minus discharges) to flag periods of **net pressure** on the care system.

2. **Forecasting capability**  
   - **Care load** (children in HHS care) and **discharge volumes** are projected **several days to several weeks** ahead.  
   - Multiple methods are compared: transparent **baselines**, **proven statistical models** (including weekly patterns), and **machine-learning** approaches.  
   - Models are checked with **time-based validation**—no random shuffling of history—so performance reflects **real forecasting conditions**.

3. **Interactive Streamlit dashboard**  
   - **EDA & insights:** summary statistics, comparative charts, correlation overview, and **auto-generated insight bullets** tied to the data you load.  
   - **Forecasts** for care load and placement demand with **uncertainty bands** where appropriate (interpret as **directional ranges**, not guarantees).  
   - **Scenario comparison** to contrast methods side-by-side.  
   - **Evaluation metrics** (accuracy-style measures by forecast horizon) to support accountability and model refresh decisions.

---

### Key takeaways for decision-makers

- **Forecasts are planning tools.** They perform best when conditions **resemble recent history**; sudden policy, operational, or border shocks can invalidate model behavior quickly.  
- **Use ranges and scenarios.** Prefer **bands** and **multiple models** over a single number when briefing leadership or partners.  
- **Pair metrics with playbooks.** Define in advance what happens if projected care load crosses **bed**, **clinical**, or **case-management** thresholds—who is notified, what surge options exist, and on what timeline.  
- **Refresh and govern data.** Reliable analytics require **stable daily data feeds**, **change logs**, and periodic **model review** after major events.

---

### Recommended next steps

| Priority | Action |
|---------|--------|
| High | Adopt **versioned daily data** and a single **source-of-truth** definition for each field. |
| High | Align dashboard **thresholds** (e.g., stress levels) with **real capacity** and policy constraints. |
| Medium | Run **structured exercises** (tabletop) using forecast scenarios to test surge protocols. |
| Medium | Document **limitations** for Congress, OIG, or public-facing summaries to avoid over-claiming precision. |
| Ongoing | **Retrain** models after known breaks; track **forecast error** by horizon as a management KPI. |

---

### Bottom line

This deliverable package—**research paper (EDA, methods, insights, recommendations)**, **Streamlit dashboard**, and this **executive summary**—gives program and government audiences a **shared, evidence-informed** basis for **anticipating** care load and placement demand. Used with appropriate **humility under uncertainty**, it supports **earlier, calmer scaling** of the resources children rely on.

---

*For technical detail, methodology, and citations, see `RESEARCH_PAPER.md`. For hands-on exploration, from the project folder run: `streamlit run app.py`.*
