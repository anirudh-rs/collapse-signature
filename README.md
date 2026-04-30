---
title: The Collapse Signature Project
emoji: ⚔
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: true
license: mit
short_description: 4999 years of civilisational collapse — analysed and mapped
---

# The Collapse Signature Project

![History in Motion](data/geospatial/gifs/collapse_medieval.gif)

> *4,999 years of rise and ruin — analysed, mapped, and questioned.*

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Live demo:** [huggingface.co/spaces/anirudh-rs/collapse-signature](https://huggingface.co/spaces/anirudh-rs/collapse-signature)

---

## The Question

Are civilisational collapses random events — or do they follow predictable patterns in time, space, causality, and resilience?

This project analyses **112 civilisations across 4,999 years** to find out.

---

## Five Key Findings

### 1. Collapses cluster in time — and it is not random
Five distinct windows exist where multiple empires collapsed simultaneously across different continents. Permutation testing confirms this is not chance (Z = 6.65, p < 0.0001).

### 2. The clusters are not regional contagion
Moran's I spatial autocorrelation is negative across all five clusters. Collapsing civilisations were geographically dispersed — not concentrated. The mechanism was shared pressure from above (climate, pandemic, migration), not spreading failure from neighbour to neighbour.

### 3. Stressors predict timing, not outcome
67% of collapse clusters align with documented climate events or pandemics. But stressors triggered pre-existing vulnerabilities rather than causing collapse directly. More stressors correlates with slower collapse (r = +0.41), not faster.

### 4. How long you rise predicts how long you last
A logistic regression model trained on all 112 civilisations achieves **92.0% leave-one-out accuracy**. The single strongest predictor: rise duration (SHAP = 2.05). Empires that built slowly survived longer — across every region and era.

### 5. Collapse chains stretch across millennia
One causal thread runs from the Neo-Babylonian Empire (626 BCE) to the Abbasid Caliphate (1258 CE) — **seven civilisations, 1,884 years**, one unbroken line of succession and conquest through the Middle East.

---

## The Counterfactual Engine

The model answers historical what-ifs with data:

| Scenario | Base Survival | Modified | Change |
|---|---|---|---|
| Qin Dynasty with Han's institutional patience | 4.6% | 86.5% | **+81.8%** |
| Napoleon with Byzantine's full resilience profile | 6.3% | 99.3% | **+93.0%** |
| Rome with Byzantine's institutional durability | 74.0% | 94.4% | +20.4% |
| Mongols with Ming administrative depth | 14.4% | 20.9% | +6.5% |

---

## The App

Three pages built in Streamlit:

**The Story**
Plain English narrative of all five findings. Every claim is backed by statistics — readable for anyone, verifiable for anyone who wants to dig deeper.

**History in Motion**
Animated territorial maps across 4,999 years showing real empire boundaries from the Historical Basemaps dataset. Four era animations (Ancient, Medieval, Early Modern, Modern) plus an interactive snapshot explorer with step-by-step navigation through exact historical boundary data.

**The Proof**
Full analytical results: temporal clustering stats, spatial autocorrelation, stressor alignment charts, network graph with centrality rankings, SHAP feature importance, counterfactual scenario engine, and complete 112-civilisation dataset browser.

---

## Technical Stack

| Layer | Tools |
|---|---|
| Data wrangling | pandas, NumPy |
| Geospatial | geopandas, Folium, Matplotlib |
| Statistics | SciPy — KDE, permutation tests, Moran's I |
| Machine Learning | scikit-learn — logistic regression, LOO-CV |
| Explainability | SHAP |
| Network Analysis | NetworkX — PageRank, betweenness centrality |
| Visualisation | Plotly, Matplotlib, Folium |
| App | Streamlit |
| Geospatial data | Historical Basemaps (Aourednik 2022) |

---

## Project Structure

```
CivilCollapse/
├── app.py                           # Streamlit dashboard (3 pages)
├── src/
│   └── geospatial.py                # Geospatial pipeline
├── data/
│   ├── raw/
│   │   ├── civilisations_master.csv # Master dataset (112 civilisations)
│   │   └── external_stressors.csv   # Climate/pandemic/war events
│   ├── processed/                   # Pipeline outputs
│   │   ├── civilisations_clean.csv
│   │   ├── collapse_clusters.csv
│   │   ├── network_centrality.csv
│   │   ├── shap_importance.csv
│   │   ├── survival_probabilities.csv
│   │   └── pillar_1_2_results.json
│   └── geospatial/
│       ├── civilisation_mapping.json  # Year-range aware repo name mapping
│       ├── historical-basemaps/       # 53 GeoJSON snapshots (48 in project scope)
│       └── gifs/                      # Pre-rendered era animations (auto-generated)
├── notebooks/
│   ├── 01_build_master_csv.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_pillar1_2_clustering.ipynb
│   ├── 05_pillar3_contextual.ipynb
│   ├── 06_pillar4_network.ipynb
│   ├── 07_pillar5_resilience.ipynb
│   ├── 08_visualization_layer.ipynb
│   ├── 09_app_prep.ipynb
│   └── 11_map_gif.ipynb
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/anirudh-rs/collapse-signature
cd collapse-signature
conda create -n collapse-env python=3.11
conda activate collapse-env
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Map animations are auto-generated on first startup — this takes approximately 2 minutes. They persist across subsequent runs.

---

## Methodology Notes

- **Collapse start year** = when sustained decline began, not a single battle date
- **BCE years** stored as negative integers (−500 = 500 BCE)
- **Population estimates** from Maddison Project, CLIO-INFRA, and historical scholarship
- **Territorial estimates** from historical atlas sources and peer-reviewed estimates
- **Collapse triggers** are the author's classification based on primary historical consensus; secondary triggers exist for most civilisations
- **Stressor events** sourced from peer-reviewed climate science and historical epidemiology
- **Geospatial boundaries** from Historical Basemaps (Aourednik 2022), 48 snapshots from 3000 BCE to 2010 CE

## Data Limitations

- n=112 is meaningful for pattern detection but small for machine learning; LOO-CV mitigates but does not eliminate overfitting risk
- Pre-1000 BCE data is significantly sparser and less reliable than later periods
- Geospatial coverage is at empire level only — sub-national divisions (occupation zones, puppet states, colonial administrations) are not modelled
- The Americas network is disconnected from the Old World by design — no pre-Columbian trans-oceanic contact is encoded before 1492

---

## Data Sources

- **CLIO-INFRA** — historical population and economic data
- **Maddison Project** (University of Groningen) — long-run GDP per capita estimates
- **Historical Basemaps** (Aourednik 2022) — GeoJSON territorial boundaries across 53 time snapshots
- **Our World in Data** — supplementary historical statistics
- Wikipedia-derived civilisation timelines, cross-referenced with primary scholarship

---

*Built as a data science hobby project. The question was simple: can 4,999 years of history be made to speak statistically? Turns out, yes.*
