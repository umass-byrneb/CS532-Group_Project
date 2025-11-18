# Cryptocurrency Analysis with PySpark
## Project Description
Cryptocurrencies are an interesting market, trading 24 hours a day 7 days a week, unlike traditional stocks which pause overnight and on weekends. This project will use a [Kaggle dataset](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory) of historical crypto prices to explore and model these unique dynamics. The goal is to apply PySpark on a local machine to perform data analysis and machine learning adapted to cryptocurrency data. We will focus on understanding price volatility and unusual market movements in top cryptocurrencies and potentially predicting price movements while emphasizing system-level performance over just predictive accuracy. 

## Project Members
Brandon Byrne   
Vishal Garimella   
Sravanthi Machcha  
Yuxin Zhang  

## How to Run

This repo contains three main entrypoints:

- `src/ms4_analysis.py` — compute regime transition stats, returns, breadth vs BTC, cluster characterization, and a logistic-regression "predictive gauge".
- `src/ms5_benchmark.py` — run `ms4_analysis` under multiple Spark configurations and record per-step timings.
- `src/ms6_make_figures.py` — generate all final figures (PNG) from the MS4/MS5 outputs.

All scripts are designed to run with `python -m src.<module>` so they can be reused as a package.

---

## 1. Setup

Tested with:

- Python 3.10
- PySpark 3.x

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
