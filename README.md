# Cryptocurrency Analysis with PySpark
## Project Description
Cryptocurrencies are an interesting market, trading 24 hours a day 7 days a week, unlike traditional stocks which pause overnight and on weekends. This project will use a [Kaggle dataset](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory) of historical crypto prices to explore and model these unique dynamics. The goal is to apply PySpark on a local machine to perform data analysis and machine learning adapted to cryptocurrency data. We will focus on understanding price volatility and unusual market movements in top cryptocurrencies and potentially predicting price movements while emphasizing system-level performance over just predictive accuracy. 

## Project Members
Brandon Byrne   
Vishal Garimella   
Sravanthi Machcha  
Yuxin Zhang  

## [CS 532 Final Project Presentation-Video.mp4](https://github.com/umass-byrneb/CS532-Group_Project/blob/b0dfb56651789f0a8319047e6ba67e1219b660ab/532_final_Cut.mp4)

## [CS 532 Final Project Presentation-Powerpoint.pptx](https://github.com/umass-byrneb/CS532-Group_Project/blob/31ea8f132088ee0ca96d781fadff3925ab3e39b5/CS%20532%20Final%20Project%20Presentation.pptx)

## How to Run

This repo contains three main entrypoints:

- `src/ms4_analysis.py` — compute regime transition stats, returns, breadth vs BTC, cluster characterization, and a logistic-regression "predictive gauge".
- `src/ms5_benchmark.py` — run `ms4_analysis` under multiple Spark configurations and record per-step timings.
- `src/ms6_make_figures.py` — generate all final figures (PNG) from the MS4/MS5 outputs.

All scripts are designed to run with `python -m src.<module>` so they can be reused as a package.

# Setup

Tested with:

- Python 3.10
- PySpark 3.12.3

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# Running The Pipeline
Main Analysis
```bash
python -m src.ms4_analysis \
  --predictions_path artifacts/ms4/predictions.parquet \
  --features_ml_path artifacts/features_ml_daily \
  --out_dir artifacts/ms4b \
  --steps 1,2,3,4,5,6,7
```
Spark benchmarks
```bash
python -m src.ms5_benchmark \
  --predictions_path artifacts/ms4/predictions.parquet \
  --features_ml_path artifacts/features_ml_daily \
  --out_dir artifacts/ms5 \
  --experiments B0,S1,S2,A1,J1,R1,G1,C1,X5,X10 \
  --steps 1,2,3,4,5,6,7 \
  --repeats 1
```

This runs ms4_analysis multiple times with different Spark configs and records:
- Per-run metrics: artifacts/ms5/<EXP>/run_1/metrics.json
- Per-step timings: artifacts/ms5/<EXP>/run_1/_timings.jsonl
- Aggregated CSVs: artifacts/ms5/summary.csv, summary_agg.csv
- JSON lines with all run metrics: artifacts/ms5/summary.jsonl

Experiments:
- B0 — Baseline: local[*], Adaptive Query Execution (AQE) ON, shuffle.partitions=200, broadcast join threshold 10MB, Kryo serializer.
- S1 — Shuffle partitions ↓ to 32.
- S2 — Shuffle partitions ↑ to 800.
- A1 — AQE OFF (adaptive planning disabled).
- J1 — Broadcast join OFF.
- R1 — Java serializer instead of Kryo.
- G1 — Whole-stage codegen OFF.
- C1 — No caching inside ms4_analysis.
- X5/X10 — Scale dataset 5× / 10× by replicating symbols.





