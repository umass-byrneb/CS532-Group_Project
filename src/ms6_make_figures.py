# Milestone 6 — Figures for final report
# Usage:
#   python -m src.ms6_make_figures \
#     --ms4_dir artifacts/ms4b \
#     --ms5_dir artifacts/ms5 \
#     --out_dir artifacts/ms6/figures \
#     --features_ml_path artifacts/features_ml_daily

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIME_ORDER = [
    "Calm / normal",
    "Choppy high-vol sideways",
    "Sell-off / drawdown",
    "Blow-off rally / parabolic",
    "Bull impulse",
]
EXPERIMENT_CODES = ["B0", "G1", "S1", "S2", "A1", "J1", "C1", "R1", "P1", "P2", "X5", "X10"]

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _save(fig, out_dir: Path, name: str):
    ensure_dir(out_dir)
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def _safer_xticklabels(ax, labels, rotation=30, ha="right"):
    labels = list(labels)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)

def _read_table(base_dir: Path, name: str) -> pd.DataFrame:
    pq_dir = base_dir / name
    csv_dir = base_dir / f"{name}_csv"
    if pq_dir.exists():
        return pd.read_parquet(pq_dir)
    csvs = sorted(csv_dir.glob("*.csv"))
    if csvs:
        return pd.read_csv(csvs[0])
    raise FileNotFoundError(f"Missing parquet/csv for '{name}' in {base_dir}")

def _load_json(base_dir: Path, name: str) -> dict:
    with (base_dir / name).open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_ms5_summary(ms5_dir: str | Path) -> pd.DataFrame:
    ms5_dir = Path(ms5_dir)
    p_jsonl = ms5_dir / "summary.jsonl"
    if p_jsonl.exists():
        rows = []
        with p_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if rows:
            return pd.DataFrame(rows)

    for fname in ("summary.csv", "summary_agg.csv"):
        p_csv = ms5_dir / fname
        if p_csv.exists():
            try:
                return pd.read_csv(p_csv, engine="python", on_bad_lines="warn")
            except Exception:
                pass

    raise FileNotFoundError("No MS5 summary found (summary.jsonl, summary.csv, or summary_agg.csv).")

def _infer_experiment_col(df: pd.DataFrame) -> str | None:
    for c in ["exp", "experiment", "name", "label", "exp_code", "experiment_code",
              "variant", "exp_id", "case", "scenario"]:
        if c in df.columns:
            return c
    pat = r"\b(" + "|".join(EXPERIMENT_CODES) + r")\b"
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) and df[c].astype(str).str.contains(pat, regex=True, na=False).any():
            return c
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["dir", "path", "log", "stdout", "stderr"]):
            s = df[c].astype(str).str.extract(pat, expand=False)
            if s.notna().any():
                df["__exp_inferred__"] = s.fillna("UNK")
                return "__exp_inferred__"
    return None

def _infer_wall_col(df: pd.DataFrame) -> str | None:
    for c in ["wall_s", "wall", "wall_time_s", "wall_seconds", "elapsed_s", "duration_s", "runtime_s"]:
        if c in df.columns:
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        lc = c.lower()
        if any(k in lc for k in ["time", "wall", "elapsed", "duration", "runtime"]):
            return c
    for c in num_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        ok = v[(v > 0.5) & (v < 10000)]
        if ok.notna().sum() > len(v) * 0.5:
            return c
    return None

def fig_transition_matrix_heatmap(ms4_dir: Path, out_dir: Path):
    df = _read_table(ms4_dir, "matrix_prob").copy().set_index("regime")
    for r in REGIME_ORDER:
        if r not in df.index:
            df.loc[r] = 0.0
        if r not in df.columns:
            df[r] = 0.0
    df = df.loc[REGIME_ORDER, REGIME_ORDER].astype(float)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(df.values, aspect="equal")
    ax.set_yticks(np.arange(len(REGIME_ORDER)))
    ax.set_yticklabels(REGIME_ORDER)
    _safer_xticklabels(ax, REGIME_ORDER, rotation=30)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Transition Probability Matrix (rows sum ≈ 1)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir, "transition_matrix_heatmap")
    print("✓ transition_matrix_heatmap")

def fig_dwell_lengths_bars(ms4_dir: Path, out_dir: Path):
    df = _read_table(ms4_dir, "dwell_overall").copy().set_index("regime").reindex(REGIME_ORDER)
    for c in ["mean_len", "p50_len", "p90_len"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")

    x = np.arange(len(df))
    w = 0.28
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - w, df["p50_len"], width=w, label="Median")
    ax.bar(x,     df["mean_len"], width=w, label="Mean")
    ax.bar(x + w, df["p90_len"],  width=w, label="P90")
    _safer_xticklabels(ax, df.index, rotation=20)
    ax.set_ylabel("Run length (days)")
    ax.set_title("Dwell Times by Regime")
    ax.legend()
    _save(fig, out_dir, "dwell_lengths_bars")
    print("✓ dwell_lengths_bars")

def fig_returns_by_regime(ms4_dir: Path, out_dir: Path):
    df0 = _read_table(ms4_dir, "returns_overall").copy().set_index("regime").reindex(REGIME_ORDER)
    df0["mean_ret"] = pd.to_numeric(df0["mean_ret"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(np.arange(len(df0)), df0["mean_ret"])
    _safer_xticklabels(ax, df0.index, rotation=20)
    ax.set_ylabel("Mean same-day return")
    ax.set_title("Returns by Regime — Same Day")
    _save(fig, out_dir, "returns_by_regime_same_day")

    df1 = _read_table(ms4_dir, "returns_fwd1_overall").copy().set_index("regime").reindex(REGIME_ORDER)
    df1["mean_ret_fwd1"] = pd.to_numeric(df1["mean_ret_fwd1"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(np.arange(len(df1)), df1["mean_ret_fwd1"])
    _safer_xticklabels(ax, df1.index, rotation=20)
    ax.set_ylabel("Mean next-day return")
    ax.set_title("Returns by Regime — Next Day")
    _save(fig, out_dir, "returns_by_regime_next_day")
    print("✓ returns_by_regime_same/next_day")

def fig_transitions_top_next_day(ms4_dir: Path, out_dir: Path, top_n: int = 15):
    df = _read_table(ms4_dir, "transitions_fwd1_overall").copy()
    if not {"transition", "mean_ret_fwd1"}.issubset(df.columns):
        return
    df["mean_ret_fwd1"] = pd.to_numeric(df["mean_ret_fwd1"], errors="coerce")
    df = df.dropna(subset=["mean_ret_fwd1"])
    df = df.reindex(df["mean_ret_fwd1"].abs().sort_values(ascending=False).index)[:top_n]
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(np.arange(len(df)), df["mean_ret_fwd1"])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["transition"])
    ax.set_xlabel("Mean next-day return")
    ax.set_title(f"Top {top_n} Transitions by Next-Day Return")
    _save(fig, out_dir, "transitions_top_next_day")
    print("✓ transitions_top_next_day")

def _load_breadth_with_btc(ms4_dir: Path, features_ml_path: str | None) -> pd.DataFrame | None:
    try:
        breadth = _read_table(ms4_dir, "breadth_daily").copy()
        breadth["Date"] = pd.to_datetime(breadth["Date"]).dt.date
    except Exception:
        return None
    if not features_ml_path:
        return None
    feats = pd.read_parquet(features_ml_path, columns=["Date", "symbol", "ret_1d_w"])
    feats["Date"] = pd.to_datetime(feats["Date"]).dt.date
    btc = feats[feats["symbol"] == "BTC"].sort_values("Date").copy()
    btc["ret_fwd1"] = btc["ret_1d_w"].shift(-1)
    return breadth.merge(btc[["Date", "ret_fwd1"]].rename(columns={"ret_fwd1": "btc_fwd1"}), on="Date", how="inner")

def fig_breadth_vs_btc(ms4_dir: Path, features_ml_path: str | None, out_dir: Path):
    df = _load_breadth_with_btc(ms4_dir, features_ml_path)
    if df is None or "btc_fwd1" not in df.columns:
        return

    if "frac_bull" in df.columns:
        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        ax.scatter(df["frac_bull"], df["btc_fwd1"], s=10, alpha=0.6)
        ax.set_xlabel("Fraction in 'Bull impulse'")
        ax.set_ylabel("BTC next-day return")
        ax.set_title("Breadth vs BTC (scatter)")
        _save(fig, out_dir, "breadth_vs_btc_scatter")

        dft = df.sort_values("Date").copy()
        dft["corr30"] = dft[["frac_bull", "btc_fwd1"]].rolling(30).corr().unstack().iloc[:, 1].values
        fig, ax = plt.subplots(figsize=(7.8, 3.8))
        ax.plot(pd.to_datetime(dft["Date"]), dft["corr30"])
        ax.set_ylabel("Rolling corr (30d)")
        ax.set_title("Rolling Correlation: frac_bull vs BTC next-day return")
        _save(fig, out_dir, "breadth_vs_btc_rolling_corr_30d")

    print("✓ breadth_vs_btc_* (if features provided)")

def fig_cluster_centroids_groupedbar(ms4_dir: Path, out_dir: Path):
    df = _read_table(ms4_dir, "step6_cluster_centroids").copy()
    avg_cols = [c for c in df.columns if c.startswith("avg_")]
    feats = [c[4:] for c in avg_cols]
    df_long = df.melt(id_vars=["cluster"], value_vars=avg_cols, var_name="feat", value_name="value")
    df_long["feat"] = df_long["feat"].str.replace(r"^avg_", "", regex=True)

    clusters = sorted(df["cluster"].unique())
    feats = list(dict.fromkeys(feats))
    x = np.arange(len(feats))
    w = 0.75 / max(1, len(clusters))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, cl in enumerate(clusters):
        vals = df_long[df_long["cluster"] == cl].set_index("feat").reindex(feats)["value"].values
        ax.bar(x + (i - (len(clusters)-1)/2)*w, vals, width=w, label=f"cluster {cl}")
    _safer_xticklabels(ax, feats, rotation=20)
    ax.set_title("Numeric Feature Centroids by Cluster")
    ax.legend(ncol=min(5, len(clusters)))
    _save(fig, out_dir, "cluster_centroids_groupedbar")
    print("✓ cluster_centroids_groupedbar")

def fig_silhouette_per_cluster(ms4_dir: Path, out_dir: Path):
    df = _read_table(ms4_dir, "step6_per_cluster_silhouette").copy().sort_values("kpred")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(np.arange(len(df)), df["mean_silhouette"])
    _safer_xticklabels(ax, df["kpred"].astype(str), rotation=0)
    ax.axhline(0, lw=1, ls="--")
    ax.set_ylabel("Mean silhouette (approx)")
    ax.set_title("Per-Cluster Silhouette (approx via centroids)")
    _save(fig, out_dir, "silhouette_per_cluster")
    print("✓ silhouette_per_cluster")

def fig_lr_coefficients(ms4_dir: Path, out_dir: Path):
    df = _read_table(ms4_dir, "gauge_coefficients").copy()
    df["absw"] = df["weight"].abs()
    df = df.sort_values("absw", ascending=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.barh(np.arange(len(df)), df["weight"])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["feature"])
    ax.axvline(0, lw=1, ls="--")
    ax.set_xlabel("Weight")
    ax.set_title("Logistic Regression Coefficients")
    _save(fig, out_dir, "lr_coefficients")
    print("✓ lr_coefficients")

def fig_threshold_sweep_curves(ms4_dir: Path, out_dir: Path):
    df = _read_table(ms4_dir, "gauge_threshold_sweep").copy()
    if "threshold" not in df.columns:
        print("[WARN] threshold_sweep_curves: no 'threshold' column found.")
        return
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df = df.dropna(subset=["threshold"]).sort_values("threshold")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    plotted = 0
    for k, label in [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("accuracy", "Accuracy")]:
        if k in df.columns:
            ax.plot(df["threshold"], pd.to_numeric(df[k], errors="coerce"), marker="o", ms=4, label=label)
            plotted += 1
    if plotted == 0:
        print("[WARN] threshold_sweep_curves: none of {f1, precision, recall, accuracy} present.")
        plt.close(fig); return

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep — F1 / Precision / Recall")
    ax.legend()
    _save(fig, out_dir, "threshold_sweep_curves")
    print("✓ threshold_sweep_curves")

def fig_gauge_auc_textbox(ms4_dir: Path, out_dir: Path):
    m = _load_json(ms4_dir, "gauge_metrics.json")
    lines = [
        f"ROC AUC: {m.get('roc_auc', np.nan):.3f}",
        f"PR  AUC: {m.get('pr_auc', np.nan):.3f}",
        "",
        f"Default 0.50 → ACC={m.get('acc_at_default', 0):.3f}, "
        f"P={m.get('precision_at_default', 0):.3f}, R={m.get('recall_at_default', 0):.3f}, "
        f"F1={m.get('f1_at_default', 0):.3f}",
        f"Best @ {m.get('best_threshold', 0):.2f} → "
        f"F1={m.get('best_f1', 0):.3f}, P={m.get('best_precision', 0):.3f}, "
        f"R={m.get('best_recall', 0):.3f}, ACC={m.get('best_accuracy', 0):.3f}",
        "",
        f"Brier: {m.get('brier', 0):.3f}",
        f"Prob q10/50/90: ({m.get('prob_quantiles',{}).get('q10',np.nan):.3f}, "
        f"{m.get('prob_quantiles',{}).get('q50',np.nan):.3f}, "
        f"{m.get('prob_quantiles',{}).get('q90',np.nan):.3f})",
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")
    ax.set_title("Predictive Gauge — Summary")
    _save(fig, out_dir, "gauge_auc_textbox")
    print("✓ gauge_auc_textbox")

def _extract_ms5_exp_and_wall(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, str, str] | None:
    df = df_raw.copy()
    exp_col = _infer_experiment_col(df)
    wall_col = _infer_wall_col(df)
    if exp_col is None or wall_col is None:
        cols_preview = ", ".join(list(df.columns)[:30])
        print(f"[DEBUG] MS5 columns available: {cols_preview}")
        return None
    df[wall_col] = pd.to_numeric(df[wall_col], errors="coerce")
    return df, exp_col, wall_col

def fig_ms5_walltime(ms5_dir: Path, out_dir: Path):
    try:
        df0 = _load_ms5_summary(ms5_dir)
        res = _extract_ms5_exp_and_wall(df0)
        if res is None:
            raise KeyError("Could not infer experiment and wall-time columns.")
        df, exp_col, wall_col = res

        g = df.groupby(exp_col)[wall_col].agg(["mean", "std"]).reset_index()
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean"]).sort_values("mean")

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.bar(np.arange(len(g)), g["mean"], yerr=g["std"].fillna(0.0), capsize=3)
        _safer_xticklabels(ax, g[exp_col], rotation=0)
        ax.set_ylabel("Wall time (s)")
        ax.set_title("Milestone 5 — Wall Time by Experiment (mean ± std)")
        _save(fig, out_dir, "ms5_walltime")
        print("✓ ms5_walltime")
    except Exception as e:
        print(f"[WARN] ms5_walltime: {e}")

def fig_ms5_speedup(ms5_dir: Path, out_dir: Path):
    try:
        df0 = _load_ms5_summary(ms5_dir)
        res = _extract_ms5_exp_and_wall(df0)
        if res is None:
            raise KeyError("Could not infer experiment and wall-time columns.")
        df, exp_col, wall_col = res

        g = df.groupby(exp_col)[wall_col].agg(["mean"]).reset_index()
        base = float(g.loc[g[exp_col].eq("B0"), "mean"].values[0]) if (g[exp_col] == "B0").any() else float(g["mean"].min())
        g["speedup_vs_B0"] = base / g["mean"]
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["speedup_vs_B0"]).sort_values("speedup_vs_B0", ascending=False)

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.bar(np.arange(len(g)), g["speedup_vs_B0"])
        _safer_xticklabels(ax, g[exp_col], rotation=0)
        ax.set_ylabel("Speedup vs B0 (×)")
        ax.set_title("Milestone 5 — Speedup vs Baseline")
        ax.axhline(1.0, lw=1, ls="--")
        _save(fig, out_dir, "ms5_speedup")
        print("✓ ms5_speedup")
    except Exception as e:
        print(f"[WARN] ms5_speedup: {e}")

def fig_ms5_per_step_speedup(ms5_dir: Path, out_dir: Path):
    """
    Heatmap of per-step speedup vs baseline (B0 if present).
    Supports steps 1..7 (and gracefully handles missing columns).
    """
    try:
        df = _load_ms5_summary(ms5_dir)
    except Exception as e:
        print(f"[WARN] ms5_per_step_speedup: {e}")
        return

    exp_col = _infer_experiment_col(df)
    if exp_col is None:
        print("[WARN] ms5_per_step_speedup: couldn't infer experiment column.")
        return

    # Accept any of step1_s..step7_s that actually exist
    all_step_cols = [f"step{i}_s" for i in range(1, 8)]
    step_cols = [c for c in all_step_cols if c in df.columns]
    if not step_cols:
        print("[WARN] ms5_per_step_speedup: no per-step columns found in MS5 summary.")
        return
    df[step_cols] = df[step_cols].apply(pd.to_numeric, errors="coerce")

    # Mean time per step per experiment
    g = df.groupby(exp_col, as_index=True)[step_cols].mean()

    # Choose baseline: B0 if available, else the min total step time row
    base_key = "B0" if "B0" in g.index else g.sum(axis=1).idxmin()
    base = g.loc[base_key]

    # Speedup = baseline_time / exp_time (per step)
    speedup = (base / g).replace([np.inf, -np.inf], np.nan)

    # Stable experiment order
    order = [c for c in EXPERIMENT_CODES if c in speedup.index]
    order += [c for c in speedup.index if c not in order]
    speedup = speedup.loc[order]

    # Pretty labels (fallback to raw column if a new step name appears)
    step_labels = {
        "step1_s": "S1 Input",
        "step2_s": "S2 Transitions",
        "step3_s": "S3 Returns",
        "step4_s": "S4 Trans→t+1",
        "step5_s": "S5 Breadth/BTC",
        "step6_s": "S6 Cluster char.",
        "step7_s": "S7 Predictive gauge",
    }
    cols = [c for c in step_cols if c in speedup.columns]
    disp_cols = [step_labels.get(c, c) for c in cols]
    Z = speedup[cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    im = ax.imshow(Z, aspect="auto", vmin=0.5, vmax=1.2, cmap="coolwarm")
    ax.set_yticks(np.arange(len(speedup.index)))
    ax.set_yticklabels(speedup.index)
    _safer_xticklabels(ax, disp_cols, rotation=15, ha="right")

    # Annotate each cell
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_title(f"Per-step Speedup vs {base_key}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speedup (×)")
    _save(fig, out_dir, "ms5_per_step_speedup")
    print("✓ ms5_per_step_speedup")

def build_all(ms4_dir: str, ms5_dir: str, out_dir: str, features_ml_path: str | None):
    ms4_dir, ms5_dir, out_dir = Path(ms4_dir), Path(ms5_dir), Path(out_dir)
    ensure_dir(out_dir)

    fig_transition_matrix_heatmap(ms4_dir, out_dir)
    fig_dwell_lengths_bars(ms4_dir, out_dir)
    fig_returns_by_regime(ms4_dir, out_dir)
    fig_transitions_top_next_day(ms4_dir, out_dir)
    fig_breadth_vs_btc(ms4_dir, features_ml_path, out_dir)
    fig_cluster_centroids_groupedbar(ms4_dir, out_dir)
    fig_silhouette_per_cluster(ms4_dir, out_dir)
    fig_lr_coefficients(ms4_dir, out_dir)
    fig_threshold_sweep_curves(ms4_dir, out_dir)
    fig_gauge_auc_textbox(ms4_dir, out_dir)
    fig_ms5_walltime(ms5_dir, out_dir)
    fig_ms5_speedup(ms5_dir, out_dir)
    fig_ms5_per_step_speedup(ms5_dir, out_dir)  

    print(f"\nAll done. Figures saved to: {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ms4_dir", required=True, help="Directory with MS4 outputs (parquet/csv/json)")
    p.add_argument("--ms5_dir", required=True, help="Directory with MS5 benchmark summaries")
    p.add_argument("--out_dir", required=True, help="Output directory for figure PNGs")
    p.add_argument("--features_ml_path", required=False, default=None,
                   help="Parquet path with features (for BTC next-day join in breadth plots)")
    args = p.parse_args()
    build_all(args.ms4_dir, args.ms5_dir, args.out_dir, args.features_ml_path)