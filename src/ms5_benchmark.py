# MS5: Benchmarks

# CLI
# ----
# python -m src.ms5_benchmark \
#   --predictions_path artifacts/ms4/predictions.parquet \
#   --features_ml_path artifacts/features_ml_daily \
#   --out_dir artifacts/ms5 \
#   --experiments B0,S1,S2,A1,J1,R1,G1,C1 \
#   --steps 1,2,3,4,5,6,7

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import subprocess
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

MAX_STEPS = 7

try:  
    import psutil  
    HAVE_PSUTIL = True
except Exception:  
    HAVE_PSUTIL = False

try:  
    import pandas as pd  
    import pyarrow as pa  
    import pyarrow.dataset as ds  
    import pyarrow.parquet as pq  
    HAVE_SCALE_DEPS = True
except Exception:  
    HAVE_SCALE_DEPS = False

@dataclass
class Experiment:
    id: str
    desc: str
    master: str = "local[*]"
    spark_conf: Dict[str, str] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    scale_factor: int = 1
    notes: Optional[str] = None

    def as_dict(self):
        d = asdict(self)
        d["spark_conf"] = self.spark_conf or {}
        d["env"] = self.env or {}
        return d

def build_experiments() -> Dict[str, Experiment]:
    base_conf = {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.shuffle.partitions": "200",
        "spark.sql.autoBroadcastJoinThreshold": str(10 * 1024 * 1024),  # 10MB
    }

    return {
        "B0": Experiment("B0", "Baseline: local[*], AQE ON, shuffle=200, broadcast=10MB, Kryo",
                         master="local[*]", spark_conf=base_conf.copy()),
        "P1": Experiment("P1", "Cores ↓ : local[1]", master="local[1]", spark_conf=base_conf.copy()),
        "P2": Experiment("P2", "Cores=4 : local[4]", master="local[4]", spark_conf=base_conf.copy()),
        "S1": Experiment("S1", "Shuffle partitions ↓ : 32",
                         spark_conf={**base_conf, "spark.sql.shuffle.partitions": "32"}),
        "S2": Experiment("S2", "Shuffle partitions ↑ : 800",
                         spark_conf={**base_conf, "spark.sql.shuffle.partitions": "800"}),
        "A1": Experiment("A1", "AQE OFF",
                         spark_conf={**base_conf, "spark.sql.adaptive.enabled": "false"}),
        "J1": Experiment("J1", "Broadcast join OFF",
                         spark_conf={**base_conf, "spark.sql.autoBroadcastJoinThreshold": "-1"}),
        "R1": Experiment("R1", "Serializer=Java (vs Kryo)",
                         spark_conf={**base_conf, "spark.serializer": "org.apache.spark.serializer.JavaSerializer"}),
        "G1": Experiment("G1", "Whole-stage codegen OFF",
                         spark_conf={**base_conf, "spark.sql.codegen.wholeStage": "false"}),
        "C1": Experiment("C1", "No caching in analysis code",
                         spark_conf=base_conf.copy(), env={"MS4_DISABLE_CACHE": "1"},
                         notes="Requires ms4_analysis to read MS4_DISABLE_CACHE=1"),
        "X5": Experiment("X5", "Scale inputs ×5 (replicate symbols)", scale_factor=5, spark_conf=base_conf.copy()),
        "X10": Experiment("X10", "Scale inputs ×10 (replicate symbols)", scale_factor=10, spark_conf=base_conf.copy()),
    }

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        prog="ms5_benchmark",
        description="Milestone 5: Run ms4_analysis under multiple Spark configurations and record performance.",
    )
    p.add_argument("--predictions_path", required=True)
    p.add_argument("--features_ml_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--experiments", default="B0,P1,P2,S1,S2,A1,J1,C1,R1,G1",
                   help='Comma-separated list of experiment IDs, or "ALL".')
    p.add_argument("--steps", default="1,2,3,4,5,6,7",
                   help="Which ms4_analysis steps to run inside each experiment (comma-separated).")
    p.add_argument("--dry_run", action="store_true",
                   help="Only list selected experiments and write a manifest. Do not execute anything.")
    p.add_argument("--analysis_module", default="src.ms4_analysis",
                   help="Python module to run for the analysis (default: src.ms4_analysis).")
    p.add_argument("--sampler_period_sec", type=float, default=1.0,
                   help="Sampling period for CPU/RSS metrics when psutil is available.")
    p.add_argument("--repeats", type=int, default=1, help="Repeat each experiment N times and aggregate results.")
    p.add_argument("--enable_scaling", action="store_true",
                   help="Enable dataset scaling for experiments with scale_factor > 1 (requires pandas+pyarrow).")
    return p.parse_args(argv)

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _build_pyspark_submit_args(master: str, conf: Dict[str, str]) -> str:
    parts = [f"--master {master}"] + [f"--conf {k}={v}" for k, v in (conf or {}).items()]
    parts.append("pyspark-shell")
    return " ".join(parts)


def _sample_process_tree_metrics(proc, period: float, acc: dict, stop: threading.Event):  # pragma: no cover - optional
    try:
        proc.cpu_percent(None)
        for c in proc.children(recursive=True):
            c.cpu_percent(None)
    except Exception:
        pass

    samples: List[Tuple[float, float, float]] = []
    while not stop.is_set():
        try:
            procs = [proc] + proc.children(recursive=True)
        except Exception:
            procs = [proc]
        rss = cpu = 0.0
        alive = False
        for p in procs:
            try:
                if p.is_running():
                    alive = True
                mi = p.memory_info()
                rss += float(mi.rss)
                cpu += float(p.cpu_percent(None))
            except Exception:
                continue
        samples.append((time.time(), rss, cpu))
        stop.wait(period)
        if not alive:
            break

    if samples:
        acc["max_rss_bytes"] = max(r for (_, r, _) in samples)
        acc["mean_cpu_percent"] = sum(c for (_, _, c) in samples) / len(samples)
        acc["num_samples"] = len(samples)
    else:
        acc.update({"max_rss_bytes": 0.0, "mean_cpu_percent": 0.0, "num_samples": 0})


def _write_summary_row(path: Path, row: dict) -> None:
    header = [
        "exp_id", "desc", "master", "steps",
        "start_utc", "end_utc", "wall_seconds", "return_code",
        "max_rss_mb", "mean_cpu_percent",
        "spark_conf_json", "env_json",
        "step1_s", "step2_s", "step3_s", "step4_s", "step5_s", "step6_s", "step7_s", "steps_sum_s",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()

        def fmt(v, nd=3):
            return f"{float(v):.{nd}f}" if isinstance(v, (int, float)) and v is not None else ""

        w.writerow({
            "exp_id": row["exp_id"],
            "desc": row["desc"],
            "master": row["master"],
            "steps": row["steps"],
            "start_utc": row["start_utc"],
            "end_utc": row["end_utc"],
            "wall_seconds": fmt(row["wall_seconds"]),
            "return_code": row["return_code"],
            "max_rss_mb": f"{row.get('max_rss_bytes', 0)/1024/1024:.2f}",
            "mean_cpu_percent": f"{row.get('mean_cpu_percent', 0.0):.2f}",
            "spark_conf_json": json.dumps(row.get("spark_conf", {})),
            "env_json": json.dumps(row.get("env", {})),
            "step1_s": fmt(row.get("step1_s")),
            "step2_s": fmt(row.get("step2_s")),
            "step3_s": fmt(row.get("step3_s")),
            "step4_s": fmt(row.get("step4_s")),
            "step5_s": fmt(row.get("step5_s")),
            "step6_s": fmt(row.get("step6_s")),
            "step7_s": fmt(row.get("step7_s")),
            "steps_sum_s": fmt(row.get("steps_sum_s")),
        })

def _collect_step_timings(timings_path: Path, run_id: Optional[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not timings_path.exists():
        return out

    def pick_step(rec: dict) -> Optional[int]:
        if isinstance(rec.get("step"), int):
            return int(rec["step"])  # ms4_analysis writes this
        # very loose fallback: parse from a text field
        for k in ("name", "title", "label", "msg", "message"):
            v = rec.get(k)
            if isinstance(v, str):
                m = re.search(r"Step\s*(\d+)", v, re.IGNORECASE)
                if m:
                    return int(m.group(1))
        return None

    def pick_secs(rec: dict) -> Optional[float]:
        for k in ("wall_s", "elapsed_s", "duration_s", "seconds"):
            v = rec.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        if isinstance(rec.get("duration_ms"), (int, float)):
            return float(rec["duration_ms"]) / 1000.0
        return None

    with timings_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if run_id and rec.get("run_id") != run_id:
                continue
            s, secs = pick_step(rec), pick_secs(rec)
            if s is not None and secs is not None:
                out[s] = out.get(s, 0.0) + float(secs)
    return out

def _scale_parquet_symbolwise(input_path: str, out_path: Path, scale_factor: int) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = None
    try:
        dataset = ds.dataset(str(input_path), format="parquet", partitioning=ds.partitioning(discover=True))
        table = dataset.to_table()
    except Exception:
        table = None

    if table is None or "symbol" not in table.schema.names:
        try:
            dataset = ds.dataset(str(input_path), format="parquet", partitioning="hive")
            table = dataset.to_table()
        except Exception:
            table = None

    if table is None or "symbol" not in table.schema.names:
        try:
            table = pq.read_table(str(input_path))
        except Exception:
            table = None

    if table is None or "symbol" not in table.schema.names:
        cols = list(table.schema.names) if table is not None else []
        raise RuntimeError(
            f"Cannot scale {input_path}: no 'symbol' column present. Detected columns={cols}."
        )

    pdf = table.to_pandas()
    parts = [pdf]
    for i in range(1, scale_factor):
        cp = pdf.copy()
        cp["symbol"] = cp["symbol"].astype(str) + f"_x{i}"
        parts.append(cp)

    out_pdf = pd.concat(parts, ignore_index=True)
    out_tbl = pa.Table.from_pandas(out_pdf, preserve_index=False)
    pq.write_table(out_tbl, out_path)
    return str(out_path)


def prepare_scaled_inputs(base_predictions: str, base_features: str, out_root: Path, factor: int) -> Tuple[str, str]:
    scaled_dir = out_root / "_scaled" / f"x{factor}"
    pred_out = scaled_dir / "predictions_scaled.parquet"
    feat_out = scaled_dir / "features_ml_scaled.parquet"

    if pred_out.exists() and feat_out.exists():
        return str(pred_out), str(feat_out)

    if not HAVE_SCALE_DEPS:
        raise RuntimeError("Scaling requires pandas + pyarrow. Install them to enable --enable_scaling.")

    print(f"[SCALE] Building x{factor} datasets → {scaled_dir}")
    p_path = _scale_parquet_symbolwise(base_predictions, pred_out, factor)
    f_path = _scale_parquet_symbolwise(base_features, feat_out, factor)
    print(f"[SCALE] Done: {p_path}, {f_path}")
    return p_path, f_path

def run_experiment(exp: Experiment, args, out_root: Path, scaled_cache: Dict[int, Tuple[str, str]], run_idx: int = 1) -> dict:
    exp_dir = out_root / exp.id / f"run_{run_idx}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = args.predictions_path
    features_ml_path = args.features_ml_path
    using_scaled = False
    if args.enable_scaling and exp.scale_factor > 1:
        if exp.scale_factor not in scaled_cache:
            scaled_cache[exp.scale_factor] = prepare_scaled_inputs(args.predictions_path, args.features_ml_path, out_root, exp.scale_factor)
        predictions_path, features_ml_path = scaled_cache[exp.scale_factor]
        using_scaled = True

    env = os.environ.copy()
    env.update(exp.env)

    run_id = f"{exp.id}_r{run_idx}_{uuid.uuid4().hex[:6]}"
    timings_path_env = exp_dir / "_timings.jsonl"
    env["MS5_RUN_ID"] = run_id
    env["MS5_TIMINGS_PATH"] = str(timings_path_env)

    pys_args = _build_pyspark_submit_args(exp.master, exp.spark_conf)
    env["PYSPARK_SUBMIT_ARGS"] = pys_args

    cmd = [
        sys.executable, "-m", args.analysis_module,
        "--predictions_path", predictions_path,
        "--features_ml_path", features_ml_path,
        "--out_dir", str(exp_dir / "analysis_out"),
        "--steps", args.steps,
    ]

    stdout_path = exp_dir / "stdout.log"
    stderr_path = exp_dir / "stderr.log"

    print(f"\n[RUN] {exp.id} r{run_idx} — {exp.desc}")
    print(f"      master={exp.master}  steps={args.steps}")
    if using_scaled:
        print(f"      scaled x{exp.scale_factor}: {predictions_path} | {features_ml_path}")
    print(f"      PYSPARK_SUBMIT_ARGS='{pys_args}'")
    if exp.env:
        print(f"      extra env={json.dumps(exp.env)}")
    print(f"      logs: {stdout_path} / {stderr_path}")

    start_ts = _utcnow_iso()
    t0 = time.perf_counter()

    with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)

        sampler = {}
        stop_evt = threading.Event()
        thr = None
        if HAVE_PSUTIL:
            try:
                p_obj = psutil.Process(proc.pid)  # type: ignore
                thr = threading.Thread(target=_sample_process_tree_metrics, kwargs=dict(proc=p_obj, period=float(args.sampler_period_sec), acc=sampler, stop=stop_evt), daemon=True)
                thr.start()
            except Exception:
                thr = None

        try:
            rc = proc.wait()
        finally:
            stop_evt.set()
            if thr:
                thr.join(timeout=5.0)

    wall = time.perf_counter() - t0
    end_ts = _utcnow_iso()

    by_step = _collect_step_timings(timings_path_env, run_id)
    if not by_step:
        by_step = _collect_step_timings(exp_dir / "analysis_out" / "_timings.jsonl", None)
    if str(args.steps).strip().lower() == "all":
        selected_steps = list(range(1, MAX_STEPS + 1))
    else:
        selected_steps = sorted({int(s) for s in re.findall(r"\d+", str(args.steps))})
        selected_steps = [s for s in selected_steps if 1 <= s <= MAX_STEPS]

    metrics = {
        "exp_id": exp.id,
        "desc": exp.desc,
        "master": exp.master,
        "steps": args.steps,
        "start_utc": start_ts,
        "end_utc": end_ts,
        "wall_seconds": wall,
        "return_code": rc,
        "spark_conf": exp.spark_conf,
        "env": exp.env,
        "pyspark_submit_args": pys_args,
        "have_psutil": HAVE_PSUTIL,
        "run_idx": run_idx,
        "scale_factor": exp.scale_factor,
        "using_scaled": using_scaled,
        "run_id": run_id,
        "timings_path": str(timings_path_env),
        "max_rss_bytes": float(sampler.get("max_rss_bytes", 0.0)),
        "mean_cpu_percent": float(sampler.get("mean_cpu_percent", 0.0)),
        "num_samples": int(sampler.get("num_samples", 0)),
    }

    for s in range(1, MAX_STEPS + 1):
        metrics[f"step{s}_s"] = None

    steps_sum = 0.0
    for s in selected_steps:
        val = by_step.get(s)
        if isinstance(val, (int, float)):
            metrics[f"step{s}_s"] = float(val)
            steps_sum += float(val)
    metrics["steps_sum_s"] = steps_sum if steps_sum > 0 else None

    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_summary_row(out_root / "summary.csv", metrics)

    rss_mb = metrics["max_rss_bytes"] / 1024 / 1024
    print(f"[DONE] {exp.id} r{run_idx} rc={rc} wall={wall:.2f}s maxRSS={rss_mb:.1f}MB meanCPU={metrics['mean_cpu_percent']:.1f}%")
    if not by_step:
        print(f"      [note] No per-step timings found at {timings_path_env} or analysis_out/_timings.jsonl")

    return metrics

def _aggregate_and_write(all_metrics: List[dict], out_dir: Path) -> None:
    import math
    from statistics import mean, pstdev

    by_exp: Dict[str, List[dict]] = {}
    for m in all_metrics:
        by_exp.setdefault(m["exp_id"], []).append(m)

    baseline = "B0" if "B0" in by_exp else next(iter(by_exp))
    base_mean_wall = mean(m["wall_seconds"] for m in by_exp[baseline])

    def mean_key(arr: List[dict], key: str) -> float:
        vals = [x.get(key) for x in arr]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        return mean(vals) if vals else float("nan")

    base_step_means = {s: mean_key(by_exp[baseline], f"step{s}_s") for s in range(1, MAX_STEPS + 1)}

    header = [
        "exp_id", "desc", "n_runs",
        "mean_wall_s", "std_wall_s", "min_wall_s", "max_wall_s",
        "mean_max_rss_mb", "mean_cpu_percent", f"speedup_vs_{baseline}",
        "step1_mean_s", "step2_mean_s", "step3_mean_s", "step4_mean_s", "step5_mean_s", "step6_mean_s", "step7_mean_s",
        f"step1_speedup_vs_{baseline}", f"step2_speedup_vs_{baseline}", f"step3_speedup_vs_{baseline}",
        f"step4_speedup_vs_{baseline}", f"step5_speedup_vs_{baseline}", f"step6_speedup_vs_{baseline}", f"step7_speedup_vs_{baseline}",
    ]

    agg_csv = out_dir / "summary_agg.csv"
    with agg_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for exp_id, arr in sorted(by_exp.items()):
            ws = [x["wall_seconds"] for x in arr]
            m = mean(ws)
            s = pstdev(ws) if len(ws) > 1 else 0.0
            mn, mx = min(ws), max(ws)
            rss_mb = [(x.get("max_rss_bytes", 0.0) / 1024 / 1024) for x in arr]
            mrss = mean(rss_mb) if rss_mb else 0.0
            cpu = [x.get("mean_cpu_percent", 0.0) for x in arr]
            mcpu = mean(cpu) if cpu else 0.0
            speedup = base_mean_wall / m if m > 0 else float("nan")
            desc = arr[0].get("desc", "")

            step_means = [mean_key(arr, f"step{s}_s") for s in range(1, MAX_STEPS + 1)]
            step_speedups = []
            for s_idx, mean_s in enumerate(step_means, start=1):
                b = base_step_means.get(s_idx)
                if isinstance(b, float) and b > 0 and isinstance(mean_s, float) and mean_s > 0:
                    step_speedups.append(b / mean_s)
                else:
                    step_speedups.append(float("nan"))

            def fmt(x, nd=3):
                return f"{x:.{nd}f}" if isinstance(x, float) and not math.isnan(x) and not math.isinf(x) else ""

            w.writerow([
                exp_id, desc, len(arr),
                f"{m:.3f}", f"{s:.3f}", f"{mn:.3f}", f"{mx:.3f}",
                f"{mrss:.2f}", f"{mcpu:.2f}", f"{speedup:.2f}",
                fmt(step_means[0]), fmt(step_means[1]), fmt(step_means[2]), fmt(step_means[3]), fmt(step_means[4]), fmt(step_means[5]), fmt(step_means[6]),
                fmt(step_speedups[0], 2), fmt(step_speedups[1], 2), fmt(step_speedups[2], 2), fmt(step_speedups[3], 2), fmt(step_speedups[4], 2), fmt(step_speedups[5], 2), fmt(step_speedups[6], 2),
            ])

    print(f"\nAggregate summary → {agg_csv}")

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = build_experiments()
    selected_ids = list(catalog.keys()) if args.experiments.strip().upper() == "ALL" else [e.strip() for e in args.experiments.split(",") if e.strip()]

    invalid = [e for e in selected_ids if e not in catalog]
    if invalid:
        print(f"[ERROR] Unknown experiment IDs: {invalid}", file=sys.stderr)
        sys.exit(2)

    selected = [catalog[e] for e in selected_ids]

    print("\nMilestone 5 — Benchmark Plan")
    print("===============================================")
    print(f"predictions_path : {args.predictions_path}")
    print(f"features_ml_path : {args.features_ml_path}")
    print(f"out_dir          : {out_dir}")
    print(f"steps            : {args.steps}")
    print(f"experiments      : {', '.join(selected_ids)}")
    print("\nExperiments:")
    for exp in selected:
        print(f"- {exp.id}: {exp.desc}")
        print(f"   master     = {exp.master}")
        print(f"   scale_x    = {exp.scale_factor}")
        if exp.env:
            print(f"   env        = {json.dumps(exp.env)}")
        print(f"   spark_conf = {json.dumps(exp.spark_conf)}")
        if exp.notes:
            print(f"   notes      = {exp.notes}")

    manifest = {
        "predictions_path": args.predictions_path,
        "features_ml_path": args.features_ml_path,
        "out_dir": str(out_dir),
        "steps": args.steps,
        "selected_experiments": [exp.as_dict() for exp in selected],
        "have_psutil": HAVE_PSUTIL,
        "sampler_period_sec": args.sampler_period_sec,
        "analysis_module": args.analysis_module,
        "enable_scaling": args.enable_scaling,
        "have_scale_deps": HAVE_SCALE_DEPS,
        "repeats": args.repeats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "ms5_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved manifest → {out_dir / 'ms5_manifest.json'}")

    if args.dry_run:
        print("\nDRY RUN requested — not executing experiments.")
        return

    if not HAVE_PSUTIL:
        print("\n[WARN] psutil not installed — CPU/RAM sampling disabled.\n       To enable, run: pip install psutil")

    if args.enable_scaling and not HAVE_SCALE_DEPS:
        print("\n[WARN] --enable_scaling requested but pandas/pyarrow not found; scaling disabled.\n       To enable scaling, run: pip install pandas pyarrow")

    all_metrics: List[dict] = []
    scaled_cache: Dict[int, Tuple[str, str]] = {}

    for exp in selected:
        for run_idx in range(1, int(args.repeats) + 1):
            m = run_experiment(exp, args, out_dir, scaled_cache, run_idx=run_idx)
            all_metrics.append(m)

    with (out_dir / "summary.jsonl").open("w") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")

    _aggregate_and_write(all_metrics, out_dir)

    print(f"\nSummary written → {out_dir / 'summary.csv'} and {out_dir / 'summary.jsonl'}")
    print("Milestone 5 Step 3 complete.")

if __name__ == "__main__":
    main()