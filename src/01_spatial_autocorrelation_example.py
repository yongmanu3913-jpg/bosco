"""
Post-processing plots for dual-unit PG-BIN SAE outputs.
Use after sae_pgbin_dual_units_0517_final.py finishes.

Outputs per unit folder:
- MCMC convergence diagnostics plots: R-hat/ESS, trace, density, rank
- Internal in-sample calibration check at gu×age level: observed vs predicted, residual plot
- Unit-level prevalence plots: histogram, ranked plot, uncertainty scatter
- Optional GeoPandas maps if shapefile paths are provided

Before running optional maps:
    pip install geopandas
"""
from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# Korean font fallback for Windows; harmless if unavailable.
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

AGE_LEVELS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def sigmoid(z):
    z = np.asarray(z, dtype="float64")
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.asarray(p, dtype="float64")
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def solve_delta_for_group(logits: np.ndarray, N: np.ndarray, target: float, max_iter: int = 70) -> float:
    lo, hi = -25.0, 25.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = np.sum(N * sigmoid(logits + mid))
        if s > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def benchmark_gu_total(unit_long: pd.DataFrame, pi_row: np.ndarray, gu_tot_map: Dict[str, int]) -> np.ndarray:
    logits = logit(np.clip(pi_row, 1e-8, 1 - 1e-8))
    N = unit_long["N"].to_numpy(dtype="float64")
    out = pi_row.copy()
    for gu, idxs in unit_long.groupby("gu_code").groups.items():
        idx = np.array(list(idxs), dtype=int)
        target = float(gu_tot_map[str(gu)])
        delta = solve_delta_for_group(logits[idx], N[idx], target)
        out[idx] = sigmoid(logits[idx] + delta)
    return out


def build_X_from_feature_cols(unit_long: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    X = np.zeros((unit_long.shape[0], len(feature_cols)), dtype="float64")
    for j, col in enumerate(feature_cols):
        if col == "intercept":
            X[:, j] = 1.0
        elif col.startswith("age_"):
            age = col.replace("age_", "")
            X[:, j] = (unit_long["agegrp6"].astype(str).to_numpy() == age).astype(float)
        elif col in unit_long.columns:
            X[:, j] = pd.to_numeric(unit_long[col], errors="coerce").fillna(0).to_numpy(dtype="float64")
        else:
            raise ValueError(f"Feature column not found in unit_long: {col}")
    return X


def safe_corr(x, y):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def rmse(x, y):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def mae(x, y):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.abs(x[ok] - y[ok]))) if ok.any() else np.nan


def load_unit_outputs(base_dir: str, unit_type: str):
    base = Path(base_dir)
    out_dir = base / f"outputs_pgbin_0517_final_{unit_type}"
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    idata_path = out_dir / "idata_pgbin.nc"
    draws_path = out_dir / "pgbin_draws.npz"
    long_path = out_dir / f"01_{unit_type}_long_unit_age.csv"
    gu_age_path = out_dir / "01_gu_age_model_totals.csv"
    pred_path = out_dir / f"{unit_type}_pred_final_PGBIN_latent_gu_total_benchmarked.csv"

    for p in [idata_path, draws_path, long_path, gu_age_path, pred_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    idata = az.from_netcdf(idata_path)
    draws = np.load(draws_path, allow_pickle=True)
    unit_long = pd.read_csv(long_path, encoding="utf-8-sig")
    gu_age = pd.read_csv(gu_age_path, encoding="utf-8-sig")
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    feature_cols = [str(x) for x in draws["feature_cols"].tolist()]
    gu_list = [str(x) for x in draws["gu_list"].tolist()]
    beta_draws = draws["beta_draws"]
    u_draws = draws["u_draws"]
    return out_dir, idata, beta_draws, u_draws, feature_cols, gu_list, unit_long, gu_age, pred


def export_mcmc_plots(out_dir: Path, idata: az.InferenceData, feature_cols: Sequence[str], top_k: int = 12):
    plot_dir = out_dir / "plots_mcmc"
    plot_dir.mkdir(exist_ok=True)

    summ = az.summary(idata, var_names=["beta", "u_g"])
    summ.to_csv(plot_dir / "mcmc_summary_arviz.csv", encoding="utf-8-sig")

    # R-hat and ESS tables/plots
    summ_plot = summ.copy().reset_index().rename(columns={"index": "parameter"})
    if "r_hat" in summ_plot.columns:
        rhat_df = summ_plot[["parameter", "r_hat"]].dropna().sort_values("r_hat", ascending=False)
        rhat_df.to_csv(plot_dir / "mcmc_rhat_sorted.csv", index=False, encoding="utf-8-sig")
        top = rhat_df.head(30).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(top))), dpi=180)
        ax.barh(top["parameter"], top["r_hat"])
        ax.axvline(1.01, linewidth=1.2, linestyle="--")
        ax.axvline(1.05, linewidth=1.2, linestyle=":")
        ax.set_xlabel("R-hat")
        ax.set_title("Worst R-hat values; lower is better")
        plt.tight_layout()
        fig.savefig(plot_dir / "mcmc_rhat_top30.png", bbox_inches="tight")
        plt.close(fig)

    ess_cols = [c for c in ["ess_bulk", "ess_tail"] if c in summ_plot.columns]
    if ess_cols:
        ess_df = summ_plot[["parameter"] + ess_cols].copy()
        ess_df["ess_min"] = ess_df[ess_cols].min(axis=1)
        ess_df = ess_df.sort_values("ess_min", ascending=True)
        ess_df.to_csv(plot_dir / "mcmc_ess_sorted.csv", index=False, encoding="utf-8-sig")
        top = ess_df.head(30).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(top))), dpi=180)
        ax.barh(top["parameter"], top["ess_min"])
        ax.set_xlabel("min(ESS bulk, ESS tail)")
        ax.set_title("Lowest effective sample sizes; higher is better")
        plt.tight_layout()
        fig.savefig(plot_dir / "mcmc_ess_lowest30.png", bbox_inches="tight")
        plt.close(fig)

    # Select beta coefficients with largest absolute posterior mean.
    beta_vals = idata.posterior["beta"].values  # chain, draw, p
    beta_flat = beta_vals.reshape(-1, beta_vals.shape[-1])
    mean_abs_order = np.argsort(np.abs(beta_flat.mean(axis=0)))[::-1]
    top_feats = [feature_cols[i] for i in mean_abs_order[: min(top_k, len(feature_cols))]]

    # Trace plot: all beta can be too dense, so top-k beta + selected u_g.
    try:
        az.plot_trace(idata, var_names=["beta"], coords={"beta_dim": top_feats}, compact=True)
        plt.tight_layout()
        plt.savefig(plot_dir / "trace_beta_top.png", bbox_inches="tight", dpi=180)
        plt.close("all")
    except Exception as e:
        (plot_dir / "trace_beta_top_ERROR.txt").write_text(repr(e), encoding="utf-8")

    try:
        az.plot_density(idata, var_names=["beta"], coords={"beta_dim": top_feats})
        plt.tight_layout()
        plt.savefig(plot_dir / "density_beta_top.png", bbox_inches="tight", dpi=180)
        plt.close("all")
    except Exception as e:
        (plot_dir / "density_beta_top_ERROR.txt").write_text(repr(e), encoding="utf-8")

    try:
        az.plot_forest(idata, var_names=["beta"], coords={"beta_dim": top_feats}, combined=True)
        plt.tight_layout()
        plt.savefig(plot_dir / "forest_beta_top.png", bbox_inches="tight", dpi=180)
        plt.close("all")
    except Exception as e:
        (plot_dir / "forest_beta_top_ERROR.txt").write_text(repr(e), encoding="utf-8")

    # Rank plot is meaningful when chains >= 2.
    try:
        if idata.posterior.sizes.get("chain", 1) >= 2:
            az.plot_rank(idata, var_names=["beta"], coords={"beta_dim": top_feats})
            plt.tight_layout()
            plt.savefig(plot_dir / "rank_beta_top.png", bbox_inches="tight", dpi=180)
            plt.close("all")
    except Exception as e:
        (plot_dir / "rank_beta_top_ERROR.txt").write_text(repr(e), encoding="utf-8")

    # Random intercept trace/density for first 10 gu.
    gu_vals = [str(x) for x in idata.posterior.coords["gu"].values]
    gu_top = gu_vals[: min(10, len(gu_vals))]
    try:
        az.plot_trace(idata, var_names=["u_g"], coords={"gu": gu_top}, compact=True)
        plt.tight_layout()
        plt.savefig(plot_dir / "trace_u_g_first10.png", bbox_inches="tight", dpi=180)
        plt.close("all")
    except Exception as e:
        (plot_dir / "trace_u_g_first10_ERROR.txt").write_text(repr(e), encoding="utf-8")

    return plot_dir


def posterior_gu_age_predictions(
    out_dir: Path,
    unit_type: str,
    beta_draws: np.ndarray,
    u_draws: np.ndarray,
    feature_cols: Sequence[str],
    gu_list: Sequence[str],
    unit_long: pd.DataFrame,
    gu_age: pd.DataFrame,
    max_draws: Optional[int] = None,
) -> pd.DataFrame:
    plot_dir = out_dir / "plots_internal_validation"
    plot_dir.mkdir(exist_ok=True)

    X = build_X_from_feature_cols(unit_long, feature_cols)
    gu_to_idx = {str(g): i for i, g in enumerate(gu_list)}
    g_idx = np.array([gu_to_idx[str(g)] for g in unit_long["gu_code"].astype(str)], dtype=int)
    N_i = unit_long["N"].to_numpy(dtype="float64")

    beta_samps = beta_draws.reshape(-1, beta_draws.shape[-1])
    u_samps = u_draws.reshape(-1, u_draws.shape[-1])
    S_total = beta_samps.shape[0]

    if max_draws is not None and S_total > max_draws:
        rng = np.random.default_rng(42)
        sel = rng.choice(S_total, size=max_draws, replace=False)
        beta_samps = beta_samps[sel]
        u_samps = u_samps[sel]

    S = beta_samps.shape[0]

    gu_age2 = gu_age.copy()
    gu_age2["gu_code"] = gu_age2["gu_code"].astype(str)
    gu_age2["agegrp6"] = gu_age2["agegrp6"].astype(str)
    gu_tot_map = gu_age2.groupby("gu_code")["y"].sum().round().astype(int).to_dict()

    keys = gu_age2[["gu_code", "agegrp6"]].drop_duplicates().copy()
    keys["ga_key"] = keys["gu_code"] + "||" + keys["agegrp6"]
    key_list = keys["ga_key"].tolist()
    key_to_idx = {k: i for i, k in enumerate(key_list)}

    row_keys = unit_long["gu_code"].astype(str).to_numpy() + "||" + unit_long["agegrp6"].astype(str).to_numpy()
    ga_idx = np.array([key_to_idx[k] for k in row_keys], dtype=int)
    K = len(key_list)

    pred_cases = np.zeros((S, K), dtype="float64")
    pred_N = np.bincount(ga_idx, weights=N_i, minlength=K)

    for s in range(S):
        eta = beta_samps[s] @ X.T + u_samps[s, g_idx]
        pi = sigmoid(eta)
        pi_b = benchmark_gu_total(unit_long, pi, gu_tot_map)
        cases = pi_b * N_i
        pred_cases[s] = np.bincount(ga_idx, weights=cases, minlength=K)

    out = keys.copy()
    out["pred_cases_mean"] = pred_cases.mean(axis=0)
    out["pred_cases_q025"] = np.quantile(pred_cases, 0.025, axis=0)
    out["pred_cases_q975"] = np.quantile(pred_cases, 0.975, axis=0)
    out["pred_N"] = pred_N
    out["pred_rate_mean"] = out["pred_cases_mean"] / np.maximum(out["pred_N"], 1)

    obs = gu_age2[["gu_code", "agegrp6", "y", "N_gu_age", "cases_obs"]].copy()
    out = out.merge(obs, on=["gu_code", "agegrp6"], how="left")
    out["obs_rate"] = out["y"] / out["N_gu_age"].replace(0, np.nan)
    out["pred_minus_obs_rate"] = out["pred_rate_mean"] - out["obs_rate"]
    out["pred_minus_obs_cases"] = out["pred_cases_mean"] - out["y"]

    out_csv = plot_dir / f"{unit_type}_internal_gu_age_observed_vs_predicted.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame([{
        "unit_type": unit_type,
        "n_gu_age_cells": int(out.shape[0]),
        "rate_mae": mae(out["obs_rate"], out["pred_rate_mean"]),
        "rate_rmse": rmse(out["obs_rate"], out["pred_rate_mean"]),
        "rate_pearson": safe_corr(out["obs_rate"], out["pred_rate_mean"]),
        "case_mae": mae(out["y"], out["pred_cases_mean"]),
        "case_rmse": rmse(out["y"], out["pred_cases_mean"]),
        "case_pearson": safe_corr(out["y"], out["pred_cases_mean"]),
        "note": "In-sample gu×age calibration check, not out-of-sample validation. The model used NHIS gu×age counts during fitting."
    }])
    metrics.to_csv(plot_dir / f"{unit_type}_internal_validation_metrics.csv", index=False, encoding="utf-8-sig")

    # observed vs predicted rate scatter
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.scatter(out["obs_rate"] * 100, out["pred_rate_mean"] * 100, s=35, alpha=0.75)
    lo = min(out["obs_rate"].min(), out["pred_rate_mean"].min()) * 100
    hi = max(out["obs_rate"].max(), out["pred_rate_mean"].max()) * 100
    ax.plot([lo, hi], [lo, hi], linewidth=1.2, linestyle="--")
    ax.set_xlabel("Observed NHIS gu×age rate (%)")
    ax.set_ylabel("Predicted gu×age rate (%)")
    ax.set_title(f"{unit_type}: internal gu×age calibration check")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(plot_dir / f"{unit_type}_internal_obs_vs_pred_rate.png", bbox_inches="tight")
    plt.close(fig)

    # residual histogram
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.hist(out["pred_minus_obs_rate"].dropna() * 100, bins=25, alpha=0.85)
    ax.axvline(0, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted - observed rate, percentage points")
    ax.set_ylabel("Number of gu×age cells")
    ax.set_title(f"{unit_type}: internal residual distribution")
    plt.tight_layout()
    fig.savefig(plot_dir / f"{unit_type}_internal_residual_histogram.png", bbox_inches="tight")
    plt.close(fig)

    return out


def export_unit_prevalence_plots(out_dir: Path, unit_type: str, pred: pd.DataFrame):
    plot_dir = out_dir / "plots_unit_prevalence"
    plot_dir.mkdir(exist_ok=True)

    p = pred.copy()
    p["pred_rate_pct"] = p["pred_rate_mean"] * 100
    p["pred_q025_pct"] = p["pred_rate_q025"] * 100
    p["pred_q975_pct"] = p["pred_rate_q975"] * 100
    p["ci_width_pctp"] = p["pred_ci_width"] * 100

    p.to_csv(plot_dir / f"{unit_type}_pred_for_plot.csv", index=False, encoding="utf-8-sig")

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.hist(p["pred_rate_pct"].dropna(), bins=30, alpha=0.85)
    ax.axvline(p["pred_rate_pct"].mean(), linewidth=1.2, linestyle="--", label="Mean")
    ax.set_xlabel("Predicted depression prevalence (%)")
    ax.set_ylabel("Number of units")
    ax.set_title(f"{unit_type}: distribution of predicted prevalence")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / f"{unit_type}_prevalence_histogram.png", bbox_inches="tight")
    plt.close(fig)

    # Ranked plot with credible interval
    pr = p.sort_values("pred_rate_pct").reset_index(drop=True)
    x = np.arange(pr.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.plot(x, pr["pred_rate_pct"], linewidth=1.5)
    ax.fill_between(x, pr["pred_q025_pct"], pr["pred_q975_pct"], alpha=0.25)
    ax.set_xlabel("Units ranked by predicted prevalence")
    ax.set_ylabel("Predicted depression prevalence (%)")
    ax.set_title(f"{unit_type}: ranked prevalence with 95% credible interval")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(plot_dir / f"{unit_type}_prevalence_ranked_ci.png", bbox_inches="tight")
    plt.close(fig)

    # Uncertainty vs prevalence
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    sizes = np.sqrt(np.maximum(p["pop_20ov"], 1))
    sizes = 10 + 90 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1e-9)
    ax.scatter(p["pred_rate_pct"], p["ci_width_pctp"], s=sizes, alpha=0.65)
    ax.set_xlabel("Predicted depression prevalence (%)")
    ax.set_ylabel("95% CI width, percentage points")
    ax.set_title(f"{unit_type}: uncertainty vs prevalence")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(plot_dir / f"{unit_type}_uncertainty_vs_prevalence.png", bbox_inches="tight")
    plt.close(fig)

    # Top and bottom units table plot via bar chart
    n = min(20, p.shape[0] // 2)
    top = p.nlargest(n, "pred_rate_pct").copy()
    bottom = p.nsmallest(n, "pred_rate_pct").copy()
    for tag, df in [("top20", top), ("bottom20", bottom)]:
        df = df.sort_values("pred_rate_pct")
        labels = df["unit_name"].astype(str).to_list()
        fig, ax = plt.subplots(figsize=(8, max(4.5, 0.28 * len(df))), dpi=180)
        ax.barh(labels, df["pred_rate_pct"])
        ax.set_xlabel("Predicted depression prevalence (%)")
        ax.set_title(f"{unit_type}: {tag} units by predicted prevalence")
        plt.tight_layout()
        fig.savefig(plot_dir / f"{unit_type}_prevalence_{tag}.png", bbox_inches="tight")
        plt.close(fig)

    return plot_dir


def export_optional_map(
    out_dir: Path,
    unit_type: str,
    pred: pd.DataFrame,
    shp_path: Optional[str],
    shp_key: Optional[str],
    pred_key: str = "unit_code",
    title: Optional[str] = None,
):
    if not shp_path or not shp_key:
        return None
    if not os.path.exists(shp_path):
        print(f"[SKIP MAP] shapefile not found: {shp_path}")
        return None
    try:
        import geopandas as gpd
    except Exception as e:
        print(f"[SKIP MAP] geopandas is not installed: {repr(e)}")
        return None

    plot_dir = out_dir / "plots_maps"
    plot_dir.mkdir(exist_ok=True)

    try:
        gdf = gpd.read_file(shp_path, encoding="cp949")
    except Exception:
        gdf = gpd.read_file(shp_path)

    if shp_key not in gdf.columns:
        raise ValueError(f"shp_key '{shp_key}' not found. Available columns: {list(gdf.columns)[:40]}")

    p = pred.copy()
    p[pred_key] = p[pred_key].astype(str).str.strip()
    p["pred_rate_pct"] = p["pred_rate_mean"] * 100
    p["ci_width_pctp"] = p["pred_ci_width"] * 100

    gdf[shp_key] = gdf[shp_key].astype(str).str.strip()
    mg = gdf.merge(p, left_on=shp_key, right_on=pred_key, how="left")

    for value_col, label in [("pred_rate_pct", "Predicted prevalence (%)"), ("ci_width_pctp", "95% CI width (percentage points)")]:
        fig, ax = plt.subplots(figsize=(9, 9), dpi=200)
        mg.plot(
            column=value_col,
            cmap="Spectral_r",
            legend=True,
            linewidth=0.2,
            edgecolor="gray",
            missing_kwds={"color": "lightgray", "label": "Missing"},
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(title or f"{unit_type}: {label}")
        plt.tight_layout()
        safe_col = value_col.replace("%", "pct")
        fig.savefig(plot_dir / f"{unit_type}_map_{safe_col}.png", bbox_inches="tight")
        plt.close(fig)

    # Save merged map attribute table without geometry for checking joins.
    mg.drop(columns="geometry").to_csv(plot_dir / f"{unit_type}_map_join_check.csv", index=False, encoding="utf-8-sig")
    return plot_dir


def postprocess_unit(
    base_dir: str,
    unit_type: str,
    shp_path: Optional[str] = None,
    shp_key: Optional[str] = None,
    max_validation_draws: Optional[int] = 800,
):
    out_dir, idata, beta_draws, u_draws, feature_cols, gu_list, unit_long, gu_age, pred = load_unit_outputs(base_dir, unit_type)
    print(f"[POST] {unit_type}: {out_dir}")

    p1 = export_mcmc_plots(out_dir, idata, feature_cols)
    print("  saved MCMC plots:", p1)

    v = posterior_gu_age_predictions(out_dir, unit_type, beta_draws, u_draws, feature_cols, gu_list, unit_long, gu_age, max_draws=max_validation_draws)
    print("  saved internal validation plots:", out_dir / "plots_internal_validation")

    p3 = export_unit_prevalence_plots(out_dir, unit_type, pred)
    print("  saved unit prevalence plots:", p3)

    p4 = export_optional_map(out_dir, unit_type, pred, shp_path, shp_key)
    if p4:
        print("  saved maps:", p4)

    return {"unit_type": unit_type, "out_dir": str(out_dir), "validation": v}


def postprocess_all(
    base_dir: str = r"D:\@260515_nhis",
    dong_shp_path: Optional[str] = None,
    dong_shp_key: Optional[str] = None,
    local_shp_path: Optional[str] = None,
    local_shp_key: Optional[str] = None,
):
    results = {}
    results["dong"] = postprocess_unit(base_dir, "dong", shp_path=dong_shp_path, shp_key=dong_shp_key)
    results["local"] = postprocess_unit(base_dir, "local", shp_path=local_shp_path, shp_key=local_shp_key)
    return results


if __name__ == "__main__":
    # Edit paths here if you want map images. Leave as None to skip maps.
    BASE_DIR = r"D:\@260515_nhis"

    # Examples. Replace with your actual files and join-key fields.
    DONG_SHP_PATH = None       # r"D:\@260515_nhis\shp\Seoul_dong_5186.shp"
    DONG_SHP_KEY = None        # "dong_code" or "ADM_DR_CD"
    LOCAL_SHP_PATH = None      # r"D:\@260515_nhis\shp\UPIS_SHP_ZON100_5186.shp"
    LOCAL_SHP_KEY = None       # "local_code" or "zone_uid"

    postprocess_all(
        base_dir=BASE_DIR,
        dong_shp_path=DONG_SHP_PATH,
        dong_shp_key=DONG_SHP_KEY,
        local_shp_path=LOCAL_SHP_PATH,
        local_shp_key=LOCAL_SHP_KEY,
    )
