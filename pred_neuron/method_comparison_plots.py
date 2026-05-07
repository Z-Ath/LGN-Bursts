from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_TO_RESULTS_ROOT = {
    "fe_v1": "./results/fe_v1",
    "fe_optuna": "./results/fe_optuna",
    "fe_randgt": "./results/fe_randgt",
}

METHOD_TO_LABEL = {
    "fe_v1": "embedding_xgboost_baseline",
    "fe_optuna": "embedding_xgboost_optuna",
    "fe_randgt": "random_target_control",
}

METHOD_TO_COLOR = {
    "fe_v1": "#4C72B0",
    "fe_optuna": "#DD8452",
    "fe_randgt": "#55A868",
}


@dataclass
class FEPlotConfig:
    methods: list[str] = field(
        default_factory=lambda: ["fe_v1", "fe_optuna", "fe_randgt"]
    )
    pred: str = "burst_B"
    pred_mean: bool = False
    random_state: int = 42
    familiar_mode: str = "none"
    deviant_mode: str = "none"
    results_roots: dict[str, str] = field(
        default_factory=lambda: dict(METHOD_TO_RESULTS_ROOT)
    )
    filter_ratio_pearson_mean_per_neuron: float | None = None
    filter_ratio_pearson_mean_per_trial: float | None = None
    output_root: str = "./results/fe_plot"
    figure_dpi: int = 180


def _control_tag(familiar_mode: str, deviant_mode: str, random_state: int) -> str:
    return f"fam-{familiar_mode}_dev-{deviant_mode}_seed{random_state}"


def _session_id_sort_key(session_id: str) -> tuple[int, str]:
    suffix = str(session_id).split("_")[-1]
    try:
        return int(suffix), str(session_id)
    except ValueError:
        return -1, str(session_id)

def _metric_columns_exist(df: pd.DataFrame) -> bool:
    required = {
        "session_id",
        "r2_mean_per_neuron",
        "r2_std_per_neuron",
        "pearson_mean_per_neuron",
        "pearson_std_per_neuron",
        "r2_mean_per_trial",
        "r2_std_per_trial",
        "pearson_mean_per_trial",
        "pearson_std_per_trial",
    }
    return required.issubset(df.columns)


def _find_first_matching(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _resolve_method_files(config: FEPlotConfig, method: str) -> tuple[Path, Path | None]:
    if method not in config.results_roots:
        raise KeyError(
            f"Unknown method={method!r}. Available roots: {sorted(config.results_roots)}"
        )

    results_root = Path(config.results_roots[method])
    control_tag = _control_tag(
        familiar_mode=config.familiar_mode,
        deviant_mode=config.deviant_mode,
        random_state=config.random_state,
    )
    results_dir = results_root / config.pred / control_tag
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    pred_tag = "mean" if config.pred_mean else "full"
    exact_eval = results_dir / f"evaluation_{config.pred}_{pred_tag}_{method}.csv"
    eval_path = exact_eval if exact_eval.exists() else _find_first_matching(
        results_dir, "evaluation_*.csv"
    )
    if eval_path is None:
        raise FileNotFoundError(f"No evaluation_*.csv found under {results_dir}")

    exact_per_dim = results_dir / f"per_dim_{config.pred}_{pred_tag}_{method}.csv"
    per_dim_path = exact_per_dim if exact_per_dim.exists() else _find_first_matching(
        results_dir, "per_dim_*.csv"
    )
    return eval_path, per_dim_path


def _summarize_per_dim_csv(per_dim_path: Path) -> pd.DataFrame:
    df = pd.read_csv(per_dim_path)
    required = {"session_id", "r2", "pearson"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {per_dim_path}: {sorted(required - set(df.columns))}"
        )

    summary = (
        df.groupby("session_id", as_index=False)
        .agg(
            r2_mean_per_neuron=("r2", lambda s: float(np.nanmean(s.to_numpy()))),
            r2_std_per_neuron=("r2", lambda s: float(np.nanstd(s.to_numpy()))),
            pearson_mean_per_neuron=(
                "pearson",
                lambda s: float(np.nanmean(s.to_numpy())),
            ),
            pearson_std_per_neuron=(
                "pearson",
                lambda s: float(np.nanstd(s.to_numpy())),
            ),
        )
        .reset_index(drop=True)
    )
    summary["r2_mean_per_trial"] = np.nan
    summary["r2_std_per_trial"] = np.nan
    summary["pearson_mean_per_trial"] = np.nan
    summary["pearson_std_per_trial"] = np.nan
    return summary


def _load_method_summary(config: FEPlotConfig, method: str) -> pd.DataFrame:
    eval_path, per_dim_path = _resolve_method_files(config=config, method=method)
    df_eval = pd.read_csv(eval_path)

    if _metric_columns_exist(df_eval):
        cols = [
            "session_id",
            "r2_mean_per_neuron",
            "r2_std_per_neuron",
            "pearson_mean_per_neuron",
            "pearson_std_per_neuron",
            "r2_mean_per_trial",
            "r2_std_per_trial",
            "pearson_mean_per_trial",
            "pearson_std_per_trial",
        ]
        summary = df_eval[cols].copy()
    else:
        if per_dim_path is None:
            raise ValueError(
                f"{eval_path} does not contain the required per-neuron/per-trial mean/std columns and no per_dim csv was found."
            )
        summary = _summarize_per_dim_csv(per_dim_path=per_dim_path)

    summary["method"] = method
    summary["method_label"] = METHOD_TO_LABEL.get(method, method)
    summary["evaluation_csv"] = str(eval_path)
    summary["per_dim_csv"] = "" if per_dim_path is None else str(per_dim_path)
    return summary


def _resolve_anchor_method(df: pd.DataFrame, methods: list[str]) -> str:
    available_methods = set(df["method"].astype(str).tolist())
    if "fe_v1" in available_methods:
        return "fe_v1"
    for method in methods:
        if method in available_methods:
            return method
    raise ValueError("No available methods found in plot table.")


def _validate_filter_ratio(ratio: float | None, field_name: str) -> float | None:
    if ratio is None:
        return None
    ratio = float(ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"{field_name} must be in the open interval (0, 1), got {ratio}."
        )
    return ratio


def _apply_filter_ratio(
    df: pd.DataFrame,
    methods: list[str],
    filter_col: str,
    filter_ratio: float | None,
) -> pd.DataFrame:
    filter_ratio = _validate_filter_ratio(filter_ratio, field_name=filter_col)
    if filter_ratio is None:
        return df

    anchor_method = _resolve_anchor_method(df=df, methods=methods)
    anchor_df = df[df["method"] == anchor_method][["session_id", filter_col]].copy()
    anchor_df = anchor_df[np.isfinite(anchor_df[filter_col].to_numpy(dtype=np.float64))]
    if anchor_df.empty:
        return df

    keep_count = int(np.ceil(len(anchor_df) * (1.0 - filter_ratio)))
    keep_count = max(1, min(keep_count, len(anchor_df)))
    keep_sessions = (
        anchor_df.sort_values(filter_col, ascending=False)
        .head(keep_count)["session_id"]
        .astype(str)
        .tolist()
    )
    return df[df["session_id"].astype(str).isin(keep_sessions)].copy()


def _filter_sessions(df: pd.DataFrame, config: FEPlotConfig) -> pd.DataFrame:
    df = _apply_filter_ratio(
        df=df,
        methods=config.methods,
        filter_col="pearson_mean_per_neuron",
        filter_ratio=config.filter_ratio_pearson_mean_per_neuron,
    )
    df = _apply_filter_ratio(
        df=df,
        methods=config.methods,
        filter_col="pearson_mean_per_trial",
        filter_ratio=config.filter_ratio_pearson_mean_per_trial,
    )
    return df


def _ordered_sessions(df: pd.DataFrame, anchor_method: str) -> list[str]:
    all_sessions = set(df["session_id"].tolist())

    anchor_df = df[df["method"] == anchor_method][
        ["session_id", "pearson_mean_per_neuron"]
    ].copy()
    anchor_rows = list(
        anchor_df[["session_id", "pearson_mean_per_neuron"]].itertuples(index=False, name=None)
    )
    anchor_rows.sort(
        key=lambda item: (
            float("-inf")
            if not np.isfinite(item[1])
            else float(item[1]),
            _session_id_sort_key(str(item[0]))[0],
        ),
        reverse=True,
    )
    ordered = [str(session_id) for session_id, _ in anchor_rows]

    missing_sessions = sorted(
        all_sessions - set(ordered),
        key=_session_id_sort_key,
        reverse=True,
    )
    return ordered + missing_sessions


def _build_plot_table(config: FEPlotConfig) -> pd.DataFrame:
    frames = [_load_method_summary(config=config, method=method) for method in config.methods]
    df = pd.concat(frames, ignore_index=True)
    df = _filter_sessions(df=df, config=config)
    anchor_method = _resolve_anchor_method(df=df, methods=config.methods)
    ordered_sessions = _ordered_sessions(df=df, anchor_method=anchor_method)
    df["session_id"] = pd.Categorical(
        df["session_id"],
        categories=ordered_sessions,
        ordered=True,
    )
    df = df.sort_values(["session_id", "method"]).reset_index(drop=True)
    return df


def _plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    methods: list[str],
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
) -> None:
    sessions = list(df["session_id"].cat.categories)
    x = np.arange(len(sessions), dtype=np.float64)
    width = 0.82 / max(len(methods), 1)

    for idx, method in enumerate(methods):
        method_df = (
            df[df["method"] == method]
            .set_index("session_id")
            .reindex(sessions)
            .reset_index()
        )
        offset = (idx - (len(methods) - 1) / 2.0) * width
        means = method_df[metric_mean_col].to_numpy(dtype=np.float64)
        stds = method_df[metric_std_col].to_numpy(dtype=np.float64)
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            color=METHOD_TO_COLOR.get(method, None),
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label=METHOD_TO_LABEL.get(method, method),
        )

    ax.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def plot_fe_session_bars(config: FEPlotConfig) -> tuple[pd.DataFrame, Path]:
    df = _build_plot_table(config=config)
    control_tag = _control_tag(
        familiar_mode=config.familiar_mode,
        deviant_mode=config.deviant_mode,
        random_state=config.random_state,
    )
    method_tag = "_vs_".join(config.methods)
    pred_tag = "mean" if config.pred_mean else "full"

    output_dir = Path(config.output_root) / config.pred / control_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(12, len(df["session_id"].cat.categories) * 0.7), 10),
        sharex=True,
        constrained_layout=True,
    )
    _plot_metric(
        ax=axes[0],
        df=df,
        methods=config.methods,
        metric_mean_col="r2_mean_per_neuron",
        metric_std_col="r2_std_per_neuron",
        ylabel="R^2 by neuron (mean ± std)",
    )
    _plot_metric(
        ax=axes[1],
        df=df,
        methods=config.methods,
        metric_mean_col="pearson_mean_per_neuron",
        metric_std_col="pearson_std_per_neuron",
        ylabel="Pearson by neuron (mean ± std)",
    )

    axes[0].set_title(
        f"FE session evaluation by neuron\n"
        f"pred={config.pred}, pred_mean={config.pred_mean}, "
        f"fam={config.familiar_mode}, dev={config.deviant_mode}, seed={config.random_state}, "
        f"filter_ratio_neuron={config.filter_ratio_pearson_mean_per_neuron}, "
        f"filter_ratio_trial={config.filter_ratio_pearson_mean_per_trial}"
    )
    axes[0].legend(ncols=max(1, len(config.methods)), frameon=False)
    axes[1].set_xlabel("Session ID (descending)")

    fig_path = output_dir / f"barplot_{config.pred}_{pred_tag}_{method_tag}.png"
    fig.savefig(fig_path, dpi=config.figure_dpi, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / f"barplot_{config.pred}_{pred_tag}_{method_tag}_summary.csv"
    export_df = df.copy()
    export_df["session_id"] = export_df["session_id"].astype(str)
    export_df.to_csv(csv_path, index=False)

    print(f"[Saved figure] {fig_path}")
    print(f"[Saved summary] {csv_path}")
    return df, fig_path


if __name__ == "__main__":
    config = FEPlotConfig(
        methods=["fe_v1","fe_optuna"],
        pred="burst_B",
        pred_mean=False,
        random_state=42,
        familiar_mode="gaussian_trials",  # none | gaussian_trials | shuffle_trials | scramble_pixels
        deviant_mode="gaussian_trials",  # none | gaussian_trials | shuffle_trials | scramble_pixels
        results_roots={
            "fe_v1": "./results/fe_v1",
            "fe_optuna": "./results/fe_optuna",
            "fe_randgt": "./results/fe_randgt",
        },
        filter_ratio_pearson_mean_per_neuron=1 / 5,
        filter_ratio_pearson_mean_per_trial=1 / 5,
        output_root="./results/fe_plot",
    )

    plot_fe_session_bars(config=config)
