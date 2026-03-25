#!/usr/bin/env python3
"""Run Bayesian analysis on laziness eval results.

Usage:
    uv run python run_analysis.py --paths logs/laziness-eval-rescored.eval
    uv run python run_analysis.py --paths logs/  # all .eval files in dir
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hibayes.analysis import AnalysisConfig, load_data, model, process_data
from hibayes.ui import ModellingDisplay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_eval_files(paths: list[str]) -> list[str]:
    """Find all .eval files from the given paths."""
    files = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".eval":
            files.append(str(path))
        elif path.is_dir():
            files.extend(str(f) for f in path.rglob("*.eval"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Run laziness eval analysis")
    parser.add_argument(
        "--paths", nargs="+", required=True, help="Paths to .eval files or directories"
    )
    parser.add_argument(
        "--config",
        default="analysis_config.yaml",
        help="Path to hibayes config YAML",
    )
    args = parser.parse_args()

    # Find eval files
    eval_files = find_eval_files(args.paths)
    if not eval_files:
        logger.error("No .eval files found")
        return
    logger.info(f"Found {len(eval_files)} eval file(s)")

    # Load config and override file paths
    config = AnalysisConfig.from_yaml(args.config)
    config.data_loader.files_to_process = eval_files

    display = ModellingDisplay()

    # Load data from eval logs
    logger.info("Loading data from eval logs...")
    state = load_data(config.data_loader, display)
    df = state.data

    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {list(df.columns)}")

    # Print raw data summary
    print("\n=== Raw Data ===")
    for col in ["score", "verification_hard", "status_incomplete", "domain_safety"]:
        if col in df.columns:
            print(f"  {col}: {df[col].value_counts().to_dict()}")

    # Print cross-tabulation
    if all(
        c in df.columns
        for c in ["score", "verification_hard", "status_incomplete", "domain_safety"]
    ):
        print("\n=== Score by condition ===")
        grouped = df.groupby(
            ["verification_hard", "status_incomplete", "domain_safety"]
        )["score"]
        for name, group in grouped:
            print(
                f"  hard={name[0]}, incomplete={name[1]}, safety={name[2]}: "
                f"mean={group.mean():.2f}, n={len(group)}"
            )

    # Process data for modelling
    logger.info("Processing data...")
    state = process_data(config.data_process, display, data=df)
    # Clear shared models list (hibayes mutable default workaround)
    state._models = []

    # Set intercept prior based on base rate
    base_rate = float(df["score"].mean())
    p = np.clip(base_rate, 1e-6, 1 - 1e-6)
    logit_base = float(np.log(p / (1 - p)))
    logger.info(f"Base rate: {base_rate:.3f}, logit: {logit_base:.3f}")

    # Fit model
    logger.info("Fitting Bayesian model...")
    state = model(
        state,
        config.models,
        config.checkers,
        config.platform,
        display,
    )

    # Print results
    print("\n=== Model Results ===")
    for model_state in state.models:
        tag = model_state.model_config.tag or model_state.model_name
        print(f"\nModel: {tag}")

        if model_state.inference_data is not None:
            import arviz as az

            summary = az.summary(model_state.inference_data, hdi_prob=0.95)
            print(summary.to_string())

        # Print diagnostics
        diag = model_state.diagnostics or {}
        if "divergences" in diag:
            print(f"  Divergences: {diag['divergences']}")
        if "r_hat" in diag:
            print(
                f"  Max R-hat: {max(diag['r_hat']) if hasattr(diag['r_hat'], '__iter__') else diag['r_hat']}"
            )

        # Generate forest plot
        plot_odds_ratio_forest(model_state, Path("plots"))


PARAM_DISPLAY_NAMES = {
    "verification_hard": "Verification Difficulty\n(hard vs easy)",
    "status_incomplete": "Task Status\n(incomplete vs complete)",
    "domain_safety": "Domain\n(safety vs capabilities)",
}


def plot_odds_ratio_forest(model_state, output_dir: Path):
    """Plot a forest plot of odds ratios with 95% HDIs for each parameter."""
    idata = model_state.inference_data
    if idata is None:
        logger.warning("No inference data, skipping forest plot")
        return

    posterior = idata.posterior

    # Extract the effect for the True level of each binary parameter
    params = []
    for var_name in PARAM_DISPLAY_NAMES:
        key = f"{var_name}_effects"
        if key not in posterior:
            continue
        da = posterior[key]
        # The coordinate dimension is the variable name itself
        coord_dim = var_name
        if coord_dim not in da.dims:
            logger.warning(f"Dimension {coord_dim} not found in {key}, skipping")
            continue
        # Select the True level (effect-coded: True = +effect)
        if True not in da.coords[coord_dim].values:
            logger.warning(f"True not in coords for {key}, skipping")
            continue
        samples = da.sel({coord_dim: True}).values.flatten()
        params.append(
            {
                "name": var_name,
                "display": PARAM_DISPLAY_NAMES[var_name],
                "log_or_samples": samples,
            }
        )

    if not params:
        logger.warning("No parameters found for forest plot")
        return

    # Convert log-odds to odds ratios
    fig, ax = plt.subplots(figsize=(10, 2.0 + len(params) * 1.0))

    y_positions = list(range(len(params)))
    for i, p in enumerate(params):
        or_samples = np.exp(p["log_or_samples"])
        median = np.median(or_samples)
        hdi_lo = np.percentile(or_samples, 2.5)
        hdi_hi = np.percentile(or_samples, 97.5)

        ax.errorbar(
            median,
            i,
            xerr=[[median - hdi_lo], [hdi_hi - median]],
            fmt="o",
            color="steelblue",
            capsize=5,
            markersize=8,
            linewidth=2,
        )

        ax.annotate(
            f"OR={median:.2f} [{hdi_lo:.2f}, {hdi_hi:.2f}]",
            xy=(hdi_hi, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    # Reference line at OR=1 (no effect)
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([p["display"] for p in params])
    ax.set_xlabel("Odds Ratio for Meeting Effort/Thoroughness Criteria (95% HDI)")
    ax.set_title(
        "Effect of Experimental Conditions on Meeting Effort/Thoroughness Criteria"
    )
    ax.invert_yaxis()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "odds_ratio_forest.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nForest plot saved to {out_path}")


if __name__ == "__main__":
    main()
