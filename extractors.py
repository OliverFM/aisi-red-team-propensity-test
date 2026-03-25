"""Custom hibayes extractors and processors for the laziness eval."""

from typing import Any

import jax.numpy as jnp
from hibayes.load import Extractor, extractor
from hibayes.process._process import DataProcessor, process
from inspect_ai.log import EvalLog, EvalSample


@extractor
def laziness_extractor() -> Extractor:
    """Extract the 3 experimental condition features from sample metadata tags."""

    def extract(sample: EvalSample, eval_log: EvalLog) -> dict[str, Any]:
        tags = (sample.metadata or {}).get("tags", [])
        return {
            "verification_hard": "verification_hard" in tags,
            "status_incomplete": "status_incomplete" in tags,
            "domain_safety": "domain_safety" in tags,
        }

    return extract


@extractor
def laziness_score_extractor() -> Extractor:
    """Extract the binary laziness_judge score as the outcome variable."""

    def extract(sample: EvalSample, eval_log: EvalLog) -> dict[str, Any]:
        if not sample.scores:
            raise ValueError("Sample has no scores")

        score_obj = sample.scores.get("laziness_judge")
        if score_obj is None:
            raise ValueError("Sample has no laziness_judge score")

        value = score_obj.value
        if value not in (0, 1):
            raise ValueError(f"laziness_judge score must be 0 or 1, got {value}")

        return {"score": int(value)}

    return extract


@extractor
def n_total_extractor() -> Extractor:
    """Add n_total=1 for Bernoulli trials (required by binomial model)."""

    def extract(sample: EvalSample, eval_log: EvalLog) -> dict[str, Any]:
        return {"n_total": 1}

    return extract


@process
def add_n_total_feature() -> DataProcessor:
    """Add n_total feature to the state for Bernoulli trials.

    Adds n_total=1 (as a JAX array) to the features dict,
    required by linear_group_binomial for non-aggregated data.
    """

    def process_fn(state: Any, display: Any = None) -> Any:
        n_samples = len(state.processed_data)
        n_total_array = jnp.ones(n_samples, dtype=jnp.int32)

        if state.features is None:
            state.features = {}
        state.features["n_total"] = n_total_array

        if display:
            display.logger.info(
                f"Added n_total feature with shape {n_total_array.shape}"
            )

        return state

    return process_fn
