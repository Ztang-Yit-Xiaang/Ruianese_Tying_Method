from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


def topk_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    classes: Sequence[str],
    k: int = 5,
) -> list[dict[str, Any]]:
    probs = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probs)[: min(k, len(classes))]
    return [{"class": classes[int(index)], "probability": float(probs[index])} for index in order]


def decode_ipa_pair(
    initial_probabilities: Sequence[float] | np.ndarray,
    final_probabilities: Sequence[float] | np.ndarray,
    initial_classes: Sequence[str],
    final_classes: Sequence[str],
    allowed_pairs: Iterable[tuple[str, str]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Decode the highest-scoring initial/final combination in the supplied pair set."""

    initial_probs = np.asarray(initial_probabilities, dtype=np.float64)
    final_probs = np.asarray(final_probabilities, dtype=np.float64)
    raw_initial_index = int(np.argmax(initial_probs))
    raw_final_index = int(np.argmax(final_probs))
    raw_pair = (initial_classes[raw_initial_index], final_classes[raw_final_index])
    pair_set = set(allowed_pairs)
    initial_index = {value: index for index, value in enumerate(initial_classes)}
    final_index = {value: index for index, value in enumerate(final_classes)}
    scored: list[tuple[float, str, str]] = []
    for initial, final in sorted(pair_set):
        if initial not in initial_index or final not in final_index:
            continue
        score = math.log(max(float(initial_probs[initial_index[initial]]), 1e-12)) + math.log(
            max(float(final_probs[final_index[final]]), 1e-12)
        )
        scored.append((score, initial, final))
    if not scored:
        raise ValueError("No allowed IPA pairs overlap the checkpoint class vocabulary")
    scored.sort(reverse=True)
    best_score, best_initial, best_final = scored[0]
    return {
        "raw_ipa_initial": raw_pair[0],
        "raw_ipa_final": raw_pair[1],
        "raw_pair_valid": raw_pair in pair_set,
        "predicted_ipa_initial": best_initial,
        "predicted_ipa_final": best_final,
        "constraint_changed_prediction": raw_pair != (best_initial, best_final),
        "pair_score": float(best_score),
        "constrained_topk_pairs": [
            {"ipa_initial": initial, "ipa_final": final, "score": float(score)}
            for score, initial, final in scored[: max(1, top_k)]
        ],
        "raw_topk_initial": topk_probabilities(initial_probs, initial_classes, top_k),
        "raw_topk_final": topk_probabilities(final_probs, final_classes, top_k),
    }
