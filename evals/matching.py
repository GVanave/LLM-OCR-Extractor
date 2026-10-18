"""Pair predicted items with ground-truth items before scoring (plan Step 4).

Items are a list, so position-by-position comparison is unfair: one missed row
would shift every later row. We use Hungarian matching on name similarity, with
a small bonus when prices agree, so the pairing is optimal over the whole receipt.
"""
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from receipt.normalize import norm_name, numbers_equal

MIN_PAIR_SIMILARITY = 0.5  # below this, two items are not the same row at all
NAME_MATCH_THRESHOLD = 0.8  # Phase 1 name judge; replaced by JEV / LLM judge in Step 6


def name_similarity(a: str, b: str) -> float:
    return fuzz.ratio(norm_name(a), norm_name(b)) / 100


def names_match(a: str, b: str) -> bool:
    """Do two names refer to the same item? Fuzzy for now (Step 6 adds a judge)."""
    na, nb = norm_name(a), norm_name(b)
    return na == nb or name_similarity(na, nb) >= NAME_MATCH_THRESHOLD


@dataclass
class Pair:
    pred_idx: int
    gt_idx: int
    similarity: float


def match_items(pred: list[dict], gt: list[dict]) -> tuple[list[Pair], list[int], list[int]]:
    """Returns (pairs, unmatched_pred_indices, unmatched_gt_indices)."""
    if not pred or not gt:
        return [], list(range(len(pred))), list(range(len(gt)))
    score = np.zeros((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            sim = name_similarity(p.get("name", ""), g.get("name", ""))
            if g.get("price") is not None and numbers_equal(p.get("price"), g.get("price")):
                sim += 0.1  # tie-breaker for repeated or similar names
            score[i, j] = sim
    rows, cols = linear_sum_assignment(-score)
    pairs = [Pair(int(i), int(j), float(score[i, j])) for i, j in zip(rows, cols)
             if score[i, j] >= MIN_PAIR_SIMILARITY]
    used_p = {p.pred_idx for p in pairs}
    used_g = {p.gt_idx for p in pairs}
    return (pairs,
            [i for i in range(len(pred)) if i not in used_p],
            [j for j in range(len(gt)) if j not in used_g])
