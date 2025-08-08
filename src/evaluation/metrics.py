from __future__ import annotations
import re
from typing import List, Dict

_ws = re.compile(r"\s+")

def _norm(s: str) -> str:
    if s is None:
        return ""
    return _ws.sub(" ", s.strip().lower())

def exact_match(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    y_true_n = [_norm(t) for t in y_true]
    y_pred_n = [_norm(p) for p in y_pred]

    matches = [1 if t == p else 0 for t, p in zip(y_true_n, y_pred_n)]
    tp = sum(matches)
    total = len(matches)
    precision = tp / total if total else 0.0
    recall = tp / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = tp / total if total else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
