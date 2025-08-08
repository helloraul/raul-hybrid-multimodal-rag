# src/evaluation/schema.py
from __future__ import annotations
from typing import Dict, List

# Canonical keys we care about in evaluations
CANON = {
    "question": ["question", "frage", "prompt", "query", "q"],
    "docref": [
        "doc_filename", "filename", "file", "path", "document", "doc", "doc_id",
        "pdf", "source", "file_name", "file_path"
    ],
    "ground_truth": [
        "ground_truth", "groundtruth", "answer", "expected", "label", "gt", "gold"
    ],
}

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("-", "_").replace(" ", "_")

def detect_columns(headers: List[str]) -> Dict[str, str]:
    """
    Detect once per file and reuse. Returns a mapping like:
      {"question": "Question", "docref": "Document", "ground_truth": "Answer"}
    Only includes keys it can actually find. Heuristics fill gaps.
    """
    norm_map = { _norm(h): h for h in headers }
    mapping: Dict[str, str] = {}

    for key, aliases in CANON.items():
        for a in aliases:
            if a in norm_map:
                mapping[key] = norm_map[a]
                break

        # heuristics if still missing
        if key not in mapping:
            if key == "docref":
                for n in norm_map:
                    if any(tok in n for tok in ["doc", "file", "pdf", "document", "path", "source"]):
                        mapping[key] = norm_map[n]
                        break
            elif key == "question":
                for n in norm_map:
                    if "question" in n or n in {"q", "query"}:
                        mapping[key] = norm_map[n]
                        break
            elif key == "ground_truth":
                for n in norm_map:
                    if "answer" in n or "ground" in n or "expected" in n or n in {"gt", "gold"}:
                        mapping[key] = norm_map[n]
                        break

    return mapping
