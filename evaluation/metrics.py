from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence


def normalize_text(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^\w\s.%+-]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def phrase_matches(text: str, patterns: Sequence[str]) -> bool:
    """Match accepted phrases with word boundaries after text normalization."""
    normalized_text = normalize_text(text)

    for pattern in patterns:
        normalized_pattern = normalize_text(str(pattern))
        if not normalized_pattern:
            continue

        expression = rf"(?<!\w){re.escape(normalized_pattern)}(?!\w)"
        if re.search(expression, normalized_text):
            return True

    return False


def exact_matches(text: str, patterns: Sequence[str]) -> bool:
    """Match the complete answer against accepted short forms such as 'No.'."""
    normalized_text = normalize_text(text).strip(" .")

    return any(
        normalized_text == normalize_text(str(pattern)).strip(" .")
        for pattern in patterns
        if pattern
    )


def evidence_matches_chunk(
    evidence: str,
    chunk: str,
    token_recall_threshold: float = 0.70,
) -> bool:
    evidence_norm = normalize_text(evidence)
    chunk_norm = normalize_text(chunk)

    if not evidence_norm:
        return False
    if evidence_norm in chunk_norm:
        return True

    evidence_tokens = set(evidence_norm.split())
    chunk_tokens = set(chunk_norm.split())

    if not evidence_tokens:
        return False

    token_recall = len(evidence_tokens & chunk_tokens) / len(evidence_tokens)
    return token_recall >= token_recall_threshold


def retrieval_scores(
    retrieved_chunks: Sequence[str],
    gold_evidence: Sequence[str],
    token_recall_threshold: float = 0.70,
) -> dict[str, float | int | None]:
    if not gold_evidence:
        return {
            "context_recall": None,
            "context_precision": None,
            "gold_evidence_hits": 0,
            "relevant_chunks": 0,
        }

    evidence_hits = sum(
        any(
            evidence_matches_chunk(evidence, chunk, token_recall_threshold)
            for chunk in retrieved_chunks
        )
        for evidence in gold_evidence
    )
    relevant_chunks = sum(
        any(
            evidence_matches_chunk(evidence, chunk, token_recall_threshold)
            for evidence in gold_evidence
        )
        for chunk in retrieved_chunks
    )

    return {
        "context_recall": evidence_hits / len(gold_evidence),
        "context_precision": (
            relevant_chunks / len(retrieved_chunks) if retrieved_chunks else 0.0
        ),
        "gold_evidence_hits": evidence_hits,
        "relevant_chunks": relevant_chunks,
    }


def fact_accuracy(answer: str, required_facts: Sequence[Any]) -> dict[str, Any]:
    if not required_facts:
        return {
            "fact_accuracy": None,
            "facts_matched": 0,
            "facts_total": 0,
            "fact_details": [],
        }

    details: list[dict[str, Any]] = []
    matched = 0

    for index, fact in enumerate(required_facts, start=1):
        if isinstance(fact, str):
            name = fact
            patterns = [fact]
            exact_patterns: list[str] = []
            forbidden_patterns: list[str] = []

        elif isinstance(fact, dict):
            name = str(fact.get("name") or f"fact_{index}")
            patterns = fact.get("patterns") or fact.get("acceptable_patterns") or []
            exact_patterns = fact.get("exact_patterns") or []
            forbidden_patterns = fact.get("forbidden_patterns") or []

            if isinstance(patterns, str):
                patterns = [patterns]
            if isinstance(exact_patterns, str):
                exact_patterns = [exact_patterns]
            if isinstance(forbidden_patterns, str):
                forbidden_patterns = [forbidden_patterns]

        else:
            raise TypeError(f"Unsupported fact definition: {fact!r}")

        positive_match = (
            phrase_matches(answer, patterns)
            or exact_matches(answer, exact_patterns)
        )
        forbidden_match = phrase_matches(answer, forbidden_patterns)
        is_match = positive_match and not forbidden_match

        matched += int(is_match)
        details.append(
            {
                "name": name,
                "matched": is_match,
                "patterns": list(patterns),
                "exact_patterns": list(exact_patterns),
                "forbidden_patterns": list(forbidden_patterns),
            }
        )

    return {
        "fact_accuracy": matched / len(required_facts),
        "facts_matched": matched,
        "facts_total": len(required_facts),
        "fact_details": details,
    }


def classification_report(
    expected: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    f1_values: list[float] = []

    for label in labels:
        tp = sum(e == label and p == label for e, p in zip(expected, predicted))
        fp = sum(e != label and p == label for e, p in zip(expected, predicted))
        fn = sum(e == label and p != label for e, p in zip(expected, predicted))
        support = sum(e == label for e in expected)

        precision_value = tp / (tp + fp) if tp + fp else 0.0
        recall_value = tp / (tp + fn) if tp + fn else 0.0
        f1_value = (
            2 * precision_value * recall_value / (precision_value + recall_value)
            if precision_value + recall_value
            else 0.0
        )

        # Exclude labels absent from a small subset/smoke test.
        if support > 0:
            f1_values.append(f1_value)

        per_label[label] = {
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    accuracy = (
        sum(e == p for e, p in zip(expected, predicted)) / len(expected)
        if expected
        else None
    )

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_label": per_label,
    }


def grouped_mean(
    rows: Sequence[dict[str, Any]],
    key: str,
    value: str,
) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        metric_value = row.get(value)
        if metric_value is not None:
            groups[str(row.get(key, "UNKNOWN"))].append(float(metric_value))

    return {group: mean(values) for group, values in groups.items()}