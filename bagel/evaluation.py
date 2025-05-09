import numpy as np

from typing import *
from sklearn.metrics import precision_recall_curve


def _adjust_scores(
        labels: np.ndarray,
        scores: np.ndarray,
        delay: int | None = None,
        inplace: bool = False,
) -> np.ndarray:
    if np.shape(scores) != np.shape(labels):
        raise ValueError('Shape mismatch between labels and scores.'
                         f'labels: {np.shape(labels)}, scores: {np.shape(scores)}')
    if delay is None:
        delay = len(scores)
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    is_anomaly = labels[0] == 1
    adjusted_scores = np.copy(scores) if not inplace else scores
    pos = 0
    for part in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, part)
            adjusted_scores[pos: ptr] = np.max(adjusted_scores[pos: ptr])
            adjusted_scores[ptr: part] = np.maximum(adjusted_scores[ptr: part], adjusted_scores[pos])
        is_anomaly = not is_anomaly
        pos = part
    part = len(labels)
    if is_anomaly:
        ptr = min(pos + delay + 1, part)
        adjusted_scores[pos: part] = np.max(adjusted_scores[pos: ptr])
    return adjusted_scores


def _ignore_missing(series_list: Sequence, missing: np.ndarray) -> tuple[np.ndarray, ...]:
    ret = []
    for series in series_list:
        series = np.copy(series)
        ret.append(series[missing != 1])
    return tuple(ret)


def _best_f1score(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=scores)
    f1score = 2 * precision * recall / np.clip(precision + recall, a_min=1e-8, a_max=None)

    best_threshold = thresholds[np.argmax(f1score)]
    best_precision = precision[np.argmax(f1score)]
    best_recall = recall[np.argmax(f1score)]

    return best_threshold, best_precision, best_recall, np.max(f1score)


def evaluate(
        labels: np.ndarray,
        scores: np.ndarray,
        missing: np.ndarray,
        window_size: int,
        delay: int | None = None,
) -> dict[str, float]:
    labels = labels[window_size - 1:]
    scores = scores[window_size - 1:]
    missing = missing[window_size - 1:]
    adjusted_scores = _adjust_scores(labels=labels, scores=scores, delay=delay)
    adjusted_labels, adjusted_scores = _ignore_missing([labels, adjusted_scores], missing=missing)
    threshold, precision, recall, f1score = _best_f1score(labels=adjusted_labels, scores=adjusted_scores)
    return {
        'threshold': float(threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1score': float(f1score),
    }
