from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def horizon_errors(y_true_by_h: dict[int, np.ndarray], y_pred_by_h: dict[int, np.ndarray]) -> dict[int, dict[str, float]]:
    out = {}
    for h, yt in y_true_by_h.items():
        if h not in y_pred_by_h:
            continue
        yp = y_pred_by_h[h]
        out[h] = {"mae": mae(yt, yp), "rmse": rmse(yt, yp), "mape": mape(yt, yp)}
    return out
