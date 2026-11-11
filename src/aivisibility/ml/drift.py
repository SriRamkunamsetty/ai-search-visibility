"""Statistical drift detection and anomaly monitoring for AI search visibility over time.
Implements Population Stability Index (PSI), 1D Wasserstein Earth Mover's Distance,
and MAD-based anomaly detection across model version releases."""
import math
from typing import Dict, List, Sequence


def calculate_psi(
    expected_dist: Sequence[float],
    actual_dist: Sequence[float],
    epsilon: float = 1e-4
) -> float:
    """Compute Population Stability Index (PSI) between two discrete probability distributions.

    Interpretations:
        PSI < 0.1: No significant drift (stable)
        0.1 <= PSI < 0.25: Moderate drift / subtle shift
        PSI >= 0.25: Significant distributional drift (action required)
    """
    if len(expected_dist) != len(actual_dist) or not expected_dist:
        return 0.0

    sum_exp = sum(expected_dist)
    sum_act = sum(actual_dist)

    if sum_exp <= 0.0 or sum_act <= 0.0:
        return 0.0

    # Normalize to probabilities
    p_exp = [(x / sum_exp) for x in expected_dist]
    p_act = [(y / sum_act) for y in actual_dist]

    psi = 0.0
    for e, a in zip(p_exp, p_act):
        # Smoothing to prevent division by zero or log(0)
        e_smooth = max(e, epsilon)
        a_smooth = max(a, epsilon)
        psi += (a_smooth - e_smooth) * math.log(a_smooth / e_smooth)

    return max(0.0, round(psi, 4))


def wasserstein_distance_1d(u_values: Sequence[float], v_values: Sequence[float]) -> float:
    """Compute the 1D Wasserstein distance (Earth Mover's Distance) between two empirical distributions.

    Measures the minimal work required to transform distribution U into distribution V.
    """
    if not u_values or not v_values:
        return 0.0

    all_vals = sorted(set(u_values) | set(v_values))
    if len(all_vals) <= 1:
        return 0.0

    u_sorted = sorted(u_values)
    v_sorted = sorted(v_values)

    n_u = len(u_sorted)
    n_v = len(v_sorted)

    # Compute empirical CDFs and integrate absolute difference
    total_work = 0.0
    for i in range(len(all_vals) - 1):
        x0, x1 = all_vals[i], all_vals[i + 1]
        width = x1 - x0

        # Empirical CDF value at x0
        import bisect
        cdf_u = bisect.bisect_right(u_sorted, x0) / n_u
        cdf_v = bisect.bisect_right(v_sorted, x0) / n_v

        total_work += abs(cdf_u - cdf_v) * width

    return round(total_work, 4)


def detect_visibility_anomalies(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    threshold_z: float = 2.0
) -> List[Dict[str, object]]:
    """Detect significant accuracy or visibility anomalies across tracked titles.

    Flags titles whose performance deviates significantly from baseline expectations.
    """
    anomalies: List[Dict[str, object]] = []

    # Calculate differences across all shared titles
    shared_titles = set(baseline_metrics.keys()) & set(current_metrics.keys())
    if not shared_titles:
        return []

    deltas = {t: current_metrics[t] - baseline_metrics[t] for t in shared_titles}
    values = list(deltas.values())

    mean_delta = sum(values) / len(values)
    variance = sum((x - mean_delta) ** 2 for x in values) / len(values)
    std_delta = math.sqrt(variance) if variance > 0 else 0.05

    for title, delta in deltas.items():
        z_score = (delta - mean_delta) / std_delta if std_delta > 0 else 0.0
        if abs(z_score) >= threshold_z:
            anomalies.append({
                "title": title,
                "baseline_value": round(baseline_metrics[title], 4),
                "current_value": round(current_metrics[title], 4),
                "delta": round(delta, 4),
                "z_score": round(z_score, 2),
                "severity": "critical" if abs(z_score) >= 3.0 else "warning",
                "direction": "improvement" if delta > 0 else "degradation"
            })

    return sorted(anomalies, key=lambda x: abs(x["z_score"]), reverse=True)
