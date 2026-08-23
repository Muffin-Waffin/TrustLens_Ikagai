"""
SynthGuard Phase 3: Forensic Decision Engine

Evaluates frame-level model predictions to produce:
- Manipulation score (median)
- Consistency
- Evidence reliability
- Evidence confidence
- Suspicious frames/segments
- Verdict: REAL / INCONCLUSIVE / LIKELY_DEEPFAKE
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SuspiciousFrame:
    frame_index: int
    timestamp_seconds: float
    score: float
    face_quality: float = 0.0
    weight: float = 0.0
    face_path: str = ""


@dataclass
class SuspiciousSegment:
    start: float
    end: float
    duration: float
    frame_count: int
    peak_score: float
    mean_score: float

    def __post_init__(self):
        """Validate segment invariants."""
        if self.start > self.end:
            raise ValueError(f"Invalid segment: start ({self.start}) > end ({self.end})")
        if self.duration < 0:
            raise ValueError(f"Invalid segment: negative duration ({self.duration})")


@dataclass
class ForensicResult:
    """Canonical forensic analysis result - single source of truth."""
    video_id: str
    verdict: str
    manipulation_score: float
    mean_score: float
    median_score: float
    max_score: float
    std_score: float
    raw_median_score: float
    weighted_mean_score: float
    weighted_median_score: float
    weighted_std_score: float
    min_frame_weight: float
    max_frame_weight: float
    mean_frame_weight: float
    consistency: float
    frame_coverage: float
    average_face_quality: float
    reliability: float
    evidence_confidence: float
    sampled_frames: int
    usable_frames: int
    suspicious_frames: list[SuspiciousFrame]
    suspicious_segments: list[SuspiciousSegment]
    explanations: list[str]
    reason_codes: list[str] = field(default_factory=list)
    average_boundary_score: Optional[float] = None
    average_frequency_anomaly: Optional[float] = None
    blink_naturalness_score: Optional[float] = None
    identity_drift_score: Optional[float] = None
    robustness_stability_score: Optional[float] = None
    signal_coverage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "verdict": self.verdict,
            "manipulation_score": self.manipulation_score,
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "max_score": self.max_score,
            "std_score": self.std_score,
            "raw_median_score": self.raw_median_score,
            "weighted_mean_score": self.weighted_mean_score,
            "weighted_median_score": self.weighted_median_score,
            "weighted_std_score": self.weighted_std_score,
            "min_frame_weight": self.min_frame_weight,
            "max_frame_weight": self.max_frame_weight,
            "mean_frame_weight": self.mean_frame_weight,
            "average_boundary_score": self.average_boundary_score,
            "average_frequency_anomaly": self.average_frequency_anomaly,
            "blink_naturalness_score": self.blink_naturalness_score,
            "identity_drift_score": self.identity_drift_score,
            "robustness_stability_score": self.robustness_stability_score,
            "signal_coverage": self.signal_coverage,
            "consistency": self.consistency,
            "frame_coverage": self.frame_coverage,
            "average_face_quality": self.average_face_quality,
            "reliability": self.reliability,
            "evidence_confidence": self.evidence_confidence,
            "sampled_frames": self.sampled_frames,
            "usable_frames": self.usable_frames,
            "suspicious_frames": [asdict(f) for f in self.suspicious_frames],
            "suspicious_segments": [asdict(s) for s in self.suspicious_segments],
            "explanations": self.explanations,
            "reason_codes": self.reason_codes,
        }


def compute_statistics(scores: np.ndarray) -> dict[str, float]:
    """Compute mean, median, max, std of frame scores."""
    if len(scores) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
        "std": float(np.std(scores, ddof=0)),  # population std
    }


def _valid_weighted_pairs(values: Any, weights: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite values paired with finite, strictly positive weights."""
    value_array = np.asarray(values, dtype=float).reshape(-1)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    count = min(value_array.size, weight_array.size)
    if count == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    value_array, weight_array = value_array[:count], weight_array[:count]
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    return value_array[valid], weight_array[valid]


def weighted_median(values: Any, weights: Any) -> float:
    """
    Compute the weighted median of values.

    The weighted median is the smallest value v such that the cumulative
    weight of all values <= v is at least half of the total weight.
    Falls back to the plain median if the total weight is zero.
    """
    values, weights = _valid_weighted_pairs(values, weights)
    if values.size == 0:
        return 0.0

    sorter = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[sorter]
    sorted_weights = np.asarray(weights, dtype=float)[sorter]

    total_weight = float(sorted_weights.sum())
    if total_weight <= 0:
        return float(np.median(sorted_values))

    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, total_weight / 2.0, side="left"))
    idx = min(idx, len(sorted_values) - 1)
    return float(sorted_values[idx])


def compute_weighted_statistics(scores: Any, weights: Any) -> dict[str, float]:
    """
    Compute quality-weighted mean, median, and std of frame scores.

    Weights are per-frame face_quality values in [0, 1]. Higher-quality
    frames contribute more to the aggregate statistics than blurry,
    low-confidence, or small-face frames.

    Uses population variance (ddof=0) to match compute_statistics.
    Falls back to uniform weights if the total weight is zero.
    """
    scores, weights = _valid_weighted_pairs(scores, weights)
    if scores.size == 0:
        return {
            "weighted_mean": 0.0,
            "weighted_median": 0.0,
            "weighted_std": 0.0,
            "min_weight": 0.0,
            "max_weight": 0.0,
            "mean_weight": 0.0,
        }

    total_weight = float(np.sum(weights, dtype=np.float64))

    weighted_mean = float(np.sum(scores * weights) / total_weight)
    w_median = weighted_median(scores, weights)
    weighted_variance = float(
        np.sum(weights * (scores - weighted_mean) ** 2) / total_weight
    )

    return {
        "weighted_mean": weighted_mean,
        "weighted_median": w_median,
        "weighted_std": float(np.sqrt(max(weighted_variance, 0.0))),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "mean_weight": float(np.mean(weights)),
    }


def compute_consistency(std_score: float, consistency_scale: float | None = None) -> float:
    """
    Convert standard deviation to consistency score [0, 1].

    Uses the quality-weighted standard deviation. ``consistency_scale`` is
    accepted only for source compatibility and is deliberately ignored.
    """
    if not np.isfinite(std_score):
        return 0.0
    consistency = 1.0 - (max(float(std_score), 0.0) / 0.5)
    return float(np.clip(consistency, 0.0, 1.0))


def compute_frame_coverage(usable_frames: int, sampled_frames: int) -> float:
    """Calculate frame coverage ratio."""
    if sampled_frames <= 0:
        return 0.0
    return float(np.clip(usable_frames / sampled_frames, 0.0, 1.0))


def compute_reliability(
    frame_coverage: float,
    average_face_quality: float,
    consistency: float,
    weights: dict[str, float],
) -> float:
    """
    Compute evidence reliability [0, 1].

    reliability = w1 * coverage + w2 * quality + w3 * consistency

    HEURISTIC: This is a prototype formula, not a statistically calibrated confidence.
    """
    reliability = (
        weights.get("coverage_weight", 0.35) * frame_coverage
        + weights.get("quality_weight", 0.35) * average_face_quality
        + weights.get("consistency_weight", 0.30) * consistency
    )
    return float(np.clip(reliability, 0.0, 1.0))


def compute_evidence_confidence(manipulation_score: float, reliability: float) -> float:
    """
    Compute evidence confidence [0, 1].

    evidence_confidence = manipulation_score * reliability

    HEURISTIC: This is a transparent prototype formula, NOT a true probability.
    """
    confidence = manipulation_score * reliability
    return float(np.clip(confidence, 0.0, 1.0))


def determine_verdict(
    evidence_confidence: float,
    reliability: float,
    thresholds: dict[str, float],
    usable_frames: int = 0,
    min_usable_frames: int = 5,
    manipulation_score: Optional[float] = None,
) -> str:
    """
    Determine final verdict based on manipulation score, evidence confidence and reliability.

    - REAL: manipulation_score < real_max (or evidence_confidence < real_max)
    - LIKELY_DEEPFAKE: manipulation_score >= deepfake_min AND reliability
      >= strong_reliability_min. Evidence confidence is a reliability-weighted
      supporting value, not a second, lower classification threshold.
    - INCONCLUSIVE: otherwise
    """
    # Minimum evidence gate
    if usable_frames < min_usable_frames:
        return "INCONCLUSIVE"

    real_max = thresholds.get("real_max", 0.35)
    deepfake_min = thresholds.get("deepfake_min", 0.65)
    strong_reliability_min = thresholds.get("strong_reliability_min", 0.50)

    if manipulation_score is not None:
        if manipulation_score >= deepfake_min and reliability >= strong_reliability_min:
            return "LIKELY_DEEPFAKE"
        elif manipulation_score < real_max and reliability >= 0.40:
            return "REAL"
        elif evidence_confidence < real_max:
            return "REAL"
        else:
            return "INCONCLUSIVE"

    if evidence_confidence < real_max:
        return "REAL"
    elif evidence_confidence >= 0.45 and reliability >= strong_reliability_min:
        return "LIKELY_DEEPFAKE"
    else:
        return "INCONCLUSIVE"


def find_suspicious_frames(
    frame_data: list[dict],
    threshold: float,
    top_k: int,
) -> list[SuspiciousFrame]:
    """Find frames with score >= threshold, return top-k by score."""
    suspicious = [
        SuspiciousFrame(
            frame_index=f["frame_index"],
            timestamp_seconds=f["timestamp_seconds"],
            score=f["score"],
            face_quality=f.get("face_quality", 0.0),
            weight=f.get("weight", f.get("face_quality", 0.0)),
            face_path=f.get("face_path", ""),
        )
        for f in frame_data
        if f.get("usable", False) and f.get("score", 0.0) >= threshold
    ]
    suspicious.sort(key=lambda f: f.score, reverse=True)
    return suspicious[:top_k]


def get_suspicious_frames_by_timestamp(
    frame_data: list[dict],
    threshold: float,
) -> list[SuspiciousFrame]:
    """Get all suspicious frames sorted by timestamp (for segment grouping)."""
    suspicious = [
        SuspiciousFrame(
            frame_index=f["frame_index"],
            timestamp_seconds=f["timestamp_seconds"],
            score=f["score"],
            face_quality=f.get("face_quality", 0.0),
            weight=f.get("weight", f.get("face_quality", 0.0)),
            face_path=f.get("face_path", ""),
        )
        for f in frame_data
        if f.get("usable", False) and f.get("score", 0.0) >= threshold
    ]
    suspicious.sort(key=lambda f: f.timestamp_seconds)
    return suspicious


def group_suspicious_segments(
    suspicious_frames: list[SuspiciousFrame],
    max_gap_seconds: float,
) -> list[SuspiciousSegment]:
    """Group nearby suspicious frames into temporal segments.

    Frames must be pre-sorted by timestamp.
    """
    if not suspicious_frames:
        return []

    segments = []
    current_segment = [suspicious_frames[0]]

    for frame in suspicious_frames[1:]:
        if frame.timestamp_seconds - current_segment[-1].timestamp_seconds <= max_gap_seconds:
            current_segment.append(frame)
        else:
            segments.append(create_segment(current_segment))
            current_segment = [frame]

    segments.append(create_segment(current_segment))
    return segments


def create_segment(frames: list[SuspiciousFrame]) -> SuspiciousSegment:
    """Create a segment from a list of frames (must be sorted by timestamp)."""
    scores = [f.score for f in frames]
    # Frames are already sorted by timestamp
    start = frames[0].timestamp_seconds
    end = frames[-1].timestamp_seconds
    return SuspiciousSegment(
        start=start,
        end=end,
        duration=end - start,
        frame_count=len(frames),
        peak_score=max(scores),
        mean_score=float(np.mean(scores)),
    )


def generate_explanations(
    result: ForensicResult,
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """
    Generate explanations grounded in actual evidence.
    Returns (explanations, reason_codes).
    Do NOT invent artifacts not detected by current models.
    """
    explanations = []
    reason_codes = []

    shift = result.weighted_median_score - result.raw_median_score
    explanations.append(
        f"Raw median manipulation score was {result.raw_median_score:.3f}; "
        f"the quality-weighted median was {result.weighted_median_score:.3f} "
        f"(shift {shift:+.3f}). Weighted mean was {result.weighted_mean_score:.3f}, "
        f"weighted standard deviation was {result.weighted_std_score:.3f}, and "
        f"consistency was {result.consistency:.3f}."
    )
    reason_codes.append("QUALITY_WEIGHTED_AGGREGATION")

    # High manipulation signal
    if result.manipulation_score >= config["forensic"]["thresholds"]["deepfake_min"]:
        explanations.append(
            "Multiple sampled face frames produced elevated manipulation scores."
        )
        reason_codes.append("HIGH_MANIPULATION_SIGNAL")

    # Consistency interpretation - state the actual weighted statistics
    if result.consistency >= 0.75:
        explanations.append(
            f"Scores clustered tightly across the usable frames "
            f"(weighted mean {result.weighted_mean_score:.2f}, weighted std {result.weighted_std_score:.2f}), "
            f"giving a consistency score of {result.consistency:.2f}."
        )
        reason_codes.append("HIGH_CONSISTENCY")
    elif result.consistency >= 0.50:
        explanations.append(
            f"Scores showed moderate spread across the usable frames "
            f"(weighted mean {result.weighted_mean_score:.2f}, weighted std {result.weighted_std_score:.2f}), "
            f"giving a consistency score of {result.consistency:.2f}."
        )
        reason_codes.append("MEDIUM_CONSISTENCY")
    else:
        explanations.append(
            f"Scores varied substantially across the video "
            f"(weighted mean {result.weighted_mean_score:.2f}, weighted std {result.weighted_std_score:.2f}), "
            f"giving a low consistency score of {result.consistency:.2f}."
        )
        reason_codes.append("LOW_CONSISTENCY")

    # Face coverage
    if result.frame_coverage >= 0.8:
        explanations.append(
            "A high proportion of sampled frames contained usable facial evidence."
        )
        reason_codes.append("HIGH_FACE_COVERAGE")
    elif result.frame_coverage < 0.5:
        explanations.append(
            "Evidence reliability was reduced because many sampled frames did not contain usable faces."
        )
        reason_codes.append("LOW_FACE_COVERAGE")

    # Face quality
    if result.average_face_quality < 0.5:
        explanations.append(
            "Evidence reliability was reduced because face quality was low."
        )
        reason_codes.append("LOW_FACE_QUALITY")
    elif result.average_face_quality >= 0.7:
        reason_codes.append("HIGH_FACE_QUALITY")

    for label, value, code in (
        ("Boundary artifact evidence", result.average_boundary_score, "BOUNDARY_ARTIFACT_EVIDENCE"),
        ("Frequency anomaly evidence", result.average_frequency_anomaly, "FREQUENCY_ANOMALY_EVIDENCE"),
        ("Blink naturalness", result.blink_naturalness_score, "BLINK_NATURALNESS_MEASURED"),
        ("Identity stability", result.identity_drift_score, "IDENTITY_STABILITY_MEASURED"),
        ("Robustness stability", result.robustness_stability_score, "ROBUSTNESS_STABILITY_MEASURED"),
    ):
        if value is not None:
            explanations.append(f"{label} was measured at {value:.3f}.")
            reason_codes.append(code)

    # Suspicious segment localization
    if result.verdict == "LIKELY_DEEPFAKE" and result.suspicious_segments:
        seg = result.suspicious_segments[0]
        explanations.append(
            f"The suspicious evidence is concentrated within a localized time segment "
            f"({seg.start:.1f}-{seg.end:.1f} seconds)."
        )
        reason_codes.append("LOCALIZED_EVIDENCE")

    # Minimum evidence gate
    if result.usable_frames < 5:
        explanations.append(
            "Insufficient usable face frames for a confident determination."
        )
        reason_codes.append("INSUFFICIENT_USABLE_FRAMES")

    # Reliability assessment
    if result.reliability >= 0.75:
        reason_codes.append("HIGH_RELIABILITY")
    elif result.reliability < 0.4:
        reason_codes.append("LOW_RELIABILITY")

    # Verdict-specific reasoning
    if result.verdict == "LIKELY_DEEPFAKE":
        reason_codes.append("EVIDENCE_CONFIDENCE_HIGH")
    elif result.verdict == "REAL":
        reason_codes.append("EVIDENCE_CONFIDENCE_LOW")

    if not explanations:
        explanations.append(
            "Insufficient evidence to reach a confident conclusion."
        )
        reason_codes.append("INSUFFICIENT_EVIDENCE")

    return explanations, reason_codes


def analyze_frame_predictions(
    frame_predictions: list[dict],
    video_id: str,
    config: dict[str, Any],
) -> ForensicResult:
    """
    Main forensic engine function.

    Args:
        frame_predictions: List of frame dicts with keys:
            - frame_index, timestamp_seconds, score, face_quality, usable
        video_id: Video identifier
        config: Configuration dict with forensic settings

    Returns:
        ForensicResult with all computed fields
    """
    forensic_config = config["forensic"]
    thresholds = forensic_config["thresholds"]
    reliability_weights = forensic_config["reliability"]

    usable_frames_data = [f for f in frame_predictions if f.get("usable", False)]
    usable_count = len(usable_frames_data)
    sampled_count = len(frame_predictions)

    if usable_count == 0:
        return ForensicResult(
            video_id=video_id,
            verdict="INCONCLUSIVE",
            manipulation_score=0.0,
            mean_score=0.0,
            median_score=0.0,
            max_score=0.0,
            std_score=0.0,
            raw_median_score=0.0,
            weighted_mean_score=0.0,
            weighted_median_score=0.0,
            weighted_std_score=0.0,
            min_frame_weight=0.0,
            max_frame_weight=0.0,
            mean_frame_weight=0.0,
            consistency=0.0,
            frame_coverage=0.0,
            average_face_quality=0.0,
            reliability=0.0,
            evidence_confidence=0.0,
            sampled_frames=sampled_count,
            usable_frames=0,
            suspicious_frames=[],
            suspicious_segments=[],
            explanations=["No usable face frames detected in the video."],
            reason_codes=["NO_USABLE_FRAMES"],
        )

    scores = np.array([f["score"] for f in usable_frames_data], dtype=float)
    face_qualities = np.array([f.get("weight", f.get("face_quality", 0.0)) for f in usable_frames_data], dtype=float)

    stats = compute_statistics(scores)
    weighted_stats = compute_weighted_statistics(scores, face_qualities)
    mean_score = stats["mean"]
    median_score = stats["median"]
    max_score = stats["max"]
    std_score = stats["std"]

    manipulation_score = weighted_stats["weighted_median"]

    consistency = compute_consistency(weighted_stats["weighted_std"])
    frame_coverage = compute_frame_coverage(usable_count, sampled_count)
    average_face_quality = float(np.mean(face_qualities))

    reliability = compute_reliability(
        frame_coverage, average_face_quality, consistency, reliability_weights
    )

    evidence_confidence = compute_evidence_confidence(manipulation_score, reliability)

    verdict = determine_verdict(
        evidence_confidence, reliability, thresholds,
        usable_frames=usable_count,
        min_usable_frames=forensic_config.get("min_usable_frames", 5),
        manipulation_score=manipulation_score,
    )

    # Get suspicious frames for display (top-k by score)
    suspicious_frames = find_suspicious_frames(
        usable_frames_data,
        forensic_config["suspicious_frame_threshold"],
        forensic_config["top_k_frames"],
    )

    # Get suspicious frames for segment grouping (sorted by timestamp)
    suspicious_frames_by_time = get_suspicious_frames_by_timestamp(
        usable_frames_data,
        forensic_config["suspicious_frame_threshold"],
    )

    suspicious_segments = group_suspicious_segments(
        suspicious_frames_by_time,
        forensic_config["max_gap_seconds"],
    )

    def aggregate_signal(name: str) -> tuple[Optional[float], int]:
        observations = [float(f[name]) for f in usable_frames_data if f.get(name) is not None and np.isfinite(f.get(name))]
        return (float(np.mean(observations)), len(observations)) if observations else (None, 0)

    boundary, boundary_coverage = aggregate_signal("boundary_score")
    frequency, frequency_coverage = aggregate_signal("frequency_anomaly")
    blink, blink_coverage = aggregate_signal("blink_naturalness")
    identity, identity_coverage = aggregate_signal("identity_drift")

    result = ForensicResult(
        video_id=video_id,
        verdict=verdict,
        manipulation_score=manipulation_score,
        mean_score=mean_score,
        median_score=median_score,
        max_score=max_score,
        std_score=std_score,
        raw_median_score=median_score,
        weighted_mean_score=weighted_stats["weighted_mean"],
        weighted_median_score=weighted_stats["weighted_median"],
        weighted_std_score=weighted_stats["weighted_std"],
        min_frame_weight=weighted_stats["min_weight"],
        max_frame_weight=weighted_stats["max_weight"],
        mean_frame_weight=weighted_stats["mean_weight"],
        consistency=consistency,
        frame_coverage=frame_coverage,
        average_face_quality=average_face_quality,
        reliability=reliability,
        evidence_confidence=evidence_confidence,
        sampled_frames=sampled_count,
        usable_frames=usable_count,
        suspicious_frames=suspicious_frames,
        suspicious_segments=suspicious_segments,
        explanations=[],
        reason_codes=[],
        average_boundary_score=boundary,
        average_frequency_anomaly=frequency,
        blink_naturalness_score=blink,
        identity_drift_score=identity,
        robustness_stability_score=None,
        signal_coverage={
            "boundary": boundary_coverage,
            "frequency": frequency_coverage,
            "blink": blink_coverage,
            "identity": identity_coverage,
        },
    )

    result.explanations, result.reason_codes = generate_explanations(result, config)

    return result


def save_forensic_result(result: ForensicResult, output_dir: Path) -> None:
    """Save forensic result to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{result.video_id}.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    csv_path = output_dir / "all_results.csv"
    row = {
        "video_id": result.video_id,
        "verdict": result.verdict,
        "manipulation_score": result.manipulation_score,
        "mean_score": result.mean_score,
        "median_score": result.median_score,
        "max_score": result.max_score,
        "std_score": result.std_score,
        "consistency": result.consistency,
        "frame_coverage": result.frame_coverage,
        "average_face_quality": result.average_face_quality,
        "reliability": result.reliability,
        "evidence_confidence": result.evidence_confidence,
        "sampled_frames": result.sampled_frames,
        "usable_frames": result.usable_frames,
    }

    import pandas as pd
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def create_timeline_plot(
    frame_predictions: list[dict],
    video_id: str,
    output_dir: Path,
    dpi: int = 150,
) -> None:
    """Create timeline plot of frame scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    usable = [f for f in frame_predictions if f.get("usable", False)]
    if not usable:
        return

    timestamps = [f["timestamp_seconds"] for f in usable]
    scores = [f["score"] for f in usable]

    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, scores, "b-", alpha=0.7, linewidth=1)
    plt.scatter(timestamps, scores, c=scores, cmap="Reds", s=20, alpha=0.8)
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Manipulation Score")
    plt.title(f"Frame-Level Manipulation Scores: {video_id}")
    plt.colorbar(label="Score")
    plt.tight_layout()
    plt.savefig(output_dir / f"{video_id}_timeline.png", dpi=dpi)
    plt.close()


def create_contact_sheet(
    frame_predictions: list[dict],
    video_id: str,
    output_dir: Path,
    cols: int = 3,
) -> None:
    """Create contact sheet of top suspicious frames."""
    try:
        import cv2
    except ImportError:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    suspicious = [f for f in frame_predictions if f.get("usable", False) and f.get("score", 0) >= 0.7]
    suspicious.sort(key=lambda f: f.get("score", 0), reverse=True)
    suspicious = suspicious[:9]  # max 9 frames

    if not suspicious:
        return

    frame_paths = [f.get("frame_path", "") for f in suspicious]
    valid_frames = [(fp, f) for fp, f in zip(frame_paths, suspicious) if fp and Path(fp).exists()]

    if not valid_frames:
        return

    images = []
    for fp, f in valid_frames:
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        text = f"{f['timestamp_seconds']:.1f}s | Score: {f['score']:.2f}"
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        images.append(img)

    if not images:
        return

    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    cv2.imwrite(str(output_dir / f"{video_id}_suspicious.jpg"), sheet)


def run_forensic_on_validation(
    validation_predictions_path: Path,
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Run forensic engine on validation predictions and summarize."""
    import pandas as pd

    df = pd.read_csv(validation_predictions_path)
    results = []

    for video_id, group in df.groupby("video_id"):
        frame_preds = group.to_dict("records")
        result = analyze_frame_predictions(frame_preds, video_id, config)
        results.append(result.to_dict())

    summary = {
        "num_videos": len(results),
        "verdicts": {
            "REAL": sum(1 for r in results if r["verdict"] == "REAL"),
            "INCONCLUSIVE": sum(1 for r in results if r["verdict"] == "INCONCLUSIVE"),
            "LIKELY_DEEPFAKE": sum(1 for r in results if r["verdict"] == "LIKELY_DEEPFAKE"),
        },
        "mean_reliability": float(np.mean([r["reliability"] for r in results])),
        "mean_consistency": float(np.mean([r["consistency"] for r in results])),
        "mean_manipulation_score": float(np.mean([r["manipulation_score"] for r in results])),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
