from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CaseStatus(str, Enum):
    """
    Status of a video analysis job.
    """

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Verdict(str, Enum):
    """
    Final forensic classification.
    """

    REAL = "REAL"
    LIKELY_DEEPFAKE = "LIKELY_DEEPFAKE"
    INCONCLUSIVE = "INCONCLUSIVE"


class SuspiciousSegment(BaseModel):
    """
    A time interval in the video that contains
    suspicious activity.
    """

    start: float = Field(
        ...,
        ge=0.0,
        description="Start timestamp in seconds.",
    )

    end: float = Field(
        ...,
        ge=0.0,
        description="End timestamp in seconds.",
    )


class SuspiciousFrame(BaseModel):
    frame_index: int
    timestamp_seconds: float
    score: float
    face_quality: float | None = None
    weight: float | None = None


class RobustnessTest(BaseModel):
    transform: str
    score: float
    difference: float
    stability: float


class RobustnessResults(BaseModel):
    original_score: float
    tests: list[RobustnessTest]
    overall_stability: float
    interpretation: str


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str


class FrameInfo(BaseModel):
    frame_index: int
    timestamp_seconds: float
    score: float | None = None
    face_quality: float | None = None
    weight: float | None = None
    boundary_score: float | None = None
    frequency_anomaly: float | None = None
    blink_naturalness: float | None = None
    identity_similarity: float | None = None
    identity_drift: float | None = None
    usable: bool
    face_found: bool


class GradCAMExplanation(BaseModel):
    frame_index: int
    timestamp_seconds: float
    score: float
    heatmap_path: str | None = None
    overlay_path: str | None = None
    original_path: str | None = None


class ForensicResult(BaseModel):
    video_id: str
    verdict: Verdict
    manipulation_score: float
    mean_score: float
    median_score: float
    max_score: float
    std_score: float
    raw_median_score: float = 0.0
    weighted_mean_score: float = 0.0
    weighted_median_score: float = 0.0
    weighted_std_score: float = 0.0
    min_frame_weight: float = 0.0
    max_frame_weight: float = 0.0
    mean_frame_weight: float = 0.0
    average_boundary_score: float | None = None
    average_frequency_anomaly: float | None = None
    blink_naturalness_score: float | None = None
    identity_drift_score: float | None = None
    robustness_stability_score: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    signal_coverage: dict[str, int] = Field(default_factory=dict)
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
    gradcam_explanations: list[GradCAMExplanation] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """
    Complete analysis result returned by synchronous /api/analyze endpoint.
    Matches the frontend's expected response format.
    """
    video_id: str
    video_metadata: VideoMetadata
    frame_infos: list[FrameInfo]
    forensic_result: ForensicResult
    robustness_results: RobustnessResults
    timestamp: str
    gradcam_explanations: list[GradCAMExplanation] = Field(default_factory=list)


class RawEvidence(BaseModel):
    """
    Output produced by the ML detection layer.

    Person 3's ML pipeline will eventually return
    this structure.

    All scores are normalized between 0 and 1.
    """

    # ---------------------------------------------------------
    # Individual forensic evidence scores
    # ---------------------------------------------------------

    visual_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Visual facial anomaly score.",
    )

    frequency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fine-detail/frequency anomaly score.",
    )

    boundary_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Face-boundary anomaly score.",
    )

    temporal_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Temporal inconsistency score.",
    )

    identity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Face identity instability score.",
    )

    # ---------------------------------------------------------
    # Frame-level information
    # ---------------------------------------------------------

    frame_scores: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Frame-level suspiciousness scores. "
            "Each item can contain frame_id, timestamp, score, "
            "and later additional evidence."
        ),
    )

    suspicious_segments: list[SuspiciousSegment] = Field(
        default_factory=list,
        description="Video intervals identified as suspicious.",
    )

    # ---------------------------------------------------------
    # Quality information
    # ---------------------------------------------------------

    usable_frames: int = Field(
        ...,
        ge=0,
        description="Number of frames suitable for forensic analysis.",
    )

    face_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality of detected/aligned faces.",
    )

    tracking_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality of tracking the same face over time.",
    )

    video_quality: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Overall quality of the source video.",
    )
    # Canonical pipeline payload retained so synchronous API construction does
    # not have to recreate or reinterpret forensic metrics.
    pipeline_forensic_result: dict[str, Any] = Field(default_factory=dict)
    pipeline_frame_infos: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_video_metadata: dict[str, Any] = Field(default_factory=dict)
    pipeline_robustness_results: dict[str, Any] = Field(default_factory=dict)


class FinalDecision(BaseModel):
    """
    Output produced by the forensic decision engine.

    Person 4's layer takes RawEvidence and generates
    this structure.
    """

    verdict: Verdict

    manipulation_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall manipulation score.",
    )

    evidence_reliability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How trustworthy the available evidence is.",
    )

    evidence_consistency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Consistency of evidence across frames.",
    )

    detector_agreement: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agreement between independent evidence sources.",
    )

    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons for the decision.",
    )

    suspicious_segments: list[SuspiciousSegment] = Field(
        default_factory=list,
        description="Video intervals supporting the decision.",
    )

    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional forensic metrics.",
    )


class AnalysisResponse(BaseModel):
    """
    Response returned immediately after a video is uploaded.
    """

    case_id: str

    status: CaseStatus

    message: str


class AnalysisStatusResponse(BaseModel):
    """
    Response returned when the frontend asks about
    the status/result of an analysis.
    """

    case_id: str

    filename: str

    status: CaseStatus

    created_at: datetime

    updated_at: datetime

    result: FinalDecision | None = None

    error_message: str | None = None
