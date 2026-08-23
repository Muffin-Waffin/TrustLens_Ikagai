import json
import logging
from datetime import datetime
from pathlib import Path
import sys

from fastapi import (
    BackgroundTasks,
    HTTPException,
    UploadFile,
)

from app.core.config import settings
from app.core.database import get_connection

from app.schemas.analysis import (
    AnalysisResult,
    AnalysisResponse,
    AnalysisStatusResponse,
    CaseStatus,
    FinalDecision,
    ForensicResult,
    FrameInfo,
    GradCAMExplanation,
    RawEvidence,
    RobustnessResults,
    RobustnessTest,
    SuspiciousFrame,
    SuspiciousSegment,
    Verdict,
    VideoMetadata,
)

from app.services.case_service import (
    new_case_id,
    sha256_file,
    utc_now,
)

from app.services.detection_service import (
    DetectionService,
)

from app.services.forensic_service import (
    ForensicService,
)

from app.services.report_service import (
    ReportService,
)

from app.services.storage_service import (
    StorageService,
)


logger = logging.getLogger(__name__)


HACKATHON_PATH = Path(__file__).parent.parent.parent.parent / "ikigai-hackathon"
sys.path.insert(0, str(HACKATHON_PATH))

from inference import analyze_video as run_full_analysis, build_analysis_result
from preprocessing import get_video_metadata
from explainability import generate_explanations_for_video, create_enhanced_timeline
from robustness import run_robustness_tests, save_robustness_report


class AnalysisService:
    """
    Main orchestration service for SynthGuard.

    Responsibilities:

        1. Accept uploaded videos
        2. Create investigation case
        3. Store original media
        4. Calculate SHA-256
        5. Run ML detection
        6. Run forensic decision engine
        7. Store results
        8. Generate evidence report
        9. Expose result to API
    """

    # Supported video formats for the MVP.
    ALLOWED_EXTENSIONS = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    }

    def __init__(self) -> None:

        self.storage = StorageService(
            upload_dir=settings.UPLOAD_DIR,
            processing_dir=settings.PROCESSING_DIR,
            evidence_dir=settings.EVIDENCE_DIR,
            report_dir=settings.REPORT_DIR,
        )

        self.detector = DetectionService()

        self.forensic = ForensicService()

        self.report_service = ReportService()

    # =========================================================
    # SYNCHRONOUS ANALYSIS (for frontend integration)
    # =========================================================

    async def analyze_video_sync(
        self,
        file: UploadFile,
        run_robustness: bool = True,
        run_explainability: bool = True,
    ) -> AnalysisResult:
        """
        Receive a video upload and run full analysis synchronously.
        Returns complete results matching frontend's AnalysisResult format.
        """
        # -----------------------------------------------------
        # 1. Validate filename
        # -----------------------------------------------------

        filename = Path(
            file.filename or "video.mp4"
        ).name

        extension = Path(
            filename
        ).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported video format. "
                    f"Allowed formats: "
                    f"{', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
                ),
            )

        # -----------------------------------------------------
        # 2. Create unique case ID
        # -----------------------------------------------------

        case_id = new_case_id()

        # -----------------------------------------------------
        # 3. Create case-specific upload directory
        # -----------------------------------------------------

        upload_dir = (
            self.storage.case_upload_dir(
                case_id
            )
        )

        video_path = (
            upload_dir / filename
        )

        # -----------------------------------------------------
        # 4. Save uploaded video
        # -----------------------------------------------------

        max_bytes = (
            settings.MAX_UPLOAD_MB
            * 1024
            * 1024
        )

        total_bytes = 0

        with video_path.open("wb") as output_file:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                # Prevent oversized uploads.
                if total_bytes > max_bytes:

                    video_path.unlink(
                        missing_ok=True
                    )

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Video exceeds the "
                            f"{settings.MAX_UPLOAD_MB} MB "
                            "upload limit."
                        ),
                    )

                output_file.write(chunk)

        # -----------------------------------------------------
        # 5. Generate case metadata
        # -----------------------------------------------------

        created_at = utc_now()

        file_hash = sha256_file(
            video_path
        )

        # -----------------------------------------------------
        # 6. Save case to database
        # -----------------------------------------------------

        with get_connection() as connection:

            connection.execute(
                """
                INSERT INTO cases (
                    case_id,
                    filename,
                    stored_path,
                    sha256,
                    status,
                    created_at,
                    updated_at,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    filename,
                    str(video_path),
                    file_hash,
                    CaseStatus.PROCESSING.value,
                    created_at.isoformat(),
                    created_at.isoformat(),
                    None,
                ),
            )

        # -----------------------------------------------------
        # 7. RUN FULL ANALYSIS PIPELINE SYNCHRONOUSLY
        # -----------------------------------------------------

        try:
            logger.info(
                "[%s] Running synchronous analysis pipeline.",
                case_id,
            )

            # Create config copy with robustness/explainability settings
            config = self.detector.config.copy()
            config["robustness"] = config.get("robustness", {}).copy()
            config["robustness"]["enabled"] = run_robustness
            config["explainability"] = config.get("explainability", {}).copy()
            config["explainability"]["enabled"] = run_explainability

            # Run full ML pipeline (preprocessing + inference + forensic + explainability + optional robustness)
            analysis_result = run_full_analysis(
                video_path=video_path,
                config=config,
                model=self.detector.model,
                face_app=self.detector.face_app,
                device=self.detector.device,
            )

            # Extract components from the analysis result
            canonical = analysis_result["canonical"]
            video_metadata = analysis_result["video_metadata"]
            frame_infos = analysis_result["frame_infos"]
            forensic_result = analysis_result["forensic_result"]
            explanations = analysis_result["explanations"]
            robustness_results = analysis_result["robustness_results"]

            # Convert forensic result to RawEvidence for the forensic service
            raw_evidence = self._forensic_result_to_raw_evidence(
                forensic_result, frame_infos, analysis_result
            )

            # Run our forensic decision engine
            final_decision = self.forensic.analyze(raw_evidence)

            # Build complete evidence payload
            evidence_payload = {

                "raw_evidence":
                    raw_evidence.model_dump(),

                "final_decision":
                    final_decision.model_dump(),

                "model_version":
                    self.detector.model_version,
            }

            # Convert to frontend format using the full analysis result
            result = self._build_frontend_result(
                case_id=case_id,
                video_path=video_path,
                video_metadata=video_metadata,
                frame_infos=frame_infos,
                forensic_result=forensic_result,
                explanations=explanations,
                robustness_results=robustness_results,
                final_decision=final_decision,
                raw_evidence=raw_evidence,
            )

            # Save to database
            timestamp = utc_now().isoformat()

            with get_connection() as connection:

                connection.execute(
                    """
                    INSERT OR REPLACE INTO results (
                        case_id,
                        verdict,
                        manipulation_score,
                        evidence_reliability,
                        evidence_consistency,
                        detector_agreement,
                        result_json,
                        created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        case_id,

                        final_decision.verdict.value,

                        final_decision.manipulation_score,

                        final_decision.evidence_reliability,

                        final_decision.evidence_consistency,

                        final_decision.detector_agreement,

                        json.dumps(
                            evidence_payload
                        ),

                        timestamp,
                    ),
                )

            # Save evidence JSON
            evidence_dir = (
                self.storage.case_evidence_dir(
                    case_id
                )
            )

            evidence_file = (
                evidence_dir
                / "evidence.json"
            )

            evidence_file.write_text(
                json.dumps(
                    evidence_payload,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            # Generate forensic report
            report_dir = (
                self.storage.case_report_dir(
                    case_id
                )
            )

            report_path = (
                report_dir
                / "report.json"
            )

            self.report_service.write_json_report(
                path=report_path,
                case={"case_id": case_id, "filename": filename, "stored_path": str(video_path), "sha256": file_hash},
                evidence=evidence_payload,
            )

            # Mark case completed
            self._set_status(
                case_id,
                CaseStatus.COMPLETED,
            )

            logger.info(
                "[%s] Synchronous analysis completed successfully.",
                case_id,
            )

            return result

        except Exception as exc:

            logger.exception(
                "[%s] Analysis failed.",
                case_id,
            )

            with get_connection() as connection:

                connection.execute(
                    """
                    UPDATE cases
                    SET
                        status = ?,
                        updated_at = ?,
                        error_message = ?
                    WHERE case_id = ?
                    """,
                    (
                        CaseStatus.FAILED.value,
                        utc_now().isoformat(),
                        str(exc),
                        case_id,
                    ),
                )

            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(exc)}",
            )

    def _forensic_result_to_raw_evidence(
        self,
        forensic_result,
        frame_infos: list[dict],
        analysis_result: dict,
    ) -> RawEvidence:
        """Convert forensic engine result to RawEvidence schema."""
        usable_frames = [f for f in frame_infos if f.get("usable", False)]
        face_qualities = [f.get("face_quality", 0.5) for f in usable_frames]
        avg_face_quality = sum(face_qualities) / len(face_qualities) if face_qualities else 0.5
        
        frame_scores = [
            {
                "frame_id": f["frame_index"],
                "timestamp": f["timestamp_seconds"],
                "score": f.get("score", 0.0),
                "face_quality": f.get("face_quality"),
                "weight": f.get("weight", f.get("face_quality")),
                "boundary_score": f.get("boundary_score"),
                "frequency_anomaly": f.get("frequency_anomaly"),
                "blink_naturalness": f.get("blink_naturalness"),
                "identity_similarity": f.get("identity_similarity"),
                "identity_drift": f.get("identity_drift"),
            }
            for f in usable_frames
        ]
        
        susp_segments_raw = getattr(forensic_result, "suspicious_segments", [])
        suspicious_segments = [
            {"start": s.start if hasattr(s, "start") else s.get("start", 0.0), "end": s.end if hasattr(s, "end") else s.get("end", 0.0)}
            for s in susp_segments_raw
        ]
        
        # Convert forensic_result to dict
        if hasattr(forensic_result, "to_dict"):
            forensic_dict = forensic_result.to_dict()
        elif isinstance(forensic_result, dict):
            forensic_dict = dict(forensic_result)
        else:
            forensic_dict = {}

        robustness_dict = analysis_result.get("robustness_results", {})
        if "overall_stability" in robustness_dict and forensic_dict.get("robustness_stability_score") is None:
            forensic_dict["robustness_stability_score"] = robustness_dict.get("overall_stability")
        
        # Extract Grad-CAM explanations from the canonical result
        gradcam_explanations = []
        canonical = analysis_result.get("canonical", {})
        explainability = canonical.get("explainability", {})
        attributions = explainability.get("attributions", [])
        for attr in attributions:
            if isinstance(attr, dict) and "frame_index" in attr:
                gradcam_explanations.append({
                    "frame_index": attr.get("frame_index", 0),
                    "timestamp_seconds": attr.get("timestamp_seconds", 0.0),
                    "score": attr.get("score", 0.0),
                    "heatmap_path": attr.get("heatmap_path"),
                    "overlay_path": attr.get("overlay_path"),
                    "original_path": attr.get("original_path"),
                })
        
        # Add Grad-CAM explanations to forensic_dict for frontend
        forensic_dict["gradcam_explanations"] = gradcam_explanations
        
        return RawEvidence(
            visual_score=forensic_dict.get("manipulation_score", 0.0),
            frequency_score=forensic_dict.get("mean_score", 0.0),
            boundary_score=forensic_dict.get("average_boundary_score") if forensic_dict.get("average_boundary_score") is not None else forensic_dict.get("manipulation_score", 0.0),
            temporal_score=1.0 - forensic_dict.get("consistency", 0.5),
            identity_score=forensic_dict.get("identity_drift_score") if forensic_dict.get("identity_drift_score") is not None else forensic_dict.get("reliability", 0.5),
            frame_scores=frame_scores,
            suspicious_segments=[SuspiciousSegment(start=s["start"], end=s["end"]) for s in suspicious_segments],
            usable_frames=forensic_dict.get("usable_frames", 0),
            face_quality=avg_face_quality,
            tracking_quality=forensic_dict.get("consistency", 0.5),
            video_quality=forensic_dict.get("frame_coverage", 0.5),
            pipeline_forensic_result=forensic_dict,
            pipeline_frame_infos=analysis_result.get("frame_infos", []),
            pipeline_video_metadata=analysis_result.get("video_metadata", {}),
            pipeline_robustness_results=robustness_dict,
        )

    def _build_frontend_result(
        self,
        case_id: str,
        video_path: Path,
        video_metadata: dict,
        frame_infos: list[dict],
        forensic_result,
        explanations: list[str],
        robustness_results: dict,
        final_decision: FinalDecision,
        raw_evidence: RawEvidence,
    ) -> AnalysisResult:
        """
        Build the complete AnalysisResult for the frontend using full pipeline results.
        """
        pipeline_forensic = raw_evidence.pipeline_forensic_result or (forensic_result.to_dict() if hasattr(forensic_result, "to_dict") else {})
        robustness = raw_evidence.pipeline_robustness_results or robustness_results or {}
        
        # Ensure robustness stability is reflected in forensic result
        if pipeline_forensic and pipeline_forensic.get("robustness_stability_score") is None:
            pipeline_forensic["robustness_stability_score"] = robustness.get("overall_stability")

        # Extract Grad-CAM explanations from pipeline result
        gradcam_explanations = []
        if "gradcam_explanations" in pipeline_forensic and isinstance(pipeline_forensic["gradcam_explanations"], list):
            for exp in pipeline_forensic["gradcam_explanations"]:
                if isinstance(exp, dict) and "frame_index" in exp:
                    gradcam_explanations.append(GradCAMExplanation(
                        frame_index=exp.get("frame_index", 0),
                        timestamp_seconds=exp.get("timestamp_seconds", 0.0),
                        score=exp.get("score", 0.0),
                        heatmap_path=exp.get("heatmap_path"),
                        overlay_path=exp.get("overlay_path"),
                        original_path=exp.get("original_path"),
                    ))
        
        # Map frame infos safely
        raw_frames = raw_evidence.pipeline_frame_infos or frame_infos or []
        frame_info_objects = [
            FrameInfo(
                frame_index=f.get("frame_index", 0),
                timestamp_seconds=f.get("timestamp_seconds", 0.0),
                score=f.get("score"),
                face_quality=f.get("face_quality"),
                weight=f.get("weight", f.get("face_quality")),
                boundary_score=f.get("boundary_score"),
                frequency_anomaly=f.get("frequency_anomaly"),
                blink_naturalness=f.get("blink_naturalness"),
                identity_similarity=f.get("identity_similarity"),
                identity_drift=f.get("identity_drift"),
                usable=f.get("usable", False),
                face_found=f.get("face_found", False),
            )
            for f in raw_frames
        ]

        # Convert suspicious frames
        susp_frames = [
            SuspiciousFrame(
                frame_index=s.get("frame_index", 0) if isinstance(s, dict) else s.frame_index,
                timestamp_seconds=s.get("timestamp_seconds", 0.0) if isinstance(s, dict) else s.timestamp_seconds,
                score=s.get("score", 0.0) if isinstance(s, dict) else s.score,
                face_quality=s.get("face_quality") if isinstance(s, dict) else getattr(s, "face_quality", None),
                weight=s.get("weight") if isinstance(s, dict) else getattr(s, "weight", None),
            )
            for s in (pipeline_forensic.get("suspicious_frames", []) or getattr(forensic_result, "suspicious_frames", []))
        ]

        # Convert suspicious segments
        susp_segs = [
            SuspiciousSegment(
                start=s.get("start", 0.0) if isinstance(s, dict) else s.start,
                end=s.get("end", 0.0) if isinstance(s, dict) else s.end,
            )
            for s in (pipeline_forensic.get("suspicious_segments", []) or getattr(forensic_result, "suspicious_segments", []))
        ]

        # Determine verdict prioritizing ML forensic engine
        verdict_str = pipeline_forensic.get("verdict") or getattr(forensic_result, "verdict", None) or final_decision.verdict.value
        try:
            verdict_val = Verdict(verdict_str)
        except Exception:
            verdict_val = Verdict.INCONCLUSIVE

        forensic_result_obj = ForensicResult(
            video_id=pipeline_forensic.get("video_id", case_id),
            verdict=verdict_val,
            manipulation_score=pipeline_forensic.get("manipulation_score", getattr(forensic_result, "manipulation_score", 0.0)),
            mean_score=pipeline_forensic.get("mean_score", getattr(forensic_result, "mean_score", 0.0)),
            median_score=pipeline_forensic.get("median_score", getattr(forensic_result, "median_score", 0.0)),
            max_score=pipeline_forensic.get("max_score", getattr(forensic_result, "max_score", 0.0)),
            std_score=pipeline_forensic.get("std_score", getattr(forensic_result, "std_score", 0.0)),
            raw_median_score=pipeline_forensic.get("raw_median_score", getattr(forensic_result, "raw_median_score", 0.0)),
            weighted_mean_score=pipeline_forensic.get("weighted_mean_score", getattr(forensic_result, "weighted_mean_score", 0.0)),
            weighted_median_score=pipeline_forensic.get("weighted_median_score", getattr(forensic_result, "weighted_median_score", 0.0)),
            weighted_std_score=pipeline_forensic.get("weighted_std_score", getattr(forensic_result, "weighted_std_score", 0.0)),
            min_frame_weight=pipeline_forensic.get("min_frame_weight", getattr(forensic_result, "min_frame_weight", 0.0)),
            max_frame_weight=pipeline_forensic.get("max_frame_weight", getattr(forensic_result, "max_frame_weight", 0.0)),
            mean_frame_weight=pipeline_forensic.get("mean_frame_weight", getattr(forensic_result, "mean_frame_weight", 0.0)),
            average_boundary_score=pipeline_forensic.get("average_boundary_score", getattr(forensic_result, "average_boundary_score", None)),
            average_frequency_anomaly=pipeline_forensic.get("average_frequency_anomaly", getattr(forensic_result, "average_frequency_anomaly", None)),
            blink_naturalness_score=pipeline_forensic.get("blink_naturalness_score", getattr(forensic_result, "blink_naturalness_score", None)),
            identity_drift_score=pipeline_forensic.get("identity_drift_score", getattr(forensic_result, "identity_drift_score", None)),
            robustness_stability_score=pipeline_forensic.get("robustness_stability_score", getattr(forensic_result, "robustness_stability_score", None)),
            reason_codes=pipeline_forensic.get("reason_codes", getattr(forensic_result, "reason_codes", [])),
            signal_coverage=pipeline_forensic.get("signal_coverage", getattr(forensic_result, "signal_coverage", {})),
            consistency=pipeline_forensic.get("consistency", getattr(forensic_result, "consistency", 0.0)),
            frame_coverage=pipeline_forensic.get("frame_coverage", getattr(forensic_result, "frame_coverage", 0.0)),
            average_face_quality=pipeline_forensic.get("average_face_quality", getattr(forensic_result, "average_face_quality", 0.0)),
            reliability=pipeline_forensic.get("reliability", getattr(forensic_result, "reliability", 0.0)),
            evidence_confidence=pipeline_forensic.get("evidence_confidence", getattr(forensic_result, "evidence_confidence", 0.0)),
            sampled_frames=pipeline_forensic.get("sampled_frames", getattr(forensic_result, "sampled_frames", 0)),
            usable_frames=pipeline_forensic.get("usable_frames", getattr(forensic_result, "usable_frames", 0)),
            suspicious_frames=susp_frames,
            suspicious_segments=susp_segs,
            explanations=pipeline_forensic.get("explanations", getattr(forensic_result, "explanations", [])),
            gradcam_explanations=gradcam_explanations,
        )

        # Build robustness results
        robustness_tests = [
            RobustnessTest(
                transform=test.get("transform", ""),
                score=test.get("score", 0.0),
                difference=test.get("difference", 0.0),
                stability=test.get("stability", 0.0),
            )
            for test in robustness.get("tests", [])
        ]

        robustness_results_obj = RobustnessResults(
            original_score=robustness.get("original_score", forensic_result_obj.manipulation_score),
            tests=robustness_tests,
            overall_stability=robustness.get("overall_stability", 0.85),
            interpretation=robustness.get("interpretation", "Robustness testing completed."),
        )

        raw_meta = raw_evidence.pipeline_video_metadata or video_metadata or {}
        video_meta = VideoMetadata(
            width=raw_meta.get("width", 1920),
            height=raw_meta.get("height", 1080),
            fps=raw_meta.get("fps", 30.0),
            frame_count=raw_meta.get("frame_count", 0),
            duration_seconds=raw_meta.get("duration_seconds", 0.0),
            codec=raw_meta.get("codec", "unknown"),
        )

        return AnalysisResult(
            video_id=case_id,
            video_metadata=video_meta,
            frame_infos=frame_info_objects,
            forensic_result=forensic_result_obj,
            robustness_results=robustness_results_obj,
            timestamp=datetime.now().isoformat(),
            gradcam_explanations=gradcam_explanations,
        )

    # =========================================================
    # BACKGROUND ANALYSIS PIPELINE (legacy async)
    # =========================================================

    def _run_analysis(
        self,
        case_id: str,
    ) -> None:
        """
        Run the actual analysis pipeline.

        This method is currently executed as a FastAPI background task.

        Later, if ML inference becomes too heavy, we can replace
        BackgroundTasks with Celery/RQ/a dedicated worker without
        changing the API contract.
        """

        try:

            # -------------------------------------------------
            # 1. Mark case as PROCESSING
            # -------------------------------------------------

            self._set_status(
                case_id,
                CaseStatus.PROCESSING,
            )

            logger.info(
                "[%s] Analysis started.",
                case_id,
            )

            # -------------------------------------------------
            # 2. Retrieve case metadata
            # -------------------------------------------------

            case = self._get_case_row(
                case_id
            )

            if case is None:

                raise RuntimeError(
                    f"Case {case_id} could not be found."
                )

            video_path = Path(
                case["stored_path"]
            )

            if not video_path.exists():

                raise RuntimeError(
                    f"Video file does not exist: "
                    f"{video_path}"
                )

            # -------------------------------------------------
            # 3. RUN ML DETECTOR
            # -------------------------------------------------

            logger.info(
                "[%s] Running detection pipeline.",
                case_id,
            )

            raw_evidence = (
                self.detector.analyze(
                    video_path
                )
            )

            logger.info(
                "[%s] Detection completed.",
                case_id,
            )

            # -------------------------------------------------
            # 4. RUN FORENSIC ENGINE
            # -------------------------------------------------

            logger.info(
                "[%s] Running forensic decision engine.",
                case_id,
            )

            final_decision = (
                self.forensic.analyze(
                    raw_evidence
                )
            )

            logger.info(
                "[%s] Verdict: %s",
                case_id,
                final_decision.verdict.value,
            )

            # -------------------------------------------------
            # 5. Build complete evidence payload
            # -------------------------------------------------

            evidence_payload = {

                "raw_evidence":
                    raw_evidence.model_dump(),

                "final_decision":
                    final_decision.model_dump(),

                "model_version":
                    self.detector.model_version,
            }

            # -------------------------------------------------
            # 6. Save result to database
            # -------------------------------------------------

            timestamp = utc_now().isoformat()

            with get_connection() as connection:

                connection.execute(
                    """
                    INSERT OR REPLACE INTO results (
                        case_id,
                        verdict,
                        manipulation_score,
                        evidence_reliability,
                        evidence_consistency,
                        detector_agreement,
                        result_json,
                        created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        case_id,

                        final_decision.verdict.value,

                        final_decision.manipulation_score,

                        final_decision.evidence_reliability,

                        final_decision.evidence_consistency,

                        final_decision.detector_agreement,

                        json.dumps(
                            evidence_payload
                        ),

                        timestamp,
                    ),
                )

            # -------------------------------------------------
            # 7. Save evidence JSON
            # -------------------------------------------------

            evidence_dir = (
                self.storage.case_evidence_dir(
                    case_id
                )
            )

            evidence_file = (
                evidence_dir
                / "evidence.json"
            )

            evidence_file.write_text(
                json.dumps(
                    evidence_payload,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            # -------------------------------------------------
            # 8. Generate forensic report
            # -------------------------------------------------

            report_dir = (
                self.storage.case_report_dir(
                    case_id
                )
            )

            report_path = (
                report_dir
                / "report.json"
            )

            self.report_service.write_json_report(
                path=report_path,
                case=dict(case),
                evidence=evidence_payload,
            )

            # -------------------------------------------------
            # 9. Mark case completed
            # -------------------------------------------------

            self._set_status(
                case_id,
                CaseStatus.COMPLETED,
            )

            logger.info(
                "[%s] Analysis completed successfully.",
                case_id,
            )

        except Exception as exc:

            # -------------------------------------------------
            # Error handling
            # -------------------------------------------------

            logger.exception(
                "[%s] Analysis failed.",
                case_id,
            )

            with get_connection() as connection:

                connection.execute(
                    """
                    UPDATE cases
                    SET
                        status = ?,
                        updated_at = ?,
                        error_message = ?
                    WHERE case_id = ?
                    """,
                    (
                        CaseStatus.FAILED.value,
                        utc_now().isoformat(),
                        str(exc),
                        case_id,
                    ),
                )

    # =========================================================
    # DATABASE HELPERS
    # =========================================================

    def _set_status(
        self,
        case_id: str,
        status: CaseStatus,
    ) -> None:
        """
        Update the processing status of a case.
        """

        with get_connection() as connection:

            connection.execute(
                """
                UPDATE cases
                SET
                    status = ?,
                    updated_at = ?
                WHERE case_id = ?
                """,
                (
                    status.value,
                    utc_now().isoformat(),
                    case_id,
                ),
            )

    def _get_case_row(
        self,
        case_id: str,
    ):
        """
        Retrieve one case from SQLite.
        """

        with get_connection() as connection:

            return connection.execute(
                """
                SELECT *
                FROM cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()

    # =========================================================
    # API READ METHODS
    # =========================================================

    def get_case(
        self,
        case_id: str,
    ) -> AnalysisStatusResponse | None:
        """
        Get current status/result of a case.
        """

        case = self._get_case_row(
            case_id
        )

        if case is None:
            return None

        result = None

        with get_connection() as connection:

            row = connection.execute(
                """
                SELECT result_json
                FROM results
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()

        if row is not None:

            payload = json.loads(
                row["result_json"]
            )

            result = FinalDecision.model_validate(
                payload["final_decision"]
            )

        return AnalysisStatusResponse(

            case_id=case["case_id"],

            filename=case["filename"],

            status=CaseStatus(
                case["status"]
            ),

            created_at=datetime.fromisoformat(
                case["created_at"]
            ),

            updated_at=datetime.fromisoformat(
                case["updated_at"]
            ),

            result=result,

            error_message=case[
                "error_message"
            ],
        )

    def get_evidence(
        self,
        case_id: str,
    ) -> dict | None:
        """
        Return the complete evidence JSON for a case.
        """

        evidence_path = (
            self.storage
            .case_evidence_dir(
                case_id
            )
            / "evidence.json"
        )

        if not evidence_path.exists():
            return None

        return json.loads(
            evidence_path.read_text(
                encoding="utf-8"
            )
        )

    def get_report_path(
        self,
        case_id: str,
    ) -> Path | None:
        """
        Return the path to a case's report.
        """

        report_path = (
            self.storage
            .case_report_dir(
                case_id
            )
            / "report.json"
        )

        if not report_path.exists():
            return None

        return report_path
