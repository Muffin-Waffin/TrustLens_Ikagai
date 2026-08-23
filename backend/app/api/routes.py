from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.schemas.analysis import (
    AnalysisResult,
    CaseStatus,
    FinalDecision,
    RawEvidence,
    SuspiciousSegment,
    Verdict,
)
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatStatusResponse,
)
from app.services.analysis_service import AnalysisService
from app.services.chat_service import ChatService, AVAILABLE_MODELS
from app.core.config import settings


router = APIRouter(
    tags=["analysis"]
)

# Services
analysis_service = AnalysisService()
chat_service = ChatService()


# ============================================================
# SYNCHRONOUS ANALYSIS (for frontend integration)
# ============================================================

@router.post(
    "/analyze",
    response_model=AnalysisResult,
    status_code=status.HTTP_200_OK,
)
async def analyze_video(
    file: UploadFile = File(...),
    run_robustness: bool = True,
    run_explainability: bool = True,
) -> AnalysisResult:
    """
    Upload a video and run full SynthGuard analysis synchronously.
    
    Returns complete analysis results including forensic verdict,
    frame-level scores, suspicious segments, and robustness results.
    """
    return await analysis_service.analyze_video_sync(file, run_robustness=run_robustness, run_explainability=run_explainability)


# ============================================================
# DEMO ENDPOINT
# ============================================================

@router.get(
    "/demo",
    response_model=AnalysisResult,
)
async def get_demo() -> AnalysisResult:
    """
    Return a demo analysis result for testing the frontend.
    """
    from datetime import datetime
    from app.schemas.analysis import (
        AnalysisResult,
        ForensicResult,
        FrameInfo,
        GradCAMExplanation,
        RobustnessResults,
        RobustnessTest,
        SuspiciousFrame,
        SuspiciousSegment,
        Verdict,
        VideoMetadata,
    )
    
    return AnalysisResult(
        video_id="demo-video",
        video_metadata=VideoMetadata(
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=900,
            duration_seconds=30.0,
            codec="h264",
        ),
        frame_infos=[
            FrameInfo(frame_index=i, timestamp_seconds=i/30.0, score=0.3 + (i % 10) * 0.05, face_quality=0.9, weight=0.72 + (i % 5) * 0.05, boundary_score=0.18 + (i % 4) * 0.03, frequency_anomaly=0.22 + (i % 3) * 0.04, blink_naturalness=0.76, identity_drift=0.92, usable=True, face_found=True)
            for i in range(0, 900, 30)
        ],
        forensic_result=ForensicResult(
            video_id="demo-video",
            verdict=Verdict.LIKELY_DEEPFAKE,
            manipulation_score=0.84,
            mean_score=0.78,
            median_score=0.82,
            max_score=0.96,
            std_score=0.12,
            raw_median_score=0.82,
            weighted_mean_score=0.85,
            weighted_median_score=0.84,
            weighted_std_score=0.09,
            min_frame_weight=0.72,
            max_frame_weight=0.95,
            mean_frame_weight=0.86,
            average_boundary_score=0.72,
            average_frequency_anomaly=0.68,
            blink_naturalness_score=0.34,
            identity_drift_score=0.48,
            robustness_stability_score=0.94,
            consistency=0.82,
            frame_coverage=0.92,
            average_face_quality=0.88,
            reliability=0.84,
            evidence_confidence=0.71,
            sampled_frames=30,
            usable_frames=28,
            suspicious_frames=[
                SuspiciousFrame(frame_index=240, timestamp_seconds=8.0, score=0.96),
                SuspiciousFrame(frame_index=270, timestamp_seconds=9.0, score=0.92),
            ],
            suspicious_segments=[
                SuspiciousSegment(start=7.5, end=9.5),
            ],
            explanations=[
                "Frame-level manipulation scores varied substantially across the video.",
                "Evidence reliability was reduced because many sampled frames did not contain usable faces.",
            ],
        ),
        robustness_results=RobustnessResults(
            original_score=0.45,
            tests=[
                RobustnessTest(transform="resize", score=0.44, difference=0.01, stability=0.95),
                RobustnessTest(transform="blur", score=0.46, difference=0.01, stability=0.93),
                RobustnessTest(transform="jpeg_compression", score=0.43, difference=0.02, stability=0.90),
                RobustnessTest(transform="brightness", score=0.45, difference=0.00, stability=0.98),
            ],
            overall_stability=0.94,
            interpretation="Model predictions are highly stable across common video transformations.",
        ),
        timestamp=datetime.now().isoformat(),
        gradcam_explanations=[],
    )


# ============================================================
# GET ANALYSIS STATUS (legacy async endpoint)
# ============================================================

@router.get(
    "/analyze/{case_id}",
)
def get_analysis(
    case_id: str,
) -> dict:
    """
    Get the current status and final result of a case.
    """
    result = analysis_service.get_case(case_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Case '{case_id}' not found.",
        )

    return result


# ============================================================
# GET RAW + FINAL EVIDENCE
# ============================================================

@router.get(
    "/evidence/{case_id}"
)
def get_evidence(
    case_id: str,
) -> dict:
    evidence = analysis_service.get_evidence(case_id)

    if evidence is None:
        raise HTTPException(
            status_code=404,
            detail=f"Evidence for case '{case_id}' was not found.",
        )

    return evidence


# ============================================================
# DOWNLOAD REPORT
# ============================================================

@router.get(
    "/report/{case_id}"
)
def get_report(
    case_id: str,
):
    report_path = analysis_service.get_report_path(case_id)

    if report_path is None or not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report for case '{case_id}' was not found.",
        )

    from fastapi.responses import FileResponse
    return FileResponse(
        path=report_path,
        media_type="application/json",
        filename=report_path.name,
    )


# ============================================================
# HISTORY
# ============================================================

@router.get(
    "/history"
)
def get_history(
    limit: int = 50,
) -> list[dict]:
    """
    Get analysis history.
    """
    return analysis_service.get_history(limit=limit)


# ============================================================
# EXPLANATION FILES (Grad-CAM heatmaps, overlays, originals)
# ============================================================

@router.get(
    "/files/explanations/{file_path:path}"
)
async def get_explanation_file(file_path: str):
    """
    Serve Grad-CAM explanation files (heatmaps, overlays, original crops).
    """
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    base_dir = Path(settings.DATA_DIR).parent / "outputs" / "explanations"
    full_path = base_dir / file_path
    
    # Security: ensure path is within the explanations directory
    try:
        full_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on extension
    ext = full_path.suffix.lower()
    media_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }.get(ext, 'application/octet-stream')
    
    return FileResponse(path=full_path, media_type=media_type)


# ============================================================
# API HEALTH
# ============================================================

@router.get(
    "/health"
)
def api_health() -> dict:
    import torch
    from app.services.detection_service import DetectionService
    detector = DetectionService()
    return {
        "status": "ok",
        "model_loaded": detector.model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# ============================================================
# CHATBOT (OpenRouter)
# ============================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat_completion(request: ChatRequest) -> ChatResponse:
    """
    Send messages to the Trustlens AI assistant powered by OpenRouter.
    Accepts conversation history, active report context, and optional custom API key/model.
    """
    return await chat_service.generate_response(request)


@router.get(
    "/chat/status",
    response_model=ChatStatusResponse,
)
def get_chat_status() -> ChatStatusResponse:
    """
    Check if an OpenRouter API key is configured on the backend
    and get available models.
    """
    return chat_service.get_status()


@router.get(
    "/chat/models",
)
def get_chat_models() -> list[dict]:
    """
    Get the list of recommended OpenRouter models for Trustlens.
    """
    return AVAILABLE_MODELS

