import json
import logging
from typing import Any
import httpx
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse, ChatStatusResponse

logger = logging.getLogger(__name__)

# List of recommended and tested models on OpenRouter
AVAILABLE_MODELS = [
    {
        "id": "google/gemini-2.0-flash-001",
        "name": "Gemini 2.0 Flash",
        "description": "Fast, highly intelligent, and cost-effective (Recommended)",
        "recommended": True,
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B Instruct",
        "description": "Open-source state-of-the-art reasoning by Meta",
        "recommended": False,
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o Mini",
        "description": "Fast and lightweight model by OpenAI",
        "recommended": False,
    },
    {
        "id": "anthropic/claude-3.5-haiku",
        "name": "Claude 3.5 Haiku",
        "description": "Ultra-fast and precise responses by Anthropic",
        "recommended": False,
    },
    {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek V3",
        "description": "Powerful general-purpose reasoning model",
        "recommended": False,
    },
]

TRUSTLENS_SYSTEM_PROMPT = """You are Trustlens AI, an expert AI Forensic Investigator and technical assistant for the Trustlens Deepfake Video Forensics Platform.

### About Trustlens
Trustlens is an explainable, multi-signal video forensic analysis system engineered to detect deepfakes, AI-generated face swaps, and facial manipulation in digital video evidence.

### Core Architecture & Technical Capabilities
1. **Feature Extraction Neural Network**: Uses a fine-tuned ConvNeXt-Tiny backbone to extract spatial and texture anomaly representations from aligned facial crops.
2. **Face Detection & Alignment**: InsightFace / RetinaFace detector performing 5-point landmark affine registration, face quality scoring, and facial tracking.
3. **Multi-Signal Forensic Analyzers**:
   - **Boundary Artifact Analysis**: Evaluates edge blending gradients, seam color transitions, and mask boundaries around the face perimeter.
   - **Frequency Domain (FFT) Analysis**: Computes 2D Fast Fourier Transform and 1D azimuthal power spectrums to spot high-frequency GAN/diffusion grid synthesis anomalies.
   - **Temporal & Blink Dynamics**: Tracks Eye Aspect Ratio (EAR) over time, blink velocity, and inter-frame motion consistency to detect unnatural blinking or robotic dynamics.
   - **Identity Consistency (Drift)**: Measures ArcFace 512-dimensional embedding similarity across sampled video frames to catch face-swap identity drift.
   - **Grad-CAM Visual Explanations**: Generates Class Activation Maps revealing which facial regions (e.g. eyes, mouth, chin perimeter) triggered deepfake classifications.
   - **Robustness Perturbation Testing**: Evaluates prediction stability against 4 common video degradations (JPEG compression, Gaussian blur, downscaling, brightness shift).
4. **Forensic Decision Engine**:
   - Scores are normalized between 0.00 (Authentic/Real) and 1.00 (Manipulated/Deepfake).
   - Aggregation uses quality-weighted median/mean to prevent false positives from low-light, occlusion, or motion blur.
   - Verdict Categories:
     - `REAL` (Manipulation score < 0.40, high consistency & stability)
     - `INCONCLUSIVE` (Manipulation score 0.40 - 0.65, or low reliability/face coverage)
     - `LIKELY_DEEPFAKE` (Manipulation score >= 0.65 with corroborating forensic signals)

### Guidelines for Your Responses
- Be professional, objective, and analytical, acting like a Senior Digital Forensics Investigator.
- If report context is provided below, reference the exact metrics, timestamps, and findings from that report.
- When explaining technical terms (e.g., FFT frequency anomalies, EAR blink naturalness, Grad-CAM, ConvNeXt), make them clear and intuitive while preserving technical accuracy.
- If asked to generate reports or executive summaries, structure them with clear headers, key findings, timestamp breakdown, and forensic conclusions.
- Format responses nicely with markdown (bullet points, bold text, tables where applicable).
"""


class ChatService:
    def __init__(self) -> None:
        self.default_model = getattr(settings, "OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
        self.base_url = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    def get_api_key(self, custom_key: str | None = None) -> str:
        if custom_key and custom_key.strip():
            return custom_key.strip()
        env_key = getattr(settings, "OPENROUTER_API_KEY", "")
        if env_key and env_key.strip():
            return env_key.strip()
        return ""

    def get_status(self) -> ChatStatusResponse:
        configured = bool(self.get_api_key())
        return ChatStatusResponse(
            configured=configured,
            default_model=self.default_model,
            available_models=AVAILABLE_MODELS,
        )

    def _format_context_prompt(self, context: dict[str, Any]) -> str:
        """Format an AnalysisResult or report JSON into a structured prompt block."""
        try:
            video_id = context.get("video_id", "Unknown")
            metadata = context.get("video_metadata", {})
            forensic = context.get("forensic_result", {})
            robustness = context.get("robustness_results", {})
            timestamp = context.get("timestamp", "N/A")

            verdict = forensic.get("verdict", "N/A")
            manipulation_score = forensic.get("manipulation_score", "N/A")
            evidence_confidence = forensic.get("evidence_confidence", "N/A")
            reliability = forensic.get("reliability", "N/A")
            consistency = forensic.get("consistency", "N/A")
            usable_frames = forensic.get("usable_frames", "N/A")
            sampled_frames = forensic.get("sampled_frames", "N/A")

            # Metrics
            boundary_score = forensic.get("average_boundary_score")
            freq_score = forensic.get("average_frequency_anomaly")
            blink_score = forensic.get("blink_naturalness_score")
            identity_score = forensic.get("identity_drift_score")

            # Segments & Frames
            segments = forensic.get("suspicious_segments", [])
            suspicious_frames = forensic.get("suspicious_frames", [])
            explanations = forensic.get("explanations", [])

            context_str = f"""
### ACTIVE VIDEO FORENSIC REPORT CONTEXT
- **Video ID**: {video_id}
- **Analysis Timestamp**: {timestamp}
- **Metadata**: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')} @ {metadata.get('fps', 'N/A')} FPS, Duration: {metadata.get('duration_seconds', 'N/A')}s, Codec: {metadata.get('codec', 'N/A')}
- **Final Verdict**: {verdict}
- **Manipulation Score**: {manipulation_score} (Scale: 0.0 = Real, 1.0 = Deepfake)
- **Evidence Confidence**: {evidence_confidence if isinstance(evidence_confidence, str) else f"{float(evidence_confidence)*100:.1f}%"}
- **Reliability Index**: {reliability if isinstance(reliability, str) else f"{float(reliability)*100:.1f}%"}
- **Temporal Consistency**: {consistency if isinstance(consistency, str) else f"{float(consistency)*100:.1f}%"}
- **Sampled Frames**: {usable_frames} usable out of {sampled_frames} sampled

#### Forensic Signals Breakdown:
- **Boundary Anomaly Score**: {boundary_score if boundary_score is not None else 'N/A'} (Elevated > 0.5 indicates mask blending artifacts)
- **Frequency Domain (FFT) Anomaly**: {freq_score if freq_score is not None else 'N/A'} (Elevated > 0.5 indicates GAN/synthesis spectral artifacts)
- **Blink Naturalness**: {blink_score if blink_score is not None else 'N/A'} (Low < 0.4 indicates abnormal/missing blinking)
- **Identity Consistency**: {identity_score if identity_score is not None else 'N/A'} (Low < 0.5 indicates identity drift across frames)

#### Robustness Perturbation Stability:
- **Overall Stability Score**: {robustness.get('overall_stability', 'N/A')}
- **Interpretation**: {robustness.get('interpretation', 'N/A')}
"""
            if segments:
                context_str += "\n#### Detected Suspicious Segments:\n"
                for s in segments:
                    context_str += f"- Interval: {s.get('start', 0):.2f}s to {s.get('end', 0):.2f}s (Peak Score: {s.get('peak_score', s.get('mean_score', 'N/A'))})\n"

            if suspicious_frames:
                context_str += "\n#### Top Suspicious Frames:\n"
                for f in suspicious_frames[:5]:
                    context_str += f"- Frame {f.get('frame_index')}: Timestamp {f.get('timestamp_seconds', 0):.2f}s (Score: {f.get('score', 0):.3f})\n"

            if explanations:
                context_str += "\n#### Engine Explanations:\n"
                for exp in explanations:
                    context_str += f"- {exp}\n"

            return context_str
        except Exception as e:
            logger.warning(f"Error formatting context prompt: {e}")
            return f"\n### ACTIVE REPORT CONTEXT:\n```json\n{json.dumps(context, default=str)[:3000]}\n```\n"

    async def generate_response(self, request: ChatRequest) -> ChatResponse:
        api_key = self.get_api_key(request.api_key)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OpenRouter API Key is missing. Please provide it in the chat settings or configure OPENROUTER_API_KEY in the backend .env file.",
            )

        model = request.model or self.default_model

        # Build message history for OpenRouter
        full_system_prompt = TRUSTLENS_SYSTEM_PROMPT
        if request.context:
            context_block = self._format_context_prompt(request.context)
            full_system_prompt += f"\n\n{context_block}"

        api_messages = [{"role": "system", "content": full_system_prompt}]
        for msg in request.messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "Trustlens Video Forensics",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": api_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="OpenRouter request timed out. Please try again or choose a faster model.",
                )
            except Exception as e:
                logger.error(f"Error contacting OpenRouter: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to communicate with OpenRouter API: {str(e)}",
                )

        if response.status_code != 200:
            logger.error(f"OpenRouter API error ({response.status_code}): {response.text}")
            try:
                err_data = response.json()
                err_msg = err_data.get("error", {}).get("message", response.text)
            except Exception:
                err_msg = response.text
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenRouter API error: {err_msg}",
            )

        try:
            data = response.json()
            choice = data["choices"][0]
            reply_content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage")

            return ChatResponse(
                message=ChatMessage(role="assistant", content=reply_content),
                model=data.get("model", model),
                finish_reason=finish_reason,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Failed to parse OpenRouter response: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process OpenRouter response: {str(e)}",
            )
