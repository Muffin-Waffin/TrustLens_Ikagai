const API_BASE = '/api';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface VideoMetadata {
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_seconds: number;
  codec: string;
}

export interface SuspiciousFrame {
  frame_index: number;
  timestamp_seconds: number;
  score: number;
  face_quality?: number;
  weight?: number;
}

export interface SuspiciousSegment {
  start: number;
  end: number;
  duration: number;
  frame_count: number;
  peak_score: number;
  mean_score: number;
}

export interface RobustnessTest {
  transform: string;
  score: number;
  difference: number;
  stability: number;
}

export interface RobustnessResults {
  original_score: number;
  tests: RobustnessTest[];
  overall_stability: number;
  interpretation: string;
}

export interface FrameInfo {
  frame_index: number;
  timestamp_seconds: number;
  score?: number;
  face_quality?: number;
  weight?: number;
  boundary_score?: number | null;
  frequency_anomaly?: number | null;
  blink_naturalness?: number | null;
  identity_similarity?: number | null;
  identity_drift?: number | null;
  usable: boolean;
  face_found: boolean;
}

export interface ForensicResult {
  video_id: string;
  verdict: 'REAL' | 'INCONCLUSIVE' | 'LIKELY_DEEPFAKE';
  manipulation_score: number;
  mean_score: number;
  median_score: number;
  max_score: number;
  std_score: number;
  raw_median_score: number;
  weighted_mean_score: number;
  weighted_median_score: number;
  weighted_std_score: number;
  min_frame_weight: number;
  max_frame_weight: number;
  mean_frame_weight: number;
  average_boundary_score?: number | null;
  average_frequency_anomaly?: number | null;
  blink_naturalness_score?: number | null;
  identity_drift_score?: number | null;
  robustness_stability_score?: number | null;
  consistency: number;
  frame_coverage: number;
  average_face_quality: number;
  reliability: number;
  evidence_confidence: number;
  sampled_frames: number;
  usable_frames: number;
  suspicious_frames: SuspiciousFrame[];
  suspicious_segments: SuspiciousSegment[];
  explanations: string[];
  reason_codes?: string[];
  gradcam_explanations: GradCAMExplanation[];
}

export interface GradCAMExplanation {
  frame_index: number;
  timestamp_seconds: number;
  score: number;
  heatmap_path: string | null;
  overlay_path: string | null;
  original_path: string | null;
}

export interface AnalysisResult {
  video_id: string;
  video_metadata: VideoMetadata;
  frame_infos: FrameInfo[];
  forensic_result: ForensicResult;
  robustness_results: RobustnessResults;
  timestamp: string;
  gradcam_explanations: GradCAMExplanation[];
}

// ─── API Functions ───────────────────────────────────────────────────────────

export interface AnalyzeOptions {
  runRobustness?: boolean;
  runExplainability?: boolean;
}

export type AnalysisStage = 'uploading' | 'preparing' | 'detecting' | 'verifying' | 'finalizing';

export interface AnalysisProgress {
  percent: number;
  stage: AnalysisStage;
  label: string;
}

const STAGE_RANKS: Record<AnalysisStage, number> = {
  uploading: 0,
  preparing: 1,
  detecting: 2,
  verifying: 3,
  finalizing: 4,
};

function getStageForPercent(percent: number): { stage: AnalysisStage; defaultLabel: string } {
  if (percent < 20) return { stage: 'uploading', defaultLabel: 'Uploading video securely' };
  if (percent < 40) return { stage: 'preparing', defaultLabel: 'Extracting & preparing frames' };
  if (percent < 70) return { stage: 'detecting', defaultLabel: 'Analyzing faces & synthetic signals' };
  if (percent < 90) return { stage: 'verifying', defaultLabel: 'Checking forensic evidence & consistency' };
  if (percent >= 100) return { stage: 'finalizing', defaultLabel: 'Analysis complete' };
  return { stage: 'finalizing', defaultLabel: 'Compiling forensic verdict' };
}

export async function analyzeVideo(
  file: File,
  onProgress?: (progress: AnalysisProgress) => void,
  options: AnalyzeOptions = {}
): Promise<AnalysisResult> {
  const params = new URLSearchParams();
  if (options.runRobustness !== undefined) params.set('run_robustness', String(options.runRobustness));
  if (options.runExplainability !== undefined) params.set('run_explainability', String(options.runExplainability));
  
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);

    xhr.open('POST', `${API_BASE}/analyze?${params.toString()}`);

    let progressTimer: number | undefined;
    let currentPercent = 0;
    let currentStage: AnalysisStage = 'uploading';
    let isUploadComplete = false;
    let hasStartedAnalysis = false;
    let isFinished = false;

    const reportProgress = (percent: number, explicitStage?: AnalysisStage, explicitLabel?: string) => {
      if (isFinished) return;

      const targetPercent = Math.max(0, Math.min(100, Math.round(percent)));
      // Strict monotonicity: percent and stage rank must NEVER decrease
      const nextPercent = Math.max(currentPercent, targetPercent);
      const stageInfo = getStageForPercent(nextPercent);
      const stageCandidate = explicitStage || stageInfo.stage;
      const stageRank = STAGE_RANKS[stageCandidate] ?? 0;
      const currentRank = STAGE_RANKS[currentStage] ?? 0;
      const nextStage = stageRank >= currentRank ? stageCandidate : currentStage;

      currentPercent = nextPercent;
      currentStage = nextStage;
      const label = explicitLabel || stageInfo.defaultLabel;

      onProgress?.({ percent: currentPercent, stage: currentStage, label });
    };

    const stopProgressTimer = () => {
      if (progressTimer !== undefined) {
        window.clearInterval(progressTimer);
        progressTimer = undefined;
      }
    };

    // Report initial state immediately
    reportProgress(0, 'uploading', 'Uploading video securely');

    // The synchronous API responds only after analysis completes. These stages
    // mirror the pipeline so users receive smooth, continuous feedback.
    const startAnalysisProgress = () => {
      if (hasStartedAnalysis || isFinished) return;
      hasStartedAnalysis = true;
      isUploadComplete = true;

      stopProgressTimer();
      reportProgress(20, 'preparing', 'Extracting & preparing frames');

      const isFast = options.runRobustness === false && options.runExplainability === false;
      const startedAt = Date.now();

      // Adaptive timeline: fast mode (~5s), full mode (~25s)
      const tPrep = isFast ? 1.5 : 3.0;
      const tDetect = isFast ? 5.0 : 16.0;
      const tVerify = isFast ? 8.0 : 26.0;

      progressTimer = window.setInterval(() => {
        if (isFinished) {
          stopProgressTimer();
          return;
        }

        const elapsedSeconds = (Date.now() - startedAt) / 1000;
        if (elapsedSeconds < tPrep) {
          const ratio = elapsedSeconds / tPrep;
          reportProgress(20 + ratio * 18, 'preparing', 'Extracting & preparing frames');
        } else if (elapsedSeconds < tDetect) {
          const ratio = (elapsedSeconds - tPrep) / (tDetect - tPrep);
          reportProgress(38 + ratio * 30, 'detecting', 'Analyzing faces & synthetic signals');
        } else if (elapsedSeconds < tVerify) {
          const ratio = (elapsedSeconds - tDetect) / (tVerify - tDetect);
          reportProgress(68 + ratio * 20, 'verifying', 'Checking forensic evidence & consistency');
        } else {
          // Asymptotically approach 94% while waiting for backend response
          const extraSeconds = elapsedSeconds - tVerify;
          const remaining = 94 - 88;
          const extraProgress = remaining * (1 - Math.exp(-extraSeconds / 15));
          reportProgress(88 + extraProgress, 'finalizing', 'Compiling forensic verdict');
        }
      }, 250);
    };

    xhr.upload.onprogress = (e) => {
      if (isUploadComplete || hasStartedAnalysis || isFinished) return;
      if (e.lengthComputable && onProgress) {
        const uploadPercent = Math.min(19, Math.round((e.loaded / e.total) * 19));
        reportProgress(uploadPercent, 'uploading', 'Uploading video securely');
      }
    };

    xhr.upload.onload = () => {
      startAnalysisProgress();
    };

    xhr.upload.onloadend = () => {
      startAnalysisProgress();
    };

    xhr.onload = () => {
      isFinished = true;
      stopProgressTimer();
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          reportProgress(100, 'finalizing', 'Analysis complete');
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error('Invalid response from server'));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.error || `Server error: ${xhr.status}`));
        } catch {
          reject(new Error(`Server error: ${xhr.status}`));
        }
      }
    };

    xhr.onerror = () => {
      isFinished = true;
      stopProgressTimer();
      reject(new Error('Network error — is the backend running?'));
    };

    xhr.ontimeout = () => {
      isFinished = true;
      stopProgressTimer();
      reject(new Error('Request timed out'));
    };

    xhr.onabort = () => {
      isFinished = true;
      stopProgressTimer();
      reject(new Error('Analysis aborted'));
    };

    xhr.timeout = 600000; // 10 minutes

    xhr.send(formData);
  });
}

export async function fetchDemoData(): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/demo`);
  if (!res.ok) throw new Error('Failed to fetch demo data');
  return res.json();
}

export async function fetchHistory(): Promise<AnalysisResult[]> {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
}

export async function checkHealth(): Promise<HealthStatus> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);
  
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data as HealthStatus;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Health check timeout - backend not responding');
    }
    throw new Error(`Backend unreachable: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
}

// ─── Chat API ───────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  context?: Record<string, unknown> | null;
  api_key?: string | null;
  model?: string | null;
  temperature?: number;
  max_tokens?: number;
}

export interface ChatResponse {
  message: ChatMessage;
  model: string;
  finish_reason?: string | null;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

export interface ChatModelOption {
  id: string;
  name: string;
  description: string;
  recommended: boolean;
}

export interface ChatStatusResponse {
  configured: boolean;
  default_model: string;
  available_models: ChatModelOption[];
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    let errorMsg = `Chat request failed (${res.status})`;
    try {
      const err = await res.json();
      errorMsg = err.detail || err.error || errorMsg;
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }

  return res.json();
}

export async function fetchChatStatus(): Promise<ChatStatusResponse> {
  const res = await fetch(`${API_BASE}/chat/status`);
  if (!res.ok) throw new Error('Failed to fetch chat status');
  return res.json();
}

export async function fetchChatModels(): Promise<ChatModelOption[]> {
  const res = await fetch(`${API_BASE}/chat/models`);
  if (!res.ok) throw new Error('Failed to fetch chat models');
  return res.json();
}
