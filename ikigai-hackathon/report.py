"""
SynthGuard Phase 4: Report Generation Module

Generates HTML forensic reports from analysis results.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SynthGuard Forensic Report - {case_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
        .container {{ background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
        h2 {{ color: #16213e; border-left: 4px solid #e94560; padding-left: 15px; margin-top: 30px; }}
        h3 {{ color: #0f3460; }}
        .verdict {{ padding: 20px; border-radius: 8px; text-align: center; font-size: 28px; font-weight: bold; margin: 20px 0; }}
        .verdict.REAL {{ background: #d4edda; color: #155724; border: 2px solid #28a745; }}
        .verdict.INCONCLUSIVE {{ background: #fff3cd; color: #856404; border: 2px solid #ffc107; }}
        .verdict.LIKELY_DEEPFAKE {{ background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #1a1a2e; }}
        .metric-label {{ font-size: 14px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }}
        .section {{ margin: 25px 0; }}
        .evidence-list {{ list-style: none; padding: 0; }}
        .evidence-list li {{ padding: 10px 15px; background: #f8f9fa; margin: 8px 0; border-radius: 4px; border-left: 3px solid #e94560; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .image-card {{ background: #f8f9fa; border-radius: 8px; overflow: hidden; }}
        .image-card img {{ width: 100%; height: auto; display: block; }}
        .image-caption {{ padding: 10px; font-size: 13px; color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #1a1a2e; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
        .badge-high {{ background: #f8d7da; color: #721c24; }}
        .badge-medium {{ background: #fff3cd; color: #856404; }}
        .badge-low {{ background: #d4edda; color: #155724; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 14px; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 15px; margin: 20px 0; }}
        .unavailable {{ color: #6c757d; font-style: italic; }}
        .chart-row {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }}
        .chart-box {{ flex: 1 1 320px; background: #f8f9fa; border-radius: 8px; padding: 15px; }}
        .chart-box svg {{ width: 100%; height: auto; display: block; }}
        .chart-caption {{ font-size: 13px; color: #6c757d; margin-top: 8px; }}
        .delta-note {{ background: #f8f9fa; border-left: 4px solid #16213e; padding: 12px 15px; margin: 15px 0; }}
        .narrative p {{ font-size: 15px; }}
        code {{ background: #eef1f6; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SynthGuard Forensic Report</h1>
        
        <div class="section">
            <h2>1. Case Information</h2>
            <table>
                <tr><th>Case ID</th><td>{case_id}</td></tr>
                <tr><th>Analysis Date</th><td>{timestamp}</td></tr>
                <tr><th>Video File</th><td>{video_filename}</td></tr>
                <tr><th>SHA-256</th><td><code>{sha256}</code></td></tr>
                <tr><th>File Size</th><td>{file_size}</td></tr>
            </table>
        </div>
<div class="section">
            <h2>2. Video Metadata</h2>
            <table>
                <tr><th>Resolution</th><td>{width_display}</td></tr>
                <tr><th>Frame Rate</th><td>{fps_display}</td></tr>
                <tr><th>Duration</th><td>{duration_display}</td></tr>
                <tr><th>Total Frames</th><td>{frame_count_display}</td></tr>
                <tr><th>Codec</th><td>{codec}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>3. Detector Information</h2>
            <table>
                <tr><th>Model</th><td>{model_name}</td></tr>
                <tr><th>Architecture</th><td>{model_architecture}</td></tr>
                <tr><th>Input Size</th><td>{image_size}x{image_size}</td></tr>
                <tr><th>Device</th><td>{device}</td></tr>
                <tr><th>Checkpoint</th><td>{checkpoint}</td></tr>
                <tr><th>Parameters</th><td>{model_parameters:,}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>4. Detection Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{manipulation_score:.3f}</div>
                    <div class="metric-label">Manipulation Score (Median)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{evidence_confidence:.3f}</div>
                    <div class="metric-label">Evidence Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{reliability:.3f}</div>
                    <div class="metric-label">Evidence Reliability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{consistency:.3f}</div>
                    <div class="metric-label">Frame Consistency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{frame_coverage:.1%}</div>
                    <div class="metric-label">Face Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_quality:.3f}</div>
                    <div class="metric-label">Avg Face Quality</div>
                </div>
            </div>
            
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Mean Score</td><td>{mean_score:.3f}</td></tr>
                <tr><td>Median Score</td><td>{median_score:.3f}</td></tr>
                <tr><td>Max Score</td><td>{max_score:.3f}</td></tr>
                <tr><td>Standard Deviation</td><td>{std_score:.3f}</td></tr>
                <tr><td>Sampled Frames</td><td>{sampled_frames}</td></tr>
                <tr><td>Usable Face Frames</td><td>{usable_frames}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>5. Forensic Decision</h2>
            <div class="verdict {verdict}">{verdict}</div>
            
            <h3>Decision Logic</h3>
            <ul class="evidence-list">
                <li>Evidence Confidence: {evidence_confidence:.3f} (threshold for DEEPFAKE: >={deepfake_min:.2f})</li>
                <li>Evidence Reliability: {reliability:.3f} (minimum for strong verdict: >={strong_reliability_min:.2f})</li>
                <li>Real threshold: <={real_max:.2f}</li>
            </ul>
            
        </div>

        <div class="section">
            <h2>6. Findings Narrative</h2>
            <div class="narrative">
                {narrative_html}
            </div>
        </div>

        <div class="section">
            <h2>7. Raw vs Quality-Weighted Evidence</h2>
            <table>
                <tr><th>Statistic</th><th>Raw</th><th>Quality-weighted</th></tr>
                <tr><td>Mean</td><td>{mean_score:.3f}</td><td>{weighted_mean_score:.3f}</td></tr>
                <tr><td>Median</td><td>{raw_median_score:.3f}</td><td>{weighted_median_score:.3f}</td></tr>
                <tr><td>Standard deviation</td><td>{std_score:.3f}</td><td>{weighted_std_score:.3f}</td></tr>
                <tr><td>Weighted median shift</td><td colspan="2">{weighted_median_shift:+.3f}</td></tr>
            </table>
            <p>Frame weights: min {min_frame_weight:.3f}, mean {mean_frame_weight:.3f}, max {max_frame_weight:.3f}. The manipulation score reported above is the quality-weighted median.</p>
            <div class="delta-note">{delta_explanation_html}</div>
        </div>

        <div class="section">
            <h2>8. Frame Score Analysis</h2>
            <div class="chart-row">
                <div class="chart-box">
                    {frame_chart_html}
                    <div class="chart-caption">Per-frame manipulation scores over time. Point size and opacity encode face-quality weight: large opaque points are sharp, confidently detected faces; faint small points are noisy frames that barely influence the aggregate.</div>
                </div>
                <div class="chart-box">
                    {distribution_chart_html}
                    <div class="chart-caption">Distribution of usable-frame scores with weighted mean {weighted_mean_score:.2f} &plusmn; weighted std {weighted_std_score:.2f}. A narrow spread around the mean corresponds to high consistency ({consistency:.2f}).</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>9. Independent Forensic Signals</h2>
            <table><tr><th>Signal</th><th>Measured value</th></tr>{signal_rows}</table>
        </div>

        <div class="section">
            <h2>10. What Would Lower Our Confidence</h2>
            <ul class="evidence-list">{confidence_limitations}</ul>
        </div>

        <div class="section">
            <h2>11. Suspicious Timeline</h2>
            <p>Frame-level manipulation scores over time. Red shaded regions indicate suspicious intervals.</p>
            {timeline_image}
            
            <h3>Suspicious Segments</h3>
            {segments_table}
        </div>

        <div class="section">
            <h2>12. Top Suspicious Frames</h2>
            <p>Frames with highest manipulation scores.</p>
            <div class="image-grid">
                {suspicious_frames_html}
            </div>
        </div>

        <div class="section">
            <h2>13. Model Attribution (Grad-CAM)</h2>
            <p class="warning">
                <strong>Note:</strong> Grad-CAM visualizations show which image regions influenced the model's prediction.
                They are explanations of model behavior, <strong>not</strong> ground-truth evidence of manipulation artifacts.
            </p>
            <div class="image-grid">
                {explanations_html_grid}
            </div>
        </div>

        <div class="section">
            <h2>14. Robustness Testing</h2>
            <p>Stability of the manipulation score under common video transformations.</p>
            <table>
                <tr><th>Transformation</th><th>Score</th><th>Difference</th><th>Stability</th></tr>
                {robustness_table}
            </table>
            
            <h3>Overall Robustness Stability: <span class="badge {robustness_badge_class}">{overall_stability:.2f}</span></h3>
            <p>{robustness_interpretation}</p>
            <p class="warning"><strong>Note:</strong> This measures score stability under the tested transformations only, not general model robustness.</p>
        </div>

        <div class="section">
            <h2>15. Technical Details</h2>
            <ul class="evidence-list">
                <li>Analysis uses a single frame-level neural network detector ({model_architecture}).</li>
                <li>No explicit frequency-domain analysis was performed.</li>
                <li>No boundary artifact detection was performed.</li>
                <li>No temporal consistency modeling (TCN) was applied.</li>
                <li>No identity consistency verification was performed.</li>
                <li>Evidence confidence is a transparent heuristic, <strong>not</strong> a calibrated probability.</li>
                <li>Cross-dataset generalization has not been established.</li>
                <li>Robustness testing covers only a limited set of common transformations.</li>
            </ul>
        </div>

        <div class="footer">
            <p>Generated by SynthGuard Phase 4 Prototype</p>
            <p>Report ID: {case_id} | {timestamp}</p>
            <p><strong>Disclaimer:</strong> This is a research prototype. Results should not be used as definitive evidence in legal or security contexts without independent verification.</p>
        </div>
    </div>
</body>
</html>
"""


def generate_case_id(prefix: str = "SG") -> str:
    """Generate a case ID."""
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    return f"{prefix}-{date_str}-{time_str[-4:]}"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def encode_image_base64(image_path: str | Path) -> str:
    """Encode image as base64 for HTML embedding."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


ACCENT = "#e94560"
DARK = "#16213e"
MUTED = "#6c757d"
GRID = "#dee2e6"


def _np_finite(value: float) -> bool:
    """Local finiteness check avoiding a hard numpy dependency in report rendering."""
    return value == value and value not in (float("inf"), float("-inf"))


def _render_frame_scatter_svg(frames: list[dict]) -> str:
    """Render an inline SVG scatter of per-frame scores.

    Point size and opacity encode each frame's face-quality weight, so
    high-quality frames visually dominate noisy ones.
    """
    points: list[tuple[float, float, float]] = []
    for f in frames:
        t = f.get("timestamp_seconds")
        s = f.get("score")
        if t is None or s is None:
            continue
        w = f.get("weight", f.get("face_quality", 0.5))
        try:
            w = min(max(float(w), 0.0), 1.0)
        except (TypeError, ValueError):
            w = 0.5
        points.append((float(t), float(s), w))

    if not points:
        return ""

    width, height = 800, 260
    left, right, top, bottom = 52, 15, 12, 40

    times = [p[0] for p in points]
    t_min, t_max = min(times), max(times)
    t_span = (t_max - t_min) or 1.0

    def x_of(t: float) -> float:
        return left + (t - t_min) / t_span * (width - left - right)

    def y_of(s: float) -> float:
        s = min(max(s, 0.0), 1.0)
        return top + (1.0 - s) * (height - top - bottom)

    grid = "".join(
        f'<line x1="{left}" y1="{y_of(v):.1f}" x2="{width - right}" y2="{y_of(v):.1f}" '
        f'stroke="{GRID}" stroke-width="1"/>'
        f'<text x="{left - 8}" y="{y_of(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{v:.2f}</text>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )

    axes = (
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="{DARK}" stroke-width="1"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="{DARK}" stroke-width="1"/>'
    )

    dots = "".join(
        f'<circle cx="{x_of(t):.1f}" cy="{y_of(s):.1f}" r="{3 + 7 * w:.1f}" '
        f'fill="{ACCENT}" fill-opacity="{0.25 + 0.65 * w:.2f}"><title>{t:.1f}s score {s:.3f} weight {w:.2f}</title></circle>'
        for t, s, w in points
    )

    x_labels = "".join(
        f'<text x="{x_of(t_min + frac * t_span):.1f}" y="{height - bottom + 18}" '
        f'text-anchor="middle" font-size="11" fill="{MUTED}">{t_min + frac * t_span:.1f}s</text>'
        for frac in (0.0, 0.5, 1.0)
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Per-frame manipulation scores; point size reflects face-quality weight">'
        f"{grid}{axes}{dots}{x_labels}</svg>"
    )


def _render_histogram_svg(scores: list[float], mean: float, std: float) -> str:
    """Render an inline SVG histogram of frame scores with a mean +/- std overlay."""
    values = [float(s) for s in scores if s is not None]
    values = [v for v in values if _np_finite(v)]
    if not values:
        return ""

    width, height = 420, 260
    left, right, top, bottom = 45, 15, 28, 40

    n_bins = min(16, max(8, int(round(len(values) ** 0.5))))
    counts = [0] * n_bins
    for v in values:
        idx = min(int(v * n_bins), n_bins - 1)
        counts[max(idx, 0)] += 1
    max_count = max(counts) or 1

    def x_of(v: float) -> float:
        v = min(max(v, 0.0), 1.0)
        return left + v * (width - left - right)

    bin_width = (width - left - right) / n_bins
    bars = "".join(
        f'<rect x="{left + i * bin_width + 1:.1f}" y="{height - bottom - (c / max_count) * (height - top - bottom):.1f}" '
        f'width="{max(bin_width - 2, 1):.1f}" height="{(c / max_count) * (height - top - bottom):.1f}" '
        f'fill="{DARK}" fill-opacity="0.75"/>'
        for i, c in enumerate(counts)
        if c > 0
    )

    band_x1 = x_of(mean - std)
    band_x2 = x_of(mean + std)
    mean_x = x_of(mean)
    overlays = (
        f'<rect x="{band_x1:.1f}" y="{top}" width="{max(band_x2 - band_x1, 0):.1f}" '
        f'height="{height - top - bottom}" fill="{ACCENT}" fill-opacity="0.10"/>'
        f'<line x1="{mean_x:.1f}" y1="{top}" x2="{mean_x:.1f}" y2="{height - bottom}" '
        f'stroke="{ACCENT}" stroke-width="2" stroke-dasharray="5,4"/>'
        f'<text x="{left + 4}" y="{top + 14}" font-size="11" fill="{MUTED}">mean {mean:.2f} &plusmn; {std:.2f}</text>'
    )

    axes = (
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="{DARK}" stroke-width="1"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="{DARK}" stroke-width="1"/>'
    )
    x_labels = (
        f'<text x="{left}" y="{height - bottom + 18}" text-anchor="middle" font-size="11" fill="{MUTED}">0</text>'
        f'<text x="{(left + width - right) / 2:.1f}" y="{height - bottom + 18}" text-anchor="middle" font-size="11" fill="{MUTED}">score</text>'
        f'<text x="{width - right}" y="{height - bottom + 18}" text-anchor="middle" font-size="11" fill="{MUTED}">1</text>'
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11" fill="{MUTED}">{max_count}</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Distribution of per-frame manipulation scores">'
        f"{bars}{overlays}{axes}{x_labels}</svg>"
    )


def _build_narrative_html(
    verdict: str,
    explanations_list: list[str],
    reason_codes: list[str],
    manipulation_score: float,
    evidence_confidence: float,
) -> str:
    """Build a plain-English narrative paragraph from explanations and reason codes."""
    leads = {
        "REAL": "The evidence indicates this video is most likely authentic.",
        "LIKELY_DEEPFAKE": "The evidence indicates this video is likely a deepfake.",
        "INCONCLUSIVE": "The available evidence does not support a confident conclusion in either direction.",
    }
    lead = leads.get(verdict, "The analysis could not reach a definitive conclusion.")
    score_sentence = (
        f"The quality-weighted manipulation score is {manipulation_score:.2f}, "
        f"with overall evidence confidence of {evidence_confidence:.2f}."
    )
    body = " ".join(str(e).strip().rstrip(".") + "." for e in explanations_list)
    chips = " ".join(f"<code>{code}</code>" for code in reason_codes)
    codes_sentence = f" Structured flags recorded during analysis: {chips}." if chips else ""
    return f"<p>{lead} {score_sentence} {body}{codes_sentence}</p>"


def _build_delta_explanation_html(raw_median: float, weighted_median: float, mean_weight: float) -> str:
    """Explain the raw-vs-weighted median delta in plain language."""
    delta = weighted_median - raw_median
    if abs(delta) < 0.02:
        summary = f"is essentially unchanged by quality weighting (delta {delta:+.3f})"
        why = (
            "low-quality frames did not systematically carry higher or lower scores than "
            "high-quality ones in this video."
        )
    elif delta > 0:
        summary = f"is {delta:+.3f} higher than the raw median"
        why = (
            "once noisy frames were down-weighted, the remaining sharp, confidently detected "
            "faces point toward more manipulation than the raw pool of frames suggested."
        )
    else:
        summary = f"is {abs(delta):.3f} lower than the raw median"
        why = (
            "part of the loudest manipulation signal came from low-quality frames; after "
            "down-weighting them, the central estimate drops."
        )
    return (
        f"<strong>Why these numbers differ:</strong> the quality-weighted median {summary}. "
        f"In other words, {why} Weights derive from per-frame face quality "
        f"(detector confidence, face dominance, sharpness); this video averaged a frame "
        f"weight of {mean_weight:.2f} on a 0&ndash;1 scale."
    )


def generate_html_report(
    video_path: str | Path,
    forensic_result: dict[str, Any],
    config: dict[str, Any],
    explanations: list[dict] | None = None,
    robustness_results: dict[str, Any] | None = None,
    timeline_path: str | None = None,
    output_path: Path | None = None,
    model_info: dict[str, Any] | None = None,
    frame_predictions: list[dict] | None = None,
) -> str:
    """
    Generate HTML forensic report.
    """
    video_path = Path(video_path)
    
    case_id = generate_case_id(config.get("report", {}).get("case_id_prefix", "SG"))
    sha256 = hashlib.sha256(video_path.read_bytes()).hexdigest() if config.get("report", {}).get("include_sha256", True) else "N/A"
    file_size = format_file_size(video_path.stat().st_size)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    video_meta = forensic_result.get("video_metadata", {})
    
    # Validate metadata - never show zeros for valid video
    width = video_meta.get("width", 0)
    height = video_meta.get("height", 0)
    fps = video_meta.get("fps", 0.0)
    duration = video_meta.get("duration_seconds", 0.0)
    frame_count = video_meta.get("frame_count", 0)
    codec = video_meta.get("codec", "unknown")
    
    metadata_available = all([width > 0, height > 0, fps > 0, frame_count > 0])
    
    # Create display values that work with template
    if metadata_available:
        width_display = f"{width}x{height}"
        fps_display = f"{fps:.2f} FPS"
        duration_display = f"{duration:.2f} seconds"
        frame_count_display = f"{frame_count:,}"
    else:
        # Mark unavailable fields with proper display strings
        width_display = f"{width if width > 0 else 'unavailable'}x{height if height > 0 else 'unavailable'}"
        fps_display = f"{fps:.2f} FPS" if fps > 0 else "unavailable"
        duration_display = f"{duration:.2f} seconds" if duration > 0 else "unavailable"
        frame_count_display = f"{frame_count:,}" if frame_count > 0 else "unavailable"
    
    verdict = forensic_result.get("verdict", "UNKNOWN")
    manipulation_score = forensic_result.get("manipulation_score", 0)
    evidence_confidence = forensic_result.get("evidence_confidence", 0)
    reliability = forensic_result.get("reliability", 0)
    consistency = forensic_result.get("consistency", 0)
    frame_coverage = forensic_result.get("frame_coverage", 0)
    avg_quality = forensic_result.get("average_face_quality", 0)
    mean_score = forensic_result.get("mean_score", 0)
    median_score = forensic_result.get("median_score", 0)
    max_score = forensic_result.get("max_score", 0)
    std_score = forensic_result.get("std_score", 0)
    sampled_frames = forensic_result.get("sampled_frames", 0)
    usable_frames = forensic_result.get("usable_frames", 0)
    raw_median_score = forensic_result.get("raw_median_score", median_score)
    weighted_mean_score = forensic_result.get("weighted_mean_score", 0.0)
    weighted_median_score = forensic_result.get("weighted_median_score", manipulation_score)
    weighted_std_score = forensic_result.get("weighted_std_score", 0.0)
    min_frame_weight = forensic_result.get("min_frame_weight", 0.0)
    max_frame_weight = forensic_result.get("max_frame_weight", 0.0)
    mean_frame_weight = forensic_result.get("mean_frame_weight", 0.0)
    def metric_text(value: Any) -> str:
        return f"{float(value):.3f}" if value is not None else "N/A (not measured)"
    signal_rows = "".join(f"<tr><td>{name}</td><td>{metric_text(forensic_result.get(key))}</td></tr>" for name, key in (
        ("Boundary Artifact", "average_boundary_score"), ("Frequency Anomaly", "average_frequency_anomaly"),
        ("Blink Naturalness", "blink_naturalness_score"), ("Identity Stability", "identity_drift_score"),
        ("Robustness Stability", "robustness_stability_score")))
    thresholds = config["forensic"]["thresholds"]
    real_max = thresholds.get("real_max", 0.35)
    deepfake_min = thresholds.get("deepfake_min", 0.70)
    strong_reliability_min = thresholds.get("strong_reliability_min", 0.60)

    limitations = []
    min_usable = config["forensic"].get("min_usable_frames", 5)
    if usable_frames < min_usable:
        limitations.append(
            f"Only {usable_frames} usable face frames were found (minimum for a confident verdict is {min_usable}); the evidence base is thin."
        )
    if frame_coverage < 0.5:
        limitations.append(
            f"Face coverage was low ({frame_coverage:.0%} of sampled frames yielded a usable face); most of the video could not be analyzed."
        )
    elif frame_coverage < 0.8:
        limitations.append(
            f"Face coverage was partial ({frame_coverage:.0%}); some sampled frames contributed no facial evidence."
        )
    if avg_quality < 0.5:
        limitations.append(
            f"Average face quality was poor ({avg_quality:.2f} on a 0-1 scale); scores from blurry or small faces are less trustworthy."
        )
    elif avg_quality < 0.7:
        limitations.append(
            f"Average face quality was moderate ({avg_quality:.2f}); some frames were blurry, small, or uncertainly detected."
        )
    if reliability < 0.4:
        limitations.append(
            f"Evidence reliability was low ({reliability:.2f} on a 0-1 scale); the verdict rests on weak evidence and should be treated with caution."
        )
    elif reliability < strong_reliability_min:
        limitations.append(
            f"Evidence reliability ({reliability:.2f}) fell below the strong-verdict bar ({strong_reliability_min:.2f})."
        )
    coverage = forensic_result.get("signal_coverage", {})
    if coverage.get("blink", 0) == 0: limitations.append("Landmark coverage was insufficient for blink analysis.")
    if coverage.get("identity", 0) == 0: limitations.append("Identity embeddings were insufficient for identity analysis.")
    if forensic_result.get("robustness_stability_score") is not None and forensic_result["robustness_stability_score"] < config.get("robustness", {}).get("thresholds", {}).get("medium", 0.65): limitations.append("Robustness stability was limited under tested transforms.")
    confidence_limitations = "".join(f"<li>{item}</li>" for item in limitations) or "<li>No material measurement limitation was identified by the configured checks.</li>"
    
    explanations_list = forensic_result.get("explanations", [])
    explanations_html = "".join(f"<li>{exp}</li>" for exp in explanations_list) if explanations_list else "<li>No specific explanations generated.</li>"
    
    reason_codes = forensic_result.get("reason_codes", [])
    reason_codes_html = "".join(f"<li><code>{code}</code></li>" for code in reason_codes) if reason_codes else "<li>No reason codes generated.</li>"

    narrative_html = _build_narrative_html(
        str(verdict), explanations_list, reason_codes,
        float(manipulation_score), float(evidence_confidence),
    )
    delta_explanation_html = _build_delta_explanation_html(
        float(raw_median_score), float(weighted_median_score), float(mean_frame_weight),
    )

    timeline_image = ""
    if timeline_path and Path(timeline_path).exists():
        b64 = encode_image_base64(timeline_path)
        if b64:
            timeline_image = f'<img src="data:image/png;base64,{b64}" alt="Timeline" style="max-width:100%; border-radius:4px;">'
    
    segments = forensic_result.get("suspicious_segments", [])
    if segments:
        segments_table = "<table><tr><th>Start</th><th>End</th><th>Duration</th><th>Frames</th><th>Peak Score</th><th>Mean Score</th></tr>"
        for seg in segments:
            segments_table += f"<tr><td>{seg['start']:.1f}s</td><td>{seg['end']:.1f}s</td><td>{seg['duration']:.1f}s</td><td>{seg['frame_count']}</td><td>{seg['peak_score']:.3f}</td><td>{seg['mean_score']:.3f}</td></tr>"
        segments_table += "</table>"
    else:
        segments_table = "<p>No suspicious segments detected.</p>"
    
    suspicious_frames = forensic_result.get("suspicious_frames", [])

    chart_pool = [f for f in (frame_predictions or []) if f.get("usable", False)]
    if not chart_pool:
        chart_pool = [f for f in suspicious_frames if f.get("score") is not None]
    frame_chart_html = _render_frame_scatter_svg(chart_pool) or "<p class='unavailable'>No usable frames available for the per-frame chart.</p>"
    hist_scores = [f.get("score") for f in chart_pool if f.get("score") is not None]
    distribution_chart_html = _render_histogram_svg(hist_scores, float(weighted_mean_score), float(weighted_std_score)) or "<p class='unavailable'>No score distribution available.</p>"

    suspicious_frames_html = ""
    for frame in suspicious_frames[:6]:
        frame_path = frame.get("frame_path", "") or frame.get("face_path", "")
        if frame_path and Path(frame_path).exists():
            b64 = encode_image_base64(frame_path)
            if b64:
                suspicious_frames_html += f"""
                <div class="image-card">
                    <img src="data:image/jpeg;base64,{b64}" alt="Frame {frame['frame_index']}">
                    <div class="image-caption">Frame {frame['frame_index']} @ {frame['timestamp_seconds']:.1f}s | Score: {frame['score']:.3f}</div>
                </div>
                """
    if not suspicious_frames_html:
        suspicious_frames_html = "<p>No suspicious frames above threshold.</p>"
    
    explanations_html_grid = ""
    if explanations:
        for exp in explanations[:6]:
            overlay_path = exp.get("overlay_path", "")
            if overlay_path and Path(overlay_path).exists():
                b64 = encode_image_base64(overlay_path)
                if b64:
                    explanations_html_grid += f"""
                    <div class="image-card">
                        <img src="data:image/jpeg;base64,{b64}" alt="Explanation for frame {exp['frame_index']}">
                        <div class="image-caption">Frame {exp['frame_index']} @ {exp['timestamp_seconds']:.1f}s | Score: {exp['score']:.3f}</div>
                    </div>
                    """
    if not explanations_html_grid:
        explanations_html_grid = "<p>No Grad-CAM explanations generated.</p>"
    
    robustness_table = ""
    robustness_badge_class = "badge-low"
    overall_stability = 0.0
    robustness_interpretation = "Robustness testing not performed."
    
    if robustness_results and robustness_results.get("tests"):
        for test in robustness_results["tests"]:
            if "stability" in test:
                stability = test["stability"]
                badge = "badge-high" if stability >= 0.85 else "badge-medium" if stability >= 0.65 else "badge-low"
                robustness_table += f"<tr><td>{test['transform']}</td><td>{test.get('score', 0):.3f}</td><td>{test.get('difference', 0):.3f}</td><td><span class='badge {badge}'>{stability:.2f}</span></td></tr>"
        
        overall_stability = robustness_results.get("overall_stability", 0)
        if overall_stability >= 0.85:
            robustness_badge_class = "badge-high"
        elif overall_stability >= 0.65:
            robustness_badge_class = "badge-medium"
        else:
            robustness_badge_class = "badge-low"
        
        robustness_interpretation = robustness_results.get("interpretation", "")
    
    if not robustness_table:
        robustness_table = "<tr><td colspan='4'>No robustness tests performed.</td></tr>"
    
    # Dynamic model information
    model_name = config["model"]["name"]
    if model_info:
        model_architecture = model_info.get("architecture", model_name)
        model_parameters = model_info.get("parameters", 0)
        device = model_info.get("device", "CUDA" if __import__("torch").cuda.is_available() else "CPU")
        checkpoint = model_info.get("checkpoint", Path(config.get("inference", {}).get("checkpoint", "models/xception_best.pth")).name)
        image_size = model_info.get("input_size", config["model"]["image_size"])
    else:
        model_architecture = model_name
        model_parameters = 0
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        checkpoint = Path(config.get("inference", {}).get("checkpoint", "models/xception_best.pth")).name
        image_size = config["model"]["image_size"]
    
    html = HTML_TEMPLATE.format(
        case_id=case_id,
        timestamp=timestamp,
        video_filename=video_path.name,
        sha256=sha256,
        file_size=file_size,
        width_display=width_display,
        fps_display=fps_display,
        duration_display=duration_display,
        frame_count_display=frame_count_display,
        codec=codec,
        model_name=model_name,
        model_architecture=model_architecture,
        model_parameters=model_parameters,
        image_size=image_size,
        device=device,
        checkpoint=checkpoint,
        manipulation_score=manipulation_score,
        evidence_confidence=evidence_confidence,
        reliability=reliability,
        consistency=consistency,
        frame_coverage=frame_coverage,
        avg_quality=avg_quality,
        mean_score=mean_score,
        median_score=median_score,
        max_score=max_score,
        std_score=std_score,
        raw_median_score=raw_median_score,
        weighted_mean_score=weighted_mean_score,
        weighted_median_score=weighted_median_score,
        weighted_std_score=weighted_std_score,
        weighted_median_shift=weighted_median_score - raw_median_score,
        min_frame_weight=min_frame_weight,
        max_frame_weight=max_frame_weight,
        mean_frame_weight=mean_frame_weight,
        signal_rows=signal_rows,
        confidence_limitations=confidence_limitations,
        narrative_html=narrative_html,
        delta_explanation_html=delta_explanation_html,
        frame_chart_html=frame_chart_html,
        distribution_chart_html=distribution_chart_html,
        sampled_frames=sampled_frames,
        usable_frames=usable_frames,
        verdict=verdict,
        real_max=real_max,
        deepfake_min=deepfake_min,
        strong_reliability_min=strong_reliability_min,
        explanations_html=explanations_html,
        reason_codes_html=reason_codes_html,
        timeline_image=timeline_image,
        segments_table=segments_table,
        suspicious_frames_html=suspicious_frames_html,
        explanations_html_grid=explanations_html_grid,
        robustness_table=robustness_table,
        overall_stability=overall_stability,
        robustness_badge_class=robustness_badge_class,
        robustness_interpretation=robustness_interpretation,
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
        print(f"[INFO] HTML report saved: {output_path}")
    
    return html


def save_json_report(
    video_path: str | Path,
    forensic_result: dict[str, Any],
    explanations: list[dict] | None = None,
    robustness_results: dict[str, Any] | None = None,
    output_path: Path | None = None,
    model_info: dict[str, Any] | None = None,
    video_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save combined results as JSON report using canonical AnalysisResult structure."""
    video_path = Path(video_path)
    
    # Use provided video_metadata or fall back to forensic_result
    meta = video_metadata or forensic_result.get("video_metadata", {})
    
    report = {
        "case": {
            "case_id": generate_case_id(),
            "filename": video_path.name,
            "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "size_bytes": video_path.stat().st_size,
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "video": {
            "fps": meta.get("fps", 0),
            "frame_count": meta.get("frame_count", 0),
            "duration_seconds": meta.get("duration_seconds", 0),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            "codec": meta.get("codec", "unknown"),
        },
        "model": {
            "architecture": model_info.get("architecture", "unknown") if model_info else "unknown",
            "checkpoint": model_info.get("checkpoint", "unknown") if model_info else "unknown",
            "device": model_info.get("device", "unknown") if model_info else "unknown",
            "input_size": model_info.get("input_size", 224) if model_info else 224,
            "parameters": model_info.get("parameters", 0) if model_info else 0,
        } if model_info else {},
        "preprocessing": {
            "sampled_frames": forensic_result.get("sampled_frames", 0),
            "faces_detected": forensic_result.get("sampled_frames", 0),  # Approximation
            "usable_face_frames": forensic_result.get("usable_frames", 0),
            "face_coverage": forensic_result.get("frame_coverage", 0),
            "average_face_quality": forensic_result.get("average_face_quality", 0),
        },
        "detection": {
            "mean_score": forensic_result.get("mean_score", 0),
            "median_score": forensic_result.get("median_score", 0),
            "max_score": forensic_result.get("max_score", 0),
            "std_score": forensic_result.get("std_score", 0),
            "raw_median_score": forensic_result.get("raw_median_score", forensic_result.get("median_score", 0)),
            "weighted_mean_score": forensic_result.get("weighted_mean_score", 0),
            "weighted_median_score": forensic_result.get("weighted_median_score", 0),
            "weighted_std_score": forensic_result.get("weighted_std_score", 0),
            "min_frame_weight": forensic_result.get("min_frame_weight", 0),
            "max_frame_weight": forensic_result.get("max_frame_weight", 0),
            "mean_frame_weight": forensic_result.get("mean_frame_weight", 0),
            "frame_predictions": forensic_result.get("frame_predictions", []),
        },
        "evidence": {
            "consistency": forensic_result.get("consistency", 0),
            "reliability": forensic_result.get("reliability", 0),
            "confidence": forensic_result.get("evidence_confidence", 0),
        },
        "decision": {
            "verdict": forensic_result.get("verdict", "UNKNOWN"),
            "reason_codes": forensic_result.get("reason_codes", []),
        },
        "suspicious": {
            "frames": forensic_result.get("suspicious_frames", []),
            "segments": forensic_result.get("suspicious_segments", []),
        },
        "explainability": {
            "attributions": explanations or [],
        },
        "robustness": {
            "tests": robustness_results.get("tests", []) if robustness_results else [],
            "overall_stability": robustness_results.get("overall_stability", 0) if robustness_results else 0,
        },
        "limitations": [
            "Analysis uses a single frame-level neural network detector.",
            "No explicit frequency-domain analysis was performed.",
            "No boundary artifact detection was performed.",
            "No temporal consistency modeling (TCN) was applied.",
            "No identity consistency verification was performed.",
            "Evidence confidence is a transparent heuristic, not a calibrated probability.",
            "Cross-dataset generalization has not been established.",
            "Robustness testing covers only a limited set of common transformations.",
        ],
    }
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] JSON report saved: {output_path}")
    
    return report
