"""
SynthGuard Phase 4: Streamlit Dashboard

Forensic dashboard for deepfake detection analysis.
The dashboard IS the forensic report.
"""

import os
import json
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from model import DeepfakeClassifier, XceptionDeepfakeClassifier, build_model, load_checkpoint
from preprocessing import (
    get_video_metadata,
    sample_frame_indices,
    initialize_face_detector,
    expand_bbox,
    compute_blur_score,
    compute_face_quality,
)
from inference import preprocess_video_for_inference, run_inference, build_analysis_result
from forensic_engine import analyze_frame_predictions
from explainability import generate_explanations_for_video, create_enhanced_timeline
from robustness import run_robustness_tests, save_robustness_report
from report import generate_html_report, save_json_report


# Page config
st.set_page_config(
    page_title="SynthGuard - Video Deepfake Forensics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .verdict-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .verdict-real { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .verdict-inconclusive { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .verdict-deepfake { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #1a1a2e; }
    .metric-label { font-size: 0.85rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1.5rem; }
    .reason-code { 
        display: inline-block; 
        background: #e9ecef; 
        padding: 0.25rem 0.75rem; 
        border-radius: 4px; 
        font-family: monospace; 
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    .signal-high { color: #dc3545; font-weight: bold; }
    .signal-medium { color: #fd7e14; font-weight: bold; }
    .signal-low { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config(config_path: str = "config.yaml"):
    """Load model and configuration."""
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(
        model_name=config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        num_classes=config["model"].get("num_classes", 2),
        device=device,
    )
    
    checkpoint_path = config.get("inference", {}).get("checkpoint", "models/xception_best.pth")
    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, device=device)
        model_loaded = True
    else:
        model_loaded = False
        st.warning(f"⚠️ Model checkpoint not found: {checkpoint_path}. Using random weights.")
    
    face_app = initialize_face_detector(config)
    
    return model, config, face_app, device, model_loaded


def process_video_pipeline(
    video_path: str,
    model: Union[DeepfakeClassifier, XceptionDeepfakeClassifier],
    config: dict,
    face_app,
    device: torch.device,
) -> dict:
    """Run full analysis pipeline on video."""
    # Phase 1 + 2: Preprocessing + Inference
    face_crops, frame_infos, video_metadata = preprocess_video_for_inference(
        video_path, config, face_app
    )
    
    scores = run_inference(face_crops, model, config, device)
    
    # Attach scores to frame infos
    score_idx = 0
    for fi in frame_infos:
        if fi.get("usable", False):
            fi["score"] = float(scores[score_idx])
            score_idx += 1
    
    # Build canonical analysis result
    return build_analysis_result(
        video_path, config, model, frame_infos, video_metadata, device
    )


def display_verdict_card(verdict: str):
    """Display verdict with styling."""
    verdict_class = {
        "REAL": "verdict-real",
        "INCONCLUSIVE": "verdict-inconclusive",
        "LIKELY_DEEPFAKE": "verdict-deepfake",
    }.get(verdict, "verdict-inconclusive")
    
    st.markdown(f"""
    <div class="verdict-card {verdict_class}">
        {verdict}
    </div>
    """, unsafe_allow_html=True)


def display_metrics_grid(metrics: dict):
    """Display metrics in a grid."""
    cols = st.columns(6)
    
    metric_items = [
        ("Manipulation Score", f"{metrics.get('manipulation_score', 0):.3f}"),
        ("Evidence Confidence", f"{metrics.get('evidence_confidence', 0):.3f}"),
        ("Reliability", f"{metrics.get('reliability', 0):.3f}"),
        ("Consistency", f"{metrics.get('consistency', 0):.3f}"),
        ("Face Coverage", f"{metrics.get('frame_coverage', 0):.1%}"),
        ("Avg Face Quality", f"{metrics.get('average_face_quality', 0):.3f}"),
    ]
    
    for col, (label, value) in zip(cols, metric_items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def display_signal_badge(label: str, value: float, high_thresh: float = 0.7, medium_thresh: float = 0.4):
    """Display a signal level badge."""
    if value >= high_thresh:
        cls = "signal-high"
        level = "HIGH"
    elif value >= medium_thresh:
        cls = "signal-medium"
        level = "MEDIUM"
    else:
        cls = "signal-low"
        level = "LOW"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {cls}">{level}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def plot_timeline(frame_infos: list, forensic_result: dict):
    """Create interactive Plotly timeline."""
    usable = [f for f in frame_infos if f.get("usable", False)]
    
    if not usable:
        st.info("No usable frames to display.")
        return
    
    timestamps = [f["timestamp_seconds"] for f in usable]
    scores = [f["score"] for f in usable]
    
    fig = go.Figure()
    
    # All scores
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=scores,
        mode='lines+markers',
        name='Frame Score',
        line=dict(color='royalblue', width=1),
        marker=dict(size=6, color=scores, colorscale='Reds', showscale=True, colorbar=dict(title="Score")),
        hovertemplate='Time: %{x:.1f}s<br>Score: %{y:.3f}<extra></extra>',
    ))
    
    # Threshold line
    threshold = 0.7
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Suspicious Threshold ({threshold})")
    
    # Mean line
    mean_score = np.mean(scores)
    fig.add_hline(y=mean_score, line_dash="dot", line_color="blue",
                  annotation_text=f"Mean ({mean_score:.3f})")
    
    # Shade suspicious segments
    for seg in forensic_result.get("suspicious_segments", []):
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor="red", opacity=0.15,
            layer="below", line_width=0,
        )
    
    fig.update_layout(
        title="Frame-Level Manipulation Scores",
        xaxis_title="Time (seconds)",
        yaxis_title="Manipulation Score",
        yaxis=dict(range=[0, 1.05]),
        height=450,
        hovermode="x unified",
        template="plotly_white",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.title("🔬 SynthGuard - Video Deepfake Forensics")
    st.markdown("*Phase 4 Prototype: Forensic Visualization + Explainability + Robustness*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        config_path = st.text_input("Config Path", "config.yaml")
        
        st.divider()
        
        st.header("📊 Model Info")
        if torch.cuda.is_available():
            st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            st.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            st.warning("⚠️ Running on CPU")
        
        st.divider()
        
        st.header("📁 Output Directories")
        st.code("outputs/explanations/")
        st.code("outputs/robustness/")
        st.code("outputs/reports/")
    
    # Load model
    with st.spinner("Loading model and detectors..."):
        model, config, face_app, device, model_loaded = load_model_and_config(config_path)
    
    if not model_loaded:
        st.warning("⚠️ Model checkpoint not found. Running with random weights (demo mode).")
    
    # Main content
    st.header("📤 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "webm", "mkv"],
        help="Upload a video to analyze for deepfake manipulation.",
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        
        # Display video info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(video_path)
        
        with col2:
            st.subheader("File Info")
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Size:** {len(uploaded_file.read()) / 1e6:.1f} MB")
            uploaded_file.seek(0)
        
        # Analyze button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Analyzing video... This may take a minute."):
                try:
                    results = process_video_pipeline(
                        video_path, model, config, face_app, device
                    )
                    
                    # Store in session state
                    st.session_state.results = results
                    st.session_state.video_path = video_path
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
        
        # Display results if available
        if "results" in st.session_state:
            results = st.session_state.results
            canonical = results["canonical"]
            forensic = results["forensic_result"]
            video_meta = results["video_metadata"]
            
            st.divider()
            
            # ============================================================
            # SECTION 1: CASE INFORMATION
            # ============================================================
            st.header("📋 Case Information")
            case = canonical["case"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Case ID:** {case['case_id']}")
                st.write(f"**File:** {case['filename']}")
            with col2:
                st.write(f"**SHA-256:** `{case['sha256'][:16]}...`" if case['sha256'] else "**SHA-256:** Computing...")
                st.write(f"**Size:** {case['size_bytes'] / 1e6:.1f} MB" if case['size_bytes'] else "**Size:** N/A")
            with col3:
                st.write(f"**Analyzed:** {case['analysis_timestamp'][:19].replace('T', ' ')}" if case['analysis_timestamp'] else "**Analyzed:** N/A")
            
            # ============================================================
            # SECTION 2: VIDEO METADATA
            # ============================================================
            st.header("🎬 Video Metadata")
            video = canonical["video"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if video['width'] and video['height']:
                    st.metric("Resolution", f"{video['width']}×{video['height']}")
                else:
                    st.metric("Resolution", "Unavailable")
            with col2:
                if video['fps']:
                    st.metric("Frame Rate", f"{video['fps']:.2f} FPS")
                else:
                    st.metric("Frame Rate", "Unavailable")
            with col3:
                if video['duration_seconds']:
                    st.metric("Duration", f"{video['duration_seconds']:.1f}s")
                else:
                    st.metric("Duration", "Unavailable")
            with col4:
                if video['frame_count']:
                    st.metric("Total Frames", f"{video['frame_count']:,}")
                else:
                    st.metric("Total Frames", "Unavailable")
            
            # ============================================================
            # SECTION 3: MODEL INFORMATION
            # ============================================================
            st.header("🤖 Model Information")
            model_info = canonical["model"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Architecture:** {model_info.get('architecture', 'Unknown')}")
            with col2:
                st.write(f"**Checkpoint:** {model_info.get('checkpoint', 'Unknown')}")
            with col3:
                st.write(f"**Device:** {model_info.get('device', 'Unknown')}")
            with col4:
                st.write(f"**Input Size:** {model_info.get('input_size', 224)}×{model_info.get('input_size', 224)}")
            
            # ============================================================
            # SECTION 4: VERDICT - MAIN FORENSIC DECISION
            # ============================================================
            st.header("🎯 Forensic Verdict")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                display_verdict_card(forensic.verdict)
            
            with col2:
                # Signal assessment
                st.subheader("Signal Assessment")
                sig_cols = st.columns(3)
                with sig_cols[0]:
                    display_signal_badge("Manipulation Signal", forensic.manipulation_score)
                with sig_cols[1]:
                    display_signal_badge("Evidence Reliability", forensic.reliability)
                with sig_cols[2]:
                    display_signal_badge("Evidence Confidence", forensic.evidence_confidence)
                
                st.write(f"**Reason:** {forensic.explanations[0] if forensic.explanations else 'N/A'}")
            
            # ============================================================
            # SECTION 5: EVIDENCE QUALITY
            # ============================================================
            st.header("📊 Evidence Quality")
            prep = canonical["preprocessing"]
            eq_cols = st.columns(4)
            with eq_cols[0]:
                st.metric("Face Coverage", f"{prep['face_coverage']:.1%}")
            with eq_cols[1]:
                st.metric("Avg Face Quality", f"{prep['average_face_quality']:.3f}")
            with eq_cols[2]:
                st.metric("Frame Consistency", f"{canonical['evidence']['consistency']:.3f}")
            with eq_cols[3]:
                st.metric("Usable Frames", f"{prep['usable_face_frames']}/{prep['sampled_frames']}")
            
            # ============================================================
            # SECTION 6: DETECTION SUMMARY
            # ============================================================
            st.header("🔍 Detection Summary")
            det = canonical["detection"]
            det_cols = st.columns(4)
            with det_cols[0]:
                st.metric("Mean Score", f"{det['mean_score']:.3f}")
            with det_cols[1]:
                st.metric("Median Score", f"{det['median_score']:.3f}")
            with det_cols[2]:
                st.metric("Max Score", f"{det['max_score']:.3f}")
            with det_cols[3]:
                st.metric("Std Dev", f"{det['std_score']:.3f}")
            
            # ============================================================
            # SECTION 7: REASON CODES
            # ============================================================
            if forensic.reason_codes:
                st.header("🏷️ Reason Codes")
                reason_html = " ".join([f'<span class="reason-code">{code}</span>' for code in forensic.reason_codes])
                st.markdown(reason_html, unsafe_allow_html=True)
            
            # ============================================================
            # TABS FOR DETAILED VIEWS
            # ============================================================
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Timeline", 
                "🎬 Suspicious Frames", 
                "🧠 Model Attention", 
                "🛡️ Robustness", 
                "📄 Export"
            ])
            
            with tab1:
                st.subheader("Manipulation Score Timeline")
                plot_timeline(results["frame_infos"], forensic.to_dict())
                
                # Suspicious segments table
                segments = forensic.suspicious_segments
                if segments:
                    st.subheader("Suspicious Segments")
                    seg_df = pd.DataFrame([{
                        "Start (s)": f"{s.start:.1f}",
                        "End (s)": f"{s.end:.1f}",
                        "Duration (s)": f"{s.duration:.1f}",
                        "Frames": s.frame_count,
                        "Peak Score": f"{s.peak_score:.3f}",
                        "Mean Score": f"{s.mean_score:.3f}",
                    } for s in segments])
                    st.dataframe(seg_df, use_container_width=True)
                else:
                    st.info("No suspicious segments detected.")
            
            with tab2:
                st.subheader("Top Suspicious Frames")
                suspicious = forensic.suspicious_frames
                
                if suspicious:
                    for i, frame in enumerate(suspicious[:6]):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            frame_path = frame.face_path if hasattr(frame, 'face_path') else frame.get("face_path", "")
                            if frame_path and Path(frame_path).exists():
                                st.image(frame_path, caption=f"Frame {frame.frame_index} @ {frame.timestamp_seconds:.1f}s")
                        with col2:
                            st.metric("Score", f"{frame.score:.3f}")
                            st.write(f"**Frame Index:** {frame.frame_index}")
                            st.write(f"**Timestamp:** {frame.timestamp_seconds:.1f}s")
                            st.write(f"**Face Quality:** {frame.face_quality:.3f}")
                else:
                    st.info("No frames above suspicious threshold.")
            
            with tab3:
                st.subheader("Grad-CAM Model Attribution")
                st.warning("""
                **Note:** Grad-CAM visualizations show which image regions influenced the model's prediction. 
                They are explanations of model behavior, **not** ground-truth evidence of manipulation artifacts.
                """)
                
                explanations = results.get("explanations", [])
                if explanations:
                    for exp in explanations:
                        col1, col2 = st.columns(2)
                        with col1:
                            if exp.get("original_path") and Path(exp["original_path"]).exists():
                                st.image(exp["original_path"], caption=f"Original - Frame {exp['frame_index']}")
                        with col2:
                            if exp.get("overlay_path") and Path(exp["overlay_path"]).exists():
                                st.image(exp["overlay_path"], caption=f"Grad-CAM Overlay - Frame {exp['frame_index']}")
                else:
                    st.info("No Grad-CAM explanations generated.")
            
            with tab4:
                st.subheader("Robustness Testing")
                
                robustness = results.get("robustness_results", {})
                if robustness and robustness.get("tests"):
                    st.write(f"**Original Score:** {robustness['original_score']:.4f}")
                    
                    rob_df = pd.DataFrame([{
                        "Transform": t["transform"],
                        "Score": f"{t.get('score', 0):.4f}",
                        "Difference": f"{t.get('difference', 0):.4f}",
                        "Stability": f"{t.get('stability', 0):.2f}",
                    } for t in robustness["tests"] if "stability" in t])
                    
                    if not rob_df.empty:
                        st.dataframe(rob_df, use_container_width=True)
                    
                    overall = robustness.get("overall_stability", 0)
                    st.metric("Overall Score Stability", f"{overall:.2f}")
                    st.write(robustness.get("interpretation", ""))
                    st.caption("This measures score stability under the tested transformations only, not general model robustness.")
                    
                    # Stability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=overall * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Stability Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 65], 'color': "lightcoral"},
                                {'range': [65, 85], 'color': "khaki"},
                                {'range': [85, 100], 'color': "lightgreen"},
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 85
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Robustness testing not performed or failed.")
            
            with tab5:
                st.subheader("Export Forensic Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📄 Generate HTML Report", use_container_width=True):
                        with st.spinner("Generating report..."):
                            report_dir = Path(config["paths"].get("reports", "./outputs/reports"))
                            report_path = report_dir / f"{results['video_id']}_report.html"
                            
                            html = generate_html_report(
                                st.session_state.video_path,
                                forensic.to_dict(),
                                config,
                                results.get("explanations"),
                                results.get("robustness_results"),
                                results.get("timeline_path"),
                                report_path,
                                model_info=results["model_info"],
                                frame_predictions=results.get("frame_infos"),
                            )
                            
                            # Also save JSON
                            json_path = report_dir / f"{results['video_id']}_report.json"
                            save_json_report(
                                st.session_state.video_path,
                                forensic.to_dict(),
                                results.get("explanations"),
                                results.get("robustness_results"),
                                json_path,
                                model_info=results["model_info"],
                                video_metadata=video_meta,
                            )
                            
                            st.success(f"✅ Report generated!")
                            st.write(f"HTML: `{report_path}`")
                            st.write(f"JSON: `{json_path}`")
                            
                            # Provide download
                            with open(report_path, "r") as f:
                                st.download_button(
                                    "📥 Download HTML Report",
                                    f.read(),
                                    file_name=f"{results['video_id']}_report.html",
                                    mime="text/html",
                                )
                
                with col2:
                    # Provide JSON download
                    canonical_json = json.dumps(canonical, indent=2)
                    st.download_button(
                        "📥 Download JSON Report",
                        canonical_json,
                        file_name=f"{results['video_id']}_forensic_report.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                
                # Show raw forensic JSON
                with st.expander("🔍 View Raw Forensic Result (JSON)"):
                    st.json(forensic.to_dict())
    
    else:
        # Welcome screen
        st.info("👆 Upload a video file to begin analysis.")
        
        st.divider()
        
        # Info about the system
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔍 Detection
            - Xception / ConvNeXt-Tiny backbone
            - ImageNet pretrained
            - 224×224 face crops
            - Frame-level scores
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Forensics
            - Median manipulation score
            - Consistency analysis
            - Evidence reliability
            - Segment grouping
            """)
        
        with col3:
            st.markdown("""
            ### 🛡️ Robustness
            - Resize stability
            - Blur resistance
            - Compression artifacts
            - Brightness invariance
            """)


if __name__ == "__main__":
    main()