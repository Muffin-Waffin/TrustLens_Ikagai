import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import type { AnalysisResult, ForensicResult } from '../services/api';

interface ReportExportProps {
  result: AnalysisResult | null;
}

const REPORT_SECTIONS = [
  { id: 'overview', label: 'Overview', icon: '📋', required: true },
  { id: 'verdict', label: 'Verdict & Scores', icon: '⚖️', required: true },
  { id: 'metrics', label: 'Detailed Metrics', icon: '📊', required: false },
  { id: 'frames', label: 'Frame Analysis', icon: '🎞️', required: false },
  { id: 'segments', label: 'Suspicious Segments', icon: '🚨', required: false },
  { id: 'explanations', label: 'Forensic Explanations', icon: '🔬', required: false },
  { id: 'gradcam', label: 'Grad-CAM Visualizations', icon: '🎯', required: false },
  { id: 'robustness', label: 'Robustness Testing', icon: '🛡️', required: false },
  { id: 'technical', label: 'Technical Appendix', icon: '⚙️', required: false },
];

export default function ReportExport({ result }: ReportExportProps) {
  const [selectedSections, setSelectedSections] = useState<Set<string>>(
    new Set(REPORT_SECTIONS.filter(s => s.required).map(s => s.id))
  );
  const [format, setFormat] = useState<'html' | 'json' | 'pdf'>('html');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedReport, setGeneratedReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggleSection = useCallback((id: string) => {
    setSelectedSections(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!result) return;
    setIsGenerating(true);
    setError(null);
    
    try {
      const report = generateReportContent(result, Array.from(selectedSections));
      
      if (format === 'html') {
        const blob = new Blob([report], { type: 'text/html' });
        downloadBlob(blob, `Trustlens_Report_${result.video_id}.html`);
      } else if (format === 'json') {
        const jsonReport = generateJSONReport(result, Array.from(selectedSections));
        const blob = new Blob([jsonReport], { type: 'application/json' });
        downloadBlob(blob, `Trustlens_Report_${result.video_id}.json`);
      } else if (format === 'pdf') {
        // For PDF, we'll generate HTML and let browser print to PDF
        const printWindow = window.open('', '_blank');
        if (printWindow) {
          printWindow.document.write(report);
          printWindow.document.close();
          printWindow.focus();
          setTimeout(() => printWindow.print(), 500);
        }
      }
      
      setGeneratedReport(report);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate report');
    } finally {
      setIsGenerating(false);
    }
  }, [result, selectedSections, format]);

  const handleDownload = useCallback(() => {
    if (!generatedReport) return;
    const blob = new Blob([generatedReport], { type: format === 'json' ? 'application/json' : 'text/html' });
    downloadBlob(blob, `Trustlens_Report_${result?.video_id}.${format}`);
  }, [generatedReport, format, result?.video_id]);

  if (!result) {
    return (
      <div style={styles.card}>
        <div style={styles.emptyState}>
          <div style={styles.emptyIcon}>📄</div>
          <h3 style={styles.emptyTitle}>No Analysis Results</h3>
          <p style={styles.emptyText}>
            Run a video analysis first to generate a forensic report.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.titleSection}>
          <h3 style={styles.title}>Generate Forensic Report</h3>
          <p style={styles.subtitle}>
            Create a comprehensive forensic analysis report for case documentation and evidence sharing.
          </p>
        </div>
        <div style={styles.badge}>Report Generator</div>
      </div>

      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Report Sections</h4>
        <p style={styles.sectionDesc}>Select sections to include in the report. Required sections cannot be removed.</p>
        <div style={styles.sectionsGrid}>
          {REPORT_SECTIONS.map(section => (
            <label key={section.id} style={styles.sectionItem}>
              <input
                type="checkbox"
                checked={selectedSections.has(section.id)}
                onChange={() => toggleSection(section.id)}
                disabled={section.required}
                style={styles.checkbox}
              />
              <div style={styles.sectionInfo}>
                <span style={styles.sectionIcon}>{section.icon}</span>
                <div>
                  <div style={styles.sectionLabel}>{section.label}</div>
                  {section.required && <span style={styles.requiredTag}>Required</span>}
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Export Format</h4>
        <div style={styles.formatSelector}>
          {(['html', 'json', 'pdf'] as const).map(fmt => (
            <button
              key={fmt}
              onClick={() => setFormat(fmt)}
              style={{
                ...styles.formatBtn,
                ...(fmt === format ? styles.formatBtnActive : {}),
              }}
            >
              <span style={styles.formatIcon}>{fmt === 'html' ? '🌐' : fmt === 'json' ? '📋' : '📄'}</span>
              <span style={styles.formatLabel}>{fmt.toUpperCase()}</span>
              <span style={styles.formatDesc}>
                {fmt === 'html' && 'Interactive web report'}
                {fmt === 'json' && 'Machine-readable data'}
                {fmt === 'pdf' && 'Print-ready document'}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div style={styles.summary}>
        <div style={styles.summaryLabel}>Report Summary</div>
        <div style={styles.summaryGrid}>
          <div style={styles.summaryItem}>
            <span style={styles.summaryKey}>Video ID</span>
            <span style={styles.summaryValue}>{result.video_id}</span>
          </div>
          <div style={styles.summaryItem}>
            <span style={styles.summaryKey}>Verdict</span>
            <span style={{ ...styles.summaryValue, ...styles[`verdict${result.forensic_result.verdict}`] }}>
              {result.forensic_result.verdict}
            </span>
          </div>
          <div style={styles.summaryItem}>
            <span style={styles.summaryKey}>Manipulation Score</span>
            <span style={styles.summaryValue}>{result.forensic_result.manipulation_score.toFixed(3)}</span>
          </div>
          <div style={styles.summaryItem}>
            <span style={styles.summaryKey}>Sections</span>
            <span style={styles.summaryValue}>{selectedSections.size} / {REPORT_SECTIONS.length}</span>
          </div>
          <div style={styles.summaryItem}>
            <span style={styles.summaryKey}>Format</span>
            <span style={styles.summaryValue}>{format.toUpperCase()}</span>
          </div>
          <div style={styles.summaryItem}>
            <span style={styles.summaryKey}>Generated</span>
            <span style={styles.summaryValue}>{new Date().toLocaleString()}</span>
          </div>
        </div>
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          style={styles.error}
        >
          {error}
        </motion.div>
      )}

      <div style={styles.actions}>
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          style={{
            ...styles.generateBtn,
            ...(isGenerating ? styles.generateBtnDisabled : {}),
          }}
        >
          {isGenerating ? (
            <>
              <span style={styles.spinner} />
              Generating...
            </>
          ) : (
            'Generate Report'
          )}
        </button>
        {generatedReport && (
          <button
            onClick={handleDownload}
            style={styles.downloadBtn}
          >
            Download {format.toUpperCase()}
          </button>
        )}
      </div>
    </div>
  );
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function generateReportContent(result: AnalysisResult, sections: string[]): string {
  const { video_id, video_metadata, frame_infos, forensic_result, robustness_results, timestamp, gradcam_explanations } = result;
  const date = new Date(timestamp).toLocaleString();
  
  const sectionsHtml = sections.map(sectionId => {
    switch (sectionId) {
      case 'overview':
        return generateOverviewSection(video_id, video_metadata, forensic_result, date);
      case 'verdict':
        return generateVerdictSection(forensic_result);
      case 'metrics':
        return generateMetricsSection(forensic_result);
      case 'frames':
        return generateFramesSection(frame_infos, forensic_result);
      case 'segments':
        return generateSegmentsSection(forensic_result);
      case 'explanations':
        return generateExplanationsSection(forensic_result);
      case 'gradcam':
        return generateGradCAMSection(gradcam_explanations);
      case 'robustness':
        return generateRobustnessSection(robustness_results);
      case 'technical':
        return generateTechnicalSection(frame_infos);
      default:
        return '';
    }
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trustlens Forensic Report - ${video_id}</title>
  <style>
    ${getReportStyles()}
  </style>
</head>
<body>
  <div class="report-container">
    <header class="report-header">
      <div class="header-content">
        <div class="logo">🛡️ Trustlens</div>
        <div class="header-text">
          <h1>Forensic Analysis Report</h1>
          <div class="meta">Video ID: ${video_id} | Generated: ${date}</div>
        </div>
        <div class="verdict-badge verdict-${forensic_result.verdict.toLowerCase().replace('_', '-')}">
          ${forensic_result.verdict}
        </div>
      </div>
    </header>
    
    <main class="report-main">
      ${sectionsHtml}
    </main>
    
    <footer class="report-footer">
      <p>Generated by Trustlens Video Deepfake Forensics Platform</p>
      <p class="disclaimer">This is a research prototype. Results should not be used as definitive evidence without independent verification.</p>
    </footer>
  </div>
</body>
</html>`;
}

function generateOverviewSection(video_id: string, video_metadata: any, forensic_result: ForensicResult, date: string): string {
  return `
    <section class="report-section">
      <h2>1. Case Overview</h2>
      <table class="info-table">
        <tr><th>Video ID</th><td>${video_id}</td></tr>
        <tr><th>Resolution</th><td>${video_metadata.width} × ${video_metadata.height}</td></tr>
        <tr><th>Frame Rate</th><td>${video_metadata.fps} fps</td></tr>
        <tr><th>Duration</th><td>${video_metadata.duration_seconds.toFixed(1)}s (${video_metadata.frame_count} frames)</td></tr>
        <tr><th>Codec</th><td>${video_metadata.codec}</td></tr>
        <tr><th>Analysis Date</th><td>${date}</td></tr>
        <tr><th>Frames Analyzed</th><td>${forensic_result.sampled_frames} sampled, ${forensic_result.usable_frames} usable</td></tr>
        <tr><th>Face Coverage</th><td>${(forensic_result.frame_coverage * 100).toFixed(1)}%</td></tr>
        <tr><th>Avg Face Quality</th><td>${forensic_result.average_face_quality.toFixed(3)}</td></tr>
      </table>
    </section>`;
}

function generateVerdictSection(forensic_result: ForensicResult): string {
  return `
    <section class="report-section">
      <h2>2. Forensic Verdict & Key Scores</h2>
      <div class="verdict-summary">
        <div class="verdict-main">
          <span class="verdict-label">Final Classification</span>
          <span class="verdict-value verdict-${forensic_result.verdict.toLowerCase().replace('_', '-')}">${forensic_result.verdict}</span>
        </div>
        <div class="score-grid">
          <div class="score-card">
            <span class="score-label">Manipulation Score</span>
            <span class="score-value">${forensic_result.manipulation_score.toFixed(3)}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Mean Score</span>
            <span class="score-value">${forensic_result.mean_score.toFixed(3)}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Median Score</span>
            <span class="score-value">${forensic_result.median_score.toFixed(3)}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Max Score</span>
            <span class="score-value">${forensic_result.max_score.toFixed(3)}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Consistency</span>
            <span class="score-value">${forensic_result.consistency.toFixed(3)}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Reliability</span>
            <span class="score-value">${forensic_result.reliability.toFixed(3)}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Evidence Confidence</span>
            <span class="score-value">${forensic_result.evidence_confidence.toFixed(3)}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Weighted Mean</span>
            <span class="score-value">${forensic_result.weighted_mean_score.toFixed(3)}</span>
          </div>
        </div>
      </div>
    </section>`;
}

function generateMetricsSection(forensic_result: ForensicResult): string {
  return `
    <section class="report-section">
      <h2>3. Detailed Forensic Metrics</h2>
      <table class="metrics-table">
        <thead>
          <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        </thead>
        <tbody>
          <tr><td>Boundary Artifact Score</td><td>${forensic_result.average_boundary_score?.toFixed(3) ?? 'N/A'}</td><td>Face-edge discontinuity evidence</td></tr>
          <tr><td>Frequency Anomaly</td><td>${forensic_result.average_frequency_anomaly?.toFixed(3) ?? 'N/A'}</td><td>Unusual high-frequency patterns</td></tr>
          <tr><td>Blink Naturalness</td><td>${forensic_result.blink_naturalness_score?.toFixed(3) ?? 'N/A'}</td><td>Physiological blink plausibility</td></tr>
          <tr><td>Identity Drift</td><td>${forensic_result.identity_drift_score?.toFixed(3) ?? 'N/A'}</td><td>Consecutive face identity stability</td></tr>
          <tr><td>Robustness Stability</td><td>${forensic_result.robustness_stability_score?.toFixed(3) ?? 'N/A'}</td><td>Stability under transformations</td></tr>
          <tr><td>Std Deviation</td><td>${forensic_result.std_score.toFixed(3)}</td><td>Score dispersion across frames</td></tr>
          <tr><td>Weighted Std Dev</td><td>${forensic_result.weighted_std_score.toFixed(3)}</td><td>Quality-weighted dispersion</td></tr>
          <tr><td>Min Frame Weight</td><td>${forensic_result.min_frame_weight.toFixed(3)}</td><td>Lowest quality frame weight</td></tr>
          <tr><td>Max Frame Weight</td><td>${forensic_result.max_frame_weight.toFixed(3)}</td><td>Highest quality frame weight</td></tr>
        </tbody>
      </table>
    </section>`;
}

function generateFramesSection(frame_infos: any[], forensic_result: ForensicResult): string {
  const usableFrames = frame_infos.filter(f => f.usable && f.score !== undefined);
  const topFrames = [...usableFrames].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 10);
  
  return `
    <section class="report-section">
      <h2>4. Frame-Level Analysis</h2>
      <p>Total usable frames: ${usableFrames.length} / ${frame_infos.length}</p>
      <table class="frames-table">
        <thead>
          <tr><th>Frame</th><th>Timestamp</th><th>Score</th><th>Face Quality</th><th>Weight</th><th>Boundary</th><th>Frequency</th><th>Blink</th><th>Identity</th></tr>
        </thead>
        <tbody>
          ${topFrames.map((f) => `
            <tr class="${forensic_result.suspicious_frames.some(s => s.frame_index === f.frame_index) ? 'suspicious' : ''}">
              <td>#${f.frame_index}</td>
              <td>${f.timestamp_seconds.toFixed(2)}s</td>
              <td><span class="score-badge score-${(f.score ?? 0) > 0.7 ? 'high' : (f.score ?? 0) > 0.4 ? 'medium' : 'low'}">${(f.score ?? 0).toFixed(3)}</span></td>
              <td>${(f.face_quality ?? 0).toFixed(3)}</td>
              <td>${(f.weight ?? f.face_quality ?? 0).toFixed(3)}</td>
              <td>${(f.boundary_score ?? 0).toFixed(3)}</td>
              <td>${(f.frequency_anomaly ?? 0).toFixed(3)}</td>
              <td>${(f.blink_naturalness ?? 0).toFixed(3)}</td>
              <td>${(f.identity_drift ?? 0).toFixed(3)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </section>`;
}

function generateSegmentsSection(forensic_result: ForensicResult): string {
  if (forensic_result.suspicious_segments.length === 0) {
    return `
      <section class="report-section">
        <h2>5. Suspicious Segments</h2>
        <p class="no-data">No suspicious segments detected.</p>
      </section>`;
  }
  
  return `
    <section class="report-section">
      <h2>5. Suspicious Segments</h2>
      <table class="segments-table">
        <thead>
          <tr><th>#</th><th>Start</th><th>End</th><th>Duration</th><th>Frames</th><th>Peak Score</th><th>Mean Score</th></tr>
        </thead>
        <tbody>
          ${forensic_result.suspicious_segments.map((seg, i) => `
            <tr>
              <td>${i + 1}</td>
              <td>${seg.start.toFixed(1)}s</td>
              <td>${seg.end.toFixed(1)}s</td>
              <td>${seg.duration.toFixed(1)}s</td>
              <td>${seg.frame_count}</td>
              <td><span class="score-badge score-high">${seg.peak_score.toFixed(3)}</span></td>
              <td>${seg.mean_score.toFixed(3)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </section>`;
}

function generateExplanationsSection(forensic_result: ForensicResult): string {
  return `
    <section class="report-section">
      <h2>6. Forensic Explanations</h2>
      <ul class="explanations-list">
        ${forensic_result.explanations.map(exp => `<li>${exp}</li>`).join('')}
      </ul>
      ${forensic_result.reason_codes && forensic_result.reason_codes.length > 0 ? `
        <h3>Reason Codes</h3>
        <ul class="reason-codes">
          ${forensic_result.reason_codes.map(code => `<li><code>${code}</code></li>`).join('')}
        </ul>
      ` : ''}
    </section>`;
}

function generateGradCAMSection(gradcam_explanations: any[]): string {
  if (!gradcam_explanations || gradcam_explanations.length === 0) {
    return `
      <section class="report-section">
        <h2>7. Grad-CAM Attention Visualizations</h2>
        <p class="no-data">No Grad-CAM explanations available. Enable explainability during analysis to generate attention heatmaps.</p>
      </section>`;
  }
  
  const fixPath = (path: string) => {
    if (!path) return '';
    let cleanPath = path.replace(/\\/g, '/');
    cleanPath = cleanPath.replace(/^outputs\//, '').replace(/^explanations\//, '');
    return `/api/files/explanations/${cleanPath}`;
  };
  
  return `
    <section class="report-section">
      <h2>7. Grad-CAM Attention Visualizations</h2>
      <p>Gradient-weighted Class Activation Mapping (Grad-CAM) shows which facial regions most influenced the model's deepfake prediction.</p>
      <div class="gradcam-grid">
        ${gradcam_explanations.map((exp) => `
          <div class="gradcam-item">
            <h4>Frame #${exp.frame_index} (${exp.timestamp_seconds.toFixed(1)}s) — Score: ${exp.score.toFixed(3)}</h4>
            <div class="gradcam-images">
              ${exp.original_path ? `<div class="gradcam-image"><img src="${fixPath(exp.original_path)}" alt="Original"><span>Original</span></div>` : ''}
              ${exp.heatmap_path ? `<div class="gradcam-image"><img src="${fixPath(exp.heatmap_path)}" alt="Heatmap"><span>Heatmap</span></div>` : ''}
              ${exp.overlay_path ? `<div class="gradcam-image"><img src="${fixPath(exp.overlay_path)}" alt="Overlay"><span>Overlay</span></div>` : ''}
            </div>
          </div>
        `).join('')}
      </div>
    </section>`;
}

function generateRobustnessSection(robustness_results: any): string {
  if (!robustness_results || !robustness_results.tests?.length) {
    return `
      <section class="report-section">
        <h2>8. Robustness Testing</h2>
        <p class="no-data">Robustness testing not performed.</p>
      </section>`;
  }
  
  return `
    <section class="report-section">
      <h2>8. Robustness Testing</h2>
      <p>Overall Stability: <strong>${robustness_results.overall_stability.toFixed(3)}</strong> — ${robustness_results.interpretation}</p>
      <table class="robustness-table">
        <thead>
          <tr><th>Transformation</th><th>Score</th><th>Difference</th><th>Stability</th></tr>
        </thead>
        <tbody>
          ${robustness_results.tests.map((test: any) => `
            <tr>
              <td>${test.transform}</td>
              <td>${test.score.toFixed(3)}</td>
              <td>${test.difference.toFixed(4)}</td>
              <td><span class="stability-badge stability-${test.stability > 0.9 ? 'high' : test.stability > 0.8 ? 'medium' : 'low'}">${test.stability.toFixed(3)}</span></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </section>`;
}

function generateTechnicalSection(frame_infos: any[]): string {
  return `
    <section class="report-section">
      <h2>9. Technical Appendix</h2>
      <h3>Model Configuration</h3>
      <table class="info-table">
        <tr><th>Architecture</th><td>ConvNeXt-Tiny / Xception</td></tr>
        <tr><th>Input Resolution</th><td>224 × 224</td></tr>
        <tr><th>Face Detector</th><td>InsightFace (RetinaFace)</td></tr>
        <tr><th>Sampling Rate</th><td>1 fps (max 300 frames)</td></tr>
        <tr><th>Min Face Confidence</th><td>0.5</td></tr>
        <tr><th>Quality Threshold</th><td>0.3</td></tr>
      </table>
      <h3>Frame Statistics</h3>
      <table class="info-table">
        <tr><th>Total Frames</th><td>${frame_infos.length}</td></tr>
        <tr><th>Usable Frames</th><td>${frame_infos.filter(f => f.usable).length}</td></tr>
        <tr><th>Frames with Faces</th><td>${frame_infos.filter(f => f.face_found).length}</td></tr>
        <tr><th>Mean Score (all)</th><td>${(frame_infos.reduce((a, b) => a + (b.score ?? 0), 0) / frame_infos.length).toFixed(4)}</td></tr>
      </table>
      <h3>Forensic Thresholds</h3>
      <table class="info-table">
        <tr><th>Suspicious Frame Threshold</th><td>0.5</td></tr>
        <tr><th>Deepfake Verdict Threshold</th><td>0.6</td></tr>
        <tr><th>Inconclusive Range</th><td>0.3 – 0.6</td></tr>
        <tr><th>Real Verdict Threshold</th><td>< 0.3</td></tr>
      </table>
    </section>`;
}

function generateJSONReport(result: AnalysisResult, sections: string[]): string {
  const reportData = {
    metadata: {
      video_id: result.video_id,
      generated_at: result.timestamp,
      generator: 'Trustlens Video Deepfake Forensics Platform',
      version: '1.0.0',
    },
    video_metadata: result.video_metadata,
    forensic_result: result.forensic_result,
    robustness_results: result.robustness_results,
    gradcam_explanations: result.gradcam_explanations,
    included_sections: sections,
  };
  return JSON.stringify(reportData, null, 2);
}

function getReportStyles(): string {
  return `
    * { box-sizing: border-box; }
    body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; color: #1a1a2e; background: #f8f9fa; margin: 0; padding: 20px; }
    .report-container { max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); overflow: hidden; }
    .report-header { background: linear-gradient(135deg, #464B71 0%, #3a3f62 50%, #464B71 100%); color: white; padding: 32px; display: flex; align-items: center; justifyContent: space-between; gap: 24; flex-wrap: wrap; }
    .header-content { display: flex; align-items: center; gap: 24; }
    .logo { font-size: 2rem; }
    .header-text h1 { margin: 0; font-size: 1.5rem; font-weight: 700; }
    .meta { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }
    .verdict-badge { padding: 12px 24px; border-radius: 8px; font-weight: 700; font-size: 1rem; text-transform: uppercase; letterSpacing: 0.05em; }
    .verdict-real { background: #22c55e; }
    .verdict-likely-deepfake { background: #ef4444; }
    .verdict-inconclusive { background: #f59e0b; }
    .report-main { padding: 32px; }
    .report-section { margin-bottom: 40px; }
    .report-section h2 { color: #464B71; font-size: 1.25rem; font-weight: 700; border-bottom: 2px solid #118AB2; padding-bottom: 8px; margin-bottom: 20px; }
    .report-section h3 { color: #464B71; font-size: 1rem; margin-top: 24px; margin-bottom: 12px; }
    .info-table, .metrics-table, .frames-table, .segments-table, .robustness-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .info-table th, .metrics-table th, .frames-table th, .segments-table th, .robustness-table th { text-align: left; padding: 10px 12px; background: #f0f2f8; color: #464B71; font-weight: 600; border-bottom: 2px solid #118AB2; }
    .info-table td, .metrics-table td, .frames-table td, .segments-table td, .robustness-table td { padding: 10px 12px; border-bottom: 1px solid #e5e7eb; }
    .info-table tr:last-child td { border-bottom: none; }
    .score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16; margin-top: 16; }
    .score-card { background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 10; padding: 16; text-align: center; }
    .score-label { display: block; font-size: 0.7rem; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6; }
    .score-value { font-size: 1.5rem; font-weight: 800; color: #464B71; font-family: 'JetBrains Mono', monospace; }
    .score-badge { padding: 4px 10px; border-radius: 6; font-weight: 700; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; }
    .score-high { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .score-medium { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .score-low { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
    .frames-table tr.suspicious { background: rgba(239, 68, 68, 0.05); }
    .frames-table tr.suspicious td { border-color: rgba(239, 68, 68, 0.1); }
    .stability-badge { padding: 4px 10px; border-radius: 6; font-weight: 700; font-size: 0.75rem; }
    .stability-high { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
    .stability-medium { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .stability-low { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .explanations-list { padding-left: 20px; }
    .explanations-list li { margin-bottom: 10px; line-height: 1.7; }
    .reason-codes li code { background: #f0f2f8; padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
    .gradcam-grid { display: grid; gap: 24; }
    .gradcam-item { border: 1px solid #e5e7eb; border-radius: 10; padding: 16; background: #fafafa; }
    .gradcam-item h4 { margin: 0 0 12px; color: #464B71; font-size: 0.9rem; }
    .gradcam-images { display: flex; gap: 16; flex-wrap: wrap; }
    .gradcam-image { text-align: center; }
    .gradcam-image img { max-width: 100%; height: auto; border-radius: 8; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .gradcam-image span { display: block; margin-top: 6px; font-size: 0.75rem; color: #6b7280; font-weight: 500; }
    .no-data { color: #9ca3af; font-style: italic; padding: 20px; background: #f8f9fa; border-radius: 8; }
    .report-footer { background: #f8f9fa; border-top: 1px solid #e5e7eb; padding: 24px 32px; text-align: center; color: #6b7280; font-size: 0.85rem; }
    .disclaimer { margin-top: 8px; font-size: 0.75rem; opacity: 0.7; }
    @media print { body { background: white; padding: 0; } .report-container { box-shadow: none; border-radius: 0; } .report-header { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
  `;
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    background: '#ffffff',
    borderRadius: 16,
    padding: 24,
    boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04)',
  },
  header: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: 24,
    gap: 16,
  },
  titleSection: { flex: 1 },
  title: { fontSize: '1.15rem', fontWeight: 700, color: '#464B71', margin: '0 0 4px' },
  subtitle: { fontSize: '0.85rem', color: '#6b7280', margin: 0, lineHeight: 1.5 },
  badge: {
    background: 'linear-gradient(135deg, #eef1fb 0%, #f5efff 100%)',
    border: '1px solid rgba(17, 138, 178, 0.2)',
    borderRadius: 8,
    padding: '4px 12px',
    fontSize: '0.65rem',
    fontWeight: 700,
    color: '#118AB2',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  section: { marginBottom: 24 },
  sectionTitle: { fontSize: '0.9rem', fontWeight: 700, color: '#464B71', margin: '0 0 8px' },
  sectionDesc: { fontSize: '0.8rem', color: '#6b7280', margin: '0 0 16px' },
  sectionsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: 10,
  },
  sectionItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '12px 14px',
    background: '#fafafa',
    border: '1px solid #e5e7eb',
    borderRadius: 10,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  checkbox: {
    width: 18,
    height: 18,
    accentColor: '#118AB2',
    cursor: 'pointer',
  },
  sectionInfo: { display: 'flex', alignItems: 'center', gap: 10 },
  sectionIcon: { fontSize: '1.2rem' },
  sectionLabel: { fontSize: '0.8rem', fontWeight: 600, color: '#464B71' },
  requiredTag: {
    fontSize: '0.55rem',
    fontWeight: 700,
    color: '#ef4444',
    background: 'rgba(239, 68, 68, 0.1)',
    padding: '2px 6px',
    borderRadius: 4,
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
  },
  formatSelector: { display: 'flex', gap: 12, flexWrap: 'wrap' },
  formatBtn: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 6,
    padding: '16px 20px',
    border: '2px solid #e5e7eb',
    borderRadius: 12,
    background: '#fafafa',
    cursor: 'pointer',
    transition: 'all 0.2s',
    minWidth: 140,
  },
  formatBtnActive: {
    borderColor: '#118AB2',
    background: 'linear-gradient(135deg, #eef1fb 0%, #f5efff 100%)',
    boxShadow: '0 4px 16px rgba(17, 138, 178, 0.15)',
  },
  formatIcon: { fontSize: '1.5rem' },
  formatLabel: { fontWeight: 700, color: '#464B71', fontSize: '0.85rem' },
  formatDesc: { fontSize: '0.65rem', color: '#9ca3af' },
  summary: {
    background: 'linear-gradient(135deg, #fefdfa 0%, #f5f0ff 100%)',
    border: '1px solid rgba(17, 138, 178, 0.15)',
    borderRadius: 12,
    padding: 20,
  },
  summaryLabel: { fontSize: '0.7rem', fontWeight: 700, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 16 },
  summaryGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
    gap: 16,
  },
  summaryItem: { display: 'flex', flexDirection: 'column', gap: 4 },
  summaryKey: { fontSize: '0.65rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.04em' },
  summaryValue: { fontSize: '0.95rem', fontWeight: 700, color: '#464B71', fontFamily: "'JetBrains Mono', monospace" },
  verdictREAL: { color: '#22c55e' },
  verdictLIKELYDEEPFAKE: { color: '#ef4444' },
  verdictINCONCLUSIVE: { color: '#f59e0b' },
  error: {
    marginTop: 16,
    padding: '12px 16px',
    background: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: 8,
    color: '#dc2626',
    fontSize: '0.8rem',
  },
  actions: {
    display: 'flex',
    gap: 12,
    marginTop: 24,
    paddingTop: 20,
    borderTop: '1px solid #e5e7eb',
    flexWrap: 'wrap',
  },
  generateBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '12px 24px',
    background: 'linear-gradient(135deg, #464B71 0%, #118AB2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: 10,
    fontSize: '0.9rem',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  generateBtnDisabled: {
    opacity: 0.7,
    cursor: 'not-allowed',
  },
  spinner: {
    width: 16,
    height: 16,
    border: '2px solid rgba(255,255,255,0.3)',
    borderTopColor: 'white',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
  downloadBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '12px 24px',
    background: 'white',
    color: '#118AB2',
    border: '2px solid #118AB2',
    borderRadius: 10,
    fontSize: '0.9rem',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  emptyState: {
    textAlign: 'center',
    padding: '48px 24px',
    color: '#9ca3af',
  },
  emptyIcon: { fontSize: '3rem', marginBottom: 12 },
  emptyTitle: { fontSize: '1rem', fontWeight: 600, color: '#464B71', margin: '0 0 8px' },
  emptyText: { fontSize: '0.85rem', lineHeight: 1.6, margin: 0 },
};