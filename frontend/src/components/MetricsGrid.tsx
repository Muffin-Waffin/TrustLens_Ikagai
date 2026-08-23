import { useEffect, useRef, useState, type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ForensicResult } from '../services/api';

interface MetricsGridProps { forensicResult: ForensicResult; }
interface MetricItem { label: string; value: number | null | undefined; format: 'decimal' | 'percent'; icon: ReactNode; tooltip: TooltipInfo; }
interface TooltipInfo {
  description: string;
  importance: string;
  usage: string;
  normalRange: string;
}

function useCountUp(target: number, duration = 1200, decimals = 3): string {
  const [value, setValue] = useState(0);
  const frameRef = useRef<number>(0);
  useEffect(() => {
    const start = performance.now();
    const animate = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      setValue(target * (1 - Math.pow(1 - progress, 3)));
      if (progress < 1) frameRef.current = requestAnimationFrame(animate);
    };
    frameRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameRef.current);
  }, [target, duration]);
  return value.toFixed(decimals);
}

const METRIC_TOOLTIPS: Record<string, TooltipInfo> = {
  'Manipulation Score': {
    description: 'Overall probability that the video contains deepfake manipulation, derived from frame-level classification scores weighted by face quality and detection confidence.',
    importance: 'Primary indicator of manipulation likelihood. High scores suggest synthetic content; low scores suggest authentic video.',
    usage: 'Used as the main decision metric for verdict classification (REAL / INCONCLUSIVE / LIKELY_DEEPFAKE).',
    normalRange: 'Real videos: 0.0–0.3 | Inconclusive: 0.3–0.6 | Deepfake: >0.6',
  },
  'Evidence Confidence': {
    description: 'Confidence level in the forensic conclusion based on the amount and quality of usable evidence (frames with clear faces, consistent detections).',
    importance: 'Indicates how trustworthy the verdict is. Low confidence means the result may be unreliable due to poor video quality or few analyzable frames.',
    usage: 'Modulates the final verdict; low confidence can downgrade a "LIKELY_DEEPFAKE" to "INCONCLUSIVE".',
    normalRange: 'High confidence: >0.7 | Moderate: 0.4–0.7 | Low: <0.4',
  },
  'Reliability': {
    description: 'Composite measure of how reliable the frame-level predictions are, combining face detection quality, temporal consistency, and model certainty.',
    importance: 'Flags cases where the model may be guessing due to blurry faces, occlusions, or distribution shift.',
    usage: 'Used to weight the manipulation score; low reliability reduces trust in extreme scores.',
    normalRange: 'Reliable: >0.7 | Moderate: 0.4–0.7 | Unreliable: <0.4',
  },
  'Consistency': {
    description: 'Temporal consistency of manipulation scores across consecutive frames. Measures whether predictions are stable or erratic over time.',
    importance: 'Deepfakes often show temporal flickering; authentic videos should have smooth, consistent scores.',
    usage: 'High consistency supports the verdict; low consistency triggers "INCONCLUSIVE" even with high mean scores.',
    normalRange: 'Consistent: >0.8 | Variable: 0.5–0.8 | Erratic: <0.5',
  },
  'Face Coverage': {
    description: 'Percentage of sampled frames where a face was successfully detected and analyzed.',
    importance: 'Low coverage means large portions of the video were not analyzed, reducing forensic completeness.',
    usage: 'Minimum threshold (typically 50%) required for a valid analysis; below this, results are marked INCONCLUSIVE.',
    normalRange: 'Good: >80% | Acceptable: 50–80% | Poor: <50%',
  },
  'Avg Face Quality': {
    description: 'Mean quality score of detected faces across all usable frames (combines sharpness, pose, illumination, and occlusion factors).',
    importance: 'Poor face quality leads to unreliable model predictions; high quality ensures the model sees clear facial features.',
    usage: 'Weights frame contributions; low-quality frames contribute less to the final manipulation score.',
    normalRange: 'High quality: >0.7 | Moderate: 0.4–0.7 | Low: <0.4',
  },
  'Boundary Artifact': {
    description: 'Average boundary artifact score measuring discontinuities at face edges (blending artifacts, color mismatches, resolution differences).',
    importance: 'Deepfake generation often leaves telltale artifacts at face boundaries where the synthetic face meets the original frame.',
    usage: 'Elevated scores (>0.5) are strong indicators of face-swap or face-reenactment manipulation.',
    normalRange: 'Clean: <0.2 | Suspicious: 0.2–0.5 | Strong artifacts: >0.5',
  },
  'Frequency Anomaly': {
    description: 'Spectral analysis score detecting unusual high-frequency patterns inconsistent with natural camera capture (compression artifacts, GAN fingerprints).',
    importance: 'Synthetic generators leave characteristic frequency-domain signatures that differ from natural sensor noise.',
    usage: 'Complementary signal; high values support manipulation hypothesis, especially for GAN-generated content.',
    normalRange: 'Natural: <0.2 | Unusual: 0.2–0.5 | Anomalous: >0.5',
  },
  'Blink Naturalness': {
    description: 'Score measuring physiological plausibility of eye blink patterns (duration, frequency, symmetry, velocity curves).',
    importance: 'Deepfakes often fail to replicate natural blink dynamics; unnatural blinks are a strong manipulation cue.',
    usage: 'Low scores (<0.4) indicate synthetic blink patterns, supporting deepfake verdict.',
    normalRange: 'Natural: >0.6 | Uncertain: 0.3–0.6 | Synthetic: <0.3',
  },
  'Identity Drift': {
    description: 'Measures identity consistency across consecutive frames using face embedding similarity. Detects face-swaps or morphing.',
    importance: 'Identity should remain stable in authentic video; sudden changes indicate face replacement.',
    usage: 'High drift (>0.4) suggests face-swap; low drift supports authentic or consistent deepfake (full-frame synthesis).',
    normalRange: 'Stable: <0.2 | Minor drift: 0.2–0.4 | Significant: >0.4',
  },
  'Robustness Stability': {
    description: 'Model prediction stability under common video transformations (compression, blur, resize, brightness).',
    importance: 'Confirms the detection is based on semantic features, not brittle artifacts that disappear under minor perturbations.',
    usage: 'Low stability warns that the result may not generalize; high stability increases verdict confidence.',
    normalRange: 'Stable: >0.9 | Moderate: 0.8–0.9 | Unstable: <0.8',
  },
};

const container = { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1, delayChildren: 0.2 } } };
const item = { hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0, transition: { duration: 0.4 } } };

const TOOLTIP_WIDTH = 360;

function TooltipContent({ label, tooltip, onClose, position }: { label: string; tooltip: TooltipInfo; onClose: (e: React.MouseEvent) => void; position: { x: number; y: number } }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: 10 }}
      style={{
        position: 'fixed',
        left: position.x,
        top: position.y,
        zIndex: 2000,
        background: 'linear-gradient(135deg, #fefdfa 0%, #f5f0ff 100%)',
        color: '#3a3f62',
        borderRadius: 14,
        padding: '18px 20px',
        width: TOOLTIP_WIDTH,
        maxHeight: '65vh',
        overflow: 'auto',
        boxShadow: '0 20px 50px rgba(70, 75, 113, 0.2), 0 0 0 1px rgba(17, 138, 178, 0.15), inset 0 1px 0 rgba(255,255,255,0.8)',
        border: '1px solid rgba(17, 138, 178, 0.2)',
        fontSize: '0.8rem',
        lineHeight: 1.6,
        fontFamily: "'Inter', sans-serif",
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, paddingBottom: 10, borderBottom: '1px solid rgba(17, 138, 178, 0.15)' }}>
        <div style={{ fontWeight: 700, color: '#464B71', fontSize: '0.9rem', maxWidth: '80%' }}>{label}</div>
        <button
          onClick={onClose}
          style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: 8,
            width: 28,
            height: 28,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            color: '#ef4444',
            fontSize: '1rem',
            lineHeight: 1,
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.2)'; e.currentTarget.style.borderColor = '#ef4444'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.2)'; }}
        >
          ×
        </button>
      </div>

      <div style={{ marginBottom: 14 }}>
        <div style={{ fontWeight: 600, color: '#118AB2', marginBottom: 6, fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>What it measures</div>
        <div style={{ color: '#4a506a', fontSize: '0.78rem' }}>{tooltip.description}</div>
      </div>

      <div style={{ marginBottom: 14 }}>
        <div style={{ fontWeight: 600, color: '#118AB2', marginBottom: 6, fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Why it matters</div>
        <div style={{ color: '#4a506a', fontSize: '0.78rem' }}>{tooltip.importance}</div>
      </div>

      <div style={{ marginBottom: 14 }}>
        <div style={{ fontWeight: 600, color: '#118AB2', marginBottom: 6, fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>How it's used</div>
        <div style={{ color: '#4a506a', fontSize: '0.78rem' }}>{tooltip.usage}</div>
      </div>

      <div style={{ borderTop: '1px solid rgba(17, 138, 178, 0.15)', paddingTop: 12 }}>
        <div style={{ fontWeight: 600, color: '#118AB2', marginBottom: 6, fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Typical ranges</div>
        <div style={{ color: '#4a506a', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.72rem', lineHeight: 1.6 }}>{tooltip.normalRange}</div>
      </div>
    </motion.div>
  );
}

function MetricCard({ label, value, format, icon, tooltip }: MetricItem & { tooltip: TooltipInfo }) {
  const available = value !== null && value !== undefined;
  const displayed = useCountUp(format === 'percent' ? (value ?? 0) * 100 : (value ?? 0), 1200, format === 'percent' ? 1 : 3);
  const [showTooltip, setShowTooltip] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const handleSeeMore = (e: React.MouseEvent) => {
    e.stopPropagation();
    const rect = cardRef.current?.getBoundingClientRect();
    if (!rect) return;

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const tooltipWidth = TOOLTIP_WIDTH;
    const tooltipHeight = 400;

    let x = rect.right + 12;
    let y = rect.top;

    if (x + tooltipWidth > viewportWidth - 12) {
      x = rect.left - tooltipWidth - 12;
    }

    if (y + tooltipHeight > viewportHeight - 12) {
      y = viewportHeight - tooltipHeight - 12;
    }
    if (y < 12) y = 12;

    setTooltipPos({ x, y });
    setShowTooltip(true);
  };

  const handleClose = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowTooltip(false);
  };

  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  return (
    <>
      <motion.div
        ref={cardRef}
        variants={item}
        whileHover={{ y: -4, boxShadow: '0 12px 32px rgba(0,0,0,0.1)' }}
        style={{
          background: '#ffffff',
          borderRadius: 14,
          padding: '20px 18px',
          borderLeft: '3px solid #118AB2',
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
          transition: 'box-shadow 0.2s, transform 0.2s',
          position: 'relative',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <div style={{ color: '#118AB2', fontSize: '1.1rem' }}>{icon}</div>
          <span style={{ fontSize: '0.72rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</span>
        </div>
        <div style={{ fontSize: '1.7rem', fontWeight: 800, color: '#464B71', fontFamily: "'JetBrains Mono', monospace" }}>{available ? <>{displayed}{format === 'percent' ? '%' : ''}</> : 'N/A'}</div>
        <button
          onClick={handleSeeMore}
          style={{
            marginTop: 12,
            padding: '6px 12px',
            background: 'linear-gradient(135deg, #eef1fb 0%, #f5efff 100%)',
            border: '1px solid rgba(17, 138, 178, 0.2)',
            borderRadius: 8,
            fontSize: '0.7rem',
            fontWeight: 600,
            color: '#118AB2',
            cursor: 'pointer',
            transition: 'all 0.2s',
            width: '100%',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'linear-gradient(135deg, #e0e6fb 0%, #ebe0ff 100%)'; e.currentTarget.style.borderColor = '#118AB2'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'linear-gradient(135deg, #eef1fb 0%, #f5efff 100%)'; e.currentTarget.style.borderColor = 'rgba(17, 138, 178, 0.2)'; }}
        >
          See more
        </button>
      </motion.div>

      <AnimatePresence>
        {showTooltip && (
          <>
            <div
              style={{
                position: 'fixed',
                inset: 0,
                zIndex: 1999,
                background: 'rgba(0,0,0,0.15)',
                backdropFilter: 'blur(1px)',
              }}
              onClick={() => setShowTooltip(false)}
            />
            <TooltipContent label={label} tooltip={tooltip} onClose={handleClose} position={tooltipPos} />
          </>
        )}
      </AnimatePresence>
    </>
  );
}

export default function MetricsGrid({ forensicResult }: MetricsGridProps) {
  const metrics: (MetricItem & { tooltip: TooltipInfo })[] = [
    { label: 'Manipulation Score', value: forensicResult.manipulation_score ?? 0, format: 'decimal', icon: <span>+</span>, tooltip: METRIC_TOOLTIPS['Manipulation Score'] },
    { label: 'Evidence Confidence', value: forensicResult.evidence_confidence ?? 0, format: 'decimal', icon: <span>✓</span>, tooltip: METRIC_TOOLTIPS['Evidence Confidence'] },
    { label: 'Reliability', value: forensicResult.reliability ?? 0, format: 'decimal', icon: <span>◈</span>, tooltip: METRIC_TOOLTIPS['Reliability'] },
    { label: 'Consistency', value: forensicResult.consistency ?? 0, format: 'decimal', icon: <span>≡</span>, tooltip: METRIC_TOOLTIPS['Consistency'] },
    { label: 'Face Coverage', value: forensicResult.frame_coverage ?? 0, format: 'percent', icon: <span>◉</span>, tooltip: METRIC_TOOLTIPS['Face Coverage'] },
    { label: 'Avg Face Quality', value: forensicResult.average_face_quality ?? 0, format: 'decimal', icon: <span>★</span>, tooltip: METRIC_TOOLTIPS['Avg Face Quality'] },
    { label: 'Boundary Artifact', value: forensicResult.average_boundary_score, format: 'decimal', icon: <span>◫</span>, tooltip: METRIC_TOOLTIPS['Boundary Artifact'] },
    { label: 'Frequency Anomaly', value: forensicResult.average_frequency_anomaly, format: 'decimal', icon: <span>⌁</span>, tooltip: METRIC_TOOLTIPS['Frequency Anomaly'] },
    { label: 'Blink Naturalness', value: forensicResult.blink_naturalness_score, format: 'decimal', icon: <span>◉</span>, tooltip: METRIC_TOOLTIPS['Blink Naturalness'] },
    { label: 'Identity Drift', value: forensicResult.identity_drift_score, format: 'decimal', icon: <span>◌</span>, tooltip: METRIC_TOOLTIPS['Identity Drift'] },
    { label: 'Robustness Stability', value: forensicResult.robustness_stability_score, format: 'decimal', icon: <span>◇</span>, tooltip: METRIC_TOOLTIPS['Robustness Stability'] },
  ];
  return <motion.div variants={container} initial="hidden" animate="visible" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>{metrics.map((metric) => <MetricCard key={metric.label} {...metric} />)}</motion.div>;
}