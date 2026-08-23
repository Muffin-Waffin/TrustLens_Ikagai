import { motion } from 'framer-motion';
import type { FrameInfo, SuspiciousFrame } from '../services/api';

interface TopSuspiciousFramesProps {
  frameInfos: FrameInfo[];
  suspiciousFrames: SuspiciousFrame[];
}

export default function TopSuspiciousFrames({ frameInfos, suspiciousFrames }: TopSuspiciousFramesProps) {
  const usableFrames = frameInfos.filter(f => f.usable && f.score !== undefined);
  const topFrames = [...usableFrames]
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
    .slice(0, 3);

  const suspiciousFrameIndices = new Set(suspiciousFrames.map(f => f.frame_index));

  if (topFrames.length === 0) {
    return (
      <div style={styles.card}>
        <div style={styles.emptyState}>
          <div style={styles.emptyIcon}>🎯</div>
          <h3 style={styles.emptyTitle}>No Suspicious Frames Detected</h3>
          <p style={styles.emptyText}>
            No frames with usable face detections and manipulation scores were found.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.titleSection}>
          <h3 style={styles.title}>Top Suspicious Frames</h3>
          <p style={styles.subtitle}>
            The {topFrames.length} frames with the highest manipulation scores. Frames marked as suspicious are flagged by the forensic engine.
          </p>
        </div>
        <div style={styles.badge}>Top 3</div>
      </div>

      <div style={styles.grid}>
        {topFrames.map((frame, idx) => (
          <motion.div
            key={frame.frame_index}
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            style={styles.frameCard}
          >
            <div style={styles.rankBadge}>{idx + 1}</div>

            <div style={styles.frameHeader}>
              <span style={styles.frameIndex}>Frame #{frame.frame_index}</span>
              <span style={{ ...styles.timestamp, ...(suspiciousFrameIndices.has(frame.frame_index) ? styles.timestampSuspicious : {}) }}>
                {suspiciousFrameIndices.has(frame.frame_index) && '⚠️ '}
                {frame.timestamp_seconds.toFixed(2)}s
              </span>
            </div>

            <div style={styles.scoreContainer}>
              <div style={styles.scoreCircle}>
                <svg viewBox="0 0 64 64" style={styles.scoreSvg}>
                  <circle
                    cx="32"
                    cy="32"
                    r="28"
                    fill="none"
                    stroke="#e5e7eb"
                    strokeWidth="4"
                  />
                  <motion.circle
                    cx="32"
                    cy="32"
                    r="28"
                    fill="none"
                    stroke={getScoreColor(frame.score ?? 0)}
                    strokeWidth="4"
                    strokeDasharray={2 * Math.PI * 28}
                    strokeDashoffset={2 * Math.PI * 28 * (1 - (frame.score ?? 0))}
                    strokeLinecap="round"
                    style={styles.scoreCircleStroke}
                    initial={{ strokeDashoffset: 2 * Math.PI * 28 }}
                    animate={{ strokeDashoffset: 2 * Math.PI * 28 * (1 - (frame.score ?? 0)) }}
                    transition={{ duration: 1, delay: 0.2 * idx, ease: 'easeOut' }}
                  />
                </svg>
                <span style={styles.scoreValue}>{(frame.score ?? 0).toFixed(3)}</span>
              </div>
              <div style={styles.scoreLabel}>
                <span style={{ color: getScoreColor(frame.score ?? 0) }}>
                  {getScoreLabel(frame.score ?? 0)}
                </span>
              </div>
            </div>

            <div style={styles.metrics}>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Face Quality</span>
                <span style={styles.metricValue}>{(frame.face_quality ?? 0).toFixed(2)}</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Weight</span>
                <span style={styles.metricValue}>{(frame.weight ?? frame.face_quality ?? 0).toFixed(2)}</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Boundary</span>
                <span style={styles.metricValue}>{(frame.boundary_score ?? 0).toFixed(2)}</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Frequency</span>
                <span style={styles.metricValue}>{(frame.frequency_anomaly ?? 0).toFixed(2)}</span>
              </div>
            </div>

            {suspiciousFrameIndices.has(frame.frame_index) && (
              <div style={styles.suspiciousBadge}>
                Flagged as Suspicious
              </div>
            )}

            <div style={styles.evidenceTags}>
              {frame.blink_naturalness !== null && frame.blink_naturalness !== undefined && (
                <span style={styles.evidenceTag}>
                  Blink: {frame.blink_naturalness < 0.4 ? 'Unnatural' : 'Natural'}
                </span>
              )}
              {frame.identity_drift !== null && frame.identity_drift !== undefined && (
                <span style={styles.evidenceTag}>
                  Identity: {frame.identity_drift > 0.4 ? 'Drift Detected' : 'Stable'}
                </span>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      <div style={styles.legend}>
        <div style={styles.legendItem}>
          <span style={{ ...styles.legendDot, background: '#ef4444' }} />
          <span>High Risk (&gt;0.7)</span>
        </div>
        <div style={styles.legendItem}>
          <span style={{ ...styles.legendDot, background: '#f59e0b' }} />
          <span>Medium Risk (0.4-0.7)</span>
        </div>
        <div style={styles.legendItem}>
          <span style={{ ...styles.legendDot, background: '#22c55e' }} />
          <span>Low Risk (&lt;0.4)</span>
        </div>
        <div style={styles.legendItem}>
          <span style={{ ...styles.legendDot, background: '#ef4444', border: '2px dashed #ef4444', backgroundColor: 'transparent' }} />
          <span>Flagged Suspicious</span>
        </div>
      </div>
    </div>
  );
}

const itemVariants = {
  hidden: { opacity: 0, y: 20, scale: 0.95 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.4, ease: 'easeOut' as const } },
};

function getScoreColor(score: number): string {
  if (score > 0.7) return '#ef4444';
  if (score > 0.4) return '#f59e0b';
  return '#22c55e';
}

function getScoreLabel(score: number): string {
  if (score > 0.7) return 'High Risk';
  if (score > 0.4) return 'Medium Risk';
  return 'Low Risk';
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
    marginBottom: 20,
    gap: 16,
  },
  titleSection: { flex: 1 },
  title: { fontSize: '1.15rem', fontWeight: 700, color: '#464B71', margin: '0 0 4px' },
  subtitle: { fontSize: '0.8rem', color: '#6b7280', margin: 0, lineHeight: 1.5 },
  badge: {
    background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
    border: '1px solid rgba(245, 158, 11, 0.2)',
    borderRadius: 8,
    padding: '4px 12px',
    fontSize: '0.65rem',
    fontWeight: 700,
    color: '#b45309',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: 16,
  },
  frameCard: {
    background: '#fafafa',
    border: '1px solid #e5e7eb',
    borderRadius: 12,
    padding: 20,
    position: 'relative',
    transition: 'all 0.2s',
  },
  rankBadge: {
    position: 'absolute',
    top: -10,
    left: 16,
    width: 28,
    height: 28,
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #464B71 0%, #118AB2 100%)',
    color: 'white',
    fontSize: '0.75rem',
    fontWeight: 700,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 2px 8px rgba(70, 75, 113, 0.3)',
  },
  frameHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  frameIndex: { fontSize: '0.75rem', fontWeight: 600, color: '#464B71', fontFamily: "'JetBrains Mono', monospace" },
  timestamp: { fontSize: '0.75rem', color: '#6b7280', fontFamily: "'JetBrains Mono', monospace" },
  timestampSuspicious: { color: '#ef4444', fontWeight: 600 },
  scoreContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginBottom: 16,
  },
  scoreCircle: {
    position: 'relative',
    width: 80,
    height: 80,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  scoreSvg: { transform: 'rotate(-90deg)' },
  scoreCircleStroke: { transition: 'stroke-dashoffset 1s ease-out' },
  scoreValue: {
    position: 'absolute',
    fontSize: '1.5rem',
    fontWeight: 800,
    color: '#464B71',
    fontFamily: "'JetBrains Mono', monospace",
  },
  scoreLabel: { marginTop: 8, fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' },
  metrics: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: 10,
    marginBottom: 12,
    padding: 12,
    background: 'white',
    borderRadius: 8,
    border: '1px solid #e5e7eb',
  },
  metric: { display: 'flex', flexDirection: 'column', gap: 2, alignItems: 'center' },
  metricLabel: { fontSize: '0.6rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.04em' },
  metricValue: { fontSize: '0.85rem', fontWeight: 700, color: '#464B71', fontFamily: "'JetBrains Mono', monospace" },
  suspiciousBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 12px',
    background: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid rgba(239, 68, 68, 0.2)',
    borderRadius: 20,
    fontSize: '0.65rem',
    fontWeight: 600,
    color: '#ef4444',
    marginBottom: 10,
    width: 'fit-content',
  },
  evidenceTags: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
  },
  evidenceTag: {
    padding: '4px 10px',
    borderRadius: 16,
    fontSize: '0.6rem',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    background: 'rgba(17, 138, 178, 0.1)',
    color: '#118AB2',
    border: '1px solid rgba(17, 138, 178, 0.15)',
  },
  legend: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 16,
    marginTop: 20,
    paddingTop: 16,
    borderTop: '1px solid #e5e7eb',
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    fontSize: '0.7rem',
    color: '#6b7280',
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: '50%',
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