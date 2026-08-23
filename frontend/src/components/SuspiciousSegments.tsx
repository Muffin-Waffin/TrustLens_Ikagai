import { motion } from 'framer-motion';
import type { SuspiciousSegment, SuspiciousFrame } from '../services/api';

interface SuspiciousSegmentsProps {
  segments: SuspiciousSegment[];
  suspiciousFrames: SuspiciousFrame[];
}

const container = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const row = {
  hidden: { opacity: 0, x: -12 },
  visible: { opacity: 1, x: 0 },
};

function ScoreBar({ score }: { score: number }) {
  const color = score >= 0.85 ? '#ef4444' : score >= 0.7 ? '#f59e0b' : '#118AB2';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div
        style={{
          width: 80,
          height: 6,
          borderRadius: 3,
          background: '#F2F2ED',
          overflow: 'hidden',
        }}
      >
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${score * 100}%` }}
          transition={{ duration: 0.8 }}
          style={{ height: '100%', borderRadius: 3, background: color }}
        />
      </div>
      <span style={{ fontSize: '0.82rem', fontWeight: 600, color, fontFamily: "'JetBrains Mono', monospace" }}>
        {score.toFixed(3)}
      </span>
    </div>
  );
}

export default function SuspiciousSegments({ segments, suspiciousFrames }: SuspiciousSegmentsProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h3
        style={{
          fontSize: '1.1rem',
          fontWeight: 700,
          color: '#464B71',
          marginBottom: 20,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="#ef4444" strokeWidth="2" />
          <path d="M12 9v4M12 17h.01" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" />
        </svg>
        Suspicious Segments
      </h3>

      {/* Segments table */}
      {segments.length > 0 ? (
        <motion.div variants={container} initial="hidden" animate="visible">
          {/* Header */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1.2fr 0.8fr 0.7fr 1.2fr 1fr',
              gap: 8,
              padding: '8px 14px',
              fontSize: '0.7rem',
              fontWeight: 600,
              color: '#9ca3af',
              textTransform: 'uppercase' as const,
              letterSpacing: '0.06em',
              borderBottom: '1px solid #e5e7eb',
            }}
          >
            <span>Time Range</span>
            <span>Duration</span>
            <span>Frames</span>
            <span>Peak Score</span>
            <span>Mean Score</span>
          </div>

          {segments.map((seg, i) => (
            <motion.div
              key={i}
              variants={row}
              whileHover={{ background: '#f8fafc' }}
              style={{
                display: 'grid',
                gridTemplateColumns: '1.2fr 0.8fr 0.7fr 1.2fr 1fr',
                gap: 8,
                padding: '12px 14px',
                borderBottom: '1px solid #f1f5f9',
                alignItems: 'center',
                borderRadius: 6,
                cursor: 'default',
              }}
            >
              <span style={{ fontWeight: 600, color: '#464B71', fontSize: '0.88rem' }}>
                {(seg.start ?? 0).toFixed(1)}s – {(seg.end ?? 0).toFixed(1)}s
              </span>
              <span style={{ color: '#6b7280', fontSize: '0.85rem' }}>{(seg.duration ?? 0).toFixed(1)}s</span>
              <span style={{ color: '#6b7280', fontSize: '0.85rem' }}>{seg.frame_count ?? 0}</span>
              <ScoreBar score={seg.peak_score ?? 0} />
              <span style={{ fontSize: '0.85rem', color: '#464B71', fontWeight: 500, fontFamily: "'JetBrains Mono', monospace" }}>
                {(seg.mean_score ?? 0).toFixed(3)}
              </span>
            </motion.div>
          ))}
        </motion.div>
      ) : (
        <p style={{ color: '#9ca3af', textAlign: 'center', padding: 20 }}>
          No suspicious segments detected.
        </p>
      )}

      {/* Suspicious frames */}
      {suspiciousFrames.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <div
            style={{
              fontSize: '0.82rem',
              fontWeight: 600,
              color: '#464B71',
              marginBottom: 12,
            }}
          >
            Top Suspicious Frames
          </div>
          <motion.div
            style={{ display: 'flex', flexWrap: 'wrap' as const, gap: 10 }}
            variants={container}
            initial="hidden"
            animate="visible"
          >
            {suspiciousFrames.map((frame, i) => (
              <motion.div
                key={i}
                variants={row}
                whileHover={{ scale: 1.03 }}
                style={{
                  background: '#F2F2ED',
                  borderRadius: 10,
                  padding: '10px 14px',
                  borderLeft: '3px solid #ef4444',
                  minWidth: 140,
                }}
              >
                <div style={{ fontSize: '0.72rem', color: '#9ca3af', marginBottom: 2 }}>
                  Frame #{frame.frame_index}
                </div>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#ef4444', fontFamily: "'JetBrains Mono', monospace" }}>
                  {(frame.score ?? 0).toFixed(3)}
                </div>
                <div style={{ fontSize: '0.72rem', color: '#6b7280' }}>
                  @ {(frame.timestamp_seconds ?? 0).toFixed(1)}s
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      )}
    </motion.div>
  );
}
