import { type ReactNode } from 'react';
import { motion } from 'framer-motion';

interface VerdictCardProps {
  verdict: string;
  manipulationScore: number;
  evidenceConfidence: number;
  reliability: number;
  onAskAi?: (prompt: string) => void;
}

const verdictConfig: Record<string, { color: string; bg: string; label: string; icon: ReactNode }> = {
  REAL: {
    color: '#22c55e',
    bg: 'rgba(34, 197, 94, 0.08)',
    label: 'AUTHENTIC',
    icon: (
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
        <path d="M12 2L4 8v8c0 7.18 5.12 13.88 12 16 6.88-2.12 12-8.82 12-16V8L12 2z" fill="none" stroke="#22c55e" strokeWidth="1.5" />
        <path d="M9 12l2 2 4-4" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  INCONCLUSIVE: {
    color: '#f59e0b',
    bg: 'rgba(245, 158, 11, 0.08)',
    label: 'INCONCLUSIVE',
    icon: (
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="#f59e0b" strokeWidth="1.5" />
        <path d="M12 8v4M12 16h.01" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round" />
      </svg>
    ),
  },
  LIKELY_DEEPFAKE: {
    color: '#ef4444',
    bg: 'rgba(239, 68, 68, 0.08)',
    label: 'DEEPFAKE DETECTED',
    icon: (
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="#ef4444" strokeWidth="1.5" />
        <path d="M12 9v4M12 17h.01" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" />
      </svg>
    ),
  },
};

export default function VerdictCard({ verdict, manipulationScore, evidenceConfidence, reliability, onAskAi }: VerdictCardProps) {
  const cfg = verdictConfig[verdict] || verdictConfig.INCONCLUSIVE;

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 200, damping: 20 }}
      style={{
        background: '#ffffff',
        borderRadius: 20,
        padding: '32px 28px',
        boxShadow: `0 4px 30px ${cfg.color}22, 0 1px 3px rgba(0,0,0,0.08)`,
        border: `1.5px solid ${cfg.color}33`,
        position: 'relative' as const,
        overflow: 'hidden',
      }}
    >
      {/* Glow accent */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 4,
          background: cfg.color,
        }}
      />

      {/* Verdict & Ask AI */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <motion.div
            animate={{ rotate: [0, -5, 5, 0] }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            {cfg.icon}
          </motion.div>
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase' as const, letterSpacing: '0.08em' }}>
              Verdict
            </div>
            <div style={{ fontSize: '1.6rem', fontWeight: 800, color: cfg.color, letterSpacing: '-0.01em' }}>
              {cfg.label}
            </div>
          </div>
        </div>

        {onAskAi && (
          <button
            onClick={() => onAskAi(`Explain why this video was determined to be ${verdict} with manipulation score ${(manipulationScore ?? 0).toFixed(3)} and evidence confidence ${((evidenceConfidence ?? 0) * 100).toFixed(1)}%.`)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: '#f1f5f9',
              border: '1px solid #e2e8f0',
              borderRadius: 10,
              padding: '6px 12px',
              fontSize: '0.78rem',
              fontWeight: 600,
              color: '#118AB2',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#e0f2fe';
              e.currentTarget.style.borderColor = '#7dd3fc';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#f1f5f9';
              e.currentTarget.style.borderColor = '#e2e8f0';
            }}
          >
            <span>✨ Ask AI Why</span>
          </button>
        )}
      </div>

      {/* Score */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase' as const, letterSpacing: '0.08em', marginBottom: 4 }}>
          Manipulation Score
        </div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          style={{ fontSize: '2.8rem', fontWeight: 800, color: '#464B71', lineHeight: 1 }}
        >
          {(manipulationScore ?? 0).toFixed(3)}
        </motion.div>
      </div>

      {/* Evidence Confidence Bar */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#6b7280' }}>Evidence Confidence</span>
          <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#464B71' }}>{((evidenceConfidence ?? 0) * 100).toFixed(1)}%</span>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: '#F2F2ED', overflow: 'hidden' }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${((evidenceConfidence ?? 0) * 100)}%` }}
            transition={{ duration: 1, delay: 0.5, ease: 'easeOut' }}
            style={{ height: '100%', borderRadius: 4, background: cfg.color }}
          />
        </div>
      </div>

      {/* Reliability Bar */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#6b7280' }}>Reliability</span>
          <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#464B71' }}>{((reliability ?? 0) * 100).toFixed(1)}%</span>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: '#F2F2ED', overflow: 'hidden' }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${((reliability ?? 0) * 100)}%` }}
            transition={{ duration: 1, delay: 0.7, ease: 'easeOut' }}
            style={{ height: '100%', borderRadius: 4, background: '#118AB2' }}
          />
        </div>
      </div>
    </motion.div>
  );
}
