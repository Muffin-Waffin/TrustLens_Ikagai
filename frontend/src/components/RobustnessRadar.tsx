import { motion } from 'framer-motion';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { RobustnessResults } from '../services/api';

interface RobustnessRadarProps {
  robustnessResults: RobustnessResults;
}

export default function RobustnessRadar({ robustnessResults }: RobustnessRadarProps) {
  const { tests, overall_stability, interpretation } = robustnessResults;

  const data = tests.map((t) => ({
    transform: t.transform.replace(/\(.*?\)/g, '').trim(),
    stability: Number(((t.stability ?? 0) * 100).toFixed(1)),
    score: Number(((t.score ?? 0) * 100).toFixed(1)),
    fullMark: 100,
  }));

  const stabilityColor =
    (overall_stability ?? 0) >= 0.85 ? '#22c55e' : (overall_stability ?? 0) >= 0.65 ? '#f59e0b' : '#ef4444';

  if (data.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: '#9ca3af' }}>
        No robustness data available.
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
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
          <path d="M12 2L4 8v8c0 5.52 3.44 10.74 8 12.8 4.56-2.06 8-7.28 8-12.8V8l-8-6z" stroke="#118AB2" strokeWidth="2" />
        </svg>
        Robustness Analysis
      </h3>

      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid stroke="#d1d5db" />
          <PolarAngleAxis
            dataKey="transform"
            tick={{ fill: '#464B71', fontSize: 12, fontWeight: 500 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: '#9ca3af', fontSize: 10 }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div
                  style={{
                    background: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: 10,
                    padding: '10px 14px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  }}
                >
                  <div style={{ fontWeight: 600, color: '#464B71', marginBottom: 4 }}>
                    {d.transform}
                  </div>
                  <div style={{ fontSize: '0.85rem', color: '#118AB2' }}>
                    Stability: {d.stability}%
                  </div>
                </div>
              );
            }}
          />
          <Radar
            name="Stability"
            dataKey="stability"
            stroke="#118AB2"
            fill="#118AB2"
            fillOpacity={0.25}
            strokeWidth={2}
            animationDuration={1000}
          />
        </RadarChart>
      </ResponsiveContainer>

      {/* Overall stability */}
      <div style={{ textAlign: 'center' as const, marginTop: 8 }}>
        <div
          style={{
            fontSize: '0.72rem',
            fontWeight: 600,
            color: '#9ca3af',
            textTransform: 'uppercase' as const,
            letterSpacing: '0.08em',
            marginBottom: 4,
          }}
        >
          Overall Stability
        </div>
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 200, delay: 0.5 }}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 72,
            height: 72,
            borderRadius: '50%',
            background: `${stabilityColor}15`,
            border: `2px solid ${stabilityColor}`,
          }}
        >
          <span style={{ fontSize: '1.3rem', fontWeight: 800, color: stabilityColor }}>
            {((overall_stability ?? 0) * 100).toFixed(0)}%
          </span>
        </motion.div>
        {interpretation && (
          <p style={{ marginTop: 12, fontSize: '0.82rem', color: '#6b7280', maxWidth: 360, margin: '12px auto 0' }}>
            {interpretation}
          </p>
        )}
      </div>
    </motion.div>
  );
}
