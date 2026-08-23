import { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { FrameInfo, ForensicResult } from '../services/api';

interface ScoreDistributionProps {
  frameInfos: FrameInfo[];
  forensicResult?: ForensicResult;
}

const BIN_COLORS = [
  '#7CD5C7', // 0.0-0.1
  '#6ECFBF', // 0.1-0.2
  '#5DC5B2', // 0.2-0.3
  '#50B8A0', // 0.3-0.4
  '#6BB88E', // 0.4-0.5
  '#A3B86C', // 0.5-0.6
  '#C9A84C', // 0.6-0.7
  '#E09436', // 0.7-0.8
  '#E87030', // 0.8-0.9
  '#ef4444', // 0.9-1.0
];

export default function ScoreDistribution({ frameInfos, forensicResult }: ScoreDistributionProps) {
  const data = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({
      range: `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`,
      count: 0,
      weightedCount: 0,
      binIndex: i,
    }));

    frameInfos
      .filter((f) => f.usable && f.score !== undefined)
      .forEach((f) => {
        const idx = Math.min(Math.floor((f.score ?? 0) * 10), 9);
        bins[idx].count++;
        bins[idx].weightedCount += f.weight ?? f.face_quality ?? 1;
      });

    return bins;
  }, [frameInfos]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
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
          <path d="M18 20V10M12 20V4M6 20v-6" stroke="#118AB2" strokeWidth="2" strokeLinecap="round" />
        </svg>
        Raw and Quality-Weighted Score Distribution
      </h3>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
          <XAxis
            dataKey="range"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            axisLine={{ stroke: '#d1d5db' }}
            interval={0}
            angle={-30}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            axisLine={{ stroke: '#d1d5db' }}
            allowDecimals={false}
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
                  <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginBottom: 2 }}>
                    Range: {d.range}
                  </div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: '#464B71' }}>
                    {d.count} frame{d.count !== 1 ? 's' : ''}
                  </div>
                </div>
              );
            }}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} animationDuration={1000}>
            {data.map((entry) => (
              <Cell key={entry.range} fill={BIN_COLORS[entry.binIndex]} />
            ))}
          </Bar>
          <Bar dataKey="weightedCount" fill="#118AB2" fillOpacity={0.35} radius={[4, 4, 0, 0]} animationDuration={1000} />
        </BarChart>
      </ResponsiveContainer>
      {forensicResult && <p style={{ color: '#6b7280', fontSize: '0.8rem', marginTop: 8 }}>
        Raw mean {forensicResult.mean_score.toFixed(3)} ± {forensicResult.std_score.toFixed(3)}; raw median {forensicResult.raw_median_score.toFixed(3)}; weighted median {forensicResult.weighted_median_score.toFixed(3)}.
      </p>}
    </motion.div>
  );
}
