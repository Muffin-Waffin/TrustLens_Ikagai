import { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
} from 'recharts';
import type { FrameInfo, SuspiciousSegment } from '../services/api';

interface TimelineChartProps {
  frameInfos: FrameInfo[];
  suspiciousSegments: SuspiciousSegment[];
  meanScore: number;
}

export default function TimelineChart({ frameInfos, suspiciousSegments, meanScore }: TimelineChartProps) {
  const data = useMemo(() => {
    return frameInfos
      .filter((f) => f.usable && f.score !== undefined)
      .map((f) => ({
        time: Number((f.timestamp_seconds ?? 0).toFixed(2)),
        score: Number((f.score ?? 0).toFixed(4)),
        weightedScore: Number(((f.score ?? 0) * (f.weight ?? f.face_quality ?? 1)).toFixed(4)),
        weight: f.weight ?? f.face_quality ?? 1,
      }))
      .sort((a, b) => a.time - b.time);
  }, [frameInfos]);

  if (data.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: '#9ca3af' }}>
        No usable frame data to display.
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
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
          <path d="M3 3v18h18" stroke="#118AB2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M7 16l4-8 4 4 6-8" stroke="#118AB2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Frame-Level Manipulation Scores
      </h3>

      <ResponsiveContainer width="100%" height={350}>
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <defs>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#118AB2" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#118AB2" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />

          <XAxis
            dataKey="time"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(v: number) => `${v.toFixed(1)}s`}
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            axisLine={{ stroke: '#d1d5db' }}
          />
          <YAxis
            domain={[0, 1.05]}
            tickFormatter={(v: number) => v.toFixed(1)}
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            axisLine={{ stroke: '#d1d5db' }}
          />

          {/* Suspicious segments shading */}
          {suspiciousSegments.map((seg, i) => (
            <ReferenceArea
              key={i}
              x1={seg.start}
              x2={seg.end}
              y1={0}
              y2={1.05}
              fill="#ef4444"
              fillOpacity={0.08}
              strokeOpacity={0}
            />
          ))}

          {/* Threshold line */}
          <ReferenceLine
            y={0.7}
            stroke="#ef4444"
            strokeDasharray="6 4"
            strokeWidth={1.5}
            label={{
              value: 'Suspicious (0.7)',
              position: 'right',
              fill: '#ef4444',
              fontSize: 11,
              fontWeight: 600,
            }}
          />

          {/* Mean line */}
          <ReferenceLine
            y={meanScore}
            stroke="#464B71"
            strokeDasharray="3 3"
            strokeWidth={1}
            label={{
              value: `Mean (${(meanScore ?? 0).toFixed(3)})`,
              position: 'right',
              fill: '#464B71',
              fontSize: 11,
              fontWeight: 500,
            }}
          />

          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload;
              const score = d.score as number;
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
                  <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginBottom: 4 }}>
                    Time: {d.time.toFixed(2)}s
                  </div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: score >= 0.7 ? '#ef4444' : '#464B71' }}>
                    Score: {score.toFixed(4)}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#6b7280' }}>Weight: {Number(d.weight).toFixed(3)} · Weighted: {Number(d.weightedScore).toFixed(4)}</div>
                </div>
              );
            }}
          />

          <Area
            type="monotone"
            dataKey="score"
            stroke="#118AB2"
            strokeWidth={2}
            fill="url(#scoreGradient)"
            dot={(props: any) => {
              const weight = Number(props.payload?.weight ?? 1);
              return <circle cx={props.cx} cy={props.cy} r={Math.max(1.5, Math.min(4.5, 1.5 + weight * 3))} fill="#118AB2" fillOpacity={Math.max(0.25, Math.min(1, weight))} stroke="white" strokeWidth={0.5} />;
            }}
            activeDot={{ r: 5, fill: '#118AB2', stroke: 'white', strokeWidth: 2 }}
            animationDuration={1200}
          />
        </AreaChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
