import { type ReactNode } from 'react';
import { motion } from 'framer-motion';
import type { VideoMetadata } from '../services/api';

interface VideoMetadataCardProps {
  metadata: VideoMetadata;
  videoId: string;
}

interface MetaItem {
  icon: ReactNode;
  label: string;
  value: string;
}

export default function VideoMetadataCard({ metadata, videoId }: VideoMetadataCardProps) {
  const items: MetaItem[] = [
    {
      label: 'Resolution',
      value: `${metadata.width ?? 0} × ${metadata.height ?? 0}`,
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <rect x="2" y="3" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2" />
          <path d="M8 21h8M12 17v4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      ),
    },
    {
      label: 'Frame Rate',
      value: `${(metadata.fps ?? 0).toFixed(1)} FPS`,
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
          <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      ),
    },
    {
      label: 'Duration',
      value: `${(metadata.duration_seconds ?? 0).toFixed(1)}s`,
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M5 3l14 9-14 9V3z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
        </svg>
      ),
    },
    {
      label: 'Codec',
      value: (metadata.codec ?? '').toUpperCase(),
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M16 18l6-6-6-6M8 6l-6 6 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ),
    },
    {
      label: 'Total Frames',
      value: (metadata.frame_count ?? 0).toLocaleString(),
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2" />
          <rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2" />
          <rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2" />
          <rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2" />
        </svg>
      ),
    },
    {
      label: 'Video ID',
      value: videoId,
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M4 7V4h3M20 7V4h-3M4 17v3h3M20 17v3h-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <rect x="7" y="7" width="10" height="10" rx="1" stroke="currentColor" strokeWidth="2" />
        </svg>
      ),
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15 }}
      style={{
        background: '#ffffff',
        borderRadius: 20,
        padding: '28px 24px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04)',
      }}
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
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" stroke="#118AB2" strokeWidth="2" />
          <path d="M14 2v6h6" stroke="#118AB2" strokeWidth="2" />
        </svg>
        Video Information
      </h3>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: 14,
        }}
      >
        {items.map((m, i) => (
          <motion.div
            key={m.label}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + i * 0.06 }}
            style={{
              background: '#F2F2ED',
              borderRadius: 10,
              padding: '12px 14px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, color: '#118AB2' }}>
              {m.icon}
              <span style={{ fontSize: '0.68rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase' as const, letterSpacing: '0.05em' }}>
                {m.label}
              </span>
            </div>
            <div
              style={{
                fontSize: m.label === 'Video ID' ? '0.78rem' : '0.95rem',
                fontWeight: 700,
                color: '#464B71',
                fontFamily: "'JetBrains Mono', monospace",
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap' as const,
              }}
            >
              {m.value}
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
