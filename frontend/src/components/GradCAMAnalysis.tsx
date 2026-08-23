import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { GradCAMExplanation } from '../services/api';

interface GradCAMAnalysisProps {
  explanations: GradCAMExplanation[];
}

const COLORMAP_LABELS = {
  jet: 'Jet (Default)',
  hot: 'Hot',
  viridis: 'Viridis',
  plasma: 'Plasma',
  inferno: 'Inferno',
  magma: 'Magma',
} as const;

type ColormapKey = keyof typeof COLORMAP_LABELS;

export default function GradCAMAnalysis({ explanations }: GradCAMAnalysisProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [colormap, setColormap] = useState<ColormapKey>('jet');
  const [viewMode, setViewMode] = useState<'overlay' | 'heatmap' | 'side-by-side'>('overlay');

  const validExplanations = explanations.filter(e => e.overlay_path || e.heatmap_path);
  const current = validExplanations[selectedIdx];

  if (validExplanations.length === 0) {
    return (
      <div style={styles.card}>
        <div style={styles.emptyState}>
          <div style={styles.emptyIcon}>🔬</div>
          <h3 style={styles.emptyTitle}>No Grad-CAM Explanations Available</h3>
          <p style={styles.emptyText}>
            Grad-CAM visualizations are generated for the most suspicious frames when explainability is enabled.
            Run analysis with explainability turned on to see model attention heatmaps.
          </p>
        </div>
      </div>
    );
  }

  const getImageUrl = (path: string | null | undefined) => {
    if (!path) return null;
    // Handle paths like "outputs/explanations/000/frame_000300_overlay.jpg" or "explanations/000/frame_000300_overlay.jpg"
    let cleanPath = path.replace(/\\/g, '/');
    cleanPath = cleanPath.replace(/^outputs\//, '').replace(/^explanations\//, '');
    return `/api/files/explanations/${cleanPath}`;
  };

  const overlaySrc = current?.overlay_path ? getImageUrl(current.overlay_path) : null;
  const heatmapSrc = current?.heatmap_path ? getImageUrl(current.heatmap_path) : null;
  const originalSrc = current?.original_path ? getImageUrl(current.original_path) : null;

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.titleSection}>
          <h3 style={styles.title}>Grad-CAM Attention Analysis</h3>
          <p style={styles.subtitle}>
            Gradient-weighted Class Activation Mapping shows which facial regions influenced the model's deepfake prediction.
          </p>
        </div>
        <div style={styles.badge}>Grad-CAM</div>
      </div>

      <div style={styles.controls}>
        <div style={styles.controlGroup}>
          <label style={styles.controlLabel}>Frame</label>
          <div style={styles.frameSelector}>
            {validExplanations.map((exp, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedIdx(idx)}
                style={{
                  ...styles.frameBtn,
                  ...(idx === selectedIdx ? styles.frameBtnActive : {}),
                }}
                title={`Frame ${exp.frame_index} at ${exp.timestamp_seconds.toFixed(1)}s — Score: ${exp.score.toFixed(3)}`}
              >
                #{idx + 1}
              </button>
            ))}
          </div>
        </div>

        <div style={styles.controlGroup}>
          <label style={styles.controlLabel}>Colormap</label>
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value as ColormapKey)}
            style={styles.select}
          >
            {Object.entries(COLORMAP_LABELS).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </div>

        <div style={styles.controlGroup}>
          <label style={styles.controlLabel}>View Mode</label>
          <div style={styles.modeSelector}>
            {(['overlay', 'heatmap', 'side-by-side'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                style={{
                  ...styles.modeBtn,
                  ...(mode === viewMode ? styles.modeBtnActive : {}),
                }}
              >
                {mode === 'overlay' ? 'Overlay' : mode === 'heatmap' ? 'Heatmap' : 'Side-by-Side'}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={styles.imageContainer}>
        <AnimatePresence mode="wait">
          {viewMode === 'overlay' && overlaySrc && (
            <motion.div
              key="overlay"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              style={styles.imageWrapper}
            >
              <div style={styles.imageLabel}>Overlay (Heatmap + Original)</div>
              <img src={overlaySrc} alt={`Grad-CAM overlay for frame ${current.frame_index}`} style={styles.image} />
            </motion.div>
          )}

          {viewMode === 'heatmap' && heatmapSrc && (
            <motion.div
              key="heatmap"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              style={styles.imageWrapper}
            >
              <div style={styles.imageLabel}>Heatmap Only</div>
              <img src={heatmapSrc} alt={`Grad-CAM heatmap for frame ${current.frame_index}`} style={styles.image} />
            </motion.div>
          )}

          {viewMode === 'side-by-side' && (
            <motion.div
              key="side-by-side"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              style={{ ...styles.imageWrapper, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}
            >
              {originalSrc && (
                <div style={styles.sideBySideItem}>
                  <div style={styles.imageLabel}>Original Face Crop</div>
                  <img src={originalSrc} alt={`Original face crop for frame ${current.frame_index}`} style={styles.image} />
                </div>
              )}
              {heatmapSrc && (
                <div style={styles.sideBySideItem}>
                  <div style={styles.imageLabel}>Heatmap</div>
                  <img src={heatmapSrc} alt={`Grad-CAM heatmap for frame ${current.frame_index}`} style={styles.image} />
                </div>
              )}
              {overlaySrc && (
                <div style={styles.sideBySideItem}>
                  <div style={styles.imageLabel}>Overlay</div>
                  <img src={overlaySrc} alt={`Grad-CAM overlay for frame ${current.frame_index}`} style={styles.image} />
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div style={styles.infoBar}>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Frame Index</span>
          <span style={styles.infoValue}>#{current?.frame_index}</span>
        </div>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Timestamp</span>
          <span style={styles.infoValue}>{current?.timestamp_seconds.toFixed(2)}s</span>
        </div>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Manipulation Score</span>
          <span style={{ ...styles.infoValue, color: current && current.score > 0.7 ? '#ef4444' : current && current.score > 0.4 ? '#f59e0b' : '#22c55e' }}>
            {(current?.score ?? 0).toFixed(3)}
          </span>
        </div>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Method</span>
          <span style={styles.infoValue}>Grad-CAM (ConvNeXt/Xception)</span>
        </div>
      </div>

      <div style={styles.legend}>
        <span style={styles.legendLabel}>Color Scale:</span>
        <div style={styles.legendGradient} />
        <span style={styles.legendLabel}>Low Attention</span>
        <span style={styles.legendLabel}>High Attention</span>
      </div>
    </div>
  );
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
  controls: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 16,
    marginBottom: 20,
    paddingBottom: 16,
    borderBottom: '1px solid #e5e7eb',
  },
  controlGroup: { display: 'flex', flexDirection: 'column', gap: 6 },
  controlLabel: { fontSize: '0.7rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em' },
  frameSelector: { display: 'flex', gap: 6, flexWrap: 'wrap' },
  frameBtn: {
    minWidth: 36,
    height: 32,
    borderRadius: 8,
    border: '1px solid rgba(17, 138, 178, 0.2)',
    background: '#fafafa',
    color: '#464B71',
    fontSize: '0.7rem',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  frameBtnActive: {
    background: 'linear-gradient(135deg, #118AB2 0%, #464B71 100%)',
    borderColor: '#118AB2',
    color: 'white',
    boxShadow: '0 2px 8px rgba(17, 138, 178, 0.3)',
  },
  select: {
    padding: '8px 12px',
    borderRadius: 8,
    border: '1px solid rgba(17, 138, 178, 0.2)',
    background: '#fafafa',
    color: '#464B71',
    fontSize: '0.75rem',
    fontWeight: 500,
    cursor: 'pointer',
    minWidth: 140,
  },
  modeSelector: { display: 'flex', gap: 6 },
  modeBtn: {
    padding: '8px 14px',
    borderRadius: 8,
    border: '1px solid rgba(17, 138, 178, 0.2)',
    background: '#fafafa',
    color: '#464B71',
    fontSize: '0.7rem',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  modeBtnActive: {
    background: 'linear-gradient(135deg, #118AB2 0%, #464B71 100%)',
    borderColor: '#118AB2',
    color: 'white',
  },
  imageContainer: {
    borderRadius: 12,
    overflow: 'hidden',
    background: '#f8f9fa',
    minHeight: 300,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageWrapper: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 12,
    padding: 16,
  },
  sideBySideItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 8,
    width: '100%',
  },
  imageLabel: { fontSize: '0.7rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em' },
  image: { maxWidth: '100%', height: 'auto', borderRadius: 8, boxShadow: '0 4px 16px rgba(0,0,0,0.1)' },
  infoBar: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 20,
    marginTop: 20,
    paddingTop: 16,
    borderTop: '1px solid #e5e7eb',
  },
  infoItem: { display: 'flex', flexDirection: 'column', gap: 4 },
  infoLabel: { fontSize: '0.65rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em' },
  infoValue: { fontSize: '0.85rem', fontWeight: 600, color: '#464B71', fontFamily: "'JetBrains Mono', monospace" },
  legend: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginTop: 16,
    paddingTop: 12,
    borderTop: '1px solid #e5e7eb',
    fontSize: '0.7rem',
    color: '#9ca3af',
  },
  legendGradient: {
    width: 120,
    height: 8,
    borderRadius: 4,
    background: 'linear-gradient(90deg, #464B71 0%, #118AB2 25%, #7CD5C7 50%, #f59e0b 75%, #ef4444 100%)',
  },
  emptyState: {
    textAlign: 'center',
    padding: '48px 24px',
    color: '#9ca3af',
  },
  emptyIcon: { fontSize: '3rem', marginBottom: 12 },
  emptyTitle: { fontSize: '1rem', fontWeight: 600, color: '#464B71', margin: '0 0 8px' },
  emptyText: { fontSize: '0.85rem', lineHeight: 1.6, margin: 0, maxWidth: 400, marginLeft: 'auto', marginRight: 'auto' },
};