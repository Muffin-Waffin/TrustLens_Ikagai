import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import type { AnalysisProgress } from '../services/api';

interface UploadZoneProps {
  onAnalyze: (file: File) => void;
  isAnalyzing: boolean;
  progress: number;
  analysisProgress: AnalysisProgress;
  onDemo: () => void;
  fastMode: boolean;
  onFastModeChange: (enabled: boolean) => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadZone({
  onAnalyze,
  isAnalyzing,
  progress,
  analysisProgress,
  onDemo,
  fastMode,
  onFastModeChange,
}: UploadZoneProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) setSelectedFile(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.mov', '.avi', '.webm', '.mkv'] },
    maxFiles: 1,
    disabled: isAnalyzing,
  });

  const handleAnalyze = () => {
    if (selectedFile && !isAnalyzing) onAnalyze(selectedFile);
  };

  // Circular progress ring
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <div style={{ maxWidth: 680, margin: '0 auto' }}>
      <motion.div
        {...(getRootProps() as Record<string, unknown>)}
        whileHover={!isAnalyzing ? { scale: 1.01 } : {}}
        whileTap={!isAnalyzing ? { scale: 0.99 } : {}}
        style={{
          border: `2px dashed ${isDragActive ? '#118AB2' : '#464B71'}`,
          borderRadius: 20,
          padding: isAnalyzing ? '40px 32px' : '48px 32px',
          textAlign: 'center' as const,
          cursor: isAnalyzing ? 'default' : 'pointer',
          background: isDragActive ? 'rgba(17, 138, 178, 0.04)' : '#ffffff',
          transition: 'all 0.3s ease',
          boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
        }}
      >
        <input {...getInputProps()} />

        {isAnalyzing ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            {/* Circular progress */}
            <svg width="100" height="100" style={{ margin: '0 auto', display: 'block' }}>
              <circle
                cx="50" cy="50" r={radius}
                fill="none" stroke="#F2F2ED" strokeWidth="6"
              />
              <motion.circle
                cx="50" cy="50" r={radius}
                fill="none" stroke="#118AB2" strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={circumference}
                animate={{ strokeDashoffset }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                style={{
                  transformOrigin: '50% 50%',
                  transform: 'rotate(-90deg)',
                }}
              />
              <text
                x="50" y="50"
                textAnchor="middle"
                dominantBaseline="central"
                style={{ fontSize: 18, fontWeight: 700, fill: '#464B71' }}
              >
                {progress}%
              </text>
            </svg>
            <p style={{ marginTop: 16, color: '#464B71', fontWeight: 600, fontSize: '1rem' }}>
              {analysisProgress.label}
            </p>
            <p style={{ color: '#9ca3af', fontSize: '0.85rem', marginTop: 4 }}>
              Step {Math.max(1, (['uploading', 'preparing', 'detecting', 'verifying', 'finalizing'] as const).indexOf(analysisProgress.stage) + 1)} of 5 · This may take a minute depending on video length
            </p>
            <div style={{ display: 'flex', gap: 6, justifyContent: 'center', marginTop: 18 }} aria-label={`Analysis progress: ${analysisProgress.label}`}>
              {(['uploading', 'preparing', 'detecting', 'verifying', 'finalizing'] as const).map((stage, index) => {
                const activeIndex = (['uploading', 'preparing', 'detecting', 'verifying', 'finalizing'] as const).indexOf(analysisProgress.stage);
                return <span key={stage} style={{ width: 42, height: 5, borderRadius: 6, background: index <= activeIndex ? '#118AB2' : '#dce1e6', transition: 'background 0.3s ease' }} />;
              })}
            </div>
          </motion.div>
        ) : selectedFile ? (
          <div>
            {/* File icon */}
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" style={{ margin: '0 auto 16px', display: 'block' }}>
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" stroke="#118AB2" strokeWidth="1.5" fill="none" />
              <path d="M14 2v6h6M10 13l2 2 4-4" stroke="#118AB2" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p style={{ fontWeight: 600, color: '#464B71', fontSize: '1.05rem' }}>
              {selectedFile.name}
            </p>
            <p style={{ color: '#9ca3af', fontSize: '0.85rem', marginTop: 4 }}>
              {formatSize(selectedFile.size)}
            </p>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12, justifyContent: 'center', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={fastMode}
                onChange={(e) => onFastModeChange(e.target.checked)}
                disabled={isAnalyzing}
                style={{ width: 18, height: 18, accentColor: '#118AB2' }}
              />
              <span style={{ fontSize: '0.85rem', color: '#464B71', fontWeight: 500 }}>
                Fast Mode (skip robustness & explainability)
              </span>
            </label>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 20 }}>
              <motion.button
                whileHover={!isAnalyzing ? { scale: 1.03 } : {}}
                whileTap={!isAnalyzing ? { scale: 0.97 } : {}}
                disabled={isAnalyzing}
                onClick={(e) => { e.stopPropagation(); handleAnalyze(); }}
                style={{
                  padding: '12px 32px',
                  background: isAnalyzing ? '#9ca3af' : '#118AB2',
                  color: 'white',
                  border: 'none',
                  borderRadius: 12,
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                  boxShadow: isAnalyzing ? 'none' : '0 4px 14px rgba(17, 138, 178, 0.35)',
                }}
              >
                🔍 Analyze Video
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }}
                style={{
                  padding: '12px 24px',
                  background: 'transparent',
                  color: '#464B71',
                  border: '1.5px solid #464B71',
                  borderRadius: 12,
                  fontSize: '0.95rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                }}
              >
                Clear
              </motion.button>
            </div>
          </div>
        ) : (
          <div>
            {/* Upload cloud icon */}
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" style={{ margin: '0 auto 16px', display: 'block' }}>
              <path
                d="M12 16V8m0 0l3 3m-3-3l-3 3"
                stroke={isDragActive ? '#118AB2' : '#464B71'}
                strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
              />
              <path
                d="M20 16.7c1.2-.9 2-2.3 2-3.9 0-2.8-2.2-5-5-5-.3 0-.6 0-.9.1C15 5.6 13.2 4 11 4 8.2 4 6 6.2 6 9c0 .3 0 .7.1 1C4.3 10.5 3 12.1 3 14c0 2.2 1.8 4 4 4h13"
                stroke={isDragActive ? '#118AB2' : '#464B71'}
                strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
              />
            </svg>
            <p style={{ fontWeight: 600, color: '#464B71', fontSize: '1.1rem' }}>
              {isDragActive ? 'Drop your video here!' : 'Drop your video here or click to browse'}
            </p>
            <p style={{ color: '#9ca3af', fontSize: '0.85rem', marginTop: 8 }}>
              Supports MP4, MOV, AVI, WebM, MKV
            </p>
          </div>
        )}
      </motion.div>

      {/* Demo button */}
      {!isAnalyzing && !selectedFile && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          style={{ textAlign: 'center' as const, marginTop: 20 }}
        >
          <span style={{ color: '#9ca3af', fontSize: '0.85rem', marginRight: 8 }}>
            No video handy?
          </span>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onDemo}
            style={{
              padding: '8px 24px',
              background: '#7CD5C7',
              color: '#1a1a2e',
              border: 'none',
              borderRadius: 10,
              fontSize: '0.85rem',
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: '0 3px 12px rgba(124, 213, 199, 0.3)',
            }}
          >
            ▶ Try Demo
          </motion.button>
        </motion.div>
      )}
    </div>
  );
}
