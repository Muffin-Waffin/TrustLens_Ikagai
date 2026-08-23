import React, { useState, useEffect, useCallback, type CSSProperties } from 'react';
import { motion, AnimatePresence, type Variants } from 'framer-motion';
import Navbar from './components/Navbar';
import UploadZone from './components/UploadZone';
import VerdictCard from './components/VerdictCard';
import MetricsGrid from './components/MetricsGrid';
import TimelineChart from './components/TimelineChart';
import ScoreDistribution from './components/ScoreDistribution';
import RobustnessRadar from './components/RobustnessRadar';
import SuspiciousSegments from './components/SuspiciousSegments';
import ExplanationPanel from './components/ExplanationPanel';
import VideoMetadataCard from './components/VideoMetadataCard';
import GradCAMAnalysis from './components/GradCAMAnalysis';
import TopSuspiciousFrames from './components/TopSuspiciousFrames';
import ReportExport from './components/ReportExport';
import ChatBot from './components/ChatBot';
import { type AnalysisResult, type AnalysisProgress, analyzeVideo, fetchDemoData } from './services/api';
import './index.css';

const styles: Record<string, CSSProperties> = {
  app: {
    minHeight: '100vh',
    background: '#F2F2ED',
  },
  main: {
    maxWidth: 1400,
    margin: '0 auto',
    padding: '88px 24px 48px',
  },
  hero: {
    textAlign: 'center' as const,
    padding: '60px 20px 40px',
  },
  heroTitle: {
    fontSize: '2.5rem',
    fontWeight: 800,
    color: '#464B71',
    marginBottom: 8,
    letterSpacing: '-0.02em',
  },
  heroSubtitle: {
    fontSize: '1.1rem',
    color: '#6b7280',
    maxWidth: 600,
    margin: '0 auto',
    lineHeight: 1.6,
  },
  heroAccent: {
    color: '#118AB2',
    fontWeight: 600,
  },
  resultsContainer: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 24,
  },
  row: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 24,
  },
  rowSingle: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: 24,
  },
  rowThree: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr',
    gap: 24,
  },
  sectionTitle: {
    fontSize: '1.5rem',
    fontWeight: 700,
    color: '#464B71',
    marginBottom: 16,
    paddingBottom: 8,
    borderBottom: '2px solid #118AB2',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  card: {
    background: '#ffffff',
    borderRadius: 16,
    padding: 24,
    boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04)',
  },
  footer: {
    textAlign: 'center' as const,
    padding: '40px 20px',
    color: '#9ca3af',
    fontSize: '0.85rem',
    borderTop: '1px solid #e5e7eb',
    marginTop: 60,
  },
  newAnalysisBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '10px 24px',
    background: '#464B71',
    color: 'white',
    border: 'none',
    borderRadius: 10,
    fontSize: '0.95rem',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
    marginBottom: 24,
  },
  errorBanner: {
    background: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: 12,
    padding: '16px 24px',
    color: '#dc2626',
    marginBottom: 24,
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
};

// Responsive style overrides
function useResponsiveStyles() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  return {
    isMobile: width < 640,
    isTablet: width >= 640 && width < 1024,
    isDesktop: width >= 1024,
  };
}

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.1 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' as const } },
};

const INITIAL_PROGRESS: AnalysisProgress = {
  percent: 0,
  stage: 'uploading',
  label: 'Uploading video securely',
};

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress>(INITIAL_PROGRESS);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fastMode, setFastMode] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [externalPrompt, setExternalPrompt] = useState<string | null>(null);
  const { isMobile, isTablet } = useResponsiveStyles();

  const handleAskAi = useCallback((prompt: string) => {
    setExternalPrompt(prompt);
    setIsChatOpen(true);
  }, []);

  const handleAnalyze = useCallback(async (file: File) => {
    if (isAnalyzing) return;
    setIsAnalyzing(true);
    setProgress(0);
    setAnalysisProgress({ percent: 0, stage: 'uploading', label: 'Uploading video securely' });
    setError(null);
    try {
      const data = await analyzeVideo(file, (update) => {
        setProgress(update.percent);
        setAnalysisProgress(update);
      }, {
        runRobustness: !fastMode,
        runExplainability: !fastMode,
      });
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
      setProgress(0);
      setAnalysisProgress(INITIAL_PROGRESS);
    }
  }, [isAnalyzing, fastMode]);

  const handleDemo = useCallback(async () => {
    if (isAnalyzing) return;
    setIsAnalyzing(true);
    setError(null);

    // Simulate all 5 steps smoothly in sequential order:
    // Step 1: Uploading (12%)
    // Step 2: Preparing (28%)
    // Step 3: Detecting (48%, 66%)
    // Step 4: Verifying (82%)
    // Step 5: Finalizing (94% -> 100%)
    const demoSteps: AnalysisProgress[] = [
      { percent: 12, stage: 'uploading', label: 'Uploading demo video' },
      { percent: 28, stage: 'preparing', label: 'Extracting & preparing frames' },
      { percent: 48, stage: 'detecting', label: 'Analyzing faces & synthetic signals' },
      { percent: 66, stage: 'detecting', label: 'Scanning manipulation features' },
      { percent: 82, stage: 'verifying', label: 'Checking forensic evidence & consistency' },
      { percent: 94, stage: 'finalizing', label: 'Compiling forensic verdict' },
    ];

    try {
      for (const stepInfo of demoSteps) {
        setProgress(stepInfo.percent);
        setAnalysisProgress(stepInfo);
        await new Promise((r) => setTimeout(r, 280));
      }
      const data = await fetchDemoData();
      setProgress(100);
      setAnalysisProgress({ percent: 100, stage: 'finalizing', label: 'Analysis complete' });
      await new Promise((r) => setTimeout(r, 350));
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load demo data.');
    } finally {
      setIsAnalyzing(false);
      setProgress(0);
      setAnalysisProgress(INITIAL_PROGRESS);
    }
  }, [isAnalyzing]);

  const handleNewAnalysis = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress(0);
    setAnalysisProgress(INITIAL_PROGRESS);
    setIsAnalyzing(false);
  }, []);

  const gridStyle = (cols: number): React.CSSProperties => {
    if (isMobile) return { display: 'grid', gridTemplateColumns: '1fr', gap: 24 };
    if (isTablet && cols > 2)
      return { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 };
    return { display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 24 };
  };

  return (
    <div style={styles.app}>
      <Navbar />

      <main style={styles.main}>
        <AnimatePresence mode="wait">
          {!result ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
            >
              <div style={styles.hero}>
                <h1 style={styles.heroTitle}>
                  Analyze Videos for <span style={styles.heroAccent}>Deepfake</span> Manipulation
                </h1>
                <p style={styles.heroSubtitle}>
                  Upload a video to run forensic analysis using our ConvNeXt-Tiny neural network.
                  Get frame-level manipulation scores, suspicious segment detection, and robustness testing.
                </p>
              </div>

              {error && (
                <motion.div
                  style={styles.errorBanner}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                >
                  <span style={{ fontSize: '1.2rem' }}>⚠️</span>
                  <span>{error}</span>
                </motion.div>
              )}

              <UploadZone
                onAnalyze={handleAnalyze}
                isAnalyzing={isAnalyzing}
                progress={progress}
                analysisProgress={analysisProgress}
                onDemo={handleDemo}
                fastMode={fastMode}
                onFastModeChange={setFastMode}
              />
            </motion.div>
          ) : (
            <motion.div
              key="results"
              style={styles.resultsContainer}
              variants={containerVariants}
              initial="hidden"
              animate="visible"
            >
              {/* Back button */}
              <motion.div variants={itemVariants}>
                <button
                  style={styles.newAnalysisBtn}
                  onClick={handleNewAnalysis}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background = '#118AB2')
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background = '#464B71')
                  }
                >
                  ← New Analysis
                </button>
              </motion.div>

              {/* Verdict + Metadata Row */}
              <motion.div variants={itemVariants} style={gridStyle(2)}>
                <VerdictCard
                  verdict={result.forensic_result.verdict}
                  manipulationScore={result.forensic_result.manipulation_score}
                  evidenceConfidence={result.forensic_result.evidence_confidence}
                  reliability={result.forensic_result.reliability}
                  onAskAi={handleAskAi}
                />
                <VideoMetadataCard
                  metadata={result.video_metadata}
                  videoId={result.video_id}
                />
              </motion.div>

              {/* Metrics Grid */}
              <motion.div variants={itemVariants}>
                <MetricsGrid forensicResult={result.forensic_result} />
              </motion.div>

              {/* Timeline Chart - Full Width */}
              <motion.div variants={itemVariants} style={styles.card}>
                <TimelineChart
                  frameInfos={result.frame_infos}
                  suspiciousSegments={result.forensic_result.suspicious_segments}
                  meanScore={result.forensic_result.mean_score}
                />
              </motion.div>

              {/* Score Distribution + Robustness Radar */}
              <motion.div variants={itemVariants} style={gridStyle(2)}>
                <div style={styles.card}>
                  <ScoreDistribution frameInfos={result.frame_infos} forensicResult={result.forensic_result} />
                </div>
                <div style={styles.card}>
                  <RobustnessRadar robustnessResults={result.robustness_results} />
                </div>
              </motion.div>

              {/* Suspicious Segments + Explanations */}
              <motion.div variants={itemVariants} style={gridStyle(2)}>
                <div style={styles.card}>
                  <SuspiciousSegments
                    segments={result.forensic_result.suspicious_segments}
                    suspiciousFrames={result.forensic_result.suspicious_frames}
                  />
                </div>
                <div style={styles.card}>
                  <ExplanationPanel
                    explanations={result.forensic_result.explanations}
                    onAskAi={handleAskAi}
                  />
                </div>
              </motion.div>

              {/* Grad-CAM Analysis + Top Suspicious Frames */}
              <motion.div variants={itemVariants} style={gridStyle(2)}>
                <div style={styles.card}>
                  <GradCAMAnalysis explanations={result.gradcam_explanations} />
                </div>
                <div style={styles.card}>
                  <TopSuspiciousFrames
                    frameInfos={result.frame_infos}
                    suspiciousFrames={result.forensic_result.suspicious_frames}
                  />
                </div>
              </motion.div>

              {/* Report Export */}
              <motion.div variants={itemVariants}>
                <ReportExport result={result} />
              </motion.div>

              {/* Footer */}
              <motion.div variants={itemVariants} style={styles.footer}>
                <p>Trustlens Forensic Analysis • Video ID: {result.video_id}</p>
                <p style={{ marginTop: 4 }}>
                  Analysis performed at {result.timestamp} • This is a research prototype. Results
                  should not be used as definitive evidence without independent verification.
                </p>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Floating ChatBot Assistant */}
      <ChatBot
        currentResult={result}
        isOpen={isChatOpen}
        onToggleOpen={setIsChatOpen}
        externalPrompt={externalPrompt}
        onClearExternalPrompt={() => setExternalPrompt(null)}
      />
    </div>
  );
}

export default App;
