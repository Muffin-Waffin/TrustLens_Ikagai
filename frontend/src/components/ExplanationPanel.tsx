import { motion } from 'framer-motion';

interface ExplanationPanelProps {
  explanations: string[];
  onAskAi?: (prompt: string) => void;
}

const container = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.12, delayChildren: 0.2 } },
};

const item = {
  hidden: { opacity: 0, x: -16 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.4 } },
};

export default function ExplanationPanel({ explanations, onAskAi }: ExplanationPanelProps) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <h3
          style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: '#464B71',
            margin: 0,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" stroke="#118AB2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Forensic Analysis
        </h3>

        {onAskAi && (
          <button
            onClick={() => onAskAi('Can you elaborate on the forensic explanations provided by the engine and give a breakdown of the detected anomalies?')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              background: '#f1f5f9',
              border: '1px solid #e2e8f0',
              borderRadius: 8,
              padding: '4px 10px',
              fontSize: '0.75rem',
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
            <span>💬 Elaborate with AI</span>
          </button>
        )}
      </div>

      {explanations.length > 0 ? (
        <motion.div
          variants={container}
          initial="hidden"
          animate="visible"
          style={{ display: 'flex', flexDirection: 'column' as const, gap: 10 }}
        >
          {explanations.map((text, i) => (
            <motion.div
              key={i}
              variants={item}
              style={{
                background: '#F2F2ED',
                borderRadius: 10,
                padding: '14px 16px',
                borderLeft: '3px solid #118AB2',
                display: 'flex',
                alignItems: 'flex-start',
                gap: 12,
              }}
            >
              <div
                style={{
                  minWidth: 24,
                  height: 24,
                  borderRadius: '50%',
                  background: '#118AB2',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginTop: 1,
                }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path d="M13 16h-1v-4h-1m1-4h.01" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
                </svg>
              </div>
              <p style={{ fontSize: '0.88rem', color: '#464B71', lineHeight: 1.55, margin: 0 }}>
                {text}
              </p>
            </motion.div>
          ))}
        </motion.div>
      ) : (
        <p style={{ color: '#9ca3af', textAlign: 'center', padding: 20 }}>
          No explanations generated.
        </p>
      )}
    </div>
  );
}
