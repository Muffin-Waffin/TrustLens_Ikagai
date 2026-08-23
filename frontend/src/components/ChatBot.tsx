import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Send,
  Sparkles,
  Settings,
  Trash2,
  Copy,
  Check,
  Bot,
  User,
  ShieldAlert,
  Key,
  ExternalLink,
  Maximize2,
  Minimize2,
} from 'lucide-react';
import {
  type AnalysisResult,
  type ChatMessage,
  type ChatModelOption,
  sendChatMessage,
  fetchChatStatus,
  fetchChatModels,
} from '../services/api';

interface ChatBotProps {
  currentResult: AnalysisResult | null;
  isOpen?: boolean;
  onToggleOpen?: (open: boolean) => void;
  externalPrompt?: string | null;
  onClearExternalPrompt?: () => void;
}

const DEFAULT_MODELS: ChatModelOption[] = [
  {
    id: 'google/gemini-2.0-flash-001',
    name: 'Gemini 2.0 Flash',
    description: 'Fast, high-quality, and cost-effective (Recommended)',
    recommended: true,
  },
  {
    id: 'meta-llama/llama-3.3-70b-instruct',
    name: 'Llama 3.3 70B Instruct',
    description: 'Open-source state-of-the-art reasoning by Meta',
    recommended: false,
  },
  {
    id: 'openai/gpt-4o-mini',
    name: 'GPT-4o Mini',
    description: 'Fast and lightweight model by OpenAI',
    recommended: false,
  },
  {
    id: 'anthropic/claude-3.5-haiku',
    name: 'Claude 3.5 Haiku',
    description: 'Ultra-fast and precise responses by Anthropic',
    recommended: false,
  },
  {
    id: 'deepseek/deepseek-chat',
    name: 'DeepSeek V3',
    description: 'Powerful general-purpose reasoning model',
    recommended: false,
  },
];

// Helper to format simple markdown nicely
function MarkdownRenderer({ content }: { content: string }) {
  // Split into lines/blocks
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let inCodeBlock = false;
  let codeBlockContent: string[] = [];

  const formatInline = (text: string): React.ReactNode => {
    // Basic inline formatting: **bold**, *italic*, `code`
    const parts: React.ReactNode[] = [];
    const regex = /(\*\*.*?\*\*|\*.*?\*|`.*?`)/g;
    let lastIdx = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIdx) {
        parts.push(text.substring(lastIdx, match.index));
      }
      const token = match[0];
      if (token.startsWith('**') && token.endsWith('**')) {
        parts.push(
          <strong key={match.index} style={{ color: '#464B71', fontWeight: 700 }}>
            {token.slice(2, -2)}
          </strong>
        );
      } else if (token.startsWith('*') && token.endsWith('*')) {
        parts.push(
          <em key={match.index} style={{ color: '#4b5563' }}>
            {token.slice(1, -1)}
          </em>
        );
      } else if (token.startsWith('`') && token.endsWith('`')) {
        parts.push(
          <code
            key={match.index}
            style={{
              background: '#e2e8f0',
              color: '#0f172a',
              padding: '2px 6px',
              borderRadius: 4,
              fontSize: '0.82em',
              fontFamily: 'monospace',
            }}
          >
            {token.slice(1, -1)}
          </code>
        );
      }
      lastIdx = regex.lastIndex;
    }

    if (lastIdx < text.length) {
      parts.push(text.substring(lastIdx));
    }

    return parts.length > 0 ? parts : text;
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre
            key={`code-${index}`}
            style={{
              background: '#1e293b',
              color: '#f8fafc',
              padding: '12px 14px',
              borderRadius: 8,
              fontSize: '0.82rem',
              overflowX: 'auto',
              margin: '8px 0',
              fontFamily: 'monospace',
            }}
          >
            <code>{codeBlockContent.join('\n')}</code>
          </pre>
        );
        codeBlockContent = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      codeBlockContent.push(line);
      return;
    }

    if (trimmed.startsWith('### ')) {
      elements.push(
        <h4
          key={index}
          style={{
            fontSize: '0.98rem',
            fontWeight: 700,
            color: '#118AB2',
            margin: '12px 0 6px',
          }}
        >
          {formatInline(trimmed.slice(4))}
        </h4>
      );
    } else if (trimmed.startsWith('## ')) {
      elements.push(
        <h3
          key={index}
          style={{
            fontSize: '1.05rem',
            fontWeight: 700,
            color: '#464B71',
            margin: '14px 0 6px',
          }}
        >
          {formatInline(trimmed.slice(3))}
        </h3>
      );
    } else if (trimmed.startsWith('# ')) {
      elements.push(
        <h2
          key={index}
          style={{
            fontSize: '1.15rem',
            fontWeight: 800,
            color: '#464B71',
            margin: '16px 0 8px',
          }}
        >
          {formatInline(trimmed.slice(2))}
        </h2>
      );
    } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      elements.push(
        <li
          key={index}
          style={{
            marginLeft: 20,
            marginBottom: 4,
            lineHeight: 1.5,
            color: '#374151',
          }}
        >
          {formatInline(trimmed.slice(2))}
        </li>
      );
    } else if (/^\d+\.\s/.test(trimmed)) {
      const dotIndex = trimmed.indexOf('.');
      elements.push(
        <div
          key={index}
          style={{
            display: 'flex',
            gap: 8,
            marginBottom: 4,
            lineHeight: 1.5,
            color: '#374151',
          }}
        >
          <span style={{ fontWeight: 700, color: '#118AB2' }}>
            {trimmed.slice(0, dotIndex + 1)}
          </span>
          <span>{formatInline(trimmed.slice(dotIndex + 1).trim())}</span>
        </div>
      );
    } else if (trimmed === '') {
      elements.push(<div key={index} style={{ height: 6 }} />);
    } else {
      elements.push(
        <p
          key={index}
          style={{
            margin: '4px 0',
            lineHeight: 1.55,
            color: '#374151',
          }}
        >
          {formatInline(line)}
        </p>
      );
    }
  });

  return <div style={{ fontSize: '0.88rem' }}>{elements}</div>;
}

export default function ChatBot({
  currentResult,
  isOpen: controlledIsOpen,
  onToggleOpen,
  externalPrompt,
  onClearExternalPrompt,
}: ChatBotProps) {
  const [internalIsOpen, setInternalIsOpen] = useState(false);
  const isOpen = controlledIsOpen !== undefined ? controlledIsOpen : internalIsOpen;

  const setIsOpen = (open: boolean) => {
    if (onToggleOpen) {
      onToggleOpen(open);
    } else {
      setInternalIsOpen(open);
    }
  };

  const [messages, setMessages] = useState<ChatMessage[]>(() => [
    {
      role: 'assistant',
      content:
        "👋 **Welcome to Trustlens AI Forensic Assistant!**\n\nI'm connected via **OpenRouter** to answer any questions about deepfake detection, model architecture, or your video forensic report.\n\nHow can I assist you today?",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [apiKey, setApiKey] = useState<string>(() => localStorage.getItem('trustlens_openrouter_key') || '');
  const [selectedModel, setSelectedModel] = useState<string>(
    () => localStorage.getItem('trustlens_openrouter_model') || 'google/gemini-2.0-flash-001'
  );
  const [models, setModels] = useState<ChatModelOption[]>(DEFAULT_MODELS);
  const [backendConfigured, setBackendConfigured] = useState<boolean>(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [includeContext, setIncludeContext] = useState<boolean>(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Load chat status & models from backend
  useEffect(() => {
    fetchChatStatus()
      .then((status) => {
        setBackendConfigured(status.configured);
        if (status.available_models && status.available_models.length > 0) {
          setModels(status.available_models);
        }
      })
      .catch(() => {
        // backend might still be loading
      });

    fetchChatModels()
      .then((fetchedModels) => {
        if (fetchedModels && fetchedModels.length > 0) {
          setModels(fetchedModels);
        }
      })
      .catch(() => {});
  }, []);

  // Handle external prompts (e.g. clicking "Ask AI" on Verdict Card)
  useEffect(() => {
    if (externalPrompt) {
      setIsOpen(true);
      setInput(externalPrompt);
      if (onClearExternalPrompt) onClearExternalPrompt();
      setTimeout(() => {
        inputRef.current?.focus();
      }, 300);
    }
  }, [externalPrompt, onClearExternalPrompt]);

  const handleSendMessage = async (textToSend?: string) => {
    const query = (textToSend || input).trim();
    if (!query || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: query };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      // Send chat request to backend
      const response = await sendChatMessage({
        messages: newMessages,
        context: includeContext && currentResult ? (currentResult as unknown as Record<string, unknown>) : null,
        api_key: apiKey.trim() || undefined,
        model: selectedModel || undefined,
      });

      setMessages((prev) => [...prev, response.message]);
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to communicate with AI Assistant.';
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ **Error:** ${errorMsg}\n\n*Tip:* Click the **⚙️ Settings** icon in the header to provide or update your OpenRouter API key.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content:
          "🧹 **Conversation cleared.**\n\nAsk any question regarding deepfake detection, forensic analysis, or your loaded report!",
      },
    ]);
  };

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleSaveSettings = () => {
    localStorage.setItem('trustlens_openrouter_key', apiKey);
    localStorage.setItem('trustlens_openrouter_model', selectedModel);
    setShowSettings(false);
  };

  // Quick suggestions based on state
  const suggestions = currentResult
    ? [
        { label: '💡 Summarize this report', prompt: 'Please summarize the forensic findings of this video analysis report.' },
        { label: `🔍 Why is verdict ${currentResult.forensic_result.verdict}?`, prompt: `Explain in detail why this video received a verdict of ${currentResult.forensic_result.verdict} with a manipulation score of ${currentResult.forensic_result.manipulation_score.toFixed(3)}.` },
        { label: '⏱️ Suspicious timestamps', prompt: 'Which exact timestamps or video segments are flagged as suspicious and what anomalies were detected there?' },
        { label: '📊 Boundary & FFT Analysis', prompt: 'Explain what the boundary artifact score and frequency anomaly score indicate for this video.' },
        { label: '🛡️ Robustness Stability', prompt: 'How robust was the model when this video underwent compression, blurring, and resizing transformations?' },
        { label: '📋 Draft formal audit report', prompt: 'Draft a formal forensic video analysis report summary suitable for presentation to an executive audit team.' },
      ]
    : [
        { label: '🧠 How does Trustlens work?', prompt: 'Can you explain the end-to-end architecture and pipeline of Trustlens video deepfake forensics?' },
        { label: '🔬 Multi-signal forensic checks', prompt: 'What specific forensic signals does Trustlens evaluate (e.g. FFT, boundary artifacts, blinks, identity drift)?' },
        { label: '📐 Neural Network & Models', prompt: 'Which neural network backbones (ConvNeXt-Tiny, InsightFace) are used in Trustlens and why?' },
        { label: '👁️ Blink & Eye Dynamics', prompt: 'How does Trustlens analyze Eye Aspect Ratio (EAR) and blink dynamics to detect deepfakes?' },
      ];

  return (
    <>
      {/* Floating Action Button */}
      <motion.div
        style={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 1100,
        }}
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 260, damping: 20 }}
      >
        <button
          onClick={() => setIsOpen(!isOpen)}
          style={{
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: isOpen ? '12px 16px' : '14px 22px',
            background: 'linear-gradient(135deg, #118AB2 0%, #0d6efd 100%)',
            color: 'white',
            border: 'none',
            borderRadius: 50,
            boxShadow: '0 8px 30px rgba(17, 138, 178, 0.4), 0 2px 8px rgba(0, 0, 0, 0.1)',
            cursor: 'pointer',
            fontSize: '0.95rem',
            fontWeight: 600,
            transition: 'all 0.25s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-2px) scale(1.03)';
            e.currentTarget.style.boxShadow = '0 12px 35px rgba(17, 138, 178, 0.55)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'none';
            e.currentTarget.style.boxShadow = '0 8px 30px rgba(17, 138, 178, 0.4)';
          }}
        >
          {/* Glowing pulse indicator */}
          <span
            style={{
              position: 'absolute',
              top: -2,
              right: -2,
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: '#7CD5C7',
              border: '2px solid white',
              boxShadow: '0 0 10px #7CD5C7',
            }}
          />
          {isOpen ? (
            <>
              <X size={20} />
              <span>Close Assistant</span>
            </>
          ) : (
            <>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(255,255,255,0.2)',
                  borderRadius: '50%',
                  padding: 4,
                }}
              >
                <Sparkles size={18} />
              </div>
              <span>Ask AI Investigator</span>
              {currentResult && (
                <span
                  style={{
                    fontSize: '0.72rem',
                    background: '#464B71',
                    padding: '2px 8px',
                    borderRadius: 10,
                    fontWeight: 700,
                    letterSpacing: '0.03em',
                  }}
                >
                  Report Active
                </span>
              )}
            </>
          )}
        </button>
      </motion.div>

      {/* Expandable Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 30, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 30, scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 300, damping: 28 }}
            style={{
              position: 'fixed',
              bottom: 84,
              right: 24,
              width: isExpanded ? 'min(780px, calc(100vw - 48px))' : 'min(460px, calc(100vw - 32px))',
              height: isExpanded ? 'min(820px, calc(100vh - 120px))' : 'min(640px, calc(100vh - 110px))',
              maxHeight: 'calc(100vh - 100px)',
              background: '#ffffff',
              borderRadius: 20,
              boxShadow: '0 20px 60px rgba(70, 75, 113, 0.25), 0 0 0 1px rgba(70, 75, 113, 0.08)',
              display: 'flex',
              flexDirection: 'column',
              zIndex: 1100,
              overflow: 'hidden',
              transition: 'width 0.3s ease, height 0.3s ease',
            }}
          >
            {/* Header */}
            <div
              style={{
                background: 'linear-gradient(135deg, #464B71 0%, #2f334d 100%)',
                color: 'white',
                padding: '16px 20px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid rgba(255,255,255,0.1)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div
                  style={{
                    position: 'relative',
                    width: 38,
                    height: 38,
                    borderRadius: 12,
                    background: 'linear-gradient(135deg, #118AB2 0%, #7CD5C7 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 4px 12px rgba(17, 138, 178, 0.3)',
                  }}
                >
                  <Bot size={22} color="white" />
                  <span
                    style={{
                      position: 'absolute',
                      bottom: -1,
                      right: -1,
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      background: '#22c55e',
                      border: '2px solid #464B71',
                    }}
                  />
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span>Trustlens AI</span>
                    <span
                      style={{
                        fontSize: '0.65rem',
                        background: 'rgba(255,255,255,0.15)',
                        padding: '1px 6px',
                        borderRadius: 6,
                        color: '#7CD5C7',
                        fontWeight: 600,
                      }}
                    >
                      OpenRouter
                    </span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#cbd5e1', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span>Model: {selectedModel.split('/').pop() || 'Gemini 2.0 Flash'}</span>
                  </div>
                </div>
              </div>

              {/* Header Action Buttons */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  title="Configure OpenRouter API Key & Model"
                  style={{
                    background: showSettings ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.1)',
                    border: 'none',
                    borderRadius: 8,
                    color: 'white',
                    padding: 7,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 0.2s',
                  }}
                >
                  <Settings size={17} />
                </button>
                <button
                  onClick={handleClearChat}
                  title="Clear Chat History"
                  style={{
                    background: 'rgba(255,255,255,0.1)',
                    border: 'none',
                    borderRadius: 8,
                    color: 'white',
                    padding: 7,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 0.2s',
                  }}
                >
                  <Trash2 size={17} />
                </button>
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  title={isExpanded ? 'Collapse' : 'Expand'}
                  style={{
                    background: 'rgba(255,255,255,0.1)',
                    border: 'none',
                    borderRadius: 8,
                    color: 'white',
                    padding: 7,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 0.2s',
                  }}
                >
                  {isExpanded ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
                </button>
                <button
                  onClick={() => setIsOpen(false)}
                  title="Close"
                  style={{
                    background: 'rgba(255,255,255,0.1)',
                    border: 'none',
                    borderRadius: 8,
                    color: 'white',
                    padding: 7,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 0.2s',
                  }}
                >
                  <X size={17} />
                </button>
              </div>
            </div>

            {/* Active Context Banner */}
            {currentResult && (
              <div
                style={{
                  background: '#f1f5f9',
                  padding: '8px 16px',
                  borderBottom: '1px solid #e2e8f0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  fontSize: '0.78rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#475569', overflow: 'hidden' }}>
                  <ShieldAlert size={15} color="#118AB2" />
                  <span style={{ fontWeight: 600 }}>Active Case:</span>
                  <span
                    style={{
                      fontFamily: 'monospace',
                      background: '#e2e8f0',
                      padding: '1px 6px',
                      borderRadius: 4,
                      maxWidth: 120,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {currentResult.video_id}
                  </span>
                  <span
                    style={{
                      fontWeight: 700,
                      color:
                        currentResult.forensic_result.verdict === 'REAL'
                          ? '#16a34a'
                          : currentResult.forensic_result.verdict === 'LIKELY_DEEPFAKE'
                          ? '#dc2626'
                          : '#d97706',
                    }}
                  >
                    [{currentResult.forensic_result.verdict}]
                  </span>
                </div>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    cursor: 'pointer',
                    fontSize: '0.74rem',
                    color: '#64748b',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={includeContext}
                    onChange={(e) => setIncludeContext(e.target.checked)}
                    style={{ accentColor: '#118AB2', cursor: 'pointer' }}
                  />
                  <span>Attach Data</span>
                </label>
              </div>
            )}

            {/* Settings Overlay Drawer */}
            <AnimatePresence>
              {showSettings && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  style={{
                    background: '#f8fafc',
                    borderBottom: '1px solid #e2e8f0',
                    padding: 16,
                    fontSize: '0.85rem',
                    overflow: 'hidden',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                    <div style={{ fontWeight: 700, color: '#1e293b', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Key size={16} color="#118AB2" />
                      <span>OpenRouter Configuration</span>
                    </div>
                    <a
                      href="https://openrouter.ai/keys"
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        color: '#118AB2',
                        fontSize: '0.78rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 3,
                        textDecoration: 'none',
                        fontWeight: 600,
                      }}
                    >
                      <span>Get API Key</span>
                      <ExternalLink size={12} />
                    </a>
                  </div>

                  {/* API Key input */}
                  <div style={{ marginBottom: 12 }}>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: 4 }}>
                      OpenRouter API Key {backendConfigured && !apiKey && '(Using Backend Default)'}
                    </label>
                    <input
                      type="password"
                      placeholder="sk-or-v1-..."
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        borderRadius: 8,
                        border: '1px solid #cbd5e1',
                        fontSize: '0.85rem',
                        outline: 'none',
                      }}
                    />
                    <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: 4 }}>
                      Key is stored securely in your local browser session.
                    </div>
                  </div>

                  {/* Model Selector */}
                  <div style={{ marginBottom: 14 }}>
                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: 4 }}>
                      Select LLM Model
                    </label>
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        borderRadius: 8,
                        border: '1px solid #cbd5e1',
                        fontSize: '0.85rem',
                        background: 'white',
                        outline: 'none',
                      }}
                    >
                      {models.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name} {m.recommended ? '⭐ (Recommended)' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
                    <button
                      onClick={() => setShowSettings(false)}
                      style={{
                        padding: '6px 14px',
                        borderRadius: 6,
                        border: '1px solid #cbd5e1',
                        background: 'white',
                        color: '#475569',
                        fontSize: '0.8rem',
                        cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSaveSettings}
                      style={{
                        padding: '6px 16px',
                        borderRadius: 6,
                        border: 'none',
                        background: '#118AB2',
                        color: 'white',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      Save Settings
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Messages Scroll Area */}
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '16px 18px',
                display: 'flex',
                flexDirection: 'column',
                gap: 16,
                background: '#fafafa',
              }}
            >
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                    maxWidth: '100%',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      gap: 8,
                      alignItems: 'flex-start',
                      maxWidth: msg.role === 'user' ? '88%' : '94%',
                      flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                    }}
                  >
                    {/* Role Icon */}
                    <div
                      style={{
                        width: 28,
                        height: 28,
                        borderRadius: '50%',
                        background: msg.role === 'user' ? '#464B71' : '#118AB2',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        flexShrink: 0,
                        marginTop: 2,
                        boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
                      }}
                    >
                      {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                    </div>

                    {/* Bubble Content */}
                    <div
                      style={{
                        background: msg.role === 'user' ? '#464B71' : '#ffffff',
                        color: msg.role === 'user' ? '#ffffff' : '#1e293b',
                        padding: '12px 16px',
                        borderRadius: 14,
                        borderTopRightRadius: msg.role === 'user' ? 2 : 14,
                        borderTopLeftRadius: msg.role === 'user' ? 14 : 2,
                        boxShadow: msg.role === 'user' ? '0 2px 8px rgba(70, 75, 113, 0.2)' : '0 2px 10px rgba(0,0,0,0.06)',
                        border: msg.role === 'user' ? 'none' : '1px solid #e2e8f0',
                        fontSize: '0.88rem',
                        lineHeight: 1.55,
                        position: 'relative',
                        wordBreak: 'break-word',
                      }}
                    >
                      {msg.role === 'user' ? (
                        <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                      ) : (
                        <div>
                          <MarkdownRenderer content={msg.content} />
                          {/* Copy Button */}
                          <button
                            onClick={() => handleCopy(msg.content, i)}
                            title="Copy response"
                            style={{
                              position: 'absolute',
                              top: 8,
                              right: 8,
                              background: 'rgba(0,0,0,0.04)',
                              border: 'none',
                              borderRadius: 6,
                              padding: 4,
                              cursor: 'pointer',
                              color: '#94a3b8',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              transition: 'all 0.2s',
                            }}
                            onMouseEnter={(e) => (e.currentTarget.style.color = '#118AB2')}
                            onMouseLeave={(e) => (e.currentTarget.style.color = '#94a3b8')}
                          >
                            {copiedIndex === i ? <Check size={13} color="#22c55e" /> : <Copy size={13} />}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}

              {/* Typing / Loading animation */}
              {loading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                >
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: '50%',
                      background: '#118AB2',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                    }}
                  >
                    <Bot size={16} />
                  </div>
                  <div
                    style={{
                      background: '#ffffff',
                      padding: '12px 18px',
                      borderRadius: 14,
                      border: '1px solid #e2e8f0',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
                    }}
                  >
                    <div style={{ display: 'flex', gap: 4 }}>
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{ repeat: Infinity, duration: 0.6, delay: 0 }}
                        style={{ width: 7, height: 7, borderRadius: '50%', background: '#118AB2' }}
                      />
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{ repeat: Infinity, duration: 0.6, delay: 0.15 }}
                        style={{ width: 7, height: 7, borderRadius: '50%', background: '#118AB2' }}
                      />
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{ repeat: Infinity, duration: 0.6, delay: 0.3 }}
                        style={{ width: 7, height: 7, borderRadius: '50%', background: '#118AB2' }}
                      />
                    </div>
                    <span style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 500 }}>
                      Consulting OpenRouter...
                    </span>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Quick Prompts Suggestions */}
            <div
              style={{
                padding: '8px 14px',
                background: '#ffffff',
                borderTop: '1px solid #e2e8f0',
                display: 'flex',
                gap: 6,
                overflowX: 'auto',
                whiteSpace: 'nowrap',
              }}
            >
              {suggestions.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSendMessage(s.prompt)}
                  disabled={loading}
                  style={{
                    padding: '5px 12px',
                    borderRadius: 20,
                    border: '1px solid #e2e8f0',
                    background: '#f8fafc',
                    color: '#464B71',
                    fontSize: '0.74rem',
                    fontWeight: 600,
                    cursor: loading ? 'not-allowed' : 'pointer',
                    transition: 'all 0.15s ease',
                    flexShrink: 0,
                  }}
                  onMouseEnter={(e) => {
                    if (!loading) {
                      e.currentTarget.style.background = '#e0f2fe';
                      e.currentTarget.style.borderColor = '#7dd3fc';
                      e.currentTarget.style.color = '#0369a1';
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = '#f8fafc';
                    e.currentTarget.style.borderColor = '#e2e8f0';
                    e.currentTarget.style.color = '#464B71';
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>

            {/* Input Bar */}
            <div
              style={{
                padding: '12px 14px',
                background: '#ffffff',
                borderTop: '1px solid #e2e8f0',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-end',
                  gap: 8,
                  background: '#f1f5f9',
                  borderRadius: 14,
                  padding: '6px 8px 6px 12px',
                  border: '1.5px solid transparent',
                  transition: 'border-color 0.2s',
                }}
                onFocus={(e) => (e.currentTarget.style.borderColor = '#118AB2')}
                onBlur={(e) => (e.currentTarget.style.borderColor = 'transparent')}
              >
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={currentResult ? 'Ask about this report or deepfakes...' : 'Ask about Trustlens project...'}
                  rows={1}
                  style={{
                    flex: 1,
                    border: 'none',
                    background: 'transparent',
                    resize: 'none',
                    outline: 'none',
                    fontSize: '0.88rem',
                    maxHeight: 90,
                    color: '#1e293b',
                    fontFamily: 'inherit',
                    lineHeight: 1.4,
                    paddingTop: 4,
                  }}
                />
                <button
                  onClick={() => handleSendMessage()}
                  disabled={!input.trim() || loading}
                  style={{
                    background: input.trim() && !loading ? '#118AB2' : '#cbd5e1',
                    color: 'white',
                    border: 'none',
                    borderRadius: 10,
                    width: 34,
                    height: 34,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
                    transition: 'all 0.2s',
                    flexShrink: 0,
                  }}
                >
                  <Send size={16} />
                </button>
              </div>

              <div
                style={{
                  fontSize: '0.68rem',
                  color: '#94a3b8',
                  textAlign: 'center',
                  marginTop: 6,
                }}
              >
                Trustlens AI can make mistakes. Verify critical forensic findings independently.
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
