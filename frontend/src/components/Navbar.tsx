export default function Navbar() {
  return (
    <nav
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: 64,
        background: 'linear-gradient(135deg, #464B71 0%, #3a3f62 50%, #464B71 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 32px',
        zIndex: 1000,
        boxShadow: '0 2px 20px rgba(70, 75, 113, 0.35)',
      }}
    >
      {/* Left: Logo + Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
          <path
            d="M16 2L4 8v8c0 7.18 5.12 13.88 12 16 6.88-2.12 12-8.82 12-16V8L16 2z"
            fill="#118AB2"
            opacity="0.9"
          />
          <path
            d="M16 5L7 9.5v6.5c0 5.74 3.84 11.1 9 12.8 5.16-1.7 9-7.06 9-12.8V9.5L16 5z"
            fill="#7CD5C7"
            opacity="0.6"
          />
          <path
            d="M14.5 17.5l-3-3 1.41-1.41L14.5 14.67l4.59-4.59L20.5 11.5l-6 6z"
            fill="white"
          />
        </svg>
        <div>
          <div
            style={{
              color: 'white',
              fontSize: '1.15rem',
              fontWeight: 700,
              letterSpacing: '-0.01em',
              lineHeight: 1.2,
            }}
          >
            Trustlens
          </div>
          <div
            style={{
              color: '#7CD5C7',
              fontSize: '0.65rem',
              fontWeight: 500,
              letterSpacing: '0.08em',
              textTransform: 'uppercase' as const,
            }}
          >
            Video Deepfake Forensics
          </div>
        </div>
      </div>
    </nav>
  );
}
