import React from 'react'

function Gauge({ fakeProbability, label }) {
  const size = 180
  const stroke = 10
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const pct = fakeProbability
  const isFake = label === 'fake'
  const color = isFake ? '#ff9f45' : '#4ce0d2'

  // ticks around the dial, oscilloscope-style
  const ticks = Array.from({ length: 24 }, (_, i) => i)

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <g transform={`translate(${size / 2},${size / 2})`}>
        {ticks.map((i) => {
          const angle = (i / ticks.length) * 360
          const active = angle <= pct * 360
          return (
            <line
              key={i}
              x1={0} y1={-(r + 6)} x2={0} y2={-(r + 12)}
              stroke={active ? color : '#232833'}
              strokeWidth={2}
              transform={`rotate(${angle})`}
            />
          )
        })}
      </g>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#1a1e27" strokeWidth={stroke} />
      <circle
        cx={size/2} cy={size/2} r={r} fill="none"
        stroke={color} strokeWidth={stroke}
        strokeDasharray={`${c * pct} ${c}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${size/2} ${size/2})`}
        style={{ transition: 'stroke-dasharray 0.6s ease' }}
      />
      <text x="50%" y="46%" textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="30" fill="#e8ebf0" fontWeight="500">
        {Math.round(pct * 100)}%
      </text>
      <text x="50%" y="60%" textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="10" fill="#7c8494" letterSpacing="1">
        FAKE PROBABILITY
      </text>
    </svg>
  )
}

export default function ResultsPanel({ result }) {
  if (!result) {
    return (
      <div className="panel">
        <p className="panel-title"><span>02 — Verdict</span></p>
        <div className="empty-state">Upload and analyze a clip to see results.</div>
      </div>
    )
  }

  const { label, fake_probability, verdict_reason, rir_features, breathing_features, duration_sec, model_status } = result

  return (
    <div className="panel">
      <p className="panel-title">
        <span>02 — Verdict</span>
        <span>{duration_sec}s clip</span>
      </p>

      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 2 }}>
        <span className={`history-tag ${model_status === 'fine-tuned' ? 'real' : 'fake'}`} style={{ fontSize: 9.5 }}>
          {model_status === 'fine-tuned' ? 'RIR + AST MODEL' : 'RIR HEURISTIC ONLY'}
        </span>
      </div>

      <div className="gauge-wrap">
        <Gauge fakeProbability={fake_probability} label={label} />
        <div className={`verdict-label ${label}`}>
          {label === 'fake' ? '⚠ Likely Synthetic' : '✓ Likely Genuine'}
        </div>
        <div className="verdict-reason">{verdict_reason}</div>
      </div>

      <div className="readout-grid">
        <div className="readout-cell">
          <div className="readout-key">RT60 Estimate</div>
          <div className="readout-val">{rir_features.rt60_estimate_sec}s</div>
        </div>
        <div className="readout-cell">
          <div className="readout-key">Spectral Decay Slope</div>
          <div className="readout-val">{rir_features.spectral_decay_slope}</div>
        </div>
        <div className="readout-cell">
          <div className="readout-key">Noise Floor</div>
          <div className="readout-val">{rir_features.noise_floor_db} dB</div>
        </div>
        <div className="readout-cell">
          <div className="readout-key">Breath Gap Regularity</div>
          <div className="readout-val">{breathing_features.breath_gap_regularity}</div>
        </div>
      </div>
    </div>
  )
}
