import React from 'react'

export default function HistoryPanel({ history }) {
  return (
    <div className="panel">
      <p className="panel-title">
        <span>03 — Analysis History</span>
        <span>{history.length} clip{history.length !== 1 ? 's' : ''}</span>
      </p>
      {history.length === 0 ? (
        <div className="empty-state">No clips analyzed yet this session.</div>
      ) : (
        history.map((h) => (
          <div className="history-row" key={h.id}>
            <span className="history-name">{h.filename}</span>
            <span className={`history-tag ${h.label}`}>
              {h.label.toUpperCase()} · {Math.round(h.confidence * 100)}%
            </span>
          </div>
        ))
      )}
    </div>
  )
}
