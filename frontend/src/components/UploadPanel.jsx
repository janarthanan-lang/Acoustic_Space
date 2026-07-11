import React, { useRef, useState } from 'react'
import RecordPanel from './RecordPanel.jsx'

export default function UploadPanel({ onFileSelected, onAnalyze, file, loading, error }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [mode, setMode] = useState('upload') // 'upload' | 'record'

  const handleFiles = (files) => {
    if (files && files[0]) onFileSelected(files[0])
  }

  return (
    <div className="panel">
      <p className="panel-title">
        <span>01 — Get Audio</span>
        <span>WAV · MP3 · FLAC · MIC</span>
      </p>

      <div className="mode-tabs">
        <button
          className={`mode-tab ${mode === 'upload' ? 'active' : ''}`}
          onClick={() => setMode('upload')}
        >
          Upload File
        </button>
        <button
          className={`mode-tab ${mode === 'record' ? 'active' : ''}`}
          onClick={() => setMode('record')}
        >
          Record Voice
        </button>
      </div>

      {mode === 'upload' && (
        <div
          className={`dropzone ${dragging ? 'dragging' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            handleFiles(e.dataTransfer.files)
          }}
        >
          <div className="dropzone-icon">◈</div>
          <div className="dropzone-label">Drop a suspect clip here, or click to browse</div>
          <div className="dropzone-hint">Max 25MB — analysis runs locally against your API</div>
          <input
            ref={inputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      )}

      {mode === 'record' && (
        <div className="dropzone" style={{ cursor: 'default' }}>
          <RecordPanel onRecordingReady={(f) => onFileSelected(f)} />
        </div>
      )}

      {file && (
        <div className="file-chip">
          ▸ {file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB
        </div>
      )}

      <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
        <button className="btn btn-primary" disabled={!file || loading} onClick={onAnalyze}>
          {loading ? 'Analyzing…' : 'Run Forensic Analysis'}
        </button>
      </div>

      {loading && (
        <div className="loading-line">
          <span className="spinner" />
          Isolating room impulse response & breathing cadence…
        </div>
      )}

      {error && <div className="error-banner">⚠ {error}</div>}
    </div>
  )
}
