import React, { useState } from 'react'
import UploadPanel from './components/UploadPanel.jsx'
import WaveformViewer from './components/WaveformViewer.jsx'
import SpectrogramView from './components/SpectrogramView.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'
import HistoryPanel from './components/HistoryPanel.jsx'

// Points at the FastAPI backend. In dev, Vite proxies /api -> localhost:8000
// (see vite.config.js). Set VITE_API_BASE to override for other deployments.
const API_BASE = import.meta.env.VITE_API_BASE || '/api'

export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])

  const handleFileSelected = (f) => {
    setFile(f)
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Server returned ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
      setHistory((h) => [{ ...data }, ...h].slice(0, 20))
    } catch (e) {
      setError(e.message || 'Analysis failed — is the backend running on :8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="brand">
          <span className="brand-mark">AST · RIR</span>
          <div>
            <h1>AcousticSpace</h1>
            <div className="brand-sub">deepfake detection via room impulse response</div>
          </div>
        </div>
        <div className="status-pill">
          <span className="status-dot" />
          analyst dashboard
        </div>
      </div>

      <div className="layout">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <UploadPanel
            file={file}
            loading={loading}
            error={error}
            onFileSelected={handleFileSelected}
            onAnalyze={handleAnalyze}
          />

          {file && (
            <div className="panel">
              <p className="panel-title"><span>Waveform</span><span>time domain</span></p>
              <WaveformViewer file={file} />

              {result?.mel_spectrogram && (
                <>
                  <p className="panel-title" style={{ marginTop: 18 }}>
                    <span>Mel Spectrogram</span><span>0–8kHz</span>
                  </p>
                  <SpectrogramView mel={result.mel_spectrogram} />
                </>
              )}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <ResultsPanel result={result} />
          <HistoryPanel history={history} />
        </div>
      </div>
    </div>
  )
}
