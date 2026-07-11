import React, { useRef, useState, useEffect } from 'react'

const MAX_SECONDS = 30

export default function RecordPanel({ onRecordingReady }) {
  const [status, setStatus] = useState('idle') // idle | requesting | recording | recorded
  const [seconds, setSeconds] = useState(0)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [errorMsg, setErrorMsg] = useState(null)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => () => {
    // cleanup mic stream + timer if the component unmounts mid-recording
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop())
    if (timerRef.current) clearInterval(timerRef.current)
  }, [])

  const startRecording = async () => {
    setErrorMsg(null)
    setStatus('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'
      const recorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType })
        const url = URL.createObjectURL(blob)
        setPreviewUrl(url)
        const file = new File([blob], `mic-recording-${Date.now()}.webm`, { type: mimeType })
        onRecordingReady(file)
        setStatus('recorded')
        stream.getTracks().forEach((t) => t.stop())
      }

      recorder.start()
      setStatus('recording')
      setSeconds(0)
      timerRef.current = setInterval(() => {
        setSeconds((s) => {
          if (s + 1 >= MAX_SECONDS) {
            stopRecording()
            return MAX_SECONDS
          }
          return s + 1
        })
      }, 1000)
    } catch (err) {
      setStatus('idle')
      setErrorMsg('Microphone access denied or unavailable. Check browser permissions.')
    }
  }

  const stopRecording = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
  }

  const reRecord = () => {
    setPreviewUrl(null)
    setStatus('idle')
    setSeconds(0)
  }

  const fmt = (s) => `0:${String(s).padStart(2, '0')}`

  return (
    <div>
      {status !== 'recorded' && (
        <div className="record-stage">
          <button
            className={`record-button ${status === 'recording' ? 'is-recording' : ''}`}
            onClick={status === 'recording' ? stopRecording : startRecording}
            disabled={status === 'requesting'}
          >
            <span className="record-dot" />
          </button>
          <div className="record-label">
            {status === 'idle' && 'Tap to start recording'}
            {status === 'requesting' && 'Requesting microphone…'}
            {status === 'recording' && `Recording… ${fmt(seconds)} / 0:${MAX_SECONDS}`}
          </div>
          {status === 'recording' && (
            <div className="record-hint">Tap again to stop, or it auto-stops at {MAX_SECONDS}s</div>
          )}
        </div>
      )}

      {status === 'recorded' && previewUrl && (
        <div className="record-stage">
          <audio controls src={previewUrl} style={{ width: '100%', marginBottom: 14 }} />
          <div style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
            <button className="btn btn-ghost" onClick={reRecord}>Re-record</button>
          </div>
        </div>
      )}

      {errorMsg && <div className="error-banner">⚠ {errorMsg}</div>}
    </div>
  )
}
