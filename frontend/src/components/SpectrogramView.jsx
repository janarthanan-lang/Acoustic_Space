import React, { useEffect, useRef } from 'react'

// dB value -> forensic-lab heatmap color (dark -> cyan -> amber for hot spots)
function colorForDb(v, min, max) {
  const t = Math.max(0, Math.min(1, (v - min) / (max - min || 1)))
  if (t < 0.5) {
    const k = t / 0.5
    const r = Math.round(10 + k * (30 - 10))
    const g = Math.round(12 + k * (60 - 12))
    const b = Math.round(18 + k * (70 - 18))
    return `rgb(${r},${g},${b})`
  }
  const k = (t - 0.5) / 0.5
  const r = Math.round(30 + k * (255 - 30))
  const g = Math.round(60 + k * (159 - 60))
  const b = Math.round(70 + k * (69 - 70))
  return `rgb(${r},${g},${b})`
}

export default function SpectrogramView({ mel }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!mel || mel.length === 0) return
    const canvas = canvasRef.current
    if (!canvas) return

    const nMels = mel.length
    const nFrames = mel[0].length
    const width = canvas.clientWidth * 2
    const height = 160
    canvas.width = width
    canvas.height = height
    const g = canvas.getContext('2d')

    let min = Infinity, max = -Infinity
    for (const row of mel) for (const v of row) { if (v < min) min = v; if (v > max) max = v }

    const cellW = width / nFrames
    const cellH = height / nMels

    for (let m = 0; m < nMels; m++) {
      for (let f = 0; f < nFrames; f++) {
        g.fillStyle = colorForDb(mel[m][f], min, max)
        g.fillRect(f * cellW, height - (m + 1) * cellH, cellW + 1, cellH + 1)
      }
    }
  }, [mel])

  return (
    <div className="canvas-wrap">
      <canvas ref={canvasRef} style={{ height: 160 }} />
    </div>
  )
}
