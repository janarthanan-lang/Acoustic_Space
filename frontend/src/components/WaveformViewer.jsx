import React, { useEffect, useRef } from 'react'

export default function WaveformViewer({ file }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!file) return
    let cancelled = false

    const draw = async () => {
      const arrayBuffer = await file.arrayBuffer()
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      const ctx = new AudioCtx()
      let audioBuffer
      try {
        audioBuffer = await ctx.decodeAudioData(arrayBuffer)
      } catch {
        ctx.close()
        return
      }
      if (cancelled) { ctx.close(); return }

      const data = audioBuffer.getChannelData(0)
      const canvas = canvasRef.current
      if (!canvas) return
      const width = canvas.clientWidth * 2
      const height = 120
      canvas.width = width
      canvas.height = height
      const g = canvas.getContext('2d')
      g.clearRect(0, 0, width, height)

      const step = Math.ceil(data.length / width)
      const mid = height / 2

      g.strokeStyle = 'rgba(76,224,210,0.15)'
      g.beginPath(); g.moveTo(0, mid); g.lineTo(width, mid); g.stroke()

      g.beginPath()
      g.strokeStyle = '#4ce0d2'
      g.lineWidth = 1.4
      for (let i = 0; i < width; i++) {
        let min = 1.0, max = -1.0
        for (let j = 0; j < step; j++) {
          const idx = i * step + j
          if (idx >= data.length) break
          const v = data[idx]
          if (v < min) min = v
          if (v > max) max = v
        }
        g.moveTo(i, mid + min * mid * 0.9)
        g.lineTo(i, mid + max * mid * 0.9)
      }
      g.stroke()
      ctx.close()
    }

    draw()
    return () => { cancelled = true }
  }, [file])

  return (
    <div className="canvas-wrap">
      <canvas ref={canvasRef} style={{ height: 120 }} />
    </div>
  )
}
