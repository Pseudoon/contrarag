import { useState } from 'react'
import { QueryPanel } from './components/QueryPanel'
import { DebateStream } from './components/DebateStream'
import { VerdictPanel } from './components/VerdictPanel'
import { SourceInspector } from './components/SourceInspector'
import { StatusBar } from './components/StatusBar'
import type { QueryResult } from './types'

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export default function App() {
  const [result, setResult] = useState<QueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [phase, setPhase] = useState<'idle' | 'retrieving' | 'debating' | 'arbitrating' | 'done'>('idle')

  const handleQuery = async (question: string, jurisdictions: string[]) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setPhase('retrieving')

    const t1 = setTimeout(() => setPhase('debating'), 1500)
    const t2 = setTimeout(() => setPhase('arbitrating'), 4000)

    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, jurisdictions })
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Query failed')
      }

      const data: QueryResult = await res.json()
      setResult(data)
      setPhase('done')
    } catch (e: any) {
      setError(e.message || 'Network error connecting to backend')
      setPhase('idle')
    } finally {
      clearTimeout(t1)
      clearTimeout(t2)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#070B14] text-slate-100 font-mono">
      {/* Grid lines background */}
      <div className="fixed inset-0 opacity-[0.03]" style={{
        backgroundImage: 'linear-gradient(#38BDF8 1px, transparent 1px), linear-gradient(90deg, #38BDF8 1px, transparent 1px)',
        backgroundSize: '48px 48px'
      }} />

      {/* Header */}
      <header className="relative border-b border-slate-800 bg-[#070B14]/90 backdrop-blur-sm px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#38BDF8] animate-pulse" />
          <span className="text-cyan-400 text-xs tracking-[0.3em] uppercase">ContraRAG</span>
          <span className="text-slate-600 text-xs">|</span>
          <span className="text-slate-500 text-xs tracking-widest">Cross-Jurisdictional AI Compliance Engine</span>
        </div>
        <StatusBar phase={phase} />
      </header>

      {/* Main bento grid */}
      <main className="relative p-4 grid grid-cols-12 gap-3 min-h-[calc(100vh-65px)]">

        {/* Left — Query Terminal */}
        <div className="col-span-3">
          <QueryPanel onSubmit={handleQuery} loading={loading} />
        </div>

        {/* Center — Debate Stream */}
        <div className="col-span-6 flex flex-col gap-3">
          <DebateStream result={result} phase={phase} error={error} />
        </div>

        {/* Right — Verdict + Sources */}
        <div className="col-span-3 flex flex-col gap-3">
          <VerdictPanel result={result} phase={phase} />
          <SourceInspector result={result} />
        </div>

      </main>
    </div>
  )
}