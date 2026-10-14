interface Props {
  phase: 'idle' | 'retrieving' | 'debating' | 'arbitrating' | 'done'
}

const PHASES = {
  idle: { label: 'System Ready', color: 'text-emerald-400', dot: 'bg-emerald-400' },
  retrieving: { label: 'Retrieving Vectors', color: 'text-cyan-400', dot: 'bg-cyan-400' },
  debating: { label: 'Agents Deliberating', color: 'text-amber-400', dot: 'bg-amber-400' },
  arbitrating: { label: 'Arbitrating', color: 'text-orange-400', dot: 'bg-orange-400' },
  done: { label: 'Ruling Complete', color: 'text-emerald-400', dot: 'bg-emerald-400' },
}

export function StatusBar({ phase }: Props) {
  const cfg = PHASES[phase]
  const isProcessing = phase !== 'idle' && phase !== 'done'

  return (
    <div className="flex items-center gap-4">
      {/* System indicators */}
      <div className="hidden md:flex items-center gap-3 text-[10px] text-slate-600">
        <span>QDRANT <span className="text-emerald-700">●</span></span>
        <span>NLI <span className="text-emerald-700">●</span></span>
        <span>GROQ <span className="text-emerald-700">●</span></span>
      </div>

      {/* Phase */}
      <div className="flex items-center gap-2 border border-slate-800 rounded px-3 py-1.5 bg-slate-900/60">
        <div className={`w-1.5 h-1.5 rounded-full ${cfg.dot} ${isProcessing ? 'animate-pulse' : ''}`} />
        <span className={`text-[10px] tracking-wider ${cfg.color}`}>{cfg.label}</span>
      </div>
    </div>
  )
}