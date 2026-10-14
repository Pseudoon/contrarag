import type { QueryResult } from '../types'

interface Props {
  result: QueryResult | null
  phase: string
}

const DECISION_CONFIG = {
  ANSWER: {
    label: 'COMPLIANT',
    color: 'text-emerald-400',
    border: 'border-emerald-800',
    bg: 'bg-emerald-950/30',
    glow: 'shadow-[0_0_20px_rgba(52,211,153,0.15)]',
    icon: '✓',
    bar: 'bg-emerald-500'
  },
  WARN: {
    label: 'ADVISORY',
    color: 'text-amber-400',
    border: 'border-amber-800',
    bg: 'bg-amber-950/30',
    glow: 'shadow-[0_0_20px_rgba(251,191,36,0.15)]',
    icon: '⚠',
    bar: 'bg-amber-500'
  },
  REFUSE: {
    label: 'DEADLOCK',
    color: 'text-red-400',
    border: 'border-red-800',
    bg: 'bg-red-950/30',
    glow: 'shadow-[0_0_20px_rgba(248,113,113,0.15)]',
    icon: '✕',
    bar: 'bg-red-500'
  }
}

export function VerdictPanel({ result, phase }: Props) {
  if (phase === 'idle' || (!result && phase !== 'done')) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
        <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-4">Supreme Arbitrator</p>
        <div className="flex items-center justify-center h-24">
          <p className="text-slate-700 text-xs">Awaiting deliberation</p>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 animate-pulse">
        <div className="w-32 h-3 bg-slate-800 rounded mb-4" />
        <div className="w-full h-16 bg-slate-800 rounded mb-3" />
        <div className="w-full h-3 bg-slate-800 rounded mb-2" />
        <div className="w-4/5 h-3 bg-slate-800 rounded" />
      </div>
    )
  }

  const decision = result.verdict.decision
  const cfg = DECISION_CONFIG[decision]
  const confidence = result.verdict.confidence
  const friction = result.verdict.friction_score

  return (
    <div className={`rounded-lg border ${cfg.border} ${cfg.bg} ${cfg.glow} p-4 flex flex-col gap-4`}>
      <p className="text-[10px] text-slate-500 uppercase tracking-widest">Supreme Arbitrator</p>

      {/* Decision badge */}
      <div className="text-center py-3 border-b border-slate-800">
        <div className={`text-3xl font-black tracking-widest ${cfg.color} mb-1`}>
          {cfg.icon} {cfg.label}
        </div>
        <div className="text-xs text-slate-500">Final Ruling</div>
      </div>

      {/* Confidence */}
      <div>
        <div className="flex justify-between text-[10px] text-slate-500 mb-1.5 uppercase tracking-wider">
          <span>Confidence</span>
          <span className={cfg.color}>{(confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${cfg.bar} transition-all duration-700`}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Friction score */}
      <div>
        <div className="flex justify-between text-[10px] text-slate-500 mb-1.5 uppercase tracking-wider">
          <span>Policy Discord</span>
          <span className="text-slate-300 font-mono">{friction} / 10</span>
        </div>
        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${friction > 5 ? 'bg-red-500' : friction > 3 ? 'bg-amber-500' : 'bg-emerald-500'} transition-all duration-700`}
            style={{ width: `${friction * 10}%` }}
          />
        </div>
      </div>

      {/* Decision reason */}
      <div className="border-t border-slate-800 pt-3">
        <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-1.5">Arbitrator Reasoning</p>
        <p className="text-xs text-slate-400 leading-relaxed">{result.verdict.decision_reason}</p>
      </div>

      {/* Regional summaries */}
      <div className="border-t border-slate-800 pt-3">
        <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-2">Jurisdiction Rulings</p>
        <div className="flex flex-col gap-2">
          {(['EU', 'US', 'UK'] as const).map(region => {
            const colors = { EU: 'text-blue-400 border-blue-900/50', US: 'text-red-400 border-red-900/50', UK: 'text-violet-400 border-violet-900/50' }
            const flags = { EU: '🇪🇺', US: '🇺🇸', UK: '🇬🇧' }
            return (
              <div key={region} className={`border rounded p-2 border-slate-800`}>
                <span className={`text-[10px] font-bold ${colors[region].split(' ')[0]}`}>
                  {flags[region]} {region}
                </span>
                <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
                  {result.verdict.regional_summaries[region] || 'No data available'}
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}