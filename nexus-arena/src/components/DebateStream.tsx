import type { QueryResult } from '../types'

interface Props {
  result: QueryResult | null
  phase: 'idle' | 'retrieving' | 'debating' | 'arbitrating' | 'done'
  error: string | null
}

const PHASE_MESSAGES = {
  idle: null,
  retrieving: 'Querying Qdrant vector store...',
  debating: 'Regional Counsel agents deliberating...',
  arbitrating: 'Supreme Arbitrator evaluating contradictions...',
  done: null,
}

function AgentCard({
  flag, name, color, borderColor, bgColor, response
}: {
  flag: string, name: string, color: string, borderColor: string, bgColor: string, response: string
}) {
  return (
    <div className={`rounded-lg border ${borderColor} ${bgColor} p-4`}>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">{flag}</span>
        <span className={`text-[10px] font-bold tracking-[0.2em] uppercase ${color}`}>{name}</span>
        <div className={`ml-auto w-1.5 h-1.5 rounded-full ${color.replace('text-', 'bg-')} animate-pulse`} />
      </div>
      <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{response}</p>
    </div>
  )
}

function Skeleton() {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 animate-pulse">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-6 h-6 rounded bg-slate-800" />
        <div className="w-32 h-3 rounded bg-slate-800" />
      </div>
      <div className="space-y-2">
        <div className="w-full h-3 rounded bg-slate-800" />
        <div className="w-5/6 h-3 rounded bg-slate-800" />
        <div className="w-4/6 h-3 rounded bg-slate-800" />
        <div className="w-full h-3 rounded bg-slate-800" />
        <div className="w-3/6 h-3 rounded bg-slate-800" />
      </div>
    </div>
  )
}

export function DebateStream({ result, phase, error }: Props) {
  const msg = PHASE_MESSAGES[phase]

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Header */}
      <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 backdrop-blur-sm px-4 py-3 flex items-center justify-between">
        <span className="text-[10px] tracking-[0.25em] text-cyan-500 uppercase">Arbitration Stream</span>
        {msg && (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse shadow-[0_0_6px_#FBBf24]" />
            <span className="text-[10px] text-amber-400">{msg}</span>
          </div>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className={`rounded-lg border p-4 ${
          error.includes('outside the scope')
            ? 'border-slate-700 bg-slate-900/60'
            : 'border-red-900 bg-red-950/40'
        }`}>
          <p className={`text-[10px] uppercase tracking-widest mb-1 ${
            error.includes('outside the scope') ? 'text-slate-500' : 'text-red-500'
          }`}>
            {error.includes('outside the scope') ? '⊘ Out of Scope' : '⚠ System Error'}
          </p>
          <p className={`text-sm ${error.includes('outside the scope') ? 'text-slate-400' : 'text-red-400'}`}>
            {error}
          </p>
        </div>
      )}

      {/* Idle state */}
      {phase === 'idle' && !error && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-8 flex flex-col items-center justify-center text-center flex-1">
          <div className="w-12 h-12 rounded-full border border-slate-700 flex items-center justify-center mb-4">
            <span className="text-2xl">⚖️</span>
          </div>
          <p className="text-slate-500 text-sm">Awaiting query input</p>
          <p className="text-slate-700 text-xs mt-1">Select jurisdictions and enter a compliance question</p>
        </div>
      )}

      {/* Loading skeletons */}
      {(phase === 'retrieving' || phase === 'debating') && (
        <>
          <Skeleton />
          <Skeleton />
        </>
      )}

      {/* Arbitrating — show EU done, US loading */}
      {phase === 'arbitrating' && result && (
        <>
          <AgentCard
            flag="🇪🇺" name="EU Regional Counsel"
            color="text-blue-400" borderColor="border-blue-900/60" bgColor="bg-blue-950/20"
            response={result.eu_response}
          />
          <Skeleton />
        </>
      )}

      {/* Done — full results */}
      {phase === 'done' && result && (
        <>
          {/* Contradiction FIRST — summary at top */}
          <div className={`rounded-lg border p-4 ${
            result.verdict.decision === 'REFUSE'
              ? 'border-red-900/60 bg-red-950/20'
              : result.verdict.decision === 'WARN'
              ? 'border-amber-900/60 bg-amber-950/20'
              : 'border-emerald-900/60 bg-emerald-950/20'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[10px] font-bold tracking-widest uppercase text-slate-400">Contradiction Analysis</span>
              <span className={`ml-auto text-[10px] font-bold px-2 py-0.5 rounded ${
                result.verdict.decision === 'REFUSE'
                  ? 'bg-red-900/50 text-red-400'
                  : result.verdict.decision === 'WARN'
                  ? 'bg-amber-900/50 text-amber-400'
                  : 'bg-emerald-900/50 text-emerald-400'
              }`}>
                {result.verdict.decision === 'REFUSE'
                  ? '⚡ CONFLICT DETECTED'
                  : result.verdict.decision === 'WARN'
                  ? '⚠ ADVISORY'
                  : '✓ NO CONFLICT'}
              </span>
            </div>
            <p className="text-sm text-slate-400">{result.contradiction.explanation}</p>
            <div className="mt-2 flex items-center gap-2">
              <span className="text-[10px] text-slate-600">NLI Score</span>
              <div className="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${
                    result.contradiction.nli_score > 0.5 ? 'bg-amber-500' : 'bg-emerald-500'
                  }`}
                  style={{ width: `${result.contradiction.nli_score * 100}%` }}
                />
              </div>
              <span className="text-[10px] text-slate-400 font-mono">{result.contradiction.nli_score.toFixed(3)}</span>
            </div>
          </div>

          {/* Sub-questions */}
          {result.is_complex && (
            <div className="rounded-lg border border-cyan-900/50 bg-cyan-950/20 px-4 py-3">
              <p className="text-[10px] text-cyan-600 uppercase tracking-widest mb-2">Query Decomposed into {result.sub_questions.length} vectors</p>
              <div className="flex flex-wrap gap-2">
                {result.sub_questions.map((q, i) => (
                  <span key={i} className="text-[10px] bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-400">
                    {i + 1}. {q}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Agent responses AFTER — for deeper reading */}
          <AgentCard
            flag="🇪🇺" name="EU Regional Counsel"
            color="text-blue-400" borderColor="border-blue-900/60" bgColor="bg-blue-950/20"
            response={result.eu_response}
          />

          <AgentCard
            flag="🇺🇸" name="US Regional Counsel"
            color="text-red-400" borderColor="border-red-900/60" bgColor="bg-red-950/20"
            response={result.us_response}
          />

          <AgentCard
            flag="🇬🇧" name="UK Regional Counsel"
            color="text-violet-400" borderColor="border-violet-900/60" bgColor="bg-violet-950/20"
            response={result.uk_response}
          />
        </>
      )}
    </div>
  )
}