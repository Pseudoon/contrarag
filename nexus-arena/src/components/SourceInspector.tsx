import { useState } from 'react'
import type { QueryResult } from '../types'

interface Props {
  result: QueryResult | null
}

export function SourceInspector({ result }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null)

  if (!result) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
        <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-3">Provenance Inspector</p>
        <p className="text-slate-700 text-xs text-center py-4">No sources retrieved</p>
      </div>
    )
  }

  const jColors: Record<string, string> = {
    EU: 'text-blue-400 bg-blue-950/50 border-blue-900',
    US: 'text-red-400 bg-red-950/50 border-red-900',
    UK: 'text-violet-400 bg-violet-950/50 border-violet-900',
  }

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] text-slate-500 uppercase tracking-widest">Provenance Inspector</p>
        <span className="text-[10px] text-slate-600">{result.sources.length} chunks</span>
      </div>

      <div className="flex flex-col gap-2 max-h-72 overflow-y-auto pr-1">
        {result.sources.map((source, i) => {
          const colorClass = jColors[source.jurisdiction] || 'text-slate-400 bg-slate-800 border-slate-700'
          const isOpen = expanded === i
          const scorePercent = Math.round(source.rerank_score * 100)

          return (
            <div
              key={i}
              className="border border-slate-800 rounded bg-slate-950/60 overflow-hidden"
            >
              <button
                onClick={() => setExpanded(isOpen ? null : i)}
                className="w-full text-left p-2.5 flex items-center gap-2 hover:bg-slate-800/40 transition-colors"
              >
                <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${colorClass}`}>
                  {source.jurisdiction}
                </span>
                <span className="text-[10px] text-slate-400 flex-1 truncate">{source.source}</span>
                <span className="text-[10px] text-slate-600 font-mono">{source.rerank_score.toFixed(3)}</span>
                <span className={`text-slate-600 text-[10px] transition-transform ${isOpen ? 'rotate-90' : ''}`}>›</span>
              </button>

              {/* Relevance bar */}
              <div className="px-2.5 pb-2">
                <div className="h-0.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-700 rounded-full"
                    style={{ width: `${Math.min(scorePercent + 50, 100)}%` }}
                  />
                </div>
              </div>

              {/* Expanded chunk text */}
              {isOpen && (
                <div className="px-2.5 pb-2.5 border-t border-slate-800 pt-2">
                  {source.sub_query && (
                    <p className="text-[9px] text-cyan-700 mb-1.5">Retrieved for: {source.sub_query}</p>
                  )}
                  <p className="text-[10px] text-slate-500 leading-relaxed">{source.text.slice(0, 400)}...</p>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}