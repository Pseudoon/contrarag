import { useState } from 'react'

interface Props {
  onSubmit: (question: string, jurisdictions: string[]) => void
  loading: boolean
}

const JURISDICTIONS = ['EU', 'US', 'UK']

const EXAMPLE_QUERIES = [
  'Is registration of high-risk AI systems mandatory?',
  'What are transparency requirements for AI providers?',
  'How do EU and US approaches to AI risk differ?',
  'What penalties exist for AI Act violations?',
]

export function QueryPanel({ onSubmit, loading }: Props) {
  const [question, setQuestion] = useState('')
  const [selected, setSelected] = useState(['EU', 'US', 'UK'])

  const toggle = (j: string) => {
    setSelected(prev =>
      prev.includes(j) ? prev.filter(x => x !== j) : [...prev, j]
    )
  }

  const submit = () => {
    if (!question.trim() || selected.length === 0 || loading) return
    onSubmit(question.trim(), selected)
  }

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Panel */}
      <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 backdrop-blur-sm p-4 flex flex-col gap-4">
        {/* Label */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] tracking-[0.25em] text-cyan-500 uppercase">Command Terminal</span>
        </div>

        {/* Jurisdiction selector */}
        <div>
          <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">Jurisdictions</p>
          <div className="flex gap-2">
            {JURISDICTIONS.map(j => (
              <button
                key={j}
                onClick={() => toggle(j)}
                className={`flex-1 py-1.5 rounded text-xs font-bold tracking-wider border transition-all duration-200
                  ${selected.includes(j)
                    ? j === 'EU' ? 'bg-blue-500/20 border-blue-500 text-blue-400'
                      : j === 'US' ? 'bg-red-500/20 border-red-500 text-red-400'
                      : 'bg-violet-500/20 border-violet-500 text-violet-400'
                    : 'bg-transparent border-slate-700 text-slate-600 hover:border-slate-500'
                  }`}
              >
                {j}
              </button>
            ))}
          </div>
        </div>

        {/* Query input */}
        <div>
          <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">Query Vector</p>
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) submit() }}
            placeholder="Enter your compliance question..."
            rows={5}
            className="w-full bg-slate-950/80 border border-slate-700 rounded p-3 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:border-cyan-600 transition-colors"
          />
          <p className="text-[10px] text-slate-600 mt-1">⌘ + Enter to submit</p>
        </div>

        {/* Submit */}
        <button
          onClick={submit}
          disabled={loading || !question.trim() || selected.length === 0}
          className={`w-full py-2.5 rounded text-xs font-bold tracking-[0.15em] uppercase transition-all duration-200 border
            ${loading
              ? 'border-cyan-800 text-cyan-800 cursor-not-allowed'
              : 'border-cyan-500 text-cyan-400 hover:bg-cyan-500/10 hover:shadow-[0_0_16px_rgba(56,189,248,0.2)]'
            }`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-3 h-3 border border-cyan-700 border-t-cyan-400 rounded-full animate-spin" />
              Processing...
            </span>
          ) : '⚡ Initiate Audit'}
        </button>
      </div>

      {/* Example queries */}
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
        <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-3">Example Queries</p>
        <div className="flex flex-col gap-2">
          {EXAMPLE_QUERIES.map((q, i) => (
            <button
              key={i}
              onClick={() => setQuestion(q)}
              className="text-left text-xs text-slate-500 hover:text-slate-300 transition-colors py-1 border-b border-slate-800/50 last:border-0"
            >
              <span className="text-cyan-800 mr-1">›</span> {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}