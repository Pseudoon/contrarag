export interface Source {
  text: string
  source: string
  jurisdiction: string
  score: number
  rerank_score: number
  sub_query: string
}

export interface Contradiction {
  found: boolean
  nli_score: number
  explanation: string
  max_pair: [string, string] | null
}

export interface Verdict {
  decision: 'ANSWER' | 'WARN' | 'REFUSE'
  confidence: number
  friction_score: number
  decision_reason: string
  regional_summaries: {
    EU: string
    US: string
    UK: string
  }
}

export interface QueryResult {
  sub_questions: string[]
  is_complex: boolean
  eu_response: string
  us_response: string
  uk_response: string
  sources: Source[]
  contradiction: Contradiction
  verdict: Verdict
  context: string
}