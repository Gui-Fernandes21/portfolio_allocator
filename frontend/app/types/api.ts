/**
 * Types mirroring the backend Pydantic schemas
 * (backend/app/schemas/recommendation.py).
 */

export type RewardVariant = 'raw' | 'dsr' | 'sortino' | 'cost'

export interface TickerWeight {
  ticker: string
  weight: number
}

export interface TraceEntry {
  agent: string
  summary: string
  used_llm: boolean
  ok: boolean
  error: string | null
}

export interface RiskAdjustment {
  rule: string
  detail: string
}

export interface RiskReport {
  adjustments: RiskAdjustment[]
  constraints: Record<string, unknown>
}

export interface Explanation {
  text: string
  generated: boolean
  fallback_reason: string | null
}

export interface MarketContext {
  available: boolean
  as_of: string | null
  narrative: string | null
  per_asset: Record<string, Record<string, number | null>>
}

export interface NewsSource {
  ticker: string
  title: string
  publisher: string
  published: string | null
  url: string | null
  sentiment: string | null
  reason: string | null
}

export interface SentimentContext {
  available: boolean
  classified: boolean
  per_asset: Record<string, Record<string, string | number>>
  sources: NewsSource[]
}

export interface RecommendationContext {
  market: MarketContext
  sentiment: SentimentContext
}

export interface Recommendation {
  as_of: string
  variant: RewardVariant
  policy_loaded: boolean
  degraded: boolean
  weights: TickerWeight[]
  risk: RiskReport
  explanation: Explanation
  context: RecommendationContext
  trace: TraceEntry[]
}

export interface PipelineDescription {
  agents: { order: number, agent: string, uses_llm: boolean }[]
  llm_free_agents: string[]
}

export interface HealthResponse {
  status: 'ok'
  available_variants: string[]
}

/** Display names for the seven ETFs in the universe. */
export const ASSET_NAMES: Record<string, string> = {
  SPY: 'US large-cap equities',
  QQQ: 'US technology equities',
  IWM: 'US small-cap equities',
  EFA: 'Developed international equities',
  EEM: 'Emerging-market equities',
  GLD: 'Gold',
  TLT: 'Long-dated US Treasuries',
}

/** Human labels for the four reward formulations from the ablation. */
export const VARIANT_NAMES: Record<RewardVariant, string> = {
  raw: 'Raw return',
  dsr: 'Differential Sharpe',
  sortino: 'Sortino (downside)',
  cost: 'Transaction-cost penalty',
}

/** Readable labels for the five agents. */
export const AGENT_LABELS: Record<string, { title: string, subtitle: string }> = {
  market_analyst: { title: 'Market analyst', subtitle: 'technical & macro read' },
  sentiment_analyser: { title: 'Sentiment analyser', subtitle: 'news signal summary' },
  execution_agent: { title: 'Execution agent', subtitle: 'frozen policy run' },
  risk_manager: { title: 'Risk manager', subtitle: 'constraints applied' },
  coordinator: { title: 'Coordinator', subtitle: 'explanation assembled' },
}
