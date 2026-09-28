/**
 * Client for the backend API.
 */
import type {
  HealthResponse,
  PipelineDescription,
  Recommendation,
  RewardVariant,
} from '~/types/api'

function apiBase(): string {
  return useRuntimeConfig().public.apiBase as string
}

// handle errors from the backend and network
function describeError(error: unknown): string {
  const anyError = error as { statusCode?: number, data?: { detail?: string }, message?: string }
  if (anyError?.data?.detail) return anyError.data.detail
  if (anyError?.statusCode === 503) {
    return 'The backend could not produce an allocation. Check that the frozen policy weights exist in backend/models/.'
  }
  if (anyError?.message?.includes('fetch')) {
    return `Could not reach the API at ${apiBase()}. Is the backend running (uvicorn app.main:app --reload)?`
  }
  return anyError?.message ?? 'Unknown error.'
}

export function useRecommendation() {
  const data = ref<Recommendation | null>(null)
  const pending = ref(false)
  const error = ref<string | null>(null)
  const variant = ref<RewardVariant>('dsr')

  async function load(next?: RewardVariant) {
    if (next) variant.value = next
    pending.value = true
    error.value = null
    try {
      data.value = await $fetch<Recommendation>('/api/v1/recommendation', {
        baseURL: apiBase(),
        method: 'POST',
        body: { variant: variant.value },
        // The first call of the day can be long
        timeout: 120_000,
      })
    }
    catch (caught) {
      error.value = describeError(caught)
      data.value = null
    }
    finally {
      pending.value = false
    }
  }

  return { data, pending, error, variant, load }
}

export function usePipelineShape() {
  return useFetch<PipelineDescription>('/api/v1/pipeline', {
    baseURL: apiBase(),
    key: 'pipeline-shape',
  })
}

export function useHealth() {
  return useFetch<HealthResponse>('/api/v1/health', {
    baseURL: apiBase(),
    key: 'health',
  })
}
