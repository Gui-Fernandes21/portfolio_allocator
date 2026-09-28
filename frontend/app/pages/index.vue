<script setup lang="ts">
import { VARIANT_NAMES, type RewardVariant } from '~/types/api'

const { data, pending, error, variant, load } = useRecommendation()

const variants = Object.keys(VARIANT_NAMES) as RewardVariant[]

onMounted(() => load())

const isPlaceholder = computed(() => data.value?.policy_loaded === false)
</script>

<template>
  <main class="page">
    <div class="card">
      <!-- Header -->
      <header class="card-header">
        <h1 class="title">
          ETF allocation advisor
        </h1>
        <p class="subtitle">
          decision support - not a trading platform
        </p>
      </header>

      <!-- Controls -->
      <div class="controls">
        <label
          for="variant"
          class="control-label"
        >Reward formulation</label>
        <select
          id="variant"
          v-model="variant"
          class="select"
          :disabled="pending"
          @change="load()"
        >
          <option
            v-for="option in variants"
            :key="option"
            :value="option"
          >
            {{ VARIANT_NAMES[option] }}
          </option>
        </select>

        <button
          type="button"
          class="button"
          :disabled="pending"
          @click="load()"
        >
          {{ pending ? 'Working…' : 'Refresh' }}
        </button>

        <span
          v-if="data"
          class="as-of"
        >Data as of {{ data.as_of }}</span>
      </div>

      <!-- States -->
      <div
        v-if="pending && !data"
        class="state-loading"
      >
        Running the agent pipeline. The first request of the day fetches
        market data and may take a few seconds.
      </div>

      <div
        v-else-if="error"
        class="state-error"
      >
        <p class="error-title">
          Could not load a recommendation
        </p>
        <p class="error-detail">
          {{ error }}
        </p>
      </div>

      <template v-else-if="data">
        <!-- never let a fallback look like a real answer -->
        <p
          v-if="isPlaceholder"
          class="banner banner-warning"
        >
          The trained policy could not be run. These are equal-weight
          placeholder values, not a model recommendation.
        </p>
        <p
          v-else-if="data.degraded"
          class="banner banner-muted"
        >
          Some parts of the pipeline were unavailable - see the reasoning
          trail. The allocation itself is unaffected.
        </p>

        <!-- Body -->
        <div class="body">
          <div class="body-main">
            <AllocationBars
              :weights="data.weights"
              :placeholder="isPlaceholder"
            />
          </div>
          <div class="body-side">
            <AgentReasoning :trace="data.trace" />
          </div>
        </div>

        <!-- Explanation -->
        <div class="section">
          <ExplanationPanel
            :explanation="data.explanation"
            :risk="data.risk"
          />
        </div>

        <!-- Context -->
        <div
          v-if="data.context.market.narrative || data.context.sentiment.sources.length"
          class="section"
        >
          <MarketContextPanel :context="data.context" />
        </div>
      </template>
    </div>
  </main>
</template>

<style scoped>
.page {
  min-height: 100vh;
  padding: var(--space-8) var(--space-4);
  background: var(--colour-page);
  color: var(--colour-text);
}

.card {
  max-width: 64rem;
  margin: 0 auto;
  overflow: hidden;
  border: 1px solid var(--colour-border);
  border-radius: var(--radius-lg);
  background: var(--colour-panel);
}

/* --- header --- */
.card-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: var(--space-4) var(--space-6);
  border-bottom: 1px solid var(--colour-border);
}

.title {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 600;
}

.subtitle {
  margin: 0;
  font-size: 0.875rem;
  color: var(--colour-text-muted);
}

/* --- controls --- */
.controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-6);
  border-bottom: 1px solid var(--colour-border-soft);
}

.control-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--colour-text-dim);
}

.select {
  padding: var(--space-1) var(--space-2);
  border: 1px solid #525252;
  border-radius: 0.25rem;
  background: var(--colour-track);
  color: #e5e5e5;
  font-size: 0.875rem;
}

.select:focus {
  outline: none;
  border-color: var(--colour-text-muted);
}

.button {
  padding: var(--space-1) var(--space-3);
  border: 1px solid #525252;
  border-radius: 0.25rem;
  background: transparent;
  color: #d4d4d4;
  font-size: 0.875rem;
  cursor: pointer;
  transition: color 150ms ease, border-color 150ms ease;
}

.button:hover:not(:disabled) {
  border-color: var(--colour-text-muted);
  color: var(--colour-text);
}

.button:disabled {
  opacity: 0.5;
  cursor: default;
}

.as-of {
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--colour-text-dim);
}

/* --- states --- */
.state-loading {
  padding: var(--space-8) var(--space-6);
  text-align: center;
  font-size: 0.875rem;
  color: var(--colour-text-dim);
}

.state-error {
  padding: var(--space-6);
}

.error-title {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--colour-error);
}

.error-detail {
  margin: var(--space-1) 0 0;
  font-size: 0.875rem;
  color: var(--colour-text-muted);
}

/* --- banners --- */
.banner {
  margin: 0;
  padding: var(--space-2) var(--space-6);
}

.banner-warning {
  border-bottom: 1px solid var(--colour-warning-border);
  background: var(--colour-warning-bg);
  font-size: 0.875rem;
  color: #fcd34d;
}

.banner-muted {
  border-bottom: 1px solid var(--colour-border);
  background: var(--colour-raised);
  font-size: 0.75rem;
  color: var(--colour-text-muted);
}

/* --- body --- */
.body {
  display: grid;
  gap: var(--space-8);
  padding: var(--space-6);
}

@media (min-width: 768px) {
  .body {
    grid-template-columns: 3fr 2fr;
  }
}

.section {
  padding: var(--space-6);
  border-top: 1px solid var(--colour-border-soft);
}
</style>
