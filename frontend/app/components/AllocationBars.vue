<script setup lang="ts">
import { ASSET_NAMES, type TickerWeight } from '~/types/api'

const props = defineProps<{
  weights: TickerWeight[]
  /** Shown when the weights are a placeholder rather than a model output. */
  placeholder?: boolean
}>()

const sorted = computed(() =>
  [...props.weights].sort((a, b) => b.weight - a.weight),
)

const largest = computed(() =>
  Math.max(...props.weights.map(w => w.weight), 0.0001),
)

function barWidth(weight: number): string {
  return `${Math.max((weight / largest.value) * 100, 1.5)}%`
}

function asPercent(weight: number): string {
  return `${(weight * 100).toFixed(1)}%`
}
</script>

<template>
  <section>
    <h2 class="heading">
      Recommended allocation
    </h2>

    <p
      v-if="placeholder"
      class="placeholder-note"
    >
      Placeholder weights - not a model recommendation.
    </p>

    <ul class="rows">
      <li
        v-for="entry in sorted"
        :key="entry.ticker"
        class="row"
      >
        <span
          class="ticker"
          :title="ASSET_NAMES[entry.ticker] ?? entry.ticker"
        >
          {{ entry.ticker }}
        </span>

        <div class="track">
          <div
            class="fill"
            :style="{ width: barWidth(entry.weight) }"
          />
        </div>

        <span class="value">{{ asPercent(entry.weight) }}</span>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.heading {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--colour-text);
}

.placeholder-note {
  margin: var(--space-2) 0 0;
  font-size: 0.75rem;
  color: var(--colour-warning);
}

.rows {
  margin: var(--space-5) 0 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.row {
  display: flex;
  align-items: center;
  gap: var(--space-4);
}

.ticker {
  flex: 0 0 2.75rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #d4d4d4;
}

.track {
  flex: 1;
  height: 1.25rem;
  overflow: hidden;
  border-radius: 0.125rem;
  background: var(--colour-track);
}

.fill {
  height: 100%;
  border-radius: 0.125rem;
  background: var(--colour-bar);
  transition: width 500ms ease;
}

.value {
  flex: 0 0 3.5rem;
  text-align: right;
  font-size: 0.875rem;
  font-variant-numeric: tabular-nums;
  color: #e5e5e5;
}

@media (prefers-reduced-motion: reduce) {
  .fill {
    transition: none;
  }
}
</style>
