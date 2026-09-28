<script setup lang="ts">
/**
 * Market and news context.
 *
 * This is the panel most at risk of being misread, so it is labelled
 * explicitly: none of it reached the policy, whose state space was fixed at
 * training time. Sentiment claims carry their source headlines, because an
 * unattributed sentiment score is not auditable.
 */
import type { RecommendationContext } from '~/types/api'

const props = defineProps<{ context: RecommendationContext }>()

const hasAnything = computed(() => !!props.context.market.narrative || props.context.sentiment.sources.length > 0)
</script>

<template>
  <section v-if="hasAnything">
    <h2 class="heading">
      Market context
    </h2>
    <p class="disclaimer">
      Provided for your own judgement. This did not influence the weights -
      the policy sees only price-derived features fixed at training time.
    </p>

    <p
      v-if="context.market.narrative"
      class="narrative"
    >
      {{ context.market.narrative }}
    </p>

    <div
      v-if="context.sentiment.sources.length"
      class="headlines"
    >
      <h3 class="subheading">
        Recent headlines
      </h3>
      <ul class="sources">
        <li
          v-for="source in context.sentiment.sources.slice(0, 6)"
          :key="source.title"
        >
          <div class="source-line">
            <span class="source-ticker">{{ source.ticker }}</span>
            <a
              v-if="source.url"
              :href="source.url"
              target="_blank"
              rel="noopener noreferrer"
              class="source-title"
            >{{ source.title }}</a>
            <span
              v-else
              class="source-title"
            >{{ source.title }}</span>
          </div>
          <div class="source-meta">
            {{ source.publisher }}
            <span
              v-if="source.sentiment"
              :class="`sentiment-${source.sentiment}`"
            >· {{ source.sentiment }}</span>
          </div>
        </li>
      </ul>
    </div>
  </section>
</template>

<style scoped>
.heading {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--colour-text);
}

.disclaimer {
  margin: var(--space-1) 0 0;
  font-size: 0.75rem;
  color: var(--colour-text-dim);
}

.narrative {
  margin: var(--space-3) 0 0;
  font-size: 0.875rem;
  line-height: 1.7;
  color: #d4d4d4;
}

.headlines {
  margin-top: var(--space-4);
}

.subheading {
  margin: 0;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--colour-text-dim);
}

.sources {
  margin: var(--space-2) 0 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.source-line {
  display: flex;
  align-items: baseline;
  gap: var(--space-2);
  font-size: 0.875rem;
}

.source-ticker {
  flex: 0 0 2.75rem;
  font-size: 0.75rem;
  color: var(--colour-text-dim);
}

.source-title {
  color: #d4d4d4;
  text-decoration: none;
  text-underline-offset: 2px;
}

a.source-title:hover {
  text-decoration: underline;
}

.source-meta {
  margin-left: 3.25rem;
  margin-top: 0.125rem;
  font-size: 0.75rem;
  color: var(--colour-text-dim);
}

.sentiment-positive {
  color: var(--colour-positive);
}

.sentiment-negative {
  color: var(--colour-negative);
}

.sentiment-neutral {
  color: var(--colour-text-muted);
}
</style>
