<script setup lang="ts">
// What each agent contributed for
import { AGENT_LABELS, type TraceEntry } from '~/types/api'

defineProps<{ trace: TraceEntry[] }>()

const expanded = ref(false)

function label(agent: string) {
  return AGENT_LABELS[agent] ?? { title: agent, subtitle: '' }
}
</script>

<template>
  <aside class="panel">
    <div class="panel-head">
      <h2 class="heading">
        Agent reasoning
      </h2>

      <button
        type="button"
        class="toggle"
        :aria-label="expanded ? 'Show agent roles' : 'Show what each agent reported'"
        :aria-expanded="expanded"
        @click="expanded = !expanded"
      >
        <svg
          viewBox="0 0 12 12"
          fill="none"
          aria-hidden="true"
        >
          <rect
            x="0.75"
            y="0.75"
            width="10.5"
            height="10.5"
            rx="1.5"
            stroke="currentColor"
            stroke-width="1.2"
          />
          <path
            v-if="expanded"
            d="M3 6h6"
            stroke="currentColor"
            stroke-width="1.2"
            stroke-linecap="round"
          />
        </svg>
      </button>
    </div>

    <ul class="entries">
      <li
        v-for="(entry, index) in trace"
        :key="entry.agent"
        class="entry"
        :class="{ divided: index > 0 }"
      >
        <div class="entry-head">
          <p class="agent-name">
            {{ label(entry.agent).title }}
          </p>

          <span
            class="tag"
            :class="entry.used_llm ? 'tag-model' : 'tag-computed'"
            :title="entry.used_llm
              ? 'Written by a language model'
              : 'Computed deterministically - no model involved'"
          >
            {{ entry.used_llm ? 'model' : 'computed' }}
          </span>
        </div>

        <p
          v-if="!entry.ok"
          class="entry-body unavailable"
        >
          Unavailable - {{ entry.summary }}
        </p>
        <p
          v-else
          class="entry-body"
        >
          {{ expanded ? entry.summary : label(entry.agent).subtitle }}
        </p>
      </li>
    </ul>
  </aside>
</template>

<style scoped>
.panel {
  border: 1px solid var(--colour-border);
  border-radius: var(--radius);
  background: var(--colour-raised);
  padding: var(--space-5);
}

.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
}

.heading {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--colour-text);
}

.toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-1);
  border: 1px solid #525252;
  border-radius: 0.25rem;
  background: transparent;
  color: var(--colour-text-muted);
  cursor: pointer;
  transition: color 150ms ease, border-color 150ms ease;
}

.toggle:hover {
  color: #e5e5e5;
  border-color: var(--colour-text-muted);
}

.toggle svg {
  display: block;
  width: 0.75rem;
  height: 0.75rem;
}

.entries {
  margin: var(--space-4) 0 0;
  padding: 0;
  list-style: none;
}

.entry {
  padding: var(--space-3) 0;
}

.entry.divided {
  border-top: 1px solid var(--colour-border-soft);
}

.entry-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-2);
}

.agent-name {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--colour-text);
}

.tag {
  flex-shrink: 0;
  font-size: 0.625rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.tag-model {
  color: var(--colour-model);
}

.tag-computed {
  color: var(--colour-computed);
}

.entry-body {
  margin: 0.125rem 0 0;
  font-size: 0.875rem;
  color: var(--colour-text-muted);
}

.entry-body.unavailable {
  color: var(--colour-warning);
}
</style>
