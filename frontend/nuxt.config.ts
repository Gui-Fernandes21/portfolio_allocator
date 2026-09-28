// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },

  // @nuxt/ui is deliberately not registered: styling is hand-written in
  // scoped blocks, so the Tailwind build it installs would be dead weight.
  modules: ['@nuxt/eslint'],

  // Reset and shared design tokens only; component styles live beside their
  // components.
  css: ['~/assets/css/main.css'],

  runtimeConfig: {
    public: {
      // Where the FastAPI backend is served. Override at run time with
      // NUXT_PUBLIC_API_BASE rather than editing this file.
      apiBase: 'http://127.0.0.1:8000',
    },
  },

  app: {
    head: {
      title: 'ETF allocation advisor',
      meta: [
        {
          name: 'description',
          content: 'Decision support for retail ETF allocation. Not a trading platform.',
        },
      ],
    },
  },
})
