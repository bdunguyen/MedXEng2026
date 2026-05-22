import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { handleInsightsRequest } from './server/geminiInsights.mjs'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  Object.assign(process.env, loadEnv(mode, process.cwd(), ''))

  return {
    plugins: [
      react(),
      {
        name: 'gemini-insights-api',
        configureServer(server) {
          server.middlewares.use('/api/insights', handleInsightsRequest)
        },
      },
    ],
  }
})
