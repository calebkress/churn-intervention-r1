import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy all /api/* requests to Express during development
      // so React never has to know about CORS or ports
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
    },
  },
})