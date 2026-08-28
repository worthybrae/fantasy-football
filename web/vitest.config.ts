import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Kept apart from vite.config.ts on purpose: that file is what builds the
// image, and it should not grow a dependency on the test runner to do it.
// Vitest prefers this file when both are present, so the plugin list is
// restated here rather than merged -- it is one plugin, and a merge would be
// more machinery than the thing it merges.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.tsx'],
    restoreMocks: true,
  },
})
