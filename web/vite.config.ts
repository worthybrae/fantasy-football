import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// A CHUNK THAT OUTLIVES A DEPLOY. Everything under `src` is hashed per
// build, so every deploy invalidates all of it -- which is correct, and is
// also why the libraries underneath should not be sitting in the same file.
// React, its DOM renderer and the router change when package.json changes
// and not otherwise. Split out, they are 73 KB gzipped that a returning
// visitor keeps across a deploy instead of downloading again, and the
// browser fetches them in parallel with the app rather than after it.
//
// MATCHED BY PATH, NOT BY PACKAGE NAME. Vite 8 bundles with rolldown, whose
// `manualChunks` takes a function and not the name-to-packages record the
// older rollup accepted. The pattern is anchored on `node_modules` and on a
// path separator at both ends of each name, so a package that merely starts
// with `react-` cannot fall in. `scheduler` and `react-router` are here
// because react-dom and react-router-dom pull them in: leaving them out
// would put two files of React's own runtime in the app chunk that is
// rebuilt every deploy.
//
// `@tanstack/react-table` is in package.json and nothing under `src`
// imports it, so a group for it would name a chunk that never forms. Add
// one here on the day the archive's data table starts using it.
const REACT = /node_modules[\\/](react|react-dom|react-router|react-router-dom|scheduler)[\\/]/

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          return REACT.test(id) ? 'react' : null
        },
      },
    },
  },
})
