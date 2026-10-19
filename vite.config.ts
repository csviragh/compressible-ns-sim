import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Relative base so the built app works both at a domain root and under a
  // sub-path such as https://<user>.github.io/compressible-ns-sim/.
  base: './',
  build: {
    target: 'es2022',
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    // a few physics tests run hundreds of solver steps; leave headroom on slow CI runners
    testTimeout: 30000,
  },
});
