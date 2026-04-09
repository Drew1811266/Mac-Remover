import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, '../src/gui/templates/dist'),
    emptyOutDir: true,
    sourcemap: false,
    target: 'es2020',
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
});
