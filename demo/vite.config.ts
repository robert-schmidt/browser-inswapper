import { createReadStream, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const models = fileURLToPath(new URL('../models/', import.meta.url));

// Multi-threaded CPU inference needs a cross-origin isolated page.
const headers = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'credentialless' };

export default defineConfig({
  base: './',
  // The dev server also serves ../models at /models/ (the files are git-ignored; see models/README.md).
  plugins: [{
    name: 'models',
    configureServer(server) {
      server.middlewares.use('/models', (req, res, next) => {
        const file = models + decodeURIComponent((req.url ?? '').split('?')[0]).replace(/^\/+/, '').replace(/\.\./g, '');
        if (!existsSync(file)) return next();
        res.setHeader('content-type', 'application/octet-stream');
        createReadStream(file).pipe(res);
      });
    },
  }],
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { headers },
  preview: { headers },
  build: { outDir: 'dist', emptyOutDir: true, target: 'es2022' },
});
