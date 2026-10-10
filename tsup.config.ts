import { defineConfig } from 'tsup';

// Two builds. The library entries keep their dependencies external (the app's bundler resolves them). The
// workers are self-contained: some bundlers (Next.js/Turbopack among them) copy a dependency's worker file as
// is instead of bundling it, so it must run without resolving any imports.
export default defineConfig([
  {
    entry: ['src/index.ts', 'src/react.ts', 'src/faces.ts'],
    format: ['esm'],
    dts: { entry: ['src/index.ts', 'src/react.ts', 'src/faces.ts'] },
    target: 'es2022',
    splitting: true,
    external: ['react', 'onnxruntime-web', '@mediapipe/tasks-vision', 'mediabunny'],
    // Sources import './x.ts'; the published files are .js (the worker URLs included).
    async onSuccess() {
      const { readFile, writeFile, readdir } = await import('node:fs/promises');
      for (const f of (await readdir('dist')).filter((n) => n.endsWith('.js')).map((n) => `dist/${n}`)) {
        await writeFile(f, (await readFile(f, 'utf8')).replace(/\.\/(\w[\w-]*)\.ts(['"])/g, './$1.js$2'));
      }
    },
  },
  {
    entry: ['src/worker.ts', 'src/segment-worker.ts'],
    format: ['esm'],
    target: 'es2022',
    platform: 'browser',
    splitting: false,
    noExternal: [/.*/],
    // App bundlers read these files too. ONNX Runtime points at its own files relative to import.meta.url (in a
    // worker self.location.href is the same URL, and the files come from wasmPaths anyway), and MediaPipe imports
    // its loader from a URL known only at run time: marked so bundlers leave it alone.
    async onSuccess() {
      const { readFile, writeFile } = await import('node:fs/promises');
      for (const f of ['dist/worker.js', 'dist/segment-worker.js']) {
        await writeFile(f, (await readFile(f, 'utf8'))
          .replace(/(new \w+\("ort[^"]+", )import\.meta\.url\)/g, '$1self.location.href)')
          .replace(/\bimport\((?!\s*\/\*)(?=\s*[^"'`\s])/g, 'import(/* webpackIgnore: true */ '));
      }
    },
  },
]);
