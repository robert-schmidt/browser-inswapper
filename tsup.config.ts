import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts', 'src/react.ts', 'src/worker.ts'],
  format: ['esm'],
  dts: { entry: ['src/index.ts', 'src/react.ts'] },
  target: 'es2022',
  clean: true,
  splitting: true,
  external: ['react', 'onnxruntime-web', '@mediapipe/tasks-vision'],
  // Sources import './x.ts'; the published files are .js (the worker URL included).
  async onSuccess() {
    const { readFile, writeFile } = await import('node:fs/promises');
    const { readdir } = await import('node:fs/promises');
    for (const f of (await readdir('dist')).filter((n) => n.endsWith('.js')).map((n) => `dist/${n}`)) {
      await writeFile(f, (await readFile(f, 'utf8')).replace(/\.\/(\w[\w-]*)\.ts(['"])/g, './$1.js$2'));
    }
  },
});
