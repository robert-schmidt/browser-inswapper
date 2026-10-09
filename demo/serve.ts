// Serves the built demo (demo/dist) and the local model files (models/) with the cross-origin isolation
// headers multi-threaded inference needs. Usage: node demo/serve.ts [port]
import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';

const root = new URL('.', import.meta.url).pathname;
const types: Record<string, string> = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.jpg': 'image/jpeg', '.mp4': 'video/mp4', '.png': 'image/png', '.css': 'text/css' };
const port = Number(process.argv[2] ?? 5180);

createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
  const file = path.startsWith('/models/') ? join(root, '..', path) : join(root, 'dist', path.endsWith('/') ? path + 'index.html' : path);
  let size = 0;
  try { size = statSync(file).size; } catch { res.writeHead(404).end('Not found'); return; }
  res.writeHead(200, {
    'content-type': types[extname(file)] ?? 'application/octet-stream',
    'content-length': size,
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'credentialless',
  });
  createReadStream(file).pipe(res);
}).listen(port, () => console.log(`demo on http://localhost:${port}/`));
