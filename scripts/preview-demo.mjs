import http from 'node:http';
import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../output/demo');
const types = {'.html': 'text/html; charset=utf-8', '.mp4': 'video/mp4', '.png': 'image/png', '.wav': 'audio/wav', '.srt': 'text/plain; charset=utf-8'};
const port = Number(process.env.PORT || 8802);
http.createServer(async (req, res) => {
  if (!['GET', 'HEAD'].includes(req.method)) {res.writeHead(405); res.end(); return;}
  try {
    const url = new URL(req.url, 'http://localhost');
    const pathname = decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname);
    const file = path.resolve(root, '.' + pathname);
    if (!file.startsWith(root + path.sep)) throw new Error('Invalid path');
    const info = await stat(file);
    if (!info.isFile()) throw new Error('Not a file');
    let start = 0, end = info.size - 1, status = 200;
    if (req.headers.range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
      if (!match || (!match[1] && !match[2])) throw new Error('Invalid range');
      if (!match[1]) start = Math.max(0, info.size - Number(match[2]));
      else {start = Number(match[1]); if (match[2]) end = Math.min(end, Number(match[2]));}
      if (!Number.isSafeInteger(start) || start < 0 || start > end) {
        res.writeHead(416, {'Content-Range': `bytes */${info.size}`}); res.end(); return;
      }
      status = 206;
      res.setHeader('Content-Range', `bytes ${start}-${end}/${info.size}`);
    }
    res.writeHead(status, {'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache'});
    if (req.method === 'HEAD') {res.end(); return;}
    const stream = createReadStream(file, {start, end});
    stream.on('error', () => res.destroy());
    res.on('close', () => stream.destroy());
    stream.pipe(res);
  } catch {res.writeHead(404); res.end('Not found. Render the demo first.');}
}).listen(port, '127.0.0.1', () => console.log(`Demo preview: http://127.0.0.1:${port}/`));
