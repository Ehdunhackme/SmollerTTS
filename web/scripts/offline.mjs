import { readdir, readFile, writeFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

async function walk(root, prefix = '') {
  const result = [];
  for (const entry of await readdir(path.join(root, prefix), { withFileTypes: true })) {
    const name = path.posix.join(prefix, entry.name);
    if (entry.isDirectory()) result.push(...await walk(root, name)); else result.push(name);
  }
  return result;
}
const files = (await walk('dist')).filter(name => !['sw.js', 'asset-report.json'].includes(name));
const digest = createHash('sha256'), entries = [];
for (const file of files) {
  const content = await readFile(path.join('dist', file));
  digest.update(file).update(content);
  entries.push({ file, bytes: content.length, sha256: createHash('sha256').update(content).digest('hex') });
}
const name = 'xiaotts-' + digest.digest('hex').slice(0, 16);
const report = { version: name, uncompressedBytes: entries.reduce((sum, item) => sum + item.bytes, 0),
  neuralGraphBytes: entries.filter(x => x.file.endsWith('.onnx')).reduce((sum, x) => sum + x.bytes, 0), entries };
await writeFile('dist/asset-report.json', JSON.stringify(report, null, 2));
await writeFile('dist/sw.js', `
const CACHE = ${JSON.stringify(name)};
const FILES = ${JSON.stringify(['./', ...files.map(name => './' + name), './asset-report.json'])};
self.addEventListener('install', event => event.waitUntil((async () => {
  const cache = await caches.open(CACHE);
  await cache.addAll(FILES);
  await self.skipWaiting();
})()));
self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const key of await caches.keys()) if (key.startsWith('xiaotts-') && key !== CACHE) await caches.delete(key);
  await self.clients.claim();
})()));
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET' || new URL(event.request.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    // All entries are versioned, same-origin static files. Development hosts
    // may set Vary: Origin; module requests add Origin while precache requests
    // omit it. That must not turn a cached module into an offline miss.
    return await cache.match(event.request, { ignoreVary: true }) || fetch(event.request);
  })());
});
`);
console.log(JSON.stringify({ offlineCache: name, rawAssetBytes: report.uncompressedBytes, neuralGraphBytes: report.neuralGraphBytes }));
