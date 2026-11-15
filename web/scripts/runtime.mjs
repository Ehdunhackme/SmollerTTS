import { copyFile, mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const runtime = path.join(root, 'public/runtime');
await mkdir(runtime, { recursive: true });
for (const name of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(path.join(root, 'node_modules/onnxruntime-web/dist', name), path.join(runtime, name));
}
await copyFile(path.join(root, 'node_modules/ephone/COPYING'), path.join(runtime, 'ephone-GPL-3.0.txt'));
await copyFile(path.join(root, 'licenses/onnxruntime-LICENSE.txt'), path.join(runtime, 'onnxruntime-LICENSE.txt'));
for (const name of ['encoder.onnx', 'decoder.onnx', 'frontend.json', ...['en','zh','es','fr','de'].map(l => l + '.lex')]) {
  await readFile(path.join(root, 'public/model', name));
}
const modelRoot = path.join(root, 'public/model');
const metadata = JSON.parse(await readFile(path.join(modelRoot, 'export.json'), 'utf8'));
const frontend = JSON.parse(await readFile(path.join(modelRoot, 'frontend.json'), 'utf8'));
const parameters = metadata.model_parameters + metadata.renderer_parameters;
if (!Number.isSafeInteger(parameters) || parameters <= 0 || parameters >= 1_000_000)
  throw new Error('The complete learned model must stay below one million parameters.');
if (JSON.stringify(frontend.languages) !== JSON.stringify(['en','zh','es','fr','de']))
  throw new Error('The browser bundle must contain all five languages.');
for (const [name, expected] of Object.entries(metadata.sha256 || {})) {
  const actual = createHash('sha256').update(await readFile(path.join(modelRoot, name))).digest('hex');
  if (actual !== expected) throw new Error('Exported model hash mismatch: ' + name);
}
if (metadata.feature_identity?.['vocabulary.json'] && metadata.feature_identity['vocabulary.json'] !== frontend.vocabulary_sha256)
  throw new Error('The pronunciation vocabulary does not match the model weights.');
console.log(`Copied matching CPU runtime; checked all five languages and ${parameters} learned parameters.`);
