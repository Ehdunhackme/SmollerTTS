import { copyFile, mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const runtime = path.join(root, 'public/runtime');
await mkdir(runtime, { recursive: true });
for (const name of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(path.join(root, 'node_modules/onnxruntime-web/dist', name), path.join(runtime, name));
}
await copyFile(path.join(root, 'node_modules/ephone/COPYING'), path.join(runtime, 'ephone-GPL-3.0.txt'));
await copyFile(path.join(root, 'licenses/onnxruntime-LICENSE.txt'), path.join(runtime, 'onnxruntime-LICENSE.txt'));
for (const name of ['encoder.onnx', 'decoder.onnx', 'frontend.json', ...['en','zh','es','fr','de'].map(l => l + '.json')]) {
  await readFile(path.join(root, 'public/model', name));
}
console.log('Copied matching CPU runtime and verified local model assets.');
