import { createFrontend } from './frontend.mjs';
import { synthesize, SAMPLE_RATE, HOP } from './dsp.mjs';

let ort, loaded, busy = false;
const json = async url => { const response = await fetch(url); if (!response.ok) throw new Error(`Could not load ${url}`); return response.json(); };
const tensor64 = (values, dims) => new ort.Tensor('int64', BigInt64Array.from(values, BigInt), dims);
async function initialize(base) {
  const started = performance.now();
  ort = await import(/* @vite-ignore */ new URL('runtime/ort.wasm.min.mjs', base).href);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL('runtime/', base).href;
  const config = await json(new URL('model/frontend.json', base));
  const lexicons = new Map();
  const loadLexicon = language => {
    if (!lexicons.has(language)) lexicons.set(language, json(new URL(`model/${language}.json`, base)));
    return lexicons.get(language);
  };
  const [encoder, decoder, frontend] = await Promise.all([
    ort.InferenceSession.create(new URL('model/encoder.onnx', base).href, { executionProviders: ['wasm'] }),
    ort.InferenceSession.create(new URL('model/decoder.onnx', base).href, { executionProviders: ['wasm'] }),
    createFrontend(config, loadLexicon),
  ]);
  return { encoder, decoder, frontend, config, initializationMs: performance.now() - started };
}

self.onmessage = async ({ data }) => {
  if (busy) { self.postMessage({ type: 'error', id: data.id, message: 'A sentence is already being generated.' }); return; }
  busy = true;
  try {
    if (!loaded) loaded = initialize(data.base).catch(error => { loaded = null; throw error; });
    const { encoder, decoder, frontend, initializationMs } = await loaded;
    if (data.type === 'load') { self.postMessage({ type: 'ready', initializationMs }); return; }
    const start = performance.now(), input = await frontend(data.text, data.language);
    const pronunciationMs = performance.now() - start, phones = input.tokens.length;
    const feeds = { tokens: tensor64(input.tokens, [1, phones]), tones: tensor64(input.tones, [1, phones]),
      stress: tensor64(input.stress, [1, phones]), language: tensor64([input.language], [1]) };
    const neuralStart = performance.now();
    const { encoded, log_duration } = await encoder.run(feeds);
    const durations = Array.from(log_duration.data, x => Math.max(1, Math.min(100, Math.round(Math.expm1(Math.max(0, Math.min(Math.log(101), x)))))));
    const frames = durations.reduce((a, b) => a + b, 0);
    if (frames < 5 || frames > Math.floor(25 * SAMPLE_RATE / HOP)) throw new Error('Predicted audio is too long or too short. Try a shorter sentence.');
    const ids = [], positions = new Float32Array(frames * 2);
    for (let i = 0, frame = 0; i < phones; i++) for (let t = 0; t < durations[i]; t++, frame++) {
      ids.push(i); positions[frame * 2] = (t + 0.5) / durations[i]; positions[frame * 2 + 1] = Math.log1p(durations[i]);
    }
    const outputs = await decoder.run({ encoded, frame_indices: tensor64(ids, [1, frames]),
      positions: new ort.Tensor('float32', positions, [1, frames, 2]),
      frame_mask: new ort.Tensor('float32', new Float32Array(frames).fill(1), [1, frames]) });
    const neuralMs = performance.now() - neuralStart, dspStart = performance.now();
    const audio = synthesize(outputs.controls.data, frames), dspMs = performance.now() - dspStart;
    let peak = 0, clipped = 0, energy = 0;
    for (const sample of audio) { peak = Math.max(peak, Math.abs(sample)); clipped += Math.abs(sample) > 1; energy += sample * sample; }
    const metrics = { pronunciationMs, neuralMs, dspMs, synthesisMs: performance.now() - start,
      initializationMs, audioSeconds: audio.length / SAMPLE_RATE, peak, clippedFraction: clipped / audio.length,
      rms: Math.sqrt(energy / audio.length), backend: 'wasm-cpu', threads: 1, phones, frames };
    metrics.realTimeFactor = metrics.synthesisMs / (metrics.audioSeconds * 1000);
    for (const tensor of [...Object.values(feeds), encoded, log_duration, ...Object.values(outputs)]) tensor.dispose();
    self.postMessage({ type: 'audio', id: data.id, audio, input, metrics }, [audio.buffer]);
  } catch (error) { self.postMessage({ type: 'error', id: data.id, message: error.message }); }
  finally { busy = false; }
};
