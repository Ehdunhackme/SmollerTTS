// Fixed source/filter synthesis; no learned weights and no network requests.
export const SAMPLE_RATE = 22050, HOP = 256, FFT_SIZE = 1024;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const f32 = Math.fround;

export function fft(real, imag, inverse = false) {
  const n = real.length;
  if (n !== imag.length || n < 2 || (n & (n - 1))) throw new Error('FFT size must be a power of two');
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
  }
  for (let size = 2; size <= n; size *= 2) {
    const angle = (inverse ? 2 : -2) * Math.PI / size;
    const wr = Math.cos(angle), wi = Math.sin(angle);
    for (let start = 0; start < n; start += size) {
      let ur = 1, ui = 0;
      for (let j = 0; j < size / 2; j++) {
        const a = start + j, b = a + size / 2;
        const tr = real[b] * ur - imag[b] * ui, ti = real[b] * ui + imag[b] * ur;
        real[b] = real[a] - tr; imag[b] = imag[a] - ti;
        real[a] += tr; imag[a] += ti;
        [ur, ui] = [ur * wr - ui * wi, ur * wi + ui * wr];
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
}

export function minimumPhase(logAmplitude) {
  const n = (logAmplitude.length - 1) * 2;
  const real = new Float64Array(n), imag = new Float64Array(n);
  real.set(logAmplitude);
  for (let i = 1; i < n / 2; i++) real[n - i] = real[i];
  fft(real, imag, true);
  for (let i = 1; i < n / 2; i++) real[i] *= 2;
  real.fill(0, n / 2 + 1); imag.fill(0);
  fft(real, imag);
  for (let i = 0; i < n; i++) {
    const amplitude = Math.exp(real[i]), phase = imag[i];
    real[i] = amplitude * Math.cos(phase); imag[i] = amplitude * Math.sin(phase);
  }
  return [real, imag];
}

export function interpolate(values, size) {
  const out = new Float32Array(size), scale = (values.length - 1) / (size - 1);
  for (let i = 0; i < size; i++) {
    const p = i * scale, low = Math.floor(p), w = p - low;
    out[i] = values[low] * (1 - w) + values[Math.min(low + 1, values.length - 1)] * w;
  }
  return out;
}

function reflect(index, size) {
  while (index < 0 || index >= size) index = index < 0 ? -index : 2 * size - 2 - index;
  return index;
}

function noiseSource(length, seed) {
  let state = seed >>> 0 || 1;
  const uniform = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return ((state >>> 0) + 0.5) / 4294967296; };
  const result = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(uniform())), angle = 2 * Math.PI * uniform();
    result[i] = radius * Math.cos(angle);
    if (i + 1 < length) result[i + 1] = radius * Math.sin(angle);
  }
  return result;
}

export function sources(controls, frames, seed = 1234) {
  const size = (frames - 1) * HOP;
  const frequencies = new Float32Array(frames), voices = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    frequencies[i] = Math.exp(clamp(controls[i * 67], Math.log(55), Math.log(750)));
    voices[i] = 1 / (1 + Math.exp(-controls[i * 67 + 1]));
  }
  const f0 = interpolate(frequencies, size), voicing = interpolate(voices, size);
  const pulse = new Float32Array(size);
  let cycles = 0;
  for (let i = 0; i < size; i++) {
    cycles += f32(f0[i] / SAMPLE_RATE);
    const phase = f32(f32(cycles) * f32(2 * Math.PI));
    const count = clamp(Math.floor(SAMPLE_RATE / 2 / f0[i]), 1, 128);
    let value = 0;
    for (let k = 1; k <= 128 && k * f0[i] < SAMPLE_RATE / 2; k++) value += Math.cos(f32(k * phase));
    pulse[i] = value * Math.sqrt(2 / count) * voicing[i];
  }
  return { pulse, noise: noiseSource(size, seed) };
}

function envelope(controls, frame, offset, bands) {
  const out = new Float64Array(FFT_SIZE / 2 + 1);
  const maxMel = Math.log10(1 + SAMPLE_RATE / 2 / 700);
  for (let i = 0; i < out.length; i++) {
    const position = Math.log10(1 + i * SAMPLE_RATE / FFT_SIZE / 700) / maxMel * (bands - 1);
    const low = clamp(Math.floor(position), 0, bands - 2), w = clamp(position - low, 0, 1);
    out[i] = controls[frame * 67 + offset + low] * (1 - w) + controls[frame * 67 + offset + low + 1] * w;
  }
  return out;
}

export function synthesize(controls, frames, { seed = 1234, excitation = null } = {}) {
  if (frames < 5 || controls.length !== frames * 67 || !controls.every(Number.isFinite)) throw new Error('Invalid acoustic controls');
  const size = (frames - 1) * HOP;
  const { pulse, noise } = excitation || sources(controls, frames, seed);
  if (pulse.length !== size || noise.length !== size) throw new Error('Invalid excitation length');
  const window = Float64Array.from({ length: FFT_SIZE }, (_, i) => (1 - Math.cos(2 * Math.PI * i / FFT_SIZE)) / 2);
  const wave = new Float64Array(size + FFT_SIZE), weights = new Float64Array(size + FFT_SIZE);
  for (let frame = 0; frame < frames; frame++) {
    const harmonic = envelope(controls, frame, 3, 40), aperiodic = envelope(controls, frame, 43, 24);
    const center = Math.max(...harmonic, ...aperiodic);
    const filters = [harmonic, aperiodic].map(values => minimumPhase(values.map(x => clamp(x - center, -12, 0))));
    const outputR = new Float64Array(FFT_SIZE), outputI = new Float64Array(FFT_SIZE);
    for (const [sourceIndex, source] of [pulse, noise].entries()) {
      const real = new Float64Array(FFT_SIZE), imag = new Float64Array(FFT_SIZE);
      for (let i = 0; i < FFT_SIZE; i++) real[i] = source[reflect(frame * HOP + i - FFT_SIZE / 2, size)] * window[i];
      fft(real, imag);
      const [fr, fi] = filters[sourceIndex];
      for (let i = 0; i < FFT_SIZE; i++) {
        outputR[i] += real[i] * fr[i] - imag[i] * fi[i];
        outputI[i] += real[i] * fi[i] + imag[i] * fr[i];
      }
    }
    fft(outputR, outputI, true);
    for (let i = 0; i < FFT_SIZE; i++) {
      wave[frame * HOP + i] += outputR[i] * window[i];
      weights[frame * HOP + i] += window[i] ** 2;
    }
  }
  const output = new Float32Array(size);
  for (let i = 0; i < size; i++) output[i] = wave[i + FFT_SIZE / 2] / weights[i + FFT_SIZE / 2];
  const rms = new Float32Array(frames), energy = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame++) {
    let total = 0;
    for (let i = 0; i < FFT_SIZE; i++) total += output[reflect(frame * HOP + i - FFT_SIZE / 2, size)] ** 2;
    rms[frame] = Math.sqrt(Math.max(total / FFT_SIZE, 1e-8));
    energy[frame] = Math.exp(clamp(controls[frame * 67 + 2], -12, 0));
  }
  const envelopeRms = interpolate(rms, size), targetRms = interpolate(energy, size);
  for (let i = 0; i < size; i++) output[i] *= Math.min(100, targetRms[i] / Math.max(envelopeRms[i], 1e-4));
  if (!output.every(Number.isFinite)) throw new Error('Synthesis produced invalid audio');
  return output;
}

export function wavBytes(audio) {
  const buffer = new ArrayBuffer(44 + audio.length * 2), view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + audio.length * 2, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true); view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, audio.length * 2, true);
  for (let i = 0; i < audio.length; i++) view.setInt16(44 + i * 2, Math.round(clamp(audio[i], -1, 1) * 32767), true);
  return buffer;
}
