import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fft, synthesize, sources, wavBytes } from '../src/dsp.mjs';

test('FFT round trips nontrivial signals', () => {
  const reference = Float64Array.from({ length: 1024 }, (_, i) => Math.sin(i * 0.17) + Math.cos(i * 0.093));
  const real = reference.slice(), imag = new Float64Array(1024);
  fft(real, imag); fft(real, imag, true);
  assert.ok(Math.max(...real.map((x, i) => Math.abs(x - reference[i]))) < 1e-11);
});

test('browser DSP matches PyTorch with identical excitation', () => {
  const fixture = JSON.parse(readFileSync(new URL('../../artifacts/browser-dsp-fixture.json', import.meta.url)));
  const audio = synthesize(Float32Array.from(fixture.controls), fixture.frames, {
    excitation: { pulse: Float32Array.from(fixture.pulse), noise: Float32Array.from(fixture.noise) },
  });
  const errors = audio.map((value, i) => value - fixture.audio[i]);
  const maxError = Math.max(...errors.map(Math.abs));
  const relativeRms = Math.sqrt(errors.reduce((sum, x) => sum + x * x, 0) / fixture.audio.reduce((sum, x) => sum + x * x, 0));
  console.log({ maxError, relativeRms });
  assert.ok(maxError < 2e-5 && relativeRms < 2e-4);
  assert.equal(new DataView(wavBytes(audio)).getUint32(40, true), audio.length * 2);
  const actualPulse = sources(Float32Array.from(fixture.controls), fixture.frames).pulse;
  const pulseError = actualPulse.reduce((sum, x, i) => sum + (x - fixture.pulse[i]) ** 2, 0);
  const pulseRelativeRms = Math.sqrt(pulseError / fixture.pulse.reduce((sum, x) => sum + x * x, 0));
  console.log({ pulseRelativeRms });
  assert.ok(pulseRelativeRms < 0.003, 'Harmonic oscillator differs from the Python renderer');
});
