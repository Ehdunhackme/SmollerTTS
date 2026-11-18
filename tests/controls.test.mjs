import test from 'node:test';
import assert from 'node:assert/strict';
import {PRESETS, validateControls, adjustDurations, adjustProsody} from '../web/controls.mjs';

test('Natural controls are an exact no-op on durations and prosody', () => {
  const durations = [0, 4, 8, 3], tokens = [1, 22, 30, 1], prosody = new Float32Array([.32, -2.1, .8]);
  assert.strictEqual(adjustDurations(durations, tokens), durations);
  assert.strictEqual(adjustProsody(prosody), prosody);
  for (const preset of Object.values(PRESETS)) assert.deepEqual(validateControls(preset), preset);
});
test('Pitch moves the physical F0 target without touching voicing or energy', () => {
  const input = new Float32Array([.4, -.2, .9, -.7, .3, -.4]);
  const output = adjustProsody(input, {pitch: 2});
  for (let i = 0; i < input.length; i += 3) {
    const hzBefore = Math.exp(input[i] * .5) * 180, hzAfter = Math.exp(output[i] * .5) * 180;
    assert.ok(Math.abs(12 * Math.log2(hzAfter / hzBefore) - 2) < 1e-6);
    assert.equal(input[i + 1], output[i + 1]); assert.equal(input[i + 2], output[i + 2]);
  }
  assert.equal(input[0], Math.fround(.4));
});
test('Energy uses the training log RMS convention without touching pitch or voicing', () => {
  const input = new Float32Array([.4, -.2, .9]);
  const output = adjustProsody(input, {energy: -1.5});
  const rmsBefore = Math.exp(input[2] * 2 - 3), rmsAfter = Math.exp(output[2] * 2 - 3);
  assert.ok(Math.abs(20 * Math.log10(rmsAfter / rmsBefore) + 1.5) < 1e-5);
  assert.equal(input[0], output[0]); assert.equal(input[1], output[1]);
});
test('Rate changes generated durations while preserving phones and zero-length pauses', () => {
  const durations = [0, 10, 20, 1, 8], tokens = [1, 22, 30, 15, 1];
  const slow = adjustDurations(durations, tokens, .85), fast = adjustDurations(durations, tokens, 1.15);
  assert.equal(slow[0], 0); assert.equal(fast[0], 0); assert.equal(fast[3], 1);
  assert.ok(slow.reduce((a, b) => a + b) > durations.reduce((a, b) => a + b));
  assert.ok(fast.reduce((a, b) => a + b) < durations.reduce((a, b) => a + b));
  assert.deepEqual(durations, [0, 10, 20, 1, 8]);
});
test('Controls reject nonfinite, out-of-range and unknown settings before inference', () => {
  for (const value of [{pitch: NaN}, {pitch: 3}, {energy: -2}, {speed: 0}, {speed: '1'}, {unknown: 1}, null])
    assert.throws(() => validateControls(value));
  assert.throws(() => adjustDurations([NaN], [1]));
  assert.throws(() => adjustProsody(new Float32Array([1, 2])));
});
