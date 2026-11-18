import test from 'node:test';
import assert from 'node:assert/strict';
import {AudioCache, audioKey, wavBlob} from '../web/audio.mjs';
test('cache evicts least recently used audio within a byte bound', () => {
  const cache = new AudioCache(16, 3), audio = () => ({wave: new Float32Array(2)});
  cache.set('a', audio()); cache.set('b', audio()); cache.get('a'); cache.set('c', audio());
  assert.equal(cache.get('b'), undefined); assert.ok(cache.get('a')); assert.equal(cache.bytes, 16);
});
test('cache enforces entry bound, replacement accounting and oversized skips', () => {
  const cache = new AudioCache(32, 1);
  cache.set('a', {wave: new Float32Array(2)}); cache.set('a', {wave: new Float32Array(3)}); assert.equal(cache.bytes, 12);
  cache.set('b', {wave: new Float32Array(2)}); assert.equal(cache.get('a'), undefined); assert.equal(cache.bytes, 8);
  cache.set('x', {wave: new Float32Array(20)}); assert.equal(cache.get('x'), undefined); assert.equal(cache.bytes, 8);
});
test('cache distinguishes every delivery setting, stress and phoneme sequence', () => {
  const chunk = {tokens: [4, 5], stress: [0, 1]}, base = {speed: 1, pitch: 0, energy: 0};
  const key = audioKey(chunk, base);
  for (const field of ['speed', 'pitch', 'energy']) assert.notEqual(key, audioKey(chunk, {...base, [field]: .9}));
  assert.notEqual(key, audioKey({...chunk, stress: [1, 0]}, base));
  assert.notEqual(key, audioKey({...chunk, tokens: [4, 6]}, base));
  assert.equal(key, audioKey(structuredClone(chunk), {...base}));
});
test('WAV encoder retains signed PCM and 22050 Hz without amplitude normalization', async () => {
  const buffer = await wavBlob(new Float32Array([-1, -.5, 0, .5, 1])).arrayBuffer(), view = new DataView(buffer);
  assert.equal(buffer.byteLength, 54); assert.equal(view.getUint32(24, true), 22050);
  assert.deepEqual(Array.from({length: 5}, (_, i) => view.getInt16(44 + i * 2, true)), [-32768, -16384, 0, 16384, 32767]);
});
