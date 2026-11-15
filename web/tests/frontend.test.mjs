import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createFrontend, tokenize } from '../src/frontend.mjs';
import { Lexicon } from '../src/lexicon.mjs';

const read = async name => JSON.parse(await readFile(new URL('../public/model/' + name, import.meta.url)));
test('browser pronunciation preserves native tokens, tones and stress in all five examples', async () => {
  const config = await read('frontend.json'), fixtures = await read('frontend-parity.json');
  const frontend = await createFrontend(config, async language => {
    const bytes = await readFile(new URL('../public/model/' + language + '.lex', import.meta.url));
    return new Lexicon(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  });
  for (const fixture of fixtures) {
    const result = await frontend(fixture.text, fixture.language);
    for (const key of ['tokens', 'tones', 'stress']) assert.deepEqual(result[key], fixture[key][0], `${fixture.language} ${key}`);
  }
  await assert.rejects(() => frontend('12 birds', 'en'), /spell numbers/);
  await assert.rejects(() => frontend('zzzxqnonword', 'en'), /Pronunciation is missing/);
});

test('Mandarin segmentation prefers known compound words', () => {
  assert.deepEqual(tokenize('今天你好', 'zh', new Map([['今天', []], ['你', []], ['好', []]])), ['今天', '你', '好']);
});
