import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {Lexicon} from '../web/lexicon.mjs';
import {pronounceNormalized} from '../web/frontend/normalization.mjs';
import {pronounceEstimated, ipaToLabels} from '../web/frontend/phonemizer.mjs';
const root = new URL('../', import.meta.url);
const bytes = readFileSync(new URL('assets/model/en.lex', root));
const lex = new Lexicon(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const {vocabulary} = JSON.parse(readFileSync(new URL('assets/model/frontend.json', root)));
const words = ['podcasting', 'microcontroller', 'millisecond', 'milliliters', "Taylor's", 'unboxing', 'Dunli', 'glorpzibble', 'ChatGPT', 'OpenAI', 'LLM', 'QZX'];
for (const word of words) test(`Unfamiliar word is spoken: ${word}`, () => {
  assert.equal(lex.has(word.toLowerCase()), false);
  const result = pronounceEstimated(`We said ${word} today.`, lex, vocabulary);
  assert.deepEqual(result.chunks.flatMap(c => c.words), ['we', 'said', word.toLowerCase(), 'today']);
  assert.equal(result.pronunciationNotes.length, 1);
  assert.ok(result.chunks.flatMap(c => c.tokens).every(n => Number.isInteger(n) && n > 0 && n !== vocabulary['<unk>']));
});
test('Known text, times, money and AM keep their accepted frontend', () => {
  const panel = JSON.parse(readFileSync(new URL('tests/fixtures/pronunciation.json', root)));
  for (const text of [...panel.samples.map(s => s.text), 'Meet at 9:05 am on March 14th. It costs $12.99.']) {
    const before = pronounceNormalized(text, lex, vocabulary), after = pronounceEstimated(text, lex, vocabulary);
    assert.equal(after.normalizedText, before.normalizedText); assert.deepEqual(after.chunks, before.chunks);
    assert.deepEqual(after.pronunciationNotes, []);
  }
});
test('Mixed word/initialism names use known stems and letter names including letter A', () => {
  assert.deepEqual(pronounceEstimated('ChatGPT', lex, vocabulary).chunks[0].tokens,
    pronounceNormalized('chat G P T', lex, vocabulary).chunks[0].tokens);
  assert.deepEqual(pronounceEstimated('OpenAI', lex, vocabulary).chunks[0].tokens,
    pronounceNormalized('open ay I', lex, vocabulary).chunks[0].tokens);
});
test('Common word plus initialism names also work when typed in lowercase', () => {
  for (const [lower, mixed] of [['chatgpt', 'ChatGPT'], ['openai', 'OpenAI']])
    assert.deepEqual(pronounceEstimated(lower, lex, vocabulary).chunks, pronounceEstimated(mixed, lex, vocabulary).chunks);
});
test('Known stems retain their exact phones under productive endings', () => {
  const expected = [...lex.get('podcast').map(p => p[0]), vocabulary['ɪ'], vocabulary['ŋ']];
  assert.deepEqual(pronounceEstimated('podcasting', lex, vocabulary).chunks[0].tokens, [1, ...expected, 1]);
});
test('Stress attaches to vowels, not onset consonants', () => {
  assert.deepEqual(ipaToLabels('ˈmɪɫɪˌsɛkənd', vocabulary).map(p => p[2]), [0,1,0,0,0,2,0,0,0,0]);
  assert.throws(() => ipaToLabels('ˈmɪ☃', vocabulary), /Unsupported/);
});
test('Invalid or empty pronunciation spells the entire word instead of dropping sounds', () => {
  for (const ipa of ['', '☃', 'ɡpt']) {
    const result = pronounceEstimated('glorpzibble', lex, vocabulary, () => ({ipa}));
    assert.equal(result.pronunciationNotes[0].method, 'spelled');
    assert.deepEqual(result.chunks[0].tokens, pronounceNormalized('g l o r p z i b b l e', lex, vocabulary).chunks[0].tokens);
  }
});
test('Repeated unfamiliar words are retained and noted once per request', () => {
  const result = pronounceEstimated('Dunli said Dunli.', lex, vocabulary);
  assert.deepEqual(result.chunks[0].words, ['dunli', 'said', 'dunli']);
  assert.equal(result.pronunciationNotes.length, 1);
});
test('Pronunciation estimation does not loosen unrelated input validation', () => {
  assert.throws(() => pronounceEstimated('03/04/2026', lex, vocabulary), /month name/);
  assert.throws(() => pronounceEstimated('中文', lex, vocabulary), /English/);
  assert.throws(() => pronounceEstimated('x'.repeat(49), lex, vocabulary), /long words/);
});
