import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Lexicon } from '../web/src/lexicon.mjs';

let count = 0;
for (const language of ['en','zh','es','fr','de']) {
  const original = JSON.parse(await readFile(`artifacts/browser-json-lexicons/${language}.json`, 'utf8'));
  const bytes = await readFile(`web/public/model/${language}.lex`);
  const lexicon = new Lexicon(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  assert.equal(lexicon.count, Object.keys(original).length);
  for (const [word, expected] of Object.entries(original)) { assert.deepEqual(lexicon.get(word), expected, word); count++; }
  console.log(language, 'all pronunciations preserved');
}
console.log('Verified entries:', count);
