import { readFile, readdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Lexicon } from '../web/src/lexicon.mjs';

const cacheRoot = process.argv[2] || 'artifacts/browser-native-lexicons';
const cacheFiles = await readdir(cacheRoot);
const exported = JSON.parse(await readFile('web/public/model/lexicon-report.json'));
const report = { status: 'passed', entries: 0, languages: {} };
for (const language of ['en','zh','es','fr','de']) {
  const matches = [];
  for (const name of cacheFiles.filter(name => name.startsWith(language + '-') && name.endsWith('.json'))) {
    const candidate = JSON.parse(await readFile(`${cacheRoot}/${name}`, 'utf8'));
    if (JSON.stringify(candidate.identity) === JSON.stringify(exported[language].identity)) matches.push(candidate);
  }
  assert.equal(matches.length, 1, `${language}: require exactly one matching native cache`);
  const original = matches[0].lexicon;
  const bytes = await readFile(`web/public/model/${language}.lex`);
  const lexicon = new Lexicon(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  assert.equal(lexicon.count, Object.keys(original).length);
  for (const [word, expected] of Object.entries(original)) { assert.deepEqual(lexicon.get(word), expected, word); report.entries++; }
  report.languages[language] = { entries: lexicon.count, identity: exported[language].identity };
  console.log(language, 'all pronunciations preserved');
}
await writeFile('artifacts/browser/lexicon-parity.json', JSON.stringify(report, null, 2));
console.log('Verified entries:', report.entries);
