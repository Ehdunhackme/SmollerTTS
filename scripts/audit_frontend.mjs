// Check the compiled browser frontend against every frozen native dev input.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createFrontend } from '../web/src/frontend.mjs';
import { Lexicon } from '../web/src/lexicon.mjs';

const panelBytes = await readFile('artifacts/evaluation/dev-v1.json');
const panel = JSON.parse(panelBytes), native = JSON.parse(await readFile('artifacts/frontend-dev-v1.json'));
assert.equal(createHash('sha256').update(panelBytes).digest('hex'), native.panel_sha256);
const config = JSON.parse(await readFile('web/public/model/frontend.json'));
const dictionaries = new Map();
const frontend = await createFrontend(config, async language => {
  if (!dictionaries.has(language)) {
    const bytes = await readFile(`web/public/model/${language}.lex`);
    dictionaries.set(language, new Lexicon(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
  }
  return dictionaries.get(language);
});
const sources = new Map(native.samples.map(item => [item.id, item]));
const report = { panel_sha256: native.panel_sha256, languages: {}, samples: [] };
for (const row of panel.rows) {
  const reference = sources.get(row.id), item = { id: row.id, language: row.language };
  let actual, failure;
  try { actual = await frontend(row.text, row.language); }
  catch (error) { failure = error.message; }
  if (reference.failure) {
    assert.ok(failure, `${row.id}: native failure was not preserved`);
    item.expected_failure = failure;
  } else {
    assert.equal(failure, undefined, `${row.id}: unexpected browser failure`);
    for (const key of ['tokens', 'tones', 'stress']) assert.deepEqual(actual[key], reference.input[key], `${row.id} ${key}`);
    assert.deepEqual(actual.words, reference.words, `${row.id} word segmentation`);
    item.exact = true;
  }
  const counts = report.languages[row.language] ||= { clips: 0, exact: 0, expected_failures: 0 };
  counts.clips++; counts.exact += Number(Boolean(item.exact)); counts.expected_failures += Number(Boolean(item.expected_failure));
  report.samples.push(item);
}
report.status = 'passed';
await mkdir('artifacts/browser', { recursive: true });
await writeFile('artifacts/browser/frontend-panel.json', JSON.stringify(report, null, 2));
console.log(JSON.stringify(report.languages, null, 2));
