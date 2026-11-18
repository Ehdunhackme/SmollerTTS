import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {normalizeEnglishText, pronounceNormalized} from '../web/frontend/normalization.mjs';
import {pronounceDictionary} from '../web/frontend/dictionary.mjs';
import {Lexicon} from '../web/lexicon.mjs';

export const example = 'The meeting moved to 3:30 pm on March 14th, so we have about 45 minutes left. It costs $12.99.';
export const spoken = 'The meeting moved to three thirty p m on March fourteenth, so we have about forty five minutes left. It costs twelve dollars and ninety nine cents.';

const cases = [
  [example, spoken],
  ['Come at 3:05 PM.', 'Come at three oh five p m.'],
  ['It starts at 12:00 am.', 'It starts at twelve a m.'],
  ['Meet at 7 pm.', 'Meet at seven p m.'],
  ['Come at 3:30 p.m. on March 14.', 'Come at three thirty p m on March fourteenth.'],
  ['Come at 9 a.m. tomorrow.', 'Come at nine a m tomorrow.'],
  ['Come at 9 a.m. Then wait.', 'Come at nine a m. Then wait.'],
  ['Meet at 15:30.', 'Meet at fifteen thirty.'],
  ['Meet at 8:00.', "Meet at eight o'clock."],
  ['March 14, 2026.', 'March fourteenth twenty twenty six.'],
  ['14 March 2026.', 'the fourteenth of March twenty twenty six.'],
  ['Feb. 29, 2024.', 'February twenty ninth twenty twenty four.'],
  ['2026-03-14.', 'March fourteenth twenty twenty six.'],
  ['July 4, 1905.', 'July fourth nineteen oh five.'],
  ['January 1, 2001.', 'January first two thousand one.'],
  ['March 14th.', 'March fourteenth.'],
  ['Wait 1 hr 45 min 30 sec.', 'Wait one hour forty five minutes thirty seconds.'],
  ['Wait 2hrs and 1min.', 'Wait two hours and one minute.'],
  ['Walk 2.5 km at 5 km/h.', 'Walk two point five kilometers at five kilometers per hour.'],
  ['Add 1 kg and 250 g.', 'Add one kilogram and two hundred fifty grams.'],
  ['Use 1 L or 2 L.', 'Use one liter or two liters.'],
  ['It is 1 °C, then -5°C.', 'It is one degree Celsius, then minus five degrees Celsius.'],
  ['It is 72°F.', 'It is seventy two degrees Fahrenheit.'],
  ['It is 2 ft and 3in.', 'It is two feet and three inches.'],
  ['We have 3 in the box.', 'We have three in the box.'],
  ['Pay $1.01 or $12.99.', 'Pay one dollar and one cent or twelve dollars and ninety nine cents.'],
  ['Pay $1,200.50.', 'Pay one thousand two hundred dollars and fifty cents.'],
  ['Pay $0.05 or $1.50.', 'Pay zero dollars and five cents or one dollar and fifty cents.'],
  ['Pay £1.01 or €2.50.', 'Pay one pound and one penny or two euros and fifty cents.'],
  ['Save 25%.', 'Save twenty five percent.'],
  ['Take .5 kg.', 'Take zero point five kilograms.'],
];
for (const [input, expected] of cases) test(input, () => {
  assert.equal(normalizeEnglishText(input), expected);
  assert.equal(normalizeEnglishText(expected), expected, 'normalization is idempotent');
});

for (const input of ['25:00', '3:75 pm', '13 pm', '0:30 am', '3:30:45', '03/04/2026',
  'April 31', 'February 29, 2025', '2026-13-14', '$12.999', '$1,23.50', '1,23 kg'])
  test(`Reject ambiguous or invalid notation: ${input}`, () => assert.throws(() => normalizeEnglishText(input)));

const root = new URL('../', import.meta.url);
const bytes = readFileSync(new URL('assets/model/en.lex', root));
const lexicon = new Lexicon(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const {vocabulary} = JSON.parse(readFileSync(new URL('assets/model/frontend.json', root)));

test('Every supported example resolves in the real deployed dictionary without dropped words', () => {
  for (const [input, expected] of cases) {
    const result = pronounceNormalized(input, lexicon, vocabulary);
    assert.deepEqual(result.chunks, pronounceNormalized(expected, lexicon, vocabulary).chunks, input);
  }
});
test('AM uses the letter A; articles retain the accepted pronunciation', () => {
  const result = pronounceNormalized('Meet at 9 am.', lexicon, vocabulary);
  const spelled = pronounceDictionary('Meet at nine ay m.', lexicon, vocabulary);
  assert.deepEqual(result.chunks[0].tokens, spelled.chunks[0].tokens);
  assert.deepEqual(pronounceNormalized('A man has a hat.', lexicon, vocabulary), pronounceDictionary('A man has a hat.', lexicon, vocabulary));
});
test('Accepted development panel retains every token, stress and chunk boundary', () => {
  const panel = JSON.parse(readFileSync(new URL('tests/fixtures/pronunciation.json', root)));
  for (const sample of panel.samples) {
    assert.deepEqual(pronounceNormalized(sample.text, lexicon, vocabulary).chunks, sample.chunks, sample.text);
  }
});
test('Unknown words are still rejected explicitly', () => {
  assert.throws(() => pronounceNormalized('3 pm glorpzibble', lexicon, vocabulary), /glorpzibble/);
});
