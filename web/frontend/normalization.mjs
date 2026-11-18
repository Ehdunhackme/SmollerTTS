// English numbers, dates, currency and units.
import {toWords, toOrdinal} from 'to-words/en-US';
import {normalizeEnglish, pronounceEnglish} from './core.mjs';
import {repairLexicon} from './dictionary.mjs';

const options = {casing: 'lower', decimalStyle: 'digit-by-digit'};
const number = value => toWords(String(value), options);
const ordinal = value => toOrdinal(String(value), options);
const amountPattern = String.raw`[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)`;
function amount(value) {
  if (!/^[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|\.\d+)$/.test(value))
    throw new Error('Please use commas in groups of three digits.');
  const digits = value.replaceAll(',', '');
  if (!Number.isFinite(Number(digits)) || Math.abs(Number(digits)) > 1e12)
    throw new Error('Please spell out very large numbers.');
  return {value: Number(digits), words: number(digits)};
}

const months = [
  ['January', 'Jan'], ['February', 'Feb'], ['March', 'Mar'], ['April', 'Apr'],
  ['May'], ['June', 'Jun'], ['July', 'Jul'], ['August', 'Aug'],
  ['September', 'Sep', 'Sept'], ['October', 'Oct'], ['November', 'Nov'], ['December', 'Dec'],
];
const monthPattern = months.flat().sort((a, b) => b.length - a.length).join('|');
function yearWords(year) {
  const y = Number(year), first = Math.floor(y / 100), last = y % 100;
  if (y >= 2000 && y < 2010) return number(y);
  if (last === 0) return `${number(first)} hundred`;
  return `${number(first)} ${last < 10 ? 'oh ' : ''}${number(last)}`;
}
function dateWords(month, day, year, dayFirst = false) {
  const index = months.findIndex(names => names.some(name => name.toLowerCase() === month.toLowerCase()));
  const y = year ? Number(year) : 2000; // With no year, allow February 29.
  const leap = y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0);
  const maximum = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][index];
  if (index < 0 || Number(day) < 1 || Number(day) > maximum)
    throw new Error('That date is not valid. Please check the day and month.');
  const spoken = dayFirst ? `the ${ordinal(day)} of ${months[index][0]}` : `${months[index][0]} ${ordinal(day)}`;
  return spoken + (year ? ` ${yearWords(year)}` : '');
}

// Match only explicit numeric quantities, so ordinary words/abbreviations do
// not change. Case-sensitive SI symbols distinguish mL from other notation.
const units = {
  sec: ['second', 'seconds'], secs: ['second', 'seconds'],
  s: ['second', 'seconds'], min: ['minute', 'minutes'], mins: ['minute', 'minutes'],
  hr: ['hour', 'hours'], hrs: ['hour', 'hours'], h: ['hour', 'hours'],
  kg: ['kilogram', 'kilograms'], g: ['gram', 'grams'],
  km: ['kilometer', 'kilometers'], m: ['meter', 'meters'], cm: ['centimeter', 'centimeters'],
  mm: ['millimeter', 'millimeters'], lb: ['pound', 'pounds'], lbs: ['pound', 'pounds'],
  oz: ['ounce', 'ounces'], ft: ['foot', 'feet'], in: ['inch', 'inches'],
  L: ['liter', 'liters'], l: ['liter', 'liters'],
  mph: ['mile per hour', 'miles per hour'],
  'km/h': ['kilometer per hour', 'kilometers per hour'],
  'm/s': ['meter per second', 'meters per second'],
};
// "in" is an ordinary preposition too; only accept the unambiguous 3in form.
const unitPattern = Object.keys(units).filter(x => x !== 'in').sort((a, b) => b.length - a.length).join('|');

export function normalizeEnglishText(input) {
  let text = input.normalize('NFKC').replace(/[’‘]/g, "'");
  const trailingStop = (match, suffix, offset, source) => {
    if (!suffix?.endsWith('.')) return '';
    const rest = source.slice(offset + match.length).trimStart();
    return !rest || /^[A-Z]/.test(rest) ? '.' : '';
  };

  // Colon pairs mean clock times here. Durations with seconds must use units;
  // never partially consume a timestamp, ratio, or an invalid minute field.
  if (/\d\s*:\s*\d+\s*:\s*\d/.test(text))
    throw new Error('Please write durations with units, for example 3 min 30 sec.');
  const meridiem = String.raw`[ap]\.?\s*m\.?`;
  const clock = new RegExp(String.raw`\b(\d{1,2}):(\d{2})(?:\s*(${meridiem}))?(?![\w:])`, 'gi');
  text = text.replace(clock, (match, hour, minute, suffix, offset, source) => {
    const h = Number(hour), m = Number(minute);
    if (m > 59 || h > (suffix ? 12 : 23) || (suffix && h < 1))
      throw new Error('That clock time is not valid. Please check the hour and minutes.');
    const tail = suffix ? ` ${suffix[0].toLowerCase()} m` : '';
    const spoken = `${number(h)}${m ? ` ${m < 10 ? 'oh ' : ''}${number(m)}` : suffix ? '' : " o'clock"}${tail}`;
    // Do not insert a sentence break into "3:30 p.m. on March 14".
    return spoken + trailingStop(match, suffix, offset, source);
  });
  text = text.replace(new RegExp(String.raw`\b(\d{1,2})\s*(${meridiem})(?!\w)`, 'gi'), (match, hour, suffix, offset, source) => {
    if (Number(hour) < 1 || Number(hour) > 12)
      throw new Error('Use an hour from 1 to 12 with a.m. or p.m.');
    return `${number(hour)} ${suffix[0].toLowerCase()} m${trailingStop(match, suffix, offset, source)}`;
  });

  // Explicit ISO dates are unambiguous. Slash dates intentionally require a
  // month name rather than silently choosing US versus day-first convention.
  text = text.replace(/\b(\d{4})-(\d{2})-(\d{2})\b/g, (_, year, month, day) => {
    if (Number(year) < 1000 || Number(month) < 1 || Number(month) > 12)
      throw new Error('Please check the ISO date (YYYY-MM-DD).');
    return dateWords(months[Number(month) - 1][0], day, year);
  });
  const day = String.raw`(\d{1,2})(?:st|nd|rd|th)?`;
  const year = String.raw`(?:,?\s+([12]\d{3}))?`;
  text = text.replace(new RegExp(String.raw`\b(${monthPattern})\.?\s+${day}${year}(?!\w)`, 'gi'),
    (_, month, d, y) => dateWords(month, d, y));
  text = text.replace(new RegExp(String.raw`\b${day}\s+(?:of\s+)?(${monthPattern})\.?${year}(?!\w)`, 'gi'),
    (_, d, month, y) => dateWords(month, d, y, true));
  if (/\d\s*\/\s*\d/.test(text))
    throw new Error('Please write dates with a month name, such as March 14, or use YYYY-MM-DD. Spell out fractions.');

  const currencies = {'$': ['dollar', 'dollars', 'cent', 'cents'], '£': ['pound', 'pounds', 'penny', 'pence'], '€': ['euro', 'euros', 'cent', 'cents']};
  text = text.replace(/([+-]?)([$£€])\s*(\d[\d,]*(?:\.\d+)?)/g, (_, sign, symbol, digits) => {
    const parts = digits.split('.');
    if (parts[1]?.length > 2) throw new Error('Currency amounts support at most two decimal places.');
    const whole = amount(parts[0]), cents = Number((parts[1] || '').padEnd(2, '0'));
    const names = currencies[symbol];
    return `${sign === '-' ? 'minus ' : sign === '+' ? 'plus ' : ''}${whole.words} ${names[whole.value === 1 ? 0 : 1]}${cents ? ` and ${number(cents)} ${names[cents === 1 ? 2 : 3]}` : ''}`;
  });
  text = text.replace(new RegExp(String.raw`(?<![\w.])(${amountPattern})\s*°\s*([CF])(?!\w)`, 'gi'), (_, digits, scale) => {
    const n = amount(digits);
    return `${n.words} ${Math.abs(n.value) === 1 ? 'degree' : 'degrees'} ${scale.toUpperCase() === 'C' ? 'Celsius' : 'Fahrenheit'}`;
  });
  if (new RegExp(String.raw`\d\s*(?:ms|mg|mL|ml)\b`).test(text))
    throw new Error('Please express this quantity in seconds, grams or liters; this dictionary lacks some milli-unit words.');
  text = text.replace(new RegExp(String.raw`(?<![\w.])(${amountPattern})(?:\s*(${unitPattern})\b|(in)\b)`, 'g'), (_, digits, unit, inches) => {
    const n = amount(digits), names = units[unit || inches];
    return `${n.words} ${names[Math.abs(n.value) === 1 ? 0 : 1]}`;
  });

  return normalizeEnglish(text);
}

export function pronounceNormalized(text, lexicon, vocabulary) {
  const normalized = normalizeEnglishText(text), repaired = repairLexicon(lexicon, vocabulary);
  const words = normalized.toLowerCase().match(/[a-z]+(?:'[a-z]+)*/g) || [];
  let index = 0;
  // Existing dictionary "a" is the article /ə/; "ay" supplies the existing
  // /ej/ letter name for A M without changing articles elsewhere or adding phones.
  return pronounceEnglish(normalized, {get(word) {
    const next = words[++index];
    return repaired.get(word === 'a' && next === 'm' ? 'ay' : word);
  }});
}
