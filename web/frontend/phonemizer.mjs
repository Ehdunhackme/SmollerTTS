import {pronounceNormalized, normalizeEnglishText} from './normalization.mjs';
import {predictWord} from '../vendor/phonemize/g2p.mjs';

// This adapter is used only for words absent from the accepted native lexicon.
// IPA spelling conventions are mapped explicitly; unknown symbols never vanish.
const IPA = {
  't͡ʃ': 'tʃ', 'd͡ʒ': 'dʒ', 'tʃ': 'tʃ', 'dʒ': 'dʒ',
  'aɪ': 'aj', 'aʊ': 'aw', 'eɪ': 'ej', 'oʊ': 'ow', 'ɔɪ': 'ɔj',
  'iː': 'iː', 'uː': 'ʉː', 'ɑː': 'ɑː', 'ɔː': 'ɔ', 'ɜː': 'ɝ', 'əɹ': 'ɚ',
  p:'p', b:'b', t:'t', d:'d', k:'k', ɡ:'ɡ', g:'ɡ', f:'f', v:'v', θ:'θ', ð:'ð',
  s:'s', z:'z', ʃ:'ʃ', ʒ:'ʒ', h:'h', m:'m', n:'n', ŋ:'ŋ', l:'l', ɫ:'ɫ',
  r:'ɹ', ɹ:'ɹ', j:'j', w:'w', ɾ:'ɾ', ʔ:'ʔ',
  i:'i', ɪ:'ɪ', ɛ:'ɛ', æ:'æ', ɑ:'ɑ', ɒ:'ɒ', ɔ:'ɔ', ʊ:'ʊ', u:'ʉː',
  ʌ:'ɐ', ɐ:'ɐ', ə:'ə', ɚ:'ɚ', ɝ:'ɝ', ɜ:'ɝ',
};
const ipaUnits = Object.keys(IPA).sort((a, b) => b.length - a.length);
const vowel = phone => /^(?:aj|aw|ej|ow|ɔj|[iɪɛæɑɒɔʊʉɐəɚɝ])/.test(phone);
export function ipaToLabels(ipa, vocabulary) {
  if (typeof ipa !== 'string' || !ipa.trim()) throw new Error('Empty fallback pronunciation.');
  const labels = []; let stress = 0;
  for (let i = 0; i < ipa.length;) {
    if (ipa[i] === 'ˈ' || ipa[i] === 'ˌ') {stress = ipa[i++] === 'ˈ' ? 1 : 2; continue;}
    if (/\s/.test(ipa[i])) {i++; continue;}
    const unit = ipaUnits.find(u => ipa.startsWith(u, i));
    if (!unit) throw new Error('Unsupported fallback phone: ' + ipa[i]);
    let phone = IPA[unit], accent = vowel(phone) ? stress : 0;
    if (phone === 'i' && accent) phone = 'iː';
    const id = vocabulary[phone];
    if (!Number.isInteger(id) || id <= 2) throw new Error('Fallback phone is outside the model vocabulary.');
    labels.push([id, 0, accent]); if (vowel(phone)) stress = 0;
    i += unit.length;
  }
  if (!labels.length || !labels.some(([id]) => Object.entries(vocabulary).some(([p, n]) => n === id && vowel(p))))
    throw new Error('No vowel in fallback pronunciation.');
  return labels;
}

const acronyms = new Set(['gpt', 'tts', 'llm', 'asr', 'html', 'css', 'sql', 'onnx', 'vr', 'ar']);
const clone = phones => phones.map(p => [...p]);

export function createFallbackLexicon(lexicon, vocabulary, surfaceForms, notes, predictor = predictWord) {
  const cache = new Map();
  const inverse = Object.fromEntries(Object.entries(vocabulary).map(([p, id]) => [id, p]));
  function letters(word) {
    const phones = [];
    for (const letter of word.replaceAll("'", '')) {
      const entry = lexicon.get(letter === 'a' ? 'ay' : letter);
      if (!entry) throw new Error('Please use English letters in this word: ' + word);
      phones.push(...clone(entry));
    }
    return phones;
  }
  function ending(base, plural) {
    const phone = inverse[base.at(-1)[0]];
    const add = plural ? (['s','z','ʃ','ʒ','tʃ','dʒ'].includes(phone) ? ['ɪ','z'] :
      ['p','pʰ','pʲ','t','tʰ','tʲ','k','kʰ','c','cʰ','f','fʲ','θ'].includes(phone) ? ['s'] : ['z']) :
      ['t','tʰ','tʲ','d','dʲ'].includes(phone) ? ['ɪ','d'] :
      ['p','pʰ','pʲ','k','kʰ','c','cʰ','f','fʲ','s','ʃ','tʃ','θ'].includes(phone) ? ['t'] : ['d'];
    return [...clone(base), ...add.map(p => [vocabulary[p], 0, 0])];
  }
  function resolve(word) {
    const known = lexicon.get(word); if (known) return known;
    if (cache.has(word)) return cache.get(word);
    const surface = surfaceForms.get(word) || word;
    if (word.length > 48) throw new Error('Please split very long words into smaller parts.');
    let phones, method, ipa, reading;
    // Common product names retain their word + initialism reading even when
    // entered entirely in lowercase. Other camel-case names use their spelling.
    const namedInitialisms = {chatgpt: ['chat', 'gpt'], openai: ['open', 'ai']};
    const named = namedInitialisms[word];
    const camel = named ? [surface, named[0], named[1].toUpperCase()] : surface.match(/^([A-Z]?[a-z]+)([A-Z]{2,8})$/);
    if (acronyms.has(word) || (/^[A-Z]{2,8}$/.test(surface) && !/[AEIOUY]/.test(surface)) || /^[bcdfghjklmnpqrstvwxz]{2,8}$/.test(word)) {
      phones = letters(word); method = 'spelled'; reading = word.toUpperCase().split('').join(' ');
    } else if (camel && lexicon.has(camel[1].toLowerCase())) {
      phones = [...clone(lexicon.get(camel[1].toLowerCase())), ...letters(camel[2].toLowerCase())];
      method = 'compound-initialism'; reading = camel[1] + ' ' + camel[2].split('').join(' ');
    } else {
      // Productive endings reuse already accepted stem sounds where possible.
      if (word.endsWith("'s") && lexicon.has(word.slice(0, -2))) {
        phones = ending(lexicon.get(word.slice(0, -2)), true); method = 'possessive';
      } else if (word.endsWith('s') && word.length > 4 && lexicon.has(word.slice(0, -1))) {
        phones = ending(lexicon.get(word.slice(0, -1)), true); method = 'plural';
      } else if (word.endsWith('ing') && word.length > 6 && lexicon.has(word.slice(0, -3))) {
        phones = [...clone(lexicon.get(word.slice(0, -3))), [vocabulary['ɪ'],0,0], [vocabulary['ŋ'],0,0]]; method = 'inflection';
      } else if (word.endsWith('ed') && word.length > 5 && lexicon.has(word.slice(0, -2))) {
        phones = ending(lexicon.get(word.slice(0, -2)), false); method = 'inflection';
      }
      if (!phones) {
        try {
          const prediction = predictor(word);
          ipa = prediction.ipa; phones = ipaToLabels(ipa, vocabulary); method = 'estimated';
        } catch {
          phones = letters(word); method = 'spelled'; reading = word.toUpperCase().replaceAll("'", '').split('').join(' ');
        }
      }
    }
    if (!phones.length || phones.length > 180) throw new Error('Please split very long words into smaller parts.');
    if (cache.size >= 256) cache.delete(cache.keys().next().value);
    cache.set(word, phones); notes.push({word: surface, method, ...(ipa && method === 'estimated' ? {ipa} : {}), ...(reading ? {reading} : {})});
    return phones;
  }
  return {get: resolve, has: word => lexicon.has(word) || /^[a-z]+(?:'[a-z]+)*$/.test(word)};
}

export function pronounceEstimated(text, lexicon, vocabulary, predictor = predictWord) {
  const normalized = normalizeEnglishText(text), notes = [];
  const surfaces = new Map((normalized.match(/[a-z]+(?:'[a-z]+)*/gi) || []).map(w => [w.toLowerCase(), w]));
  const fallback = createFallbackLexicon(lexicon, vocabulary, surfaces, notes, predictor);
  return {...pronounceNormalized(text, fallback, vocabulary), pronunciationNotes: notes};
}
