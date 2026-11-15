import createEphone, { en_us, roa, gmw } from 'ephone';

const voices = { en: 'en-US', es: 'es', fr: 'fr', de: 'de' };
const vowelCharacters = /[aeiouyɑɐɒæɛɜɞəɚɝɘɤɨɪɔœøɵʉʊʌɯɶɷ\u0329]/u;
const punct = /^[,.;:!?，。！？；：]$/u;
const diphthongs = ['aɪ', 'aʊ', 'ɔɪ', 'oʊ', 'eɪ', 'əʊ', 'ɪə', 'eə', 'ʊə', 'ɑu', 'ɔø'];

export function wordStress(ipa) {
  const stress = [];
  let pending = 0;
  for (let i = 0; i < ipa.length; i++) {
    if (ipa[i] === 'ˈ') pending = 1;
    else if (ipa[i] === 'ˌ') pending = 2;
    else if (vowelCharacters.test(ipa[i])) {
      stress.push(pending); pending = 0;
      const diphthong = diphthongs.find(value => ipa.startsWith(value, i));
      if (diphthong) i += diphthong.length - 1;
    }
  }
  // Native annotation invokes eSpeak on an isolated word. The browser wrapper
  // adds a clause terminator, which de-stresses function words. Restore the
  // isolated-word primary accent when it emitted none (or only secondary).
  if (stress.length && !stress.includes(1)) {
    const secondary = stress.indexOf(2);
    stress[secondary < 0 ? stress.length - 1 : secondary] = 1;
  }
  return stress;
}

export function tokenize(text, language, lexicon) {
  text = text.toLowerCase().replace(/[’‘]/g, "'");
  if (language !== 'zh') return text.match(/[\p{L}\p{M}]+(?:'[\p{L}\p{M}]+)*|[0-9]+/gu) || [];
  const chunks = text.match(/[\u3400-\u9fff]+|[a-z]+|[0-9]+/g) || [];
  const result = [];
  for (const chunk of chunks) for (let i = 0; i < chunk.length;) {
    let length = Math.min(12, chunk.length - i);
    while (length > 1 && !lexicon.has(chunk.slice(i, i + length))) length--;
    result.push(chunk.slice(i, i + length)); i += length;
  }
  return result;
}

export async function createFrontend(config, loadLexicon) {
  const ephone = await createEphone([en_us, roa, gmw]);
  const vowels = new Set(config.vowels), cache = new Map();
  return async function pronounce(text, language) {
    if (!config.languages.includes(language)) throw new Error('Choose a supported language.');
    text = text.normalize('NFKC');
    if (/\d/u.test(text)) throw new Error('Please spell numbers out for this early demo.');
    const lexicon = await loadLexicon(language);
    const labels = [[config.vocabulary['<sil>'], 0, 0]], missing = [], words = [];
    if (language !== 'zh') ephone.setVoice(voices[language]);
    for (const chunk of text.split(/([,.;:!?，。！？；：])/u)) {
      if (!chunk.trim()) continue;
      if (punct.test(chunk)) {
        if (labels.at(-1)[0] !== config.vocabulary['<sil>']) labels.push([config.vocabulary['<sil>'], 0, 0]);
        continue;
      }
      for (const word of tokenize(chunk, language, lexicon)) {
        const entry = lexicon.get(word);
        if (!entry) { missing.push(word); continue; }
        const marked = entry.map(label => label.slice());
        if (language !== 'zh') {
          const key = language + ':' + word;
          if (!cache.has(key)) cache.set(key, wordStress(ephone.textToIpa(word)));
          const stress = cache.get(key), indices = marked.flatMap((label, i) => vowels.has(label[0]) ? [i] : []);
          if (indices.length === stress.length) indices.forEach((index, i) => { marked[index][2] = stress[i] || marked[index][2]; });
        }
        labels.push(...marked); words.push(word);
      }
    }
    if (missing.length) throw new Error(`Pronunciation is missing for: ${[...new Set(missing)].join(', ')}. Try another word.`);
    if (!words.length) throw new Error('Type a sentence first.');
    if (labels.at(-1)[0] !== config.vocabulary['<sil>']) labels.push([config.vocabulary['<sil>'], 0, 0]);
    if (labels.length > 250) throw new Error('Try a shorter sentence (up to 250 sounds).');
    return { tokens: labels.map(x => x[0]), tones: labels.map(x => x[1]), stress: labels.map(x => x[2]),
      language: config.languages.indexOf(language), words };
  };
}
