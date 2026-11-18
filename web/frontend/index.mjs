import {pronounceWithOverrides} from './overrides.mjs';
import {createFallbackLexicon} from './phonemizer.mjs';
import {normalizeEnglishText, pronounceNormalized} from './normalization.mjs';
import {parsePronunciations} from './rules.mjs';

// Only promote individual compounds supported by the development audio checks.
// Broad prefix rules regressed on unseen words, including nano- compounds.
// Native dictionary entries always win; manual rules remain an explicit override.
const testedCompounds = new Map([
  ['microcontroller', ['micro', 'controller']],
  ['microcamera', ['micro', 'camera']],
  ['microplastics', ['micro', 'plastics']],
  ['nanofibers', ['nano', 'fibers']],
]);
export function selectiveParts(word, lexicon, surface = word) {
  if (lexicon.has(word) || /^[A-Z]+$/.test(surface)) return null;
  const parts = testedCompounds.get(word);
  return parts?.every(part => lexicon.has(part)) ? parts : null;
}

// Recover punctuation boundaries from the normalized text without changing the
// accepted chunker or inserting new phones. Numeric punctuation is already gone.
export function attachPunctuation(result) {
  const events = [], words = []; let sentenceStart = 0;
  for (const item of result.normalizedText.match(/[a-z]+(?:'[a-z]+)*|[,.;:!?]+/gi) || []) {
    if (/^[a-z]/i.test(item)) {words.push(item.toLowerCase()); continue;}
    if (!words.length) continue;
    const previous = events.at(-1);
    const symbols = (previous?.after === words.length ? events.pop().symbols : '') + item;
    const question = previous?.after === words.length ? previous.question :
      /^(what|where|when|why|who|whom|whose|which|how)(?:'s)?$/.test(words[sentenceStart] || '') ? 'wh' : 'rise';
    events.push({after: words.length, symbols, question});
    if (/[.!?]/.test(item)) sentenceStart = words.length;
  }
  let offset = 0;
  return {...result, chunks: result.chunks.map(chunk => {
    const end = offset + chunk.words.length;
    const boundaries = events.filter(e => e.after > offset && e.after <= end);
    const silences = chunk.tokens.flatMap((id, i) => id === 1 && i > 0 ? [i] : []);
    const punctuation = boundaries.map((event, i) => ({
      index: silences[i], mark: event.symbols.includes('?') ? '?' : event.symbols.includes('!') ? '!' :
        event.symbols.includes('.') ? '.' : event.symbols.includes(';') ? ';' : event.symbols.includes(':') ? ':' : ',',
      ...(event.symbols.includes('?') ? {question: event.question} : {}),
    }));
    if (punctuation.some(cue => cue.index === undefined)) throw new Error('Could not place a punctuation boundary. Please simplify the sentence.');
    offset = end;
    return {...chunk, punctuation};
  })};
}

export function pronounceEnglish(text, lexicon, vocabulary, options = {}) {
  for (const name of ['splitCompounds', 'punctuation']) {
    if (options[name] !== undefined && typeof options[name] !== 'boolean') throw new Error('Invalid pronunciation setting.');
  }
  const rules = parsePronunciations(options.pronunciations ?? '');
  const automatic = options.splitCompounds !== false;
  const normalized = normalizeEnglishText(text);
  const surfaces = new Map((normalized.match(/[a-z]+(?:'[a-z]+)*/gi) || []).map(w => [w.toLowerCase(), w]));
  const candidates = new Map();
  if (automatic) for (const [word, surface] of surfaces) {
    if (rules.has(word)) continue;
    const parts = selectiveParts(word, lexicon, surface);
    if (parts) candidates.set(word, parts);
  }
  let result;
  if (!candidates.size) result = pronounceWithOverrides(text, lexicon, vocabulary, {splitCompounds: false, pronunciations: options.pronunciations ?? ''});
  else {
    const notes = [], changes = new Map(), aliases = new Map();
    // Fallback stem lookups only see the original lexicon, never the aliases.
    const fallback = createFallbackLexicon(lexicon, vocabulary, surfaces, notes);
    const overlay = {get(word) {
      const manual = rules.get(word), parts = manual || candidates.get(word);
      if (!parts) return fallback.get(word);
      if (aliases.has(word)) return aliases.get(word);
      let phones, estimatedParts = [];
      if (parts.every(p => lexicon.has(p))) phones = parts.flatMap((part, i) => lexicon.get(part).map(([id, tone, stress]) =>
        // Prefixes keep secondary stress; the root keeps its dictionary stress.
        [id, tone, !manual && i < parts.length - 1 && stress === 1 ? 2 : stress]));
      else {
        const reading = pronounceWithOverrides(parts.join(' '), lexicon, vocabulary);
        if (reading.chunks.length !== 1) throw new Error('Use a shorter pronunciation for: ' + word);
        const c = reading.chunks[0]; phones = c.tokens.flatMap((id, i) => id === 1 ? [] : [[id, 0, c.stress[i]]]);
        estimatedParts = reading.pronunciationNotes;
      }
      if (phones.length > 180) throw new Error('Use a shorter pronunciation for: ' + word);
      aliases.set(word, phones);
      changes.set(word, {word: surfaces.get(word) || word, method: manual ? 'custom' : 'compound', reading: parts.join(' '), estimatedParts});
      return phones;
    }};
    result = {...pronounceNormalized(text, overlay, vocabulary), pronunciationNotes: [...changes.values(), ...notes]};
  }
  return options.punctuation === false ? result : attachPunctuation(result);
}
