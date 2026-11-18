// Whole-word overrides are isolated from dictionary and stem lookup.
// Overrides are applied at the surface-word boundary, outside the base fallback,
// so an alias cannot change a stem lookup for a different word.
import {pronounceEstimated, createFallbackLexicon} from './phonemizer.mjs';
import {normalizeEnglishText, pronounceNormalized} from './normalization.mjs';
import {parsePronunciations} from './rules.mjs';

export function pronounceWithOverrides(text, lexicon, vocabulary, options = {}) {
  const rules = parsePronunciations(options.pronunciations ?? '');
  if (!rules.size) return pronounceEstimated(text, lexicon, vocabulary);

  const normalized = normalizeEnglishText(text);
  const surfaces = new Map((normalized.match(/[a-z]+(?:'[a-z]+)*/gi) || []).map(w => [w.toLowerCase(), w]));
  if (![...rules.keys()].some(word => surfaces.has(word))) return pronounceEstimated(text, lexicon, vocabulary);
  const notes = [], changes = new Map(), aliases = new Map();
  const fallback = createFallbackLexicon(lexicon, vocabulary, surfaces, notes);
  const overlay = {
    has: word => rules.has(word) || fallback.has(word),
    get(word) {
      const manual = rules.get(word);
      const parts = manual;
      if (!parts) return fallback.get(word);
      if (aliases.has(word)) return aliases.get(word);
      let phones, estimatedParts = [];
      if (parts.every(p => lexicon.has(p))) phones = parts.flatMap(p => lexicon.get(p).map(x => [...x]));
      else {
        const pronounced = pronounceEstimated(parts.join(' '), lexicon, vocabulary);
        if (pronounced.chunks.length !== 1) throw new Error('Use a shorter pronunciation for: ' + word);
        const chunk = pronounced.chunks[0];
        phones = chunk.tokens.flatMap((p, i) => p === 1 ? [] : [[p, 0, chunk.stress[i]]]);
        estimatedParts = pronounced.pronunciationNotes;
      }
      if (phones.length > 180) throw new Error('Use a shorter pronunciation for: ' + word);
      aliases.set(word, phones);
      changes.set(word, {word: surfaces.get(word) || word, method: manual ? 'custom' : 'compound', reading: parts.join(' '), estimatedParts});
      return phones;
    }
  };
  const result = pronounceNormalized(text, overlay, vocabulary);
  return {...result, pronunciationNotes: [...changes.values(), ...notes]};
}
