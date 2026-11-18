const WORD = /^[a-z]+(?:'[a-z]+)*$/;

export function parsePronunciations(text = '') {
  if (typeof text !== 'string' || text.length > 2000) throw new Error('Use up to 2,000 characters for pronunciation rules.');
  const rules = new Map();
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const parts = line.split('=');
    if (parts.length !== 2) throw new Error('Use one rule per line: name = spoken parts');
    const word = parts[0].trim().toLowerCase(), reading = parts[1].trim().toLowerCase().replaceAll('-', ' ');
    const units = reading.split(/\s+/);
    if (!WORD.test(word) || word.length > 48 || !units.length || units.length > 12 || units.some(w => !WORD.test(w) || w.length > 48))
      throw new Error('Pronunciation rules need an English word and up to 12 spoken parts.');
    if (rules.has(word)) throw new Error('Keep one pronunciation rule for: ' + word);
    rules.set(word, units);
    if (rules.size > 20) throw new Error('Use up to 20 pronunciation rules at a time.');
  }
  return rules;
}