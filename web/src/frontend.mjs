const punct = /^[,.;:!?，。！？；：]$/u;

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
  if (config.stress_mode !== 'precomputed-native') throw new Error('Matching pronunciation tables are required. Rebuild the model bundle.');
  return async function pronounce(text, language) {
    if (!config.languages.includes(language)) throw new Error('Choose a supported language.');
    text = text.normalize('NFKC');
    if (/\d/u.test(text)) throw new Error('Please spell numbers out for this early demo.');
    const lexicon = await loadLexicon(language);
    const labels = [[config.vocabulary['<sil>'], 0, 0]], missing = [], words = [];
    for (const chunk of text.split(/([,.;:!?，。！？；：])/u)) {
      if (!chunk.trim()) continue;
      if (punct.test(chunk)) {
        if (labels.at(-1)[0] !== config.vocabulary['<sil>']) labels.push([config.vocabulary['<sil>'], 0, 0]);
        continue;
      }
      for (const word of tokenize(chunk, language, lexicon)) {
        const entry = lexicon.get(word);
        if (!entry) { missing.push(word); continue; }
        labels.push(...entry); words.push(word);
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
