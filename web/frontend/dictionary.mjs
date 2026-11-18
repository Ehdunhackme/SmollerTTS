// One documented dictionary-variant repair; applied before pronunciation lookup.
// Both variants exist in our source MFA dictionary. Training contains 6 nasal
// taps versus 6,414 palatal nasals. This is not a general phoneme substitution.
import {pronounceEnglish} from './core.mjs';

export function repairLexicon(lexicon, vocabulary) {
  const original = ['fʲ','ɪ','ɾ̃','ɪ','ʃ'].map(x=>vocabulary[x]);
  const replacement = vocabulary['ɲ'];
  if(original.some(x=>!Number.isInteger(x))||!Number.isInteger(replacement))
    throw new Error('Dictionary pronunciation repair requires the original phone inventory.');
  return {
    has: word=>lexicon.has(word),
    get(word) {
      const phones=lexicon.get(word);
      if(word!=='finish'||!phones)return phones;
      if(phones.length!==original.length||phones.some((p,i)=>p[0]!==original[i]))
        throw new Error('The source pronunciation of finish changed; re-audit the repair.');
      return phones.map((p,i)=>i===2?[replacement,p[1],p[2]]:[...p]);
    },
  };
}

export function pronounceDictionary(text,lexicon,vocabulary) {
  return pronounceEnglish(text,repairLexicon(lexicon,vocabulary));
}
