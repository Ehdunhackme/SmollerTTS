import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {Lexicon} from '../web/lexicon.mjs';
import {pronounceEstimated as baseline} from '../web/frontend/phonemizer.mjs';
import {pronounceEnglish as current, selectiveParts} from '../web/frontend/index.mjs';
import {applyPunctuation} from '../web/punctuation.mjs';
import {audioKey} from '../web/audio.mjs';
const root = new URL('../assets/model/', import.meta.url);
const bytes = readFileSync(new URL('en.lex', root));
const lex = new Lexicon(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const vocabulary = JSON.parse(readFileSync(new URL('frontend.json', root))).vocabulary;
const phones = result => result.chunks.map(({tokens, stress, words}) => ({tokens, stress, words}));
const controls = {speed: 1, pitch: 0, energy: 0};

test('punctuation is carried separately: same phones, distinct cache entries', () => {
  const chunks = ['You are ready.', 'You are ready?', 'You are ready!'].map(t => current(t, lex, vocabulary).chunks[0]);
  assert.deepEqual(chunks[0].tokens, chunks[1].tokens);
  assert.deepEqual(chunks[0].tokens, chunks[2].tokens);
  assert.equal(new Set(chunks.map(c => audioKey(c, controls))).size, 3);
  assert.equal(chunks[1].punctuation[0].question, 'rise');
  assert.equal(current('Where are you?', lex, vocabulary).chunks[0].punctuation[0].question, 'wh');
});

test('commas, quotes, repeated punctuation and capacity boundaries stay attached to silence', () => {
  for (const text of ['Hello, world! Are you ready?', 'What happened? !!', '"Wait," she said. "Why?"',
    'Dr. Smith paid $12.99 at 3:30 pm.', ('The small bird flew across the quiet garden, then returned home. ').repeat(8),
    ('The small bird flew across the quiet garden ').repeat(9) + '?']) {
    const result = current(text, lex, vocabulary);
    assert.deepEqual(phones(result), phones(baseline(text, lex, vocabulary)), text);
    for (const chunk of result.chunks) for (const cue of chunk.punctuation) assert.equal(chunk.tokens[cue.index], 1, text);
  }
  const money = current('It costs $12.99.', lex, vocabulary);
  assert.equal(money.chunks.length, 1);
  assert.deepEqual(money.chunks[0].punctuation.map(p => p.mark), ['.']);
});

test('no splitting of familiar words, arbitrary substrings, short parts, or acronyms', () => {
  for (const word of ['internationalization','extraordinary','therapist','manuscript','photography','superb','notebook','ChatGPT','OpenAI','SQL','glorpzibble','quuxbird','bioprinter','cyberattack','VoiceBox','nanomaterials','nanocrystals','micromotors']) {
    assert.equal(selectiveParts(word.toLowerCase(), lex, word), null, word);
    assert.deepEqual(phones(current(word,lex,vocabulary)),phones(baseline(word,lex,vocabulary)),word);
  }
});

test('selective compounds reuse real component phones, without inserted silences', () => {
  for (const [word, parts] of [['microcontroller','micro controller'],['microcamera','micro camera'],['microplastics','micro plastics'],['nanofibers','nano fibers']]) {
    const changed=current(word,lex,vocabulary);
    assert.equal(changed.pronunciationNotes[0].reading,parts);
    assert.deepEqual(changed.chunks[0].tokens,baseline(parts,lex,vocabulary).chunks[0].tokens);
  }
  const text = 'The microcontroller is connected.';
  const result = current(text, lex, vocabulary);
  assert.equal(result.pronunciationNotes.find(n => n.method === 'compound').reading, 'micro controller');
  assert.deepEqual(result.chunks[0].tokens, baseline('The micro controller is connected.',lex,vocabulary).chunks[0].tokens);
  const prefixStress=lex.get('micro').map(p=>p[2]);
  assert.ok(prefixStress.includes(1));
  const single=current('microcontroller',lex,vocabulary).chunks[0];
  assert.deepEqual(single.stress.slice(1,1+prefixStress.length),prefixStress.map(n=>n===1?2:n));
  assert.deepEqual(phones(current(text,lex,vocabulary,{splitCompounds:false})),phones(baseline(text,lex,vocabulary)));
});

test('manual aliases remain scoped and never change fallback stem lookups', () => {
  const result=current('The podcast is about podcasting. The microcontroller is here.',lex,vocabulary,{pronunciations:'podcast = dog'});
  const expected=current('The dog is about podcasting. The microcontroller is here.',lex,vocabulary);
  const sounds = r => r.chunks.map(({tokens, stress}) => ({tokens, stress}));
  assert.deepEqual(sounds(result),sounds(expected));
  assert.throws(()=>current('Hello.',lex,vocabulary,{pronunciations:'word another'}), /name = spoken parts/);
  assert.deepEqual(current('Hello?',lex,vocabulary,{splitCompounds:false,punctuation:false}),baseline('Hello?',lex,vocabulary));
});

test('punctuation controls preserve plain periods and the original tensors', () => {
  const tokens=[1,4,5,6,1], stress=[0,0,1,0,0], durations=[1,5,6,4,1], data=new Float32Array(15);
  const plain=applyPunctuation(tokens,stress,durations,data,[{index:4,mark:'.'}],controls);
  assert.equal(plain.durations,durations); assert.equal(plain.prosody,data);
  const q=applyPunctuation(tokens,stress,durations,data,[{index:4,mark:'?',question:'rise'}],controls);
  const ex=applyPunctuation(tokens,stress,durations,data,[{index:4,mark:'!'}],controls);
  assert.ok(q.durations[4]>=20); assert.ok(q.prosody[9]>q.prosody[3]);
  assert.ok(ex.prosody.some((v,i)=>i%3===2&&v>0));
  assert.deepEqual(data,new Float32Array(15)); assert.deepEqual(durations,[1,5,6,4,1]);
  assert.deepEqual(q.prosody.filter((_,i)=>i%3===1),new Float32Array(5));
  const bounded=applyPunctuation(tokens,stress,durations,data,[{index:4,mark:'?',question:'rise'}],{pitch:2,energy:1.5});
  assert.deepEqual(bounded.prosody,new Float32Array(15));
});
