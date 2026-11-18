import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Lexicon } from '../web/lexicon.mjs';

test('compact pronunciation lookup preserves boundary and Unicode entries', async () => {
  for (const language of ['en']) {
    const bytes = await readFile(new URL('../assets/model/' + language + '.lex', import.meta.url));
    const lexicon = new Lexicon(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const decoder = new TextDecoder();
    for (const i of [0, Math.floor(lexicon.count / 2), lexicon.count - 1]) {
      const position = lexicon.start + lexicon.view.getUint32(8 + i * 4, true);
      const length = lexicon.view.getUint16(position, true);
      const word = decoder.decode(lexicon.bytes.subarray(position + 4, position + 4 + length));
      assert.equal(lexicon.find(word), position);
      assert.ok(lexicon.get(word).length > 0);
    }
    assert.equal(lexicon.get('thiswordisnotinthedictionaryzzxq'), undefined);
  }
  assert.throws(() => new Lexicon(new ArrayBuffer(16)), /Invalid/);
});
