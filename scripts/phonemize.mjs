import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {parseArgs} from 'node:util';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {Lexicon} from '../web/lexicon.mjs';
import {pronounceEnglish} from '../web/frontend/index.mjs';

const {values}=parseArgs({options:{text:{type:'string'},output:{type:'string'}}});
if(!values.text?.trim()||!values.output)throw new Error('Usage: node scripts/phonemize.mjs --text "Hello there." --output runs/prompt.json');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const lock=JSON.parse(await readFile(path.join(root,'model/assets.lock.json'),'utf8'));
const files={};
for(const name of ['frontend.json','en.lex']){
  const bytes=await readFile(path.join(root,'assets/model',name));
  if(bytes.length!==lock.files[name].bytes||createHash('sha256').update(bytes).digest('hex')!==lock.files[name].sha256)
    throw new Error(`Unverified ${name}. Run npm run setup.`);
  files[name]=bytes;
}
const lex=files['en.lex'];
const lexicon=new Lexicon(lex.buffer.slice(lex.byteOffset,lex.byteOffset+lex.byteLength));
const {vocabulary}=JSON.parse(files['frontend.json'].toString('utf8'));
const result=pronounceEnglish(values.text,lexicon,vocabulary);
const output=path.resolve(values.output);
await mkdir(path.dirname(output),{recursive:true});
await writeFile(output,JSON.stringify({purpose:'Native listening input; browser delivery controls are not applied',
  samples:[{text:values.text,normalized_text:result.normalizedText,
    chunks:result.chunks.map(({tokens,stress})=>({tokens,stress}))}],pronunciation_notes:result.pronunciationNotes},null,2)+'\n',{flag:'wx'});
console.log(output);
