import {readFile, writeFile, mkdir, copyFile, readdir, rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'vite';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
process.chdir(root);
const source=path.join(root,'web'),assets=path.join(root,'assets/model');
const output=path.join(root,'dist/projects/smollertts');
const hash=b=>createHash('sha256').update(b).digest('hex');
const lock=JSON.parse(await readFile('model/assets.lock.json','utf8'));
const model=JSON.parse(await readFile('model/config.json','utf8'));
for(const [name,info] of Object.entries(lock.files)){
  let data;try{data=await readFile(path.join(assets,name));}catch{throw new Error(`Missing ${name}. Run npm run setup first.`);}
  if(data.length!==info.bytes||hash(data)!==info.sha256)throw new Error(`Unverified model asset: ${name}. Run npm run setup.`);
}
// The generated output is the only directory this build removes.
if(!output.startsWith(path.join(root,'dist')+path.sep))throw new Error('Invalid output directory');
await rm(output,{recursive:true,force:true});await mkdir(output,{recursive:true});
const fingerprint=createHash('sha256');
for(const base of ['web','model','benchmarks']){
  for(const file of (await readdir(base,{recursive:true,withFileTypes:true})).filter(f=>f.isFile()).sort((a,b)=>path.join(a.parentPath,a.name).localeCompare(path.join(b.parentPath,b.name)))){
    const name=path.join(file.parentPath,file.name);fingerprint.update(path.relative(root,path.resolve(name)).replaceAll('\\','/'));fingerprint.update(await readFile(name));
  }
}
const buildId=fingerprint.digest('hex').slice(0,12);
const cacheKeys=text=>text.replace(/(['"])(\.\/[^'"?]+\.(?:mjs|css|json))\1/g,(_,quote,name)=>`${quote}${name}?h=${buildId}${quote}`);
await build({root,configFile:false,publicDir:false,logLevel:'warn',build:{lib:{entry:path.join(source,'frontend/index.mjs'),formats:['es'],fileName:()=> 'frontend.bundle.mjs'},outDir:output,emptyOutDir:false,target:'es2022',minify:true}});
for(const name of Object.keys(lock.files)){
  const dest=path.join(output,name);await mkdir(path.dirname(dest),{recursive:true});await copyFile(path.join(assets,name),dest);
}
const partials=Object.fromEntries(await Promise.all(['header','footer'].map(async name=>[name,await readFile(path.join(source,'partials',name+'.html'),'utf8')])));
for(const file of await readdir(source,{withFileTypes:true})){
  if(!file.isFile() || file.name.endsWith('.md'))continue;
  const name=file.name,text=await readFile(path.join(source,name),'utf8');
  if(name.endsWith('.html'))await writeFile(path.join(output,name),cacheKeys(text.replace('<!-- site-header -->',partials.header).replace('<!-- site-footer -->',partials.footer)));
  else await writeFile(path.join(output,name),name.endsWith('.mjs')?cacheKeys(text):text);
}
for(const folder of ['fonts','icons']){
  await mkdir(path.join(output,folder),{recursive:true});
  for(const name of await readdir(path.join(source,folder)))await copyFile(path.join(source,folder,name),path.join(output,folder,name));
}
await copyFile('benchmarks/overview.json',path.join(output,'benchmark.json'));
await copyFile('benchmarks/wer.json',path.join(output,'smollertts-wer.json'));
await copyFile('LICENSE',path.join(output,'LICENSE'));
await copyFile('THIRD_PARTY.md',path.join(output,'THIRD_PARTY.md'));
await mkdir(path.join(output,'model'),{recursive:true});
await copyFile('model/assets.lock.json',path.join(output,'model/assets.lock.json'));
await mkdir(path.join(output,'licenses'),{recursive:true});
for(const name of await readdir('licenses'))await copyFile(path.join('licenses',name),path.join(output,'licenses',name));
const weightFiles=['encoder.onnx','frames.onnx','vocoder.onnx'];
const voiceFiles=[...weightFiles,'en.lex','frontend.json','frontend.bundle.mjs','lexicon.mjs','engine.mjs','worker.mjs','audio.mjs','punctuation.mjs','controls.mjs','ort/ort.wasm.min.mjs','ort/ort-wasm-simd-threaded.mjs','ort/ort-wasm-simd-threaded.wasm'];
const sizes=Object.fromEntries(await Promise.all(voiceFiles.map(async name=>[name,(await readFile(path.join(output,name))).length])));
await writeFile(path.join(output,'model-size.json'),JSON.stringify({parameters:model.parameters,acoustic_parameters:model.acoustic_parameters,vocoder_parameters:model.vocoder_parameters,weight_bytes:weightFiles.reduce((n,k)=>n+sizes[k],0),voice_asset_bytes:Object.values(sizes).reduce((a,b)=>a+b,0),files:sizes,units:'Decimal MB; uncompressed file sizes. HTTP compression, page assets and benchmark files are excluded. This is not RAM usage.'},null,2)+'\n');
await writeFile(path.join(output,'model.json'),JSON.stringify({...model,build:buildId},null,2)+'\n');
const manifest={name:'SmollerTTS',parameters:model.parameters,build:buildId,files:{}};
for(const entry of await readdir(output,{recursive:true,withFileTypes:true})){
  if(!entry.isFile())continue;
  const filename=path.join(entry.parentPath,entry.name),relative=path.relative(output,filename).replaceAll('\\','/');
  const bytes=await readFile(filename);manifest.files[relative]={bytes:bytes.length,sha256:hash(bytes)};
}
await writeFile(path.join(output,'assets.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(`Built ${output}\n${model.parameters.toLocaleString()} parameters · ${(Object.values(sizes).reduce((a,b)=>a+b,0)/1e6).toFixed(2)} MB voice assets`);
