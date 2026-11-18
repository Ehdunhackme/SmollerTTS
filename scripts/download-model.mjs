import {readFile, writeFile, mkdir, rename, rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const lock=JSON.parse(await readFile(path.join(root,'model/assets.lock.json'),'utf8'));
const destination=path.join(root,'assets/model');
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
for(const [name,info] of Object.entries(lock.files)){
  const target=path.resolve(destination,name);
  if(!target.startsWith(destination+path.sep))throw new Error('Invalid asset path');
  try{
    const existing=await readFile(target);
    if(existing.length===info.bytes&&hash(existing)===info.sha256){console.log(`Verified ${name}`);continue;}
  }catch(error){if(error.code!=='ENOENT')throw error;}
  const url=`https://huggingface.co/${lock.repository}/resolve/${lock.revision}/${name}`;
  let bytes;
  for(let attempt=0;attempt<3;attempt++){
    try{
      const response=await fetch(url,{signal:AbortSignal.timeout(180000)});
      if(!response.ok)throw new Error(`${response.status} downloading ${name}`);
      bytes=Buffer.from(await response.arrayBuffer());
      if(bytes.length!==info.bytes||hash(bytes)!==info.sha256)throw new Error(`Integrity check failed: ${name}`);
      break;
    }catch(error){if(attempt===2)throw error;console.warn(`Retrying ${name}: ${error.message}`);}
  }
  await mkdir(path.dirname(target),{recursive:true});
  await writeFile(target+'.download',bytes);
  // Replace only this validated asset; interrupted downloads never become inputs.
  await rm(target,{force:true});await rename(target+'.download',target);
  console.log(`Downloaded ${name} (${(bytes.length/1e6).toFixed(2)} MB)`);
}
console.log('Model assets are ready. Run npm run build.');
