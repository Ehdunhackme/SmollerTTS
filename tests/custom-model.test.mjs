import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {readCustomModel} from '../scripts/custom-model.mjs';

test('custom model requires a matching export report and unchanged graphs',async()=>{
  const directory=await mkdtemp(path.join(tmpdir(),'smollertts-custom-'));
  try{
    const graphs={};
    for(const name of ['encoder','frames','vocoder']){
      const bytes=Buffer.from(`test graph ${name}`);
      graphs[name]={bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')};
      await writeFile(path.join(directory,`${name}.onnx`),bytes);
    }
    const model={parameters:2808792,acoustic_parameters:1882807,vocoder_parameters:925985,sample_rate:22050,hop:256,checkpoint_sha256:'fixture',graphs};
    const report={parity_passed:true,checkpoint_sha256:'fixture',graphs,cases:[{}]};
    await writeFile(path.join(directory,'model.json'),JSON.stringify(model));
    await writeFile(path.join(directory,'export.json'),JSON.stringify(report));
    assert.equal((await readCustomModel(directory)).experimental,true);
    await writeFile(path.join(directory,'export.json'),JSON.stringify({...report,parity_passed:false}));
    await assert.rejects(readCustomModel(directory),/parity report/);
    await writeFile(path.join(directory,'export.json'),JSON.stringify(report));
    await writeFile(path.join(directory,'encoder.onnx'),'changed');
    await assert.rejects(readCustomModel(directory),/changed after export/);
  }finally{
    // mkdtemp returned this exact test-owned directory.
    assert.ok(directory.startsWith(path.join(tmpdir(),'smollertts-custom-')));
    await rm(directory,{recursive:true,force:true});
  }
});
