import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';

// Contributor builds must have passed native/ONNX parity and retain the released
// architecture/frontend contract. Changing that contract needs an explicit port.
export async function readCustomModel(directory) {
  const model=JSON.parse(await readFile(path.join(directory,'model.json'),'utf8'));
  const report=JSON.parse(await readFile(path.join(directory,'export.json'),'utf8'));
  if(report.parity_passed!==true||report.checkpoint_sha256!==model.checkpoint_sha256||!report.cases?.length)
    throw new Error('Custom export has no matching successful parity report. Run python -m smollertts.export.');
  if(model.parameters!==2808792||model.acoustic_parameters!==1882807||model.vocoder_parameters!==925985||model.sample_rate!==22050||model.hop!==256)
    throw new Error('Custom model does not match the supported browser architecture.');
  for(const name of ['encoder','frames','vocoder']) {
    const bytes=await readFile(path.join(directory,`${name}.onnx`));
    const digest=createHash('sha256').update(bytes).digest('hex');
    for(const info of [model.graphs?.[name],report.graphs?.[name]]) {
      if(!info||info.bytes!==bytes.length||info.sha256!==digest)throw new Error(`Custom graph changed after export: ${name}`);
    }
  }
  return {...model,experimental:true};
}
