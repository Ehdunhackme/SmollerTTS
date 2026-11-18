import test from 'node:test';
import assert from 'node:assert/strict';
import {hasMetric, promptValues, bounds, frontier} from '../web/benchmark-metrics.mjs';

test('missing WER is excluded, but a genuine zero is retained',()=>{
  assert.equal(hasMetric({wer:null},'wer'),false);
  assert.equal(hasMetric({wer:0},'wer'),true);
  assert.deepEqual(promptValues({per_prompt:[{wer:''},{wer:null},{wer:'0.0'},{wer:'0.25'}]},'wer'),[0,.25]);
});
test('frontier follows the correct metric direction and handles equal budgets',()=>{
  const rows=[{id:'a',parameters_m:1,utmos:3,wer:.1},{id:'b',parameters_m:1,utmos:4,wer:.2},{id:'c',parameters_m:2,utmos:3.8,wer:.05},{id:'d',parameters_m:3,utmos:4.2,wer:.07},{id:'e',parameters_m:4,utmos:4,wer:null}];
  assert.deepEqual(frontier(rows,'utmos').map(r=>r.id),['b','d']);
  assert.deepEqual(frontier(rows,'wer').map(r=>r.id),['a','c']);
});
test('prompt-range axes include outliers, including WER over 100 percent',()=>{
  const rows=[{wer:.2,per_prompt:[{wer:'1.27'},{wer:'0.0'}]}];
  assert.equal(bounds(rows,'wer',false).max,.2);
  assert.ok(bounds(rows,'wer',true).max>=1.27);
  assert.equal(bounds([{utmos:3,per_prompt:[{utmos:1.2},{utmos:4.8}]}],'utmos',true).min,1);
});
