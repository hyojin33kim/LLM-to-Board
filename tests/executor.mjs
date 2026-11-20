import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {generatePlan} from '../dist/planner.mjs';
import {executePlan,compareGolden,compareArray} from '../dist/executor.mjs';
const file=p=>{const b=readFileSync(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};
const base=JSON.parse(readFileSync('dist/config.json'));base.concurrency=1;base.hardware.ddr_bytes_per_tile=2*1048576;
const weights=file('dist/fixtures/weights.bin');
function* splits(n){if(n===0){yield [];return}for(let i=1;i<=n;i++)for(const tail of splits(n-i))yield [i,...tail]}
const runs=[];
for(const context of [1,128,512])for(const split of splits(5)){const c={...base,context,tiles:split.length},plan=generatePlan(c,split),result=executePlan(plan,weights,file(`dist/fixtures/prefix_${context}.bin`)),comparison=compareGolden(result,file(`dist/fixtures/golden_${context}.bin`),context);assert.ok(comparison.passed,JSON.stringify({context,split,comparison}));runs.push({context,split,comparison})}
const c={...base,context:128,tiles:2},plan=generatePlan(c,[3,2]),bad=plan.descriptors.find(d=>d.opcode==='ADD'&&d.graph_nodes.some(n=>n.includes('layer00.ffn_residual.')));const fault=compareGolden(executePlan(plan,weights,file('dist/fixtures/prefix_128.bin'),{faultDescriptor:bad.id}),file('dist/fixtures/golden_128.bin'),128);assert.equal(fault.passed,false);assert.equal(fault.first_failed_layer,0);
let prefix=new ArrayBuffer(0),result,maxAbs=0,maxL2=0;const all=new Float32Array(file('validation/all_logits.bin'));
for(let context=1;context<=512;context++){const p=generatePlan({...base,context,tiles:2},[3,2]);result=executePlan(p,weights,prefix);const cmp=compareArray(result.logits,all.subarray((context-1)*512,context*512));assert.ok(cmp.passed,`continuous context ${context}: ${JSON.stringify(cmp)}`);maxAbs=Math.max(maxAbs,cmp.max_abs_error);maxL2=Math.max(maxL2,cmp.relative_l2);prefix=result.kv.buffer;}
const finalKV=compareArray(result.kv,new Float32Array(file('validation/final_kv.bin')));assert.ok(finalKV.passed);
const report={schema:'tile-validation-0.5',backend:'Node functional interpreter; not hardware timing',input:'token ID 1 repeated 512 times',source_commit:plan.source_commit,model_sha256:plan.model_sha256,fixture_cases:runs.length,runs,fault_injection:{descriptor_id:bad.id,graph_nodes:bad.graph_nodes,detected:!fault.passed,first_failed_layer:fault.first_failed_layer},continuous:{steps:512,split:[3,2],prefix:'previous interpreter result (no Golden KV injection)',max_abs_logit_error:maxAbs,max_relative_l2_logit:maxL2,final_kv:finalKV}};
writeFileSync('dist/fixtures/validation-report.json',JSON.stringify(report,null,2));console.log(JSON.stringify({...report,runs:undefined},null,2));
