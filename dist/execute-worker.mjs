import {executePlan,compareGolden} from './executor.mjs';
self.onmessage=async({data})=>{try{
 const {plan,injectFault}=data;
 if(![1,128,512].includes(plan.context)||plan.token_id!==1)throw Error('Golden supports contexts 1/128/512 and token ID 1');
 const mr=await fetch('./fixtures/manifest.json');if(!mr.ok)throw Error('Manifest fetch failed');const manifest=await mr.json();
 const get=async name=>{const r=await fetch('./fixtures/'+name);if(!r.ok)throw Error(`Fixture fetch failed: ${name}`);const b=await r.arrayBuffer(),sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b)),x=>x.toString(16).padStart(2,'0')).join('');if(b.byteLength!==manifest.files[name].bytes||sha!==manifest.files[name].sha256)throw Error(`Fixture integrity failed: ${name}`);return b};
 self.postMessage({type:'progress',message:'가중치·C Golden 무결성 확인 중'});
 const [weights,prefix,golden]=await Promise.all(['weights.bin',`prefix_${plan.context}.bin`,`golden_${plan.context}.bin`].map(get));
 const fault=injectFault?plan.descriptors.find(d=>d.opcode==='ADD'&&d.graph_nodes.some(n=>n.includes('layer00.ffn_residual.'))):null;
 const start=performance.now(),result=executePlan(plan,weights,prefix,{faultDescriptor:fault?.id}),walltime=performance.now()-start,comparison=compareGolden(result,golden,plan.context);
 self.postMessage({type:'result',result:{schema:'tile-validation-0.5',backend:'browser functional interpreter',not_hardware_timing:true,fixture_input:'single step with C-computed repeat-token-1 KV prefix',context:plan.context,split:plan.split,source_commit:plan.source_commit,token_id:1,descriptors_executed:result.descriptors_executed,comparison,walltime_ms:walltime,fault_injection:fault?{descriptor_id:fault.id,graph_nodes:fault.graph_nodes}:null,trace:result.trace}});
 }catch(e){self.postMessage({type:'error',message:e.message,descriptor_id:e.descriptor_id,graph_nodes:e.graph_nodes})}};
