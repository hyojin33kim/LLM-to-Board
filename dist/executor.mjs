import {validatePlan} from './planner.mjs';
const f=Math.fround;
// Physical tile-local SRAM/DDR arrays exercise the planner's address reuse.
export function executePlan(plan,weights,prefix,{faultDescriptor=null}={}){
 const check=validatePlan(plan);if(!check.passed)throw Error(JSON.stringify(check.errors.slice(0,3)));
 if(weights.byteLength!==1056540)throw Error('Checkpoint size mismatch');
 const memory=new Map(plan.memory.map(m=>[`${m.tile}:${m.space}`,new Float32Array(Math.ceil(m.required_bytes/4))]));
 const buffers=new Map(plan.buffers.map(b=>[b.id,b]));
 const storage=b=>memory.get(`${b.tile}:${b.space}`);
 const read=r=>{const b=buffers.get(r.buffer),a=storage(b),v=new Float32Array(r.rows*r.bytes/4);for(let row=0;row<r.rows;row++){const start=(b.address+r.offset+row*r.stride)/4;v.set(a.subarray(start,start+r.bytes/4),row*r.bytes/4)}return v};
 const write=(r,v)=>{if(v.length!==r.rows*r.bytes/4)throw Error('Output shape mismatch');const b=buffers.get(r.buffer),a=storage(b);for(let row=0;row<r.rows;row++)a.set(v.subarray(row*r.bytes/4,(row+1)*r.bytes/4),(b.address+r.offset+row*r.stride)/4)};
 const full=id=>({buffer:id,offset:0,bytes:buffers.get(id).size,rows:1,stride:buffers.get(id).size});
 for(const b of plan.buffers.filter(b=>b.kind==='weight'))new Uint8Array(storage(b).buffer,b.address,b.size).set(new Uint8Array(weights,b.source_offset,b.size));
 const past=plan.token_index*32,prior=new Float32Array(prefix);if(prior.length!==5*2*past)throw Error('KV prefix size mismatch');let cursor=0;
 for(let l=0;l<5;l++)for(const kind of ['K','V']){const b=plan.buffers.find(b=>b.id.endsWith(`.L${l}.${kind}`));storage(b).set(prior.subarray(cursor,cursor+past),b.address/4);cursor+=past;}
 const trace=[],checkpoints=[];let current;
 try{for(const ds of plan.descriptors){current=ds;trace.push({event:'START',descriptor_id:ds.id,tile:ds.tile,graph_nodes:ds.graph_nodes});const ins=ds.inputs.map(read),p=ds.params;let out=[];
 switch(ds.opcode){
 case 'DMA_LOAD':case 'LINK_COPY':case 'KV_COMMIT':out=[ins[0]];break;
 case 'GEMV_ROWS':{const y=new Float32Array(p.M);for(let r=0;r<p.M;r++){let s=0;for(let k=0;k<p.K;k++)s=f(s+f(ins[0][r*p.K+k]*ins[1][k]));y[r]=s}out=[y];break;}
 case 'RMSNORM':{const [x,w]=ins;let s=0;for(const v of x)s=f(s+f(v*v));s=f(s/x.length);s=f(s+f(p.epsilon));s=f(1/f(Math.sqrt(s)));out=[x.map((v,i)=>f(w[i]*f(s*v)))];break;}
 case 'ROPE':{const [q,k]=ins;for(let i=0;i<q.length;i+=2){const freq=f(1/f(Math.pow(p.theta,f((i%p.head_size)/p.head_size)))),v=f(p.position*freq),cr=f(Math.cos(v)),ci=f(Math.sin(v));for(const vec of [q,k])if(i<vec.length){const a=vec[i],b=vec[i+1];vec[i]=f(f(a*cr)-f(b*ci));vec[i+1]=f(f(a*ci)+f(b*cr))}}out=[q,k];break;}
 case 'QK':{const y=new Float32Array(p.context),[q,k]=ins;for(let t=0;t<p.context;t++){let s=0;for(let i=0;i<p.head_size;i++)s=f(s+f(q[i]*k[t*p.head_size+i]));y[t]=f(s/f(Math.sqrt(p.head_size)))}out=[y];break;}
 case 'SOFTMAX':{const x=ins[0];let max=x[0];for(const v of x)if(v>max)max=v;let sum=0;for(let i=0;i<x.length;i++){x[i]=f(Math.exp(f(x[i]-max)));sum=f(sum+x[i])}out=[x.map(v=>f(v/sum))];break;}
 case 'AV':{const y=new Float32Array(p.head_size),[a,v]=ins;for(let t=0;t<p.context;t++)for(let i=0;i<p.head_size;i++)y[i]=f(y[i]+f(a[t]*v[t*p.head_size+i]));out=[y];break;}
 case 'ADD':out=[ins[0].map((v,i)=>f(v+ins[1][i]))];break;
 case 'SWIGLU':out=[ins[0].map((v,i)=>f(f(v*f(1/f(1+f(Math.exp(-v)))))*ins[1][i]))];break;
 case 'OUTPUT_READY':break;
 default:throw Error(`Unsupported opcode ${ds.opcode}`);
 }
 if(ds.id===faultDescriptor&&out[0]?.length)out[0][0]=f(out[0][0]+1);
 ds.outputs.forEach((r,i)=>write(r,out[i]));
 const layerNode=ds.graph_nodes.find(n=>n.includes('.ffn_residual.'));if(layerNode)checkpoints[Number(layerNode.match(/^layer(\d+)/)[1])]=out[0].slice();
 trace.push({event:'END',descriptor_id:ds.id,tile:ds.tile,graph_nodes:ds.graph_nodes});
 }}catch(e){e.descriptor_id=current?.id;e.graph_nodes=current?.graph_nodes;throw e;}
 const logits=read(full(plan.buffers.find(b=>b.id.endsWith('.logits')).id)),kv=new Float32Array(5*2*plan.context*32);cursor=0;
 for(let l=0;l<5;l++)for(const kind of ['K','V']){const b=plan.buffers.find(b=>b.id.endsWith(`.L${l}.${kind}`)),v=read(full(b.id));kv.set(v,cursor);cursor+=v.length;}
 return {logits,kv,checkpoints,trace,descriptors_executed:plan.descriptors.length};
}
export function compareArray(actual,expected,{atol=1e-5,rtol=1e-4}={}){
 if(actual.length!==expected.length)throw Error('Comparison size mismatch');let maxAbs=0,sq=0,refSq=0,failed=0,first=null;
 for(let i=0;i<actual.length;i++){const a=actual[i],b=expected[i],diff=Math.abs(a-b);if(!Number.isFinite(a)||!Number.isFinite(b)||diff>atol+rtol*Math.abs(b)){failed++;if(first===null)first=i}maxAbs=Math.max(maxAbs,diff);sq+=diff*diff;refSq+=b*b;}
 return {passed:failed===0,elements:actual.length,max_abs_error:maxAbs,relative_l2:Math.sqrt(sq/Math.max(refSq,1e-30)),failed_elements:failed,first_failed_index:first,atol,rtol};
}
export function compareGolden(result,golden,context){const ref=new Float32Array(golden),kvEnd=512+5*2*context*32;if(ref.length!==kvEnd+320)throw Error('Golden size mismatch');const logits=compareArray(result.logits,ref.subarray(0,512)),kv=compareArray(result.kv,ref.subarray(512,kvEnd));const layers=result.checkpoints.map((x,i)=>({layer:i,...compareArray(x,ref.subarray(kvEnd+i*64,kvEnd+(i+1)*64))}));return {passed:logits.passed&&kv.passed&&layers.every(x=>x.passed),logits,kv,layers,first_failed_layer:layers.find(x=>!x.passed)?.layer??null};}
