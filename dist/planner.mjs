// Declarative, serial FP32 decode-step plan. No HW execution or numerical claims.
const ALIGN=64, align=n=>Math.ceil(n/ALIGN)*ALIGN;
const gid=(l,op,h=-1)=>`layer${String(l).padStart(2,'0')}.${op}.head${h<0?'-1':String(h).padStart(2,'0')}`;
const slice=(buffer,offset,bytes,rows=1,stride=bytes)=>({buffer,offset,bytes,rows,stride});
export function generatePlan(c,split,{tokenId=1}={}){
 const m=c.model,h=c.hardware,d=m.dim,H=m.hidden_dim,L=m.n_layers,heads=m.n_heads,kvHeads=m.n_kv_heads,hd=d/heads,kd=hd*kvHeads,C=c.context,pos=C-1;
 if(c.concurrency!==1)throw Error('실행 계획 v0.4는 단일 요청 전용입니다. 동시 요청을 1로 바꾼 뒤 자동 분할을 실행하세요.');
 if(d!==64||H!==172||L!==5||heads!==8||kvHeads!==4||m.vocab_size!==512||m.seq_len!==512)throw Error('검증된 Stories260K 구조만 지원합니다.');
 if(!Number.isInteger(C)||C<1||C>m.seq_len)throw Error('context 범위 오류');
 if(!Number.isInteger(tokenId)||tokenId<0||tokenId>=m.vocab_size)throw Error('token ID 범위 오류');
 if(split.length!==c.tiles||split.some(n=>!Number.isInteger(n)||n<1)||split.reduce((a,b)=>a+b,0)!==L)throw Error('분할이 전체 layer를 정확히 포함해야 합니다.');
 if([h.sram_bytes_per_tile,h.ddr_bytes_per_tile,h.debug_bytes_per_tile,h.reserve_fraction].some(v=>!Number.isFinite(v)||v<0)||h.reserve_fraction>=1)throw Error('메모리 설정 오류');
 const buffers=[],descriptors=[],buf=new Map();
 const add=(id,tile,space,size,kind='scratch',initial=[],sourceOffset=null)=>{if(buf.has(id))throw Error('buffer 중복');const b={id,tile,space,size,kind,initial,source_offset:sourceOffset};buffers.push(b);buf.set(id,b);return id};
 const full=id=>slice(id,0,buf.get(id).size);
 const emit=(tile,opcode,nodes,inputs,outputs,params={})=>{const id=descriptors.length;descriptors.push({id,tile,opcode,graph_nodes:nodes,wait_for:id?[id-1]:[],inputs,outputs,params});};
 const tensor=(tile,name,n)=>add(`T${tile}.${name}`,tile,'SRAM',n*4);
 const offsets={};let off=28;
 for(const [name,n] of [['embedding',m.vocab_size*d],['att_norm',L*d],['q',L*d*d],['k',L*d*kd],['v',L*d*kd],['o',L*d*d],['ffn_norm',L*d],['w1',L*d*H],['w2',L*d*H],['w3',L*d*H],['final_norm',d]]){offsets[name]=off;off+=n*4;}
 const layerWords={att_norm:d,q:d*d,k:d*kd,v:d*kd,o:d*d,ffn_norm:d,w1:d*H,w2:d*H,w3:d*H};
 const weight=(tile,l,name,n)=>{const id=`T${tile}.W.${l}.${name}`;if(buf.has(id))return id;const src=name==='embedding'?offsets.embedding:name==='final_norm'?offsets.final_norm:offsets[name]+l*layerWords[name]*4;return add(id,tile,'DDR',n*4,'weight',[[0,n*4]],src)};
 let current=null,lastTile=null,layer=0;
 for(let t=0;t<split.length;t++){
  add(`T${t}.debug`,t,'SRAM',h.debug_bytes_per_tile,'debug');
  const staging=add(`T${t}.weight_staging`,t,'SRAM',4096,'scratch');
  let x=tensor(t,'input',d);
  if(t===0){const emb=weight(t,-1,'embedding',m.vocab_size*d);emit(t,'DMA_LOAD',[gid(-1,'embedding')],[slice(emb,tokenId*d*4,d*4)],[full(x)],{token_id:tokenId});}
  else emit(t,'LINK_COPY',[`link.T${lastTile}.T${t}`],[full(current)],[full(x)],{source_tile:lastTile,destination_tile:t,bytes:d*4});
  const mat=(l,name,input,output,n,outs)=>{
   const w=weight(t,l,name,n*outs);const rowsPer=Math.floor(4096/(n*4));if(!rowsPer)throw Error('GEMV row exceeds staging');
   for(let r=0;r<outs;r+=rowsPer){const count=Math.min(rowsPer,outs-r),bytes=count*n*4;
    emit(t,'DMA_LOAD',[gid(l,name)],[slice(w,r*n*4,bytes)],[slice(staging,0,bytes)]);
    emit(t,'GEMV_ROWS',[gid(l,name)],[slice(staging,0,bytes),full(input)],[slice(output,r*4,count*4)],{M:count,K:n,output_row:r,dtype:'FP32',accumulator:'FP32',reduction:'ascending K, no FMA'});
   }
  };
  const norm=(l,name,input,output)=>{const w=weight(t,l,name,d);emit(t,'DMA_LOAD',[gid(l,name)],[full(w)],[slice(staging,0,d*4)]);emit(t,'RMSNORM',[gid(l,name)],[full(input),slice(staging,0,d*4)],[full(output)],{epsilon:1e-5,elements:d,reduction:'ascending index'});};
  for(let count=0;count<split[t];count++,layer++){
   const l=layer,pre=tensor(t,`L${l}.norm`,d),q=tensor(t,`L${l}.q`,d);
   const keys=add(`T${t}.L${l}.K`,t,'SRAM',C*kd*4,'kv',pos?[[0,pos*kd*4]]:[]),values=add(`T${t}.L${l}.V`,t,'SRAM',C*kd*4,'kv',pos?[[0,pos*kd*4]]:[]);
   const kt=tensor(t,`L${l}.k_current`,kd),vt=tensor(t,`L${l}.v_current`,kd);
   norm(l,'att_norm',x,pre);mat(l,'q',pre,q,d,d);mat(l,'k',pre,kt,d,kd);mat(l,'v',pre,vt,d,kd);
   emit(t,'ROPE',[gid(l,'rope')],[full(q),full(kt)],[full(q),full(kt)],{position:pos,head_size:hd,theta:10000});
   emit(t,'KV_COMMIT',[gid(l,'k'),gid(l,'rope')],[full(kt)],[slice(keys,pos*kd*4,kd*4)],{position:pos});
   emit(t,'KV_COMMIT',[gid(l,'v')],[full(vt)],[slice(values,pos*kd*4,kd*4)],{position:pos});
   const av=tensor(t,`L${l}.attention_out`,d);
   for(let head=0;head<heads;head++){
    const scores=tensor(t,`L${l}.head${head}.scores`,C),khead=Math.floor(head/(heads/kvHeads)),offset=khead*hd*4;
    emit(t,'QK',[gid(l,'qk',head)],[slice(q,head*hd*4,hd*4),slice(keys,offset,hd*4,C,kd*4)],[full(scores)],{context:C,head_size:hd,scale:`1/sqrt(${hd})`});
    emit(t,'SOFTMAX',[gid(l,'softmax',head)],[full(scores)],[full(scores)],{elements:C,stable:true,reduction:'ascending token'});
    emit(t,'AV',[gid(l,'av',head)],[full(scores),slice(values,offset,hd*4,C,kd*4)],[slice(av,head*hd*4,hd*4)],{context:C,head_size:hd,reduction:'ascending token'});
   }
   const o=tensor(t,`L${l}.o`,d),res=tensor(t,`L${l}.residual`,d);mat(l,'o',av,o,d,d);emit(t,'ADD',[gid(l,'att_residual')],[full(x),full(o)],[full(res)]);
   const fn=tensor(t,`L${l}.ffn_norm`,d),u=tensor(t,`L${l}.w1`,H),v=tensor(t,`L${l}.w3`,H),fout=tensor(t,`L${l}.w2`,d),next=tensor(t,`L${l}.output`,d);
   norm(l,'ffn_norm',res,fn);mat(l,'w1',fn,u,d,H);mat(l,'w3',fn,v,d,H);
   emit(t,'SWIGLU',[gid(l,'swiglu')],[full(u),full(v)],[full(u)],{elements:H,formula:'(u * (1 / (1 + exp(-u)))) * v',fusion_group:`L${l}.FFN_local`});
   mat(l,'w2',u,fout,H,d);emit(t,'ADD',[gid(l,'ffn_residual')],[full(res),full(fout)],[full(next)]);x=next;
  }
  current=x;lastTile=t;
  if(t===split.length-1){const normout=tensor(t,'final_norm',d),logits=tensor(t,'logits',m.vocab_size);norm(-1,'final_norm',x,normout);
   // classifier shares embedding weights in this checkpoint, replicate table across tiles only.
   const emb=weight(t,-1,'embedding',m.vocab_size*d),per=Math.floor(4096/(d*4));
   for(let r=0;r<m.vocab_size;r+=per){const count=Math.min(per,m.vocab_size-r),bytes=count*d*4;emit(t,'DMA_LOAD',[gid(-1,'classifier')],[slice(emb,r*d*4,bytes)],[slice(staging,0,bytes)]);emit(t,'GEMV_ROWS',[gid(-1,'classifier')],[slice(staging,0,bytes),full(normout)],[slice(logits,r*4,count*4)],{M:count,K:d,output_row:r,dtype:'FP32',accumulator:'FP32',reduction:'ascending K, no FMA'});}
   emit(t,'OUTPUT_READY',['output.logits'],[full(logits)],[],{bytes:m.vocab_size*4});
  }
 }
 // Derive live intervals, then first-fit aligned allocation; persistent KV/debug stay alive.
 const end=descriptors.length;for(const b of buffers){b.first=end;b.last=-1;}
 for(const ds of descriptors)for(const r of [...ds.inputs,...ds.outputs]){const b=buf.get(r.buffer);b.first=Math.min(b.first,ds.id);b.last=Math.max(b.last,ds.id)}
 for(const b of buffers)if(['weight','kv','debug'].includes(b.kind)){b.first=0;b.last=end;}
 const memory=[];
 for(let t=0;t<split.length;t++)for(const space of ['SRAM','DDR']){
  const bs=buffers.filter(b=>b.tile===t&&b.space===space).sort((a,b)=>a.first-b.first||b.last-a.last||a.id.localeCompare(b.id));const placed=[];
  for(const b of bs){let addr=0;const live=placed.filter(x=>!(x.last<b.first||b.last<x.first)).sort((a,b)=>a.address-b.address);for(const x of live){if(addr+align(b.size)<=x.address)break;addr=Math.max(addr,x.address+align(x.size));}b.address=align(addr);b.allocated_bytes=align(b.size);placed.push(b);}
  const required=Math.max(0,...bs.map(b=>b.address+b.allocated_bytes)),capacity=space==='SRAM'?h.sram_bytes_per_tile:h.ddr_bytes_per_tile;
  memory.push({tile:t,space,required_bytes:required,capacity_bytes:capacity,usable_bytes:Math.floor(capacity*(1-h.reserve_fraction))});
 }
 const plan={schema:'tile-plan-0.4',status:'declarative_not_executed',model:'Stories260K',source_commit:'661292a84ec02d5fda4ec5f61ea03f3a8e4a23cf',model_sha256:'b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696',dtype:'FP32',context:C,token_index:pos,token_id:tokenId,requests:1,split,execution_policy:'strict serial, completion includes all output writes; no overlap',preconditions:['All DDR weight regions loaded from source_offset in the pinned checkpoint','KV prefix [0, token_index) supplied by the preceding token steps on each owning tile','New session must start at token_index 0; a nonzero index is a continuation, not a cold boot'],buffers,descriptors,memory};
 plan.validation=validatePlan(plan);return plan;
}
export function validatePlan(plan){
 const errors=[],map=new Map(),ids=new Set(),initialized=new Map();
 const err=(code,detail)=>errors.push({code,detail});
 const merge=(a)=>{a.sort((x,y)=>x[0]-y[0]);const out=[];for(const [s,e] of a){if(out.length&&s<=out.at(-1)[1])out.at(-1)[1]=Math.max(e,out.at(-1)[1]);else out.push([s,e]);}return out};
 for(const b of plan.buffers){if(map.has(b.id))err('BUFFER_DUPLICATE',b.id);map.set(b.id,b);initialized.set(b.id,(b.initial||[]).map(x=>[...x]));if(!Number.isInteger(b.address)||b.address<0||b.address%ALIGN)err('ALIGNMENT',b.id);if(b.size<=0||b.allocated_bytes<b.size)err('ALLOCATION_SIZE',b.id);}
 const actualLife=new Map(plan.buffers.map(b=>[b.id,{first:Infinity,last:-1}]));
 for(let i=0;i<plan.descriptors.length;i++){
  const ds=plan.descriptors[i];if(ids.has(ds.id)||ds.id!==i)err('DESCRIPTOR_ID',String(ds.id));ids.add(ds.id);
  if(JSON.stringify(ds.wait_for)!==JSON.stringify(i?[i-1]:[]))err('DEPENDENCY',String(ds.id));
  for(const direction of ['inputs','outputs'])for(const r of ds[direction]){
   const b=map.get(r.buffer);if(!b){err('UNKNOWN_BUFFER',r.buffer);continue;}
   const life=actualLife.get(b.id);life.first=Math.min(life.first,i);life.last=Math.max(life.last,i);
   if(b.tile!==ds.tile&&!(ds.opcode==='LINK_COPY'&&direction==='inputs'&&b.tile===ds.params.source_tile))err('TILE_OWNERSHIP',`${i}:${b.id}`);
   if(![r.offset,r.bytes,r.rows,r.stride].every(Number.isInteger)||r.offset<0||r.bytes<=0||r.rows<1||r.stride<r.bytes||r.offset+(r.rows-1)*r.stride+r.bytes>b.size){err('BOUNDS',`${i}:${b.id}`);continue;}
   const spans=Array.from({length:r.rows},(_,n)=>[r.offset+n*r.stride,r.offset+n*r.stride+r.bytes]);
   if(direction==='inputs')for(const [s,e] of spans)if(!initialized.get(b.id).some(([a,z])=>a<=s&&z>=e)){err('READ_BEFORE_WRITE',`${i}:${b.id} [${s},${e})`);break;}
   if(direction==='outputs'){if(b.space==='DDR')err('WEIGHT_WRITE',`${i}:${b.id}`);initialized.set(b.id,merge([...initialized.get(b.id),...spans]));}
  }
 }
 for(const b of plan.buffers){const life=actualLife.get(b.id);if(['weight','kv','debug'].includes(b.kind)){life.first=0;life.last=plan.descriptors.length;}if(b.first!==life.first||b.last!==life.last)err('LIFETIME',b.id);}
 for(let i=0;i<plan.buffers.length;i++)for(let j=i+1;j<plan.buffers.length;j++){
  const a=plan.buffers[i],b=plan.buffers[j];if(a.tile!==b.tile||a.space!==b.space)continue;
  const la=actualLife.get(a.id),lb=actualLife.get(b.id);if(a.address<b.address+b.allocated_bytes&&b.address<a.address+a.allocated_bytes&&la.first<=lb.last&&lb.first<=la.last)err('LIVE_OVERLAP',`${a.id} / ${b.id}`);
 }
 for(const m of plan.memory){const required=Math.max(0,...plan.buffers.filter(b=>b.tile===m.tile&&b.space===m.space).map(b=>b.address+b.allocated_bytes));if(required!==m.required_bytes)err('MEMORY_SUMMARY',`${m.tile}:${m.space}`);if(required>m.usable_bytes)err('CAPACITY',`T${m.tile} ${m.space}: ${required} > ${m.usable_bytes}`);}
 return {passed:errors.length===0,checks:['bounds','alignment','strict dependency chain','read-before-write under declared preconditions','recomputed lifetimes','live address overlap','capacity','tile ownership'],errors};
}
