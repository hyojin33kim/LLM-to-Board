export function solve(c,rows){
 const h=c.hardware,m=c.model,N=c.tiles,L=m.n_layers,ctx=c.context;
 if(!Number.isInteger(N)||N<1||N>L||!Number.isInteger(c.concurrency)||c.concurrency<1)throw Error('Tile 수와 동시 요청 수를 확인하세요.');
 for(const [k,v] of Object.entries(h))if(!Number.isFinite(v)||v<0)throw Error('HW 설정은 유효한 양수여야 합니다.');
 for(const k of ['clock_MHz','mac_per_cycle','vector_elements_per_cycle','exp_per_cycle','ddr_bytes_per_cycle','sram_bytes_per_cycle','link_bytes_per_cycle'])if(h[k]<=0)throw Error('처리율과 클록은 0보다 커야 합니다.');
 const rr=rows.filter(r=>r.pos===ctx-1);if(!rr.length)throw Error('지원하지 않는 context입니다.');
 const cuts=[];function comb(a,left,start){if(!left){cuts.push(a);return}for(let i=start;i<L;i++)comb([...a,i],left-1,i+1)}comb([],N-1,1);
 const d=m.dim,kd=d*m.n_kv_heads/m.n_heads,f=m.hidden_dim,v=m.vocab_size,lw=(2*d*d+2*d*kd+3*d*f+2*d)*4,emb=d*v*4,link=Math.ceil(d*4/h.link_bytes_per_cycle)+h.link_latency_cycles;
 const candidates=cuts.map(cut=>{const b=[0,...cut,L],reasons=[],stages=[];for(let t=0;t<N;t++){
 const lo=b[t],hi=b[t+1],r=rr.filter(x=>(x.layer>=lo&&x.layer<hi)||(x.op==='embedding'&&t===0)||(['final_norm','classifier'].includes(x.op)&&t===N-1));
 const cycles=r.reduce((sum,x)=>{const work=(x.mac?x.mac/h.mac_per_cycle:Math.max(x.m*x.n,1)/h.vector_elements_per_cycle)+x.exp_calls/h.exp_per_cycle;return sum+Math.ceil(h.launch_cycles+Math.max(work,x.weight_bytes/h.ddr_bytes_per_cycle,(x.read_bytes+x.write_bytes-x.weight_bytes)/h.sram_bytes_per_cycle))},0);
 const weight=(hi-lo)*lw+((t===0||t===N-1)?emb:0)+(t===N-1?d*4:0),kv=(hi-lo)*2*ctx*kd*4*c.concurrency,sram=kv+h.scratch_bytes_per_tile+h.dma_buffer_bytes_per_tile+h.debug_bytes_per_tile;
 if(sram>h.sram_bytes_per_tile*(1-h.reserve_fraction))reasons.push(`T${t}: SRAM 부족 (${(sram/1048576).toFixed(2)} MiB 필요)`);
 if(weight>h.ddr_bytes_per_tile*(1-h.reserve_fraction))reasons.push(`T${t}: DDR 부족 (${(weight/1048576).toFixed(2)} MiB 필요)`);
 stages.push({tile:t,layers:Array.from({length:hi-lo},(_,i)=>lo+i),cycles,weight_bytes:weight,kv_bytes:kv,sram_required_bytes:sram});}
 const latency=stages.reduce((s,x)=>s+x.cycles,0)+(N-1)*link,interval=Math.max(...stages.map(x=>x.cycles+(x.tile<N-1?link:0))),agg=Math.min(c.concurrency/latency,1/interval)*h.clock_MHz*1e6;
 return {cuts:cut,split:b.slice(1).map((x,i)=>x-b[i]),stages,reasons,feasible:!reasons.length,latency_cycles:latency,pipeline_interval_cycles:interval,estimated_aggregate_tps:agg,estimated_per_user_tps:agg/c.concurrency}});
 const score=x=>c.objective==='latency'?[x.latency_cycles,x.pipeline_interval_cycles,...x.cuts]:[-x.estimated_aggregate_tps,x.latency_cycles,x.pipeline_interval_cycles,...x.cuts];
 const ok=candidates.filter(x=>x.feasible).sort((a,b)=>{const x=score(a),y=score(b);for(let i=0;i<x.length;i++)if(x[i]!==y[i])return x[i]-y[i];return 0});return {candidates,selected:ok[0]||null};
}
