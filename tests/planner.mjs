import assert from 'node:assert/strict';import {readFileSync,writeFileSync} from 'node:fs';import {generatePlan,validatePlan} from '../dist/planner.mjs';import {solve} from '../dist/engine.mjs';
const base=JSON.parse(readFileSync(new URL('../dist/config.json',import.meta.url))),rows=JSON.parse(readFileSync(new URL('../dist/profile.json',import.meta.url)));base.concurrency=1;let tested=0;
for(const context of [1,128,512])for(const tiles of [1,2,3,4,5]){const c=structuredClone(base);c.context=context;c.tiles=tiles;c.hardware.ddr_bytes_per_tile=2*1048576;for(const x of solve(c,rows).candidates){const p=generatePlan(c,x.split);assert.ok(p.validation.passed,JSON.stringify(p.validation.errors.slice(0,3)));const gids=new Set(p.descriptors.flatMap(d=>d.graph_nodes));for(const r of rows.filter(r=>r.pos===context-1)){const id=`layer${String(r.layer).padStart(2,'0')}.${r.op}.head${r.head<0?'-1':String(r.head).padStart(2,'0')}`;assert.ok(gids.has(id),id)}assert.equal(p.descriptors.filter(x=>x.opcode==='LINK_COPY').reduce((s,x)=>s+x.params.bytes,0),(tiles-1)*256);assert.equal(p.buffers.filter(x=>x.kind==='kv').reduce((s,b)=>s+b.size,0),5*2*context*32*4);tested++}}
const c=structuredClone(base),p=generatePlan(c,[3,2]);assert.ok(p.validation.passed);const reject=(modify,code)=>{const q=structuredClone(p);modify(q);assert.ok(validatePlan(q).errors.some(e=>e.code===code),code)};
reject(q=>{q.descriptors[2].wait_for=[]},'DEPENDENCY');
reject(q=>{q.descriptors[0].inputs[0].offset=1e9},'BOUNDS');
reject(q=>{q.buffers.filter(b=>b.kind==='weight')[0].initial=[]},'READ_BEFORE_WRITE');
reject(q=>{const b=q.buffers.filter(b=>b.tile===0&&b.kind==='kv');b[1].address=b[0].address},'LIVE_OVERLAP');
reject(q=>{q.buffers.find(b=>b.kind==='kv').last=0},'LIFETIME');
reject(q=>{q.descriptors[1].tile=1},'TILE_OWNERSHIP');
const small=structuredClone(c);small.hardware.sram_bytes_per_tile=1024;assert.ok(!generatePlan(small,[3,2]).validation.passed);
const multi=structuredClone(c);multi.concurrency=8;assert.throws(()=>generatePlan(multi,[3,2]),/단일 요청/);
writeFileSync(new URL('./plan-example.json',import.meta.url),JSON.stringify(p,null,2));console.log(`PASS ${tested} complete layer partitions; 8 negative checks; all 188 profile graph nodes covered`);console.log(JSON.stringify({descriptors:p.descriptors.length,memory:p.memory,weight_dma_bytes:p.descriptors.filter(d=>d.opcode==='DMA_LOAD').reduce((s,d)=>s+d.inputs[0].bytes,0)},null,2));
