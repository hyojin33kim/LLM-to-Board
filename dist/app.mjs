import {solve} from './engine.mjs';
import {generatePlan} from './planner.mjs';
const $=id=>document.getElementById(id);let base,rows,c,result,chosen,plan,validationResult,executionWorker;
const fields={context:'context',users:'concurrency',tiles:'tiles',objective:'objective'};
const hw={sram:['sram_bytes_per_tile',1048576],ddr:['ddr_bytes_per_tile',1048576],clock:['clock_MHz',1],mac:['mac_per_cycle',1],bw:['ddr_bytes_per_cycle',1],link:['link_bytes_per_cycle',1]};
function form(){let x=structuredClone(base);for(const [id,k] of Object.entries(fields))x[k]=id==='objective'?$(id).value:Number($(id).value);for(const [id,[k,m]] of Object.entries(hw))x.hardware[k]=Number($(id).value)*m;return x}
function fill(x){if(!x||x.model?.dim!==64||x.model?.n_layers!==5)throw Error('Stories260K 설정 JSON을 선택하세요.');for(const [id,k] of Object.entries(fields))$(id).value=x[k];for(const [id,[k,m]] of Object.entries(hw))$(id).value=x.hardware[k]/m;}
function clearPlan(){clearExecution();plan=null;$('downloadPlan').disabled=true;$('planStatus').textContent='배치를 선택한 뒤 계획을 생성하세요.';$('planMemory').innerHTML='';for(const id of ['planPreconditions','planBuffers','planDescriptors'])$(id).hidden=true;}
function render(){clearPlan();const x=chosen;const number=v=>v.toLocaleString('ko-KR',{maximumFractionDigits:1});$('metrics').innerHTML=[['선택한 배치',x?x.split.join(' + '):'배치 불가','연속 Transformer layers'],['예상 지연',x?number(x.latency_cycles/c.hardware.clock_MHz)+' µs':'—','가정 기반 · 1 token / 요청'],['예상 합산 처리량',x&&x.feasible?number(x.estimated_aggregate_tps):'—','가정 기반 · token/s']].map(a=>`<div class="card metric"><span class="muted">${a[0]}</span><strong>${a[1]}</strong><small>${a[2]}</small></div>`).join('');
$('selectedLabel').textContent=!x?'적합 후보 없음':x===result.selected?'자동 추천':'후보 검토';
$('tilesView').innerHTML=x?x.stages.map(s=>{const perc=s.sram_required_bytes/(c.hardware.sram_bytes_per_tile*.9)*100;return `<div class="tile"><b>Tile ${s.tile}</b><div class="layers">${s.layers.map(l=>`<span>L${l}</span>`).join('')}</div><div>${number(s.cycles)} cycles</div><p class="muted">SRAM ${(s.sram_required_bytes/1048576).toFixed(2)} / ${(c.hardware.sram_bytes_per_tile*.9/1048576).toFixed(2)} MiB 유효</p><div class="bar"><i style="width:${Math.min(100,perc)}%;background:${perc>100?'#c45d4c':'#20a5a0'}"></i></div><small>Weight ${(s.weight_bytes/1024).toFixed(1)} KiB<br>KV ${(s.kv_bytes/1024).toFixed(1)} KiB</small></div>`}).join(''):'메모리 용량을 늘리거나 동시 요청 수를 줄여 다시 비교하세요.';
$('candidates').innerHTML=result.candidates.map((v,i)=>`<tr class="${v===x?'active':''}"><td><b>${v.split.join(' + ')}</b>${v===result.selected?' ★':''}</td><td class="${v.feasible?'':'bad'}">${v.feasible?'적합':'용량 초과'}</td><td>${number(v.latency_cycles/c.hardware.clock_MHz)}</td><td>${v.feasible?number(v.estimated_aggregate_tps):'—'}</td><td>${v.feasible?number(v.estimated_per_user_tps):'—'}</td><td><button data-index="${i}">선택</button></td></tr>`).join('');
$('reasons').textContent=x?.reasons.join(' / ')||(!result.selected?'모든 후보가 메모리 제약을 초과합니다. 후보를 선택하면 Tile별 원인을 볼 수 있습니다.':'');$('reasons').className='error';
$('candidates').querySelectorAll('button').forEach(b=>b.onclick=()=>{chosen=result.candidates[Number(b.dataset.index)];render()});}
function run(){clearPlan();try{c=form();result=solve(c,rows);chosen=result.selected;$('err').textContent='';render();$('status').textContent=`${result.candidates.length}개 후보 계산 완료`;}catch(e){$('err').textContent=e.message;}}
$('run').onclick=run;
$('save').onclick=()=>{try{localStorage.setItem('llm-tile-config-v1',JSON.stringify(form()));$('status').textContent='이 브라우저에 설정을 저장했습니다.';}catch{$('err').textContent='브라우저 저장을 사용할 수 없습니다. JSON 다운로드를 사용하세요.'}};
$('restore').onclick=()=>{try{const s=localStorage.getItem('llm-tile-config-v1');if(!s)throw Error('저장된 설정이 없습니다.');fill(JSON.parse(s));run();}catch(e){$('err').textContent=e.message}};
$('export').onclick=()=>{try{const reviewed=chosen?.split?.join(',');run();if($('err').textContent)return;if(reviewed){chosen=result.candidates.find(v=>v.split.join(',')===reviewed)||result.selected;render();}const blob=new Blob([JSON.stringify({schema_version:'web-0.3',config:c,result,reviewed_split:chosen?.split||null,evidence:'assumed_cycle_model_not_measured'},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='llm-tile-design.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}catch(e){$('err').textContent=e.message}};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{const j=JSON.parse(await $('file').files[0].text());fill(j.config||j);run();}catch(e){$('err').textContent='가져오기 실패: '+e.message}};

function downloadJSON(value,name){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
$('makePlan').onclick=()=>{clearExecution();try{
 const current=form();if(JSON.stringify(current)!==JSON.stringify(c))throw Error('조건이 변경되었습니다. 자동 분할을 다시 실행한 뒤 계획을 생성하세요.');
 if(!chosen)throw Error('먼저 검토할 배치를 선택하세요.');
 plan=generatePlan(c,chosen.split);const v=plan.validation;
 $('planStatus').textContent=v.passed?`구조 검증 통과 · ${plan.descriptors.length} descriptors · token index ${plan.token_index} · 수치 실행 미검증`:`계획 검증 실패: ${v.errors.map(e=>e.code+': '+e.detail).join(' / ')}`;
 $('planStatus').className=v.passed?'tag':'error';$('downloadPlan').disabled=!v.passed;$('executePlan').disabled=!v.passed;$('injectFault').disabled=!v.passed;
 $('planMemory').innerHTML='<div class="tablewrap"><table><thead><tr><th>Tile</th><th>주소 공간</th><th>배치 필요량</th><th>유효 용량</th></tr></thead><tbody>'+plan.memory.map(m=>`<tr><td>T${m.tile}</td><td>${m.space}</td><td>${(m.required_bytes/1024).toFixed(1)} KiB</td><td>${(m.usable_bytes/1024).toFixed(1)} KiB</td></tr>`).join('')+'</tbody></table></div><p class="muted">상단 후보 비교는 예약량 기준, 이 표는 수명에 따른 주소 재사용을 반영한 실제 계획 기준입니다. Weight staging은 4 KiB, FFN 중간 데이터는 SRAM에 유지합니다. 상단 시간 추정에는 새 descriptor의 실행 비용이 아직 반영되지 않았습니다.</p>';
 const hex=n=>'0x'+n.toString(16).padStart(8,'0');
 $('bufferTable').innerHTML='<table><thead><tr><th>Buffer</th><th>공간</th><th>상대 주소</th><th>Bytes</th><th>수명 (descriptor)</th></tr></thead><tbody>'+plan.buffers.map(b=>`<tr><td>${b.id}</td><td>${b.space}</td><td>${hex(b.address)}</td><td>${b.size}</td><td>${b.first}–${b.last}</td></tr>`).join('')+'</tbody></table>';
 $('descriptorTable').innerHTML='<table><thead><tr><th>ID</th><th>Tile</th><th>Opcode</th><th>Graph node</th><th>선행 ID</th></tr></thead><tbody>'+plan.descriptors.slice(0,40).map(d=>`<tr><td>${d.id}</td><td>T${d.tile}</td><td>${d.opcode}</td><td>${d.graph_nodes.join(', ')}</td><td>${d.wait_for.join(', ')||'—'}</td></tr>`).join('')+'</tbody></table>';
 for(const id of ['planPreconditions','planBuffers','planDescriptors'])$(id).hidden=false;
 }catch(e){clearPlan();$('planStatus').className='error';$('planStatus').textContent=e.message;}};
$('downloadPlan').onclick=()=>{if(plan?.validation.passed)downloadJSON(plan,'stories260k-tile-plan.json');};
for(const id of [...Object.keys(fields),...Object.keys(hw)])$(id).addEventListener('input',()=>{clearPlan();$('status').textContent='조건 변경됨 · 자동 분할을 다시 실행하세요.'});

try{[base,rows]=await Promise.all(['config.json','profile.json'].map(async url=>{const r=await fetch(url);if(!r.ok)throw Error('설정 로딩 실패');return r.json()}));fill(base);run();}catch(e){$('err').textContent=e.message}

function clearExecution(){executionWorker?.terminate();executionWorker=null;validationResult=null;for(const id of ['executePlan','injectFault','downloadValidation'])$(id).disabled=true;$('executionStatus').textContent='실행 계획을 먼저 생성하세요.';$('comparisonTable').innerHTML='';}
function runValidation(injectFault){
 if(!plan?.validation.passed)return;
 validationResult=null;$('downloadValidation').disabled=true;$('comparisonTable').innerHTML='';$('executePlan').disabled=true;$('injectFault').disabled=true;
 $('executionStatus').textContent='SW 실행 준비 중';
 const worker=new Worker(new URL('./execute-worker.mjs',import.meta.url),{type:'module'});executionWorker=worker;
 const finish=()=>{worker.terminate();if(executionWorker===worker)executionWorker=null;$('executePlan').disabled=false;$('injectFault').disabled=false;};
 worker.onmessage=({data})=>{if(data.type==='progress'){$('executionStatus').textContent=data.message;return}if(data.type==='error'){$('executionStatus').textContent='실행 실패: '+data.message;finish();return}
 validationResult=data.result;const v=validationResult.comparison;
 $('executionStatus').textContent=(injectFault?(v.passed?'오류 주입 후 미검출':'오류 주입 검출 · 최초 불일치 layer '+v.first_failed_layer):(v.passed?'Golden 비교 PASS':'Golden 비교 FAIL'))+' · '+validationResult.descriptors_executed+' descriptors';
 const comparisons=[['logits',v.logits],['KV',v.kv],...v.layers.map(x=>['Layer '+x.layer,x])];
 $('comparisonTable').innerHTML='<table><thead><tr><th>관측 지점</th><th>결과</th><th>최대 절대 오차</th><th>상대 L2</th><th>불일치 수</th></tr></thead><tbody>'+comparisons.map(([name,x])=>`<tr><td>${name}</td><td>${x.passed?'PASS':'FAIL'}</td><td>${x.max_abs_error.toExponential(3)}</td><td>${x.relative_l2.toExponential(3)}</td><td>${x.failed_elements}</td></tr>`).join('')+'</tbody></table>';
 $('downloadValidation').disabled=false;finish();};
 worker.onerror=e=>{$('executionStatus').textContent='Worker 실행 실패: '+e.message;finish()};worker.postMessage({plan,injectFault});
}
$('executePlan').onclick=()=>runValidation(false);$('injectFault').onclick=()=>runValidation(true);
$('downloadValidation').onclick=()=>{if(validationResult)downloadJSON(validationResult,'stories260k-validation-trace.json')};
