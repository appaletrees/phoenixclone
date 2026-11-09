'use strict';
const $=id=>document.getElementById(id);
let catalog=[], status=null, points=[], after=-1, historyKey='', frameChoiceKey='', cameraMode='recorded', pending=false, polling=false;
const charts=[];
const controlFields={mode:'controlMode',kp:'mitKp',kd:'mitKd',vel_ref:'mitVelRef',t_ref:'mitTRef',speed_limit:'mitSpeedLimit',robot_rate:'mitRobotRate'};
let controlDirty=false, controlRequested=null;
let recordingDirty=false, recordingRequested=null;
function controlValues(){return Object.fromEntries(Object.entries(controlFields).map(([key,id])=>[key,key==='mode'?$(id).value:Number($(id).value)]));}
function controlPanel(){$('mitParameters').hidden=$('controlMode').value!=='mit';}
const phaseNames={idle:'待机',aligning:'起点对齐',moving:'移动到指定帧',holding:'保持指定帧目标',playing:'回放中',paused:'已暂停',completed:'回放完成',disabled:'已失能',error:'设备异常'};
for(let joint=0;joint<6;joint++)for(let arm=0;arm<2;arm++){
 const dim=arm*7+joint, card=document.createElement('div');card.className='chart-card';
 const heading=document.createElement('h3');heading.textContent=`${arm?'右臂':'左臂'} · 关节 ${joint+1}`;
 const value=document.createElement('span');heading.append(value);const canvas=document.createElement('canvas');
 card.append(heading,canvas);$('charts').append(card);charts.push({canvas,value,dim,title:heading.firstChild.textContent});
}
async function command(name,body={}){
 try{pending=true;const response=await fetch('/api/command/'+name,{method:'POST',headers:{'Content-Type':'application/json','X-Replay-Token':window.REPLAY_TOKEN},body:JSON.stringify(body)});
 const data=await response.json();if(!response.ok)throw Error(data.error||'操作失败');await poll();return data;}
 catch(e){$('notice').textContent=e.message;$('notice').classList.add('error');return null;}
 finally{pending=false;}
}
function chooseEpisodes(){
 const dataset=catalog.find(d=>d.id===$('dataset').value);$('episode').replaceChildren();
 if(!dataset)return;for(const e of dataset.episodes){const o=document.createElement('option');o.value=e.index;o.textContent=`第 ${e.index} 轮 · ${e.duration.toFixed(1)} 秒 · ${e.frames} 帧`;$('episode').append(o);}rateDescription();
}
function rateDescription(){const fps=status?.episode?.fps||catalog.find(d=>d.id===$('dataset').value)?.fps||30;const hz=Number($('hz').value);$('rateDescription').textContent=`${hz} Hz = 原速 ${(hz/fps).toFixed(2)}×`;
 document.querySelectorAll('[data-hz]').forEach(b=>b.classList.toggle('selected',Math.abs(Number(b.dataset.hz)-hz)<.001));}
function cameras(mode){cameraMode=mode;$('recorded').classList.toggle('active',mode==='recorded');$('live').classList.toggle('active',mode==='live');
 for(let i=0;i<3;i++){$('cam'+i).src=`/stream/${mode}/${i}`;$('camInfo'+i).textContent=mode==='recorded'?'原始采集录像 · 与当前帧位置同步':'现场实时画面 · 10 fps 网页预览';} }
function f(v,n=3){return Number.isFinite(v)?v.toFixed(n):'—';}
function update(s){
 const key=[s.last_run?.run_id,s.episode?.id,s.episode?.episode,s.episode?.field,s.history_epoch].join(':');
 if(key!==historyKey){points=[];historyKey=key;}
 for(const p of s.samples){if(!points.length||p.seq>points[points.length-1].seq)points.push(p);}if(points.length>36000)points.splice(0,points.length-36000);
 after=s.latest_sequence;status=s;
 const real=s.mode==='hardware',active=['playing','aligning','moving'].includes(s.phase);
 const settings=s.control_settings;
 if(settings){
  if(controlRequested&&Object.keys(controlFields).every(k=>settings[k]===controlRequested[k])){controlDirty=false;controlRequested=null;}
  if(!controlDirty)for(const [key,id] of Object.entries(controlFields))$(id).value=settings[key];
  const a=s.robot_control;
  $('controlSummary').textContent=a.mode==='mit'?`已应用 MIT · Kp ${a.mit.kp} / Kd ${a.mit.kd} · 驱动设定 ${a.controller_hz_requested} Hz`:'已应用 Joint 位置控制 · 驱动设定 100 Hz';
  $('controlHint').textContent=s.connected?'当前已连接；先断开设备，再修改模式或参数。':controlDirty?'配置尚未应用；点击“应用控制配置”后连接设备。':'连接前可修改；应用配置不会使能或移动机械臂。';
 }else{$('controlSummary').textContent='服务版本未更新';$('controlHint').textContent='原终端 Ctrl+C 后按原命令重启服务，再刷新页面以启用 MIT 配置。';}
 for(const id of Object.values(controlFields))$(id).disabled=!settings||s.connected||s.busy;
 $('applyControl').disabled=!settings||s.connected||s.busy;
 controlPanel();
 const recording=s.recording;
 if(recording?.format==='lerobot_v2.1'){
  const config=recording.settings;
  if(recordingRequested&&!s.busy&&config.mode===recordingRequested.mode&&config.image_hz===recordingRequested.image_hz&&config.output_dir===recordingRequested.output_dir&&!s.error){recordingDirty=false;recordingRequested=null;}
  if(!recordingDirty){$('recordMode').value=config.mode;$('recordImageHz').value=config.image_hz;$('recordOutputDir').value=config.output_dir;}
  for(const id of ['recordMode','recordImageHz','recordOutputDir','applyRecording'])$(id).disabled=recording.active||recording.saving||s.busy;
  $('recordImageHz').disabled||=$('recordMode').value!=='full';
  const images=recording.images;
  $('recordingStatus').textContent=`${recording.active?'记录中':recording.saving?'后台保存中':'待开始 / 已保存'} · 状态 ${recording.recorded_states} 条 · 下发 ${recording.recorded_actions} 条`+(images?` · 完整三路图像 ${images.complete_groups} 组 · 图像采样记录 ${images.written} 组 · 队列丢弃 ${images.dropped} · 采样迟到 ${images.skipped_ticks}`:'');
  if(images?.last_missing_cameras?.length)$('recordingStatus').textContent+=` · 缺失现场相机：${images.last_missing_cameras.join('、')}`;
  if(images?.error)$('recordingStatus').textContent+=` · 保存失败：${images.error}`;
  if(recording.lerobot?.status==='complete')$('recordingStatus').textContent+=` · LeRobot v2.1 已保存 ${recording.lerobot.frames} 帧`;
  if(recording.lerobot?.status==='error')$('recordingStatus').textContent+=` · LeRobot 导出失败：${recording.lerobot.error}`;
  $('recordingHint').textContent=recordingDirty?'配置尚未应用，点击“应用记录配置”后开始。':recording.active?'正在记录；暂停仍会记录。切换单帧目标或修改配置前先停止本次记录。':'开始回放或移动到指定帧时记录，停止或完成时保存。图像取现场相机；演示模式没有现场图像。';
 }else{for(const id of ['recordMode','recordImageHz','recordOutputDir','applyRecording'])$(id).disabled=true;$('recordingStatus').textContent='当前服务仍是旧记录格式，请重启服务以启用 LeRobot v2.1 和保存目录设置';}
 $('modeBadge').textContent=real?'真机模式 · 页面按钮控制运动':'离线演示 · 反馈为模拟值';$('modeBadge').classList.toggle('hardware',real);
 $('connection').textContent=s.connected?'● 设备已连接':'● 未连接设备';$('connection').classList.toggle('on',s.connected);
 $('phase').textContent=s.busy?'处理中…':phaseNames[s.phase]||s.phase;$('phaseDot').classList.toggle('active',active);
 const cameraErrors=Object.entries(s.video_errors||{}).map(([k,v])=>`${k}: ${v}`).join('；');
 $('notice').textContent=s.error||cameraErrors||s.message;$('notice').classList.toggle('error',Boolean(s.error||cameraErrors));
 $('start').textContent=s.phase==='paused'?'继续回放':s.phase==='completed'?'从头再次回放':real?'开始真机回放':'开始演示回放';
 $('start').disabled=s.busy||active||!s.episode||!s.connected||recordingDirty||s.recording?.saving;$('load').disabled=s.busy||active;
 for(const id of ['dataset','episode','field'])$(id).disabled=s.busy||active;
 $('connect').disabled=s.busy||s.connected||controlDirty;$('connect').textContent=real?'连接设备与相机':'连接演示设备';
 $('pause').disabled=!active;$('stop').disabled=!s.connected;$('disable').disabled=!s.connected;$('disconnect').disabled=!s.connected||s.busy;
 $('seek').disabled=!s.episode||active||s.busy;
 $('frameNumber').disabled=$('previewFrame').disabled=!s.episode||active||s.busy;
 $('gotoFrame').disabled=!s.episode||!s.connected||active||s.busy||recordingDirty||s.recording?.saving;
 if(s.episode){const e=s.episode;$('episodeInfo').replaceChildren();const lines=[`${e.title} · 原始轨迹 ${e.episode}`,`${e.frames} 帧 / ${(e.frames/e.fps).toFixed(1)} 秒 / ${e.fps} Hz`,`${e.field} → 双臂绝对关节位置`,e.task];
 for(const line of lines){const p=document.createElement('div');p.textContent=line;$('episodeInfo').append(p);}
 $('seek').max=e.frames-1;if(document.activeElement!==$('seek'))$('seek').value=s.frame;
 $('frameNumber').max=e.frames-1;const choice=[e.id,e.episode,e.field,s.history_epoch].join(':');if(choice!==frameChoiceKey){$('frameNumber').value=s.frame;frameChoiceKey=choice;}
 $('frameLabel').textContent=`原始第 ${s.frame} 帧 / ${e.frames-1}`;$('timeLabel').textContent=`${(s.frame/e.fps).toFixed(2)} / ${((e.frames-1)/e.fps).toFixed(2)} s`;
 $('progressLabel').textContent=`已下发 ${s.sent_frames} 帧 · 辅助第 15 维 ${s.auxiliary===null?'无':s.auxiliary}（不下发）`;}
 $('requestedHz').textContent=`${s.hz} Hz`;$('actualHz').textContent=s.actual_hz?`${s.actual_hz.toFixed(1)} Hz`:'—';$('sent').textContent=s.sent_frames;
 $('grippers').textContent=`${f(s.target?.[6],2)} / ${f(s.current?.[6],2)} · ${f(s.target?.[13],2)} / ${f(s.current?.[13],2)}`;
 $('feedbackLegend').textContent=real?'当前关节反馈 · 采样点':'模拟反馈 · 非真机测量';
 $('chartMeaning').textContent=real?'横轴为本次运行经过时间。蓝线是回放目标，黑线和圆点是当前关节反馈。':'演示模式：三路采集录像是真实数据；黑线是模拟反馈，用于验证页面流程。';
 const keys=['cam_external','cam_left_wrist','cam_right_wrist'];for(let i=0;i<3;i++)$('camState'+i).textContent=cameraMode==='recorded'?'录像':s.connected&&s.camera_status[keys[i]]?'实时':'未连接';
 $('recording').textContent=s.recording?.format!=='lerobot_v2.1'?'当前服务仍是旧记录格式，请停止原进程并重启后使用新格式。':s.recording?.directory?`记录目录：${s.recording.directory} · LeRobot v2.1：meta/ + data/ + videos/ · 同时保留原始诊断记录`:'开始回放后自动记录，停止后生成 LeRobot v2.1 数据集。';
 $('diagnostics').textContent=`调度迟到 ${s.late_ticks} · 驱动限位修正 ${s.clamped_ticks}`;
 if(s.robot_control)$('diagnostics').textContent+=` · ${s.robot_control.mode.toUpperCase()} / 驱动设定 ${s.robot_control.controller_hz_requested} Hz`;
 drawAll();
}
async function poll(){if(polling)return;polling=true;try{const r=await fetch('/api/status?after='+after);if(!r.ok)throw Error('连接中断');update(await r.json());}
 catch(e){$('connection').textContent='● 页面服务不可达';$('connection').classList.remove('on');}finally{polling=false;}}
function drawChart(chart){
 const canvas=chart.canvas,rect=canvas.getBoundingClientRect(),scale=window.devicePixelRatio||1;
 const w=Math.max(200,Math.round(rect.width)),h=Math.round(rect.height||128);if(canvas.width!==Math.round(w*scale)||canvas.height!==Math.round(h*scale)){canvas.width=Math.round(w*scale);canvas.height=Math.round(h*scale);}
 const ctx=canvas.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
 const unit=$('unit').value,k=unit==='deg'?180/Math.PI:1,windowSize=Number($('window').value);
 const end=points.length?points[points.length-1].t:30,start=windowSize?Math.max(0,end-windowSize):(points[0]?.t||0),right=Math.max(start+1,end);
 const visible=points.filter(p=>p.t>=start),values=[];for(const p of visible)for(const a of [p.target,p.current])if(a&&Number.isFinite(a[chart.dim]))values.push(a[chart.dim]*k);
 let lo=values.length?Math.min(...values):-1,hi=values.length?Math.max(...values):1;let pad=Math.max(unit==='deg'?1:.02,(hi-lo)*.15);lo-=pad;hi+=pad;
 const L=40,R=w-11,T=8,B=h-23,X=t=>L+(t-start)/(right-start)*(R-L),Y=v=>B-(v*k-lo)/(hi-lo)*(B-T);
 ctx.strokeStyle='#edf1f4';ctx.lineWidth=.7;ctx.fillStyle='#91a0aa';ctx.font='9px system-ui';
 for(let j=0;j<=3;j++){const y=T+j*(B-T)/3;ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(R,y);ctx.stroke();ctx.textAlign='right';ctx.fillText((hi-j*(hi-lo)/3).toFixed(unit==='deg'?1:2),L-6,y+3);}
 for(let j=0;j<=4;j++){const t=start+j*(right-start)/4,x=X(t);ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,B);ctx.stroke();ctx.textAlign='center';ctx.fillText(t.toFixed(1),x,B+15);}
 ctx.save();ctx.beginPath();ctx.rect(L,T,R-L,B-T);ctx.clip();
 for(const [field,color] of [['target','#268ab9'],['current','#30343b']]){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=field==='target'?1.2:1;let pen=false;
 for(const p of visible){const v=p[field]?.[chart.dim];if(!Number.isFinite(v)){pen=false;continue;}const x=X(p.t),y=Y(v);if(!pen)ctx.moveTo(x,y);else ctx.lineTo(x,y);pen=true;}ctx.stroke();}
 ctx.fillStyle='#30343b';const stride=Math.max(1,Math.ceil(visible.length/180));for(let i=0;i<visible.length;i+=stride){const p=visible[i],v=p.current?.[chart.dim];if(Number.isFinite(v)){ctx.beginPath();ctx.arc(X(p.t),Y(v),1.45,0,2*Math.PI);ctx.fill();}}ctx.restore();
 const last=visible[visible.length-1];chart.value.textContent=`目标 ${f(last?.target?.[chart.dim]*k)} / 当前 ${f(last?.current?.[chart.dim]*k)} ${unit==='deg'?'°':'rad'}`;
 if(!visible.length){ctx.fillStyle='#a5b1b9';ctx.textAlign='center';ctx.fillText('连接设备后显示采样点',w/2,h/2);}
}
function drawAll(){for(const c of charts)drawChart(c);}
for(const id of Object.values(controlFields))$(id).addEventListener('input',()=>{controlDirty=true;controlRequested=null;controlPanel();$('connect').disabled=true;$('controlHint').textContent='配置尚未应用；点击“应用控制配置”后连接设备。';});
$('applyControl').onclick=()=>{for(const id of Object.values(controlFields)){if(!$(id).reportValidity())return;}controlRequested=controlValues();command('control',controlRequested);};
for(const id of ['recordMode','recordImageHz','recordOutputDir'])$(id).addEventListener('input',()=>{recordingDirty=true;recordingRequested=null;$('start').disabled=$('gotoFrame').disabled=true;$('recordImageHz').disabled=$('recordMode').value!=='full';$('recordingHint').textContent='配置尚未应用，点击“应用记录配置”后开始。';});
$('applyRecording').onclick=async()=>{if(!$('recordImageHz').reportValidity())return;recordingRequested=null;const result=await command('recording',{mode:$('recordMode').value,image_hz:Number($('recordImageHz').value),output_dir:$('recordOutputDir').value});if(result){recordingRequested=result.normalized;await poll();}};
$('dataset').addEventListener('change',chooseEpisodes);$('hz').addEventListener('input',rateDescription);
document.querySelectorAll('[data-hz]').forEach(b=>b.onclick=()=>{$('hz').value=b.dataset.hz;rateDescription();if(status?.phase==='playing')command('rate',{hz:Number(b.dataset.hz)});});
$('load').onclick=()=>command('select',{dataset:$('dataset').value,episode:Number($('episode').value),field:$('field').value});
$('connect').onclick=()=>command('connect');$('disconnect').onclick=()=>command('disconnect');
$('start').onclick=()=>command('start',{hz:Number($('hz').value),align_seconds:Number($('align').value)});
for(const name of ['pause','stop','disable'])$(name).onclick=()=>command(name);
$('applyRate').onclick=()=>command('rate',{hz:Number($('hz').value)});
$('seek').onchange=()=>{$('frameNumber').value=$('seek').value;command('seek',{frame:Number($('seek').value)});};
$('previewFrame').onclick=()=>command('seek',{frame:Number($('frameNumber').value)});
$('gotoFrame').onclick=()=>command('goto_frame',{frame:Number($('frameNumber').value),align_seconds:Number($('align').value)});
$('recorded').onclick=()=>cameras('recorded');$('live').onclick=()=>cameras('live');$('unit').onchange=drawAll;$('window').onchange=drawAll;
$('exportPlot').onclick=()=>{const c=document.createElement('canvas');c.width=1600;c.height=1900;const ctx=c.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);
 ctx.fillStyle='#263746';ctx.font='bold 30px sans-serif';ctx.fillText(`双臂 Piper | ${status?.episode?.title||'回放'} | 轨迹 ${status?.episode?.episode??'—'}`,70,60);
 ctx.font='18px sans-serif';ctx.fillStyle='#71838f';ctx.fillText(`蓝线：回放目标；黑线与圆点：${status?.mode==='hardware'?'当前关节反馈':'模拟反馈（非真机）'} | ${status?.hz||30} Hz`,70,98);
 charts.forEach((chart,i)=>{const x=60+(i%2)*780,y=135+Math.floor(i/2)*280;ctx.fillStyle='#526a78';ctx.font='19px sans-serif';ctx.fillText(chart.title,x+10,y+20);ctx.drawImage(chart.canvas,x,y+35,740,220);});
 ctx.fillStyle='#71838f';ctx.font='17px sans-serif';ctx.fillText('横轴：本次运行经过时间（s）；未进行时间偏移或事后平滑。',70,1860);
 c.toBlob(blob=>{if(!blob)return;const link=document.createElement('a');link.download=`piper-replay-${status?.last_run?.run_id||'preview'}.png`;link.href=URL.createObjectURL(blob);document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(link.href),60000);},'image/png');};
new ResizeObserver(drawAll).observe($('charts'));
(async()=>{const r=await fetch('/api/catalog'),data=await r.json();catalog=data.datasets;
 for(const d of catalog){const o=document.createElement('option');o.value=d.id;o.textContent=`${d.title} · ${d.count} 条轨迹`;$('dataset').append(o);}
 if(catalog.some(d=>d.id==='piper_clean_table_lerobot'))$('dataset').value='piper_clean_table_lerobot';chooseEpisodes();cameras('recorded');
 await poll();if(status?.episode){$('dataset').value=status.episode.id;chooseEpisodes();$('episode').value=status.episode.episode;$('field').value=status.episode.field;$('hz').value=status.hz;$('align').value=status.align_seconds;rateDescription();}
 else await command('select',{dataset:$('dataset').value,episode:Number($('episode').value),field:'action'});
 setInterval(()=>{if(!pending)poll();},200);
})().catch(e=>{$('notice').textContent=e.message;});
