#!/usr/bin/env python3
"""Final runtime-v2 figures, read only from archived JSON evidence (figures4papers style)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'figures';DATA=ROOT/'runtime-v2/runs'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Helvetica'],
 'font.size':12,'axes.titlesize':14,'axes.labelsize':12,'axes.linewidth':1.8,'axes.spines.right':False,
 'axes.spines.top':False,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
NAMES=['Frozen v1 + MTP','v2 exact + MTP','v2 fast + MTP'];DIRS=['final_frozen_v1','final_v2_exact','final_v2_fast'];COLORS=['#767676','#3775BA','#0F4D92']
def load(d,f):return json.loads((DATA/d/f).read_text())
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig=plt.figure(figsize=(13,8.4));gs=fig.add_gridspec(2,2,left=.085,right=.98,top=.83,bottom=.23,wspace=.30,hspace=.63)
a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[1,1])
fig.suptitle('Runtime v2 — faster serving, unchanged NVFP4 weights',y=.98,fontsize=19,fontweight='bold')
fig.text(.5,.925,'Final validation on one DGX Spark • W4A4 + MTP • October 31, 2026',ha='center',color='#4D4D4D')
x=np.arange(3);w=.24
for i,(d,col,lab) in enumerate(zip(DIRS,COLORS,NAMES)):
 vals=[load(d,'perf_matched.json')['levels'][0]['decode_tok_s_per_req']['p50']]+[load(d,f'screen_{k}.json')['levels'][0]['decode_tok_s_per_req_p50'] for k in ['code','doc']]
 bars=a.bar(x+(i-1)*w,vals,w,color=col,edgecolor='black',linewidth=.8,label=lab)
 for q,v in zip(bars,vals):a.text(q.get_x()+q.get_width()/2,v+.55,f'{v:.1f}',ha='center',fontsize=11,fontweight='bold')
a.set(title='A  Single-request decode (not aggregate throughput)',ylabel='Median decode tokens/s',xticks=x,xticklabels=['Matched synthetic (~1K input)','Short code (~90 input)','Document (~4K input)'],ylim=(0,44));a.legend(loc='upper left',ncol=3,fontsize=10);a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
for d,col,lab,m in zip(DIRS,COLORS,NAMES,['o','s','^']):
 data=load(d,'perf_matched.json')['levels'];xs=[t['concurrency'] for t in data]
 b.plot(xs,[t['aggregate_output_tok_s'] for t in data],color=col,marker=m,lw=2.5,ms=7,label=lab)
 c.plot(xs,[t['ttft_s']['p50'] for t in data],color=col,marker=m,lw=2.5,ms=7)
b.set(title='B  Matched-workload concurrency',xlabel='Concurrent requests',ylabel='Aggregate output tokens/s',xticks=[1,2,4,8],ylim=(0,145));b.grid(axis='y',alpha=.16)
c.set(title='C  Matched-workload first token',xlabel='Concurrent requests',ylabel='Median TTFT (seconds)',xticks=[1,2,4,8],ylim=(0,6));c.grid(axis='y',alpha=.16)
fig.text(.5,.075,'Exactly 256 output tokens • greedy • no prefix-cache hits • 0 errors in final performance runs\nMatched: 8 requests/level; code/document: 4 at c=1 • final results, not earlier screening peaks\nExact preserves reference routing math; fast changes FP32 reduction order and may change generated text.\nSource: runtime-v2/runs/final_*/{perf_matched,screen_code,screen_doc}.json',ha='center',va='center',fontsize=10,color='#4D4D4D',linespacing=1.6)
save(fig,'runtime-v2-performance')
keys=['ifeval','gsm8k','mmlu_pro','humaneval','tools','vqa'];labels=['IFEval*','GSM8K','MMLU-Pro','HumanEval','Tools**','TextVQA']
fig,axes=plt.subplots(1,2,figsize=(13.8,6.9),gridspec_kw={'width_ratios':[1.6,1]});fig.subplots_adjust(left=.07,right=.98,top=.77,bottom=.37,wspace=.29)
fig.suptitle('Runtime v2 quality — preserved evidence, explicit trade-offs',y=.98,fontsize=18,fontweight='bold')
fig.text(.5,.914,'Same 300-item suite • thinking off • greedy • evaluation concurrency 16',ha='center',color='#4D4D4D')
x=np.arange(6);w=.24
for i,(d,col,lab) in enumerate(zip(DIRS,COLORS,NAMES)):
 j=load(d,'eval/summary.json');vals=[100*j['tasks'][k]['accuracy'] for k in keys]
 axes[0].bar(x+(i-1)*w,vals,w,color=col,edgecolor='black',linewidth=.7,label=lab)
axes[0].set(title='A  First complete regression run',ylabel='Accuracy (%)',xticks=x,xticklabels=labels,ylim=(0,105));axes[0].tick_params(axis='x',labelrotation=25);fig.legend(handles=axes[0].get_legend_handles_labels()[0],labels=NAMES,loc='upper center',bbox_to_anchor=(.5,.865),ncol=3,fontsize=10);axes[0].grid(axis='y',alpha=.16);axes[0].set_axisbelow(True)
for i,(d,col) in enumerate(zip(DIRS[1:],COLORS[1:])):
 q=load(d,'eval/compare_vs_ref.json')['overall_micro'];v=q['delta']*100;lo,hi=np.array(q['delta_boot95'])*100
 axes[1].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=col,capsize=5,lw=2)
 axes[1].text(-5.5,i+.22,f'{v:+.2f} pp [{lo:+.2f}, {hi:+.2f}]',fontsize=10)
axes[1].axvline(0,color='#767676',lw=1,ls='--');axes[1].set(title='B  Pooled change vs frozen v1',xlabel='Percentage points (paired-bootstrap 95% CI)',yticks=[0,1],yticklabels=['v2 exact','v2 fast'],xlim=(-6,3),ylim=(1.55,-.55));axes[1].grid(axis='x',alpha=.16)
fig.text(.5,.16,'Frozen: 251/300 (83.7%) • exact: 249/300 (83.0%) • fast: 247/300 (82.3%)\nFast MMLU-Pro first run: 58.3% vs 70.0% frozen (−11.7 pp); fast reruns: 71.7%, 65.0%.\nFast truncations: 20 vs 14 frozen. Fast fails original first-run per-task/stability tripwires; not “lossless.”\n*Re-implemented IFEval checker. **20 hand-written tool cases. Small subsets, not full-benchmark results.\nSource: runtime-v2/runs/final_*/eval; all reruns retained in runtime-v2/runs/var and RESULTS.md.',ha='center',va='center',fontsize=10,color='#4D4D4D',linespacing=1.7)
save(fig,'runtime-v2-quality')
print('Generated final v2 performance/quality PNG, PDF, SVG from JSONs.')
