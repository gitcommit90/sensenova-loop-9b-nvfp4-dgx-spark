#!/usr/bin/env python3
"""Regenerate figures from checked-in measurements; figures4papers house conventions.
https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making
Usage: python3 -m pip install -r figures/requirements.txt; python3 figures/plot_benchmarks.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Helvetica'],
 'font.size':12,'axes.titlesize':14,'axes.labelsize':12,'axes.linewidth':1.8,
 'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False,
 'svg.fonttype':'none','pdf.fonttype':42})
COLORS=['#767676','#B64342','#3775BA','#0F4D92']
LABELS=['BF16','BF16 + MTP','NVFP4','NVFP4 + MTP']
FILES=['perf_sglang_bf16.json','perf_sglang_bf16_mtp.json','perf_sglang_nvfp4_v1.json','perf_sglang_nvfp4_v1_mtp.json']
data=[json.loads((ROOT/'evaluation'/f).read_text()) for f in FILES]
for d in data:
 assert [l['concurrency'] for l in d['levels']]==[1,2,4,8]
 assert all(l['n_errors']==0 and l['output_tokens_mean']==256 for l in d['levels'])
def save(fig,name):
 for ext in ['png','pdf','svg']:
  fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig=plt.figure(figsize=(12.8,8.1))
gs=fig.add_gridspec(2,2,left=.09,right=.97,top=.84,bottom=.17,wspace=.32,hspace=.55)
ax=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,:])
fig.suptitle('SenseNova Loop 9B NVFP4 — one DGX Spark',y=.98,fontsize=19,fontweight='bold')
fig.text(.5,.927,'Functional release baseline • native W4A4 • original weights, PTQ only',ha='center',color='#4D4D4D',fontsize=12)
x=np.arange(4)
for a,key,title,ylabel in [(ax,'decode_tok_s_per_req','A  Single-request decode','Median decode tokens/s'),(ax2,'ttft_s','B  Time to first token (one request)','Median TTFT (seconds)')]:
 vals=[d['levels'][0][key]['p50'] for d in data]
 bars=a.bar(x,vals,color=COLORS,edgecolor='black',linewidth=1)
 for b,v in zip(bars,vals):a.text(b.get_x()+b.get_width()/2,v+max(vals)*.035,f'{v:.2f}',ha='center',fontsize=11,fontweight='bold')
 a.set(title=title,ylabel=ylabel,xticks=x,xticklabels=['BF16','BF16\n+ MTP','NVFP4','NVFP4\n+ MTP'],ylim=(0,max(vals)*1.24))
 a.grid(axis='y',alpha=.16); a.set_axisbelow(True)
for d,c,label,marker in zip(data,COLORS,LABELS,['o','s','^','D']):
 vals=[l['aggregate_output_tok_s'] for l in d['levels']]
 ax3.plot([1,2,4,8],vals,color=c,label=label,marker=marker,lw=2.5,ms=7)
ax3.set(title='C  Concurrency scales aggregate throughput — not individual-user speed',xlabel='Concurrent requests',ylabel='Aggregate output tokens/s',xticks=[1,2,4,8],ylim=(0,78))
ax3.legend(loc='upper left',ncol=2,fontsize=10);ax3.grid(axis='y',alpha=.16)
ax3.annotate('64.79 without MTP / 64.43 with MTP',xy=(8,64.79),xytext=(4.0,72),fontsize=10,arrowprops={'arrowstyle':'-','color':'#767676'})
fig.text(.5,.075,'1,051 prompt tokens → 256 output tokens • greedy • unique prompts, no prefix-cache hits\n16 requests/level without MTP; 8 with MTP • CUDA graphs • 0 request errors/OOMs\nSource: evaluation/perf_sglang_*.json • Measurements: October 30, 2026',ha='center',va='center',fontsize=10,color='#4D4D4D',linespacing=1.6)
save(fig,'performance-summary')
comp=json.loads((ROOT/'evaluation/compare_vs_ref.json').read_text())
keys=['ifeval','gsm8k','mmlu_pro','humaneval','tools','vqa']
names=['IFEval*','GSM8K','MMLU-Pro','HumanEval','Tools**','TextVQA']
fig,axes=plt.subplots(1,2,figsize=(13.5,6.6),gridspec_kw={'width_ratios':[1.15,1]})
fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.36,wspace=.37)
fig.suptitle('Quality after NVFP4 quantization — small regression suite',y=.98,fontsize=18,fontweight='bold')
fig.text(.5,.903,'300 fixed items • thinking off • paired comparison to BF16 • not full-benchmark scores',ha='center',fontsize=12,color='#4D4D4D')
x=np.arange(len(keys));w=.36
for off,k,col,label in [(-w/2,'ref_acc',COLORS[0],'BF16'),(w/2,'cand_acc',COLORS[2],'NVFP4')]:
 vals=[100*comp['tasks'][t][k] for t in keys]
 axes[0].bar(x+off,vals,w,color=col,edgecolor='black',linewidth=.8,label=label)
axes[0].set(title='A  Task accuracy',ylabel='Accuracy (%)',xticks=x,xticklabels=names,ylim=(0,105));axes[0].tick_params(axis='x',labelrotation=25)
axes[0].legend(loc='lower left',fontsize=10);axes[0].grid(axis='y',alpha=.16);axes[0].set_axisbelow(True)
ys=np.arange(6)
for i,t in enumerate(keys):
 q=comp['tasks'][t];v=q['delta']*100;lo,hi=[v*100 for v in q['delta_boot95']]
 axes[1].errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=COLORS[2],capsize=4,lw=2)
axes[1].axvline(0,color='#767676',lw=1,ls='--')
axes[1].set(title='B  Change and paired-bootstrap 95% CI',xlabel='NVFP4 − BF16 (percentage points)',yticks=ys,yticklabels=[f'{n} (n={comp["tasks"][k]["n"]})' for n,k in zip(names,keys)],xlim=(-18,18))
axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.16)
fig.text(.5,.12,'Pooled accuracy: BF16 82.7% → NVFP4 83.0% • Δ +0.3 pp [−3.0, +4.0]\n*IFEval subset uses a re-implemented checker. **Tools: 20 hand-written BFCL-style cases.\nSubsets of 20–80 items/task; uncertainty is wide. No claim of losslessness or intelligence improvement.\nSource: evaluation/compare_vs_ref.json; see model card for truncations and methodology.',ha='center',va='center',fontsize=10,color='#4D4D4D',linespacing=1.65)
save(fig,'quality-summary')
print('Rendered performance and quality figures from measured JSONs.')
