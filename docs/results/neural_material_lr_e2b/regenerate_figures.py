"""Regenerate E2b paper figures from tracked CSVs only; no model or raw checkpoint queries."""
import argparse
import csv
import importlib.abc
import math
from pathlib import Path
import os
import sys

class NoScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith(('torch.', 'experiments.')):
            raise RuntimeError('CSV figure regeneration forbids scientific imports: ' + fullname)
sys.meta_path.insert(0, NoScientificImports())

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--output-dir', type=Path, required=True, help='Fresh directory, preferably outside Git; never overwrites existing figures')
args = parser.parse_args()
OUT = args.output_dir.expanduser().absolute()
if OUT.resolve() != OUT or OUT.exists():
    parser.error('Output must be a fresh, unaliased directory')
OUT.mkdir(parents=True, exist_ok=False)
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES = ('M1/101', 'M1/103', 'M2/103', 'M3/101', 'M3/103')
HEADS = ('mx_predictor', 'my_predictor', 'metallic_predictor', 'rotation_predictor', 'kd_predictor')
def load(name, integers, floats):
    with (args.data_dir / name).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in integers:row[key] = int(row[key])
        for key in floats:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):raise ValueError('Nonfinite plotting value: ' + key)
    return rows
trajectories = load('trajectories.csv', ['state'], ['lr', 'loss', 'original_gradient_norm', 'Q_relative'] + [h + '_logit_gradient_L2' for h in HEADS])
saturation = load('sigmoid_boundaries.csv', ['state', 'boundary_channels'], ['lr'])
motions = load('adam_displacements.csv', ['update', 'current_gradient_nonzero'], ['lr', 'actual_parameter_L2', 'bitwise_changed_fraction', 'parameter_gradient_nonzero_fraction'])
assert len(trajectories) == 110 and len(saturation) == 550 and len(motions) == 100
series = {(case, lr): [r for r in trajectories if r['case'] == case and r['lr'] == lr] for case in CASES for lr in (.01, .001)}
assert all([r['state'] for r in rows] == list(range(11)) for rows in series.values())

# Plotting statements below are copied verbatim from the executed analysis.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={.01:'#D55E00',.001:'#0072B2'}
def emit(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(5,3,figsize=(14,15),sharex=True)
for i,case in enumerate(CASES):
    for lr in (.01,.001):
        rows=series[case,lr];x=[r['state'] for r in rows]
        for j,field in enumerate(('loss','original_gradient_norm','Q_relative')):
            axes[i,j].plot(x,[r[field] for r in rows],'-o',ms=3,color=colors[lr],label='LR '+str(lr))
            axes[i,j].grid(alpha=.2)
    axes[i,0].set_yscale('log');axes[i,1].set_yscale('symlog',linthresh=1e-9);axes[i,2].set_yscale('log')
    axes[i,0].set_ylabel(case);axes[i,1].set_ylabel('symlog; exact zero retained')
for j,title in enumerate(('Fitting MSE, original tapes','Original N1 gradient norm','Relative anisotropy-tensor Q error')):axes[0,j].set_title(title)
for ax in axes[-1]:ax.set_xlabel('Completed updates / pre-update state');ax.set_xticks(range(0,11,2))
axes[0,0].legend();fig.suptitle('Recorded early trajectories — same initial states, only LR differs',y=1.005)
fig.tight_layout();emit(fig,'early_trajectories')

positive=[row[h+'_logit_gradient_L2'] for row in trajectories for h in HEADS if row[h+'_logit_gradient_L2']>0]
vmin=math.floor(math.log10(min(positive)));vmax=math.ceil(math.log10(max(positive)))
fig,axes=plt.subplots(5,2,figsize=(14,13),sharex=True,sharey=True,constrained_layout=True)
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#c9c9c9')
for i,case in enumerate(CASES):
    for j,lr in enumerate((.01,.001)):
        rows=series[case,lr];values=np.array([[row[h+'_logit_gradient_L2'] for row in rows] for h in HEADS]);masked=np.ma.masked_all(values.shape)
        nonzero=values>0;masked[nonzero]=np.log10(values[nonzero])
        im=axes[i,j].imshow(masked,aspect='auto',origin='upper',cmap=cmap,vmin=vmin,vmax=vmax,extent=(-.5,10.5,4.5,-.5))
        for hidx,h in enumerate(HEADS):
            for row in saturation:
                if row['case']==case and row['lr']==lr and row['head']==h and row['boundary_channels']:
                    axes[i,j].text(row['state'],hidx,'B',ha='center',va='center',color='red',weight='bold',fontsize=8)
        axes[i,j].set_title(case+' | LR '+str(lr));axes[i,j].set_yticks(range(5));axes[i,j].set_yticklabels(['mx','my','metallic','rotation','albedo'])
        axes[i,j].set_xticks(range(0,11,2))
for ax in axes[-1]:ax.set_xlabel('Pre-update state (state 10 is terminal)')
fig.colorbar(im,ax=axes.ravel().tolist(),label='log10 actual logit-loss-gradient L2; gray = exact zero',shrink=.7)
fig.suptitle('Actual graph gradients at calibrated logits; B = at least one exact sigmoid boundary')
emit(fig,'logit_gradients_and_boundaries')

fig,axes=plt.subplots(5,2,figsize=(13,14),sharex=True)
for i,case in enumerate(CASES):
    for lr in (.01,.001):
        rows=[r for r in motions if r['case']==case and r['lr']==lr];x=[r['update'] for r in rows]
        axes[i,0].plot(x,[r['actual_parameter_L2'] for r in rows],'-o',ms=3,color=colors[lr],label='LR '+str(lr))
        axes[i,1].plot(x,[100*r['bitwise_changed_fraction'] for r in rows],'-',color=colors[lr],label=str(lr)+' weights changed')
        axes[i,1].plot(x,[100*r['parameter_gradient_nonzero_fraction'] for r in rows],'--',color=colors[lr],label=str(lr)+' nonzero gradients')
        for row in rows:
            if row['current_gradient_nonzero']==0:axes[i,0].plot(row['update'],row['actual_parameter_L2'],'x',color='black',ms=8)
    axes[i,0].set_ylabel(case+' | L2 displacement');axes[i,1].set_ylabel('% of 1,485,584 parameters')
    for ax in axes[i]:ax.grid(alpha=.2)
axes[0,0].set_title('Actual float32 parameter movement; x = zero current gradient')
axes[0,1].set_title('Gradient zeros and unchanged weights are different observations')
axes[0,0].legend();axes[0,1].legend(fontsize=8)
for ax in axes[-1]:ax.set_xlabel('Update number (gradient measured at pre-state update−1)')
fig.tight_layout();emit(fig,'adam_motion_vs_gradient')


assert 'torch' not in sys.modules
print('CSV-only figures written to ' + str(OUT))
