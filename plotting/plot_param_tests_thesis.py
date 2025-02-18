import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
import pandas as pd
import sys
import os
import numpy as np

def plotDiff(x, y1, y2, y1_label, y2_label, xlabel, ylabel, savepath):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10,8))

  axs[0].scatter(x, y1, label=y1_label)
  axs[0].scatter(x, y2, label=y2_label)

  axs[0].set_ylabel(ylabel)

  axs[0].legend()

  axs[1].scatter(x, 100*(y1-y2))
  axs[1].set_xlabel(xlabel)
  axs[1].set_ylabel("%s - %s (%%)"%(y1_label, y2_label))
  axs[1].axhline(-1, color="0.8", linestyle="--")
  axs[1].axhline(1, color="0.8", linestyle="--")
  if "skip" in savepath:
    axs[1].set_ylim(-3, 3)
    assert max(abs(100*(y1-y2))) < 3
  else:
    axs[1].set_ylim(-5, 5)
    assert max(abs(100*(y1-y2))) < 5
  axs[0].set_ylim(0.4, 1.09)

  f.tight_layout()
  f.savefig(savepath+".png")
  f.savefig(savepath+".pdf")
  print(savepath+".png")
  plt.clf()
  
def plot2DDiff(x1_x2, y1, y2, y1_label, y2_label, xlabel, ylabel, savepath):
  assert min(y1) > 0.4
  assert min(y2) > 0.4

  x = np.arange(0, len(x1_x2))

  mxs = sorted(np.unique(x1_x2[:,0]))
  ncol = len(mxs)

  f, axs = plt.subplots(2, ncol, sharex="col", sharey="row", gridspec_kw={'height_ratios': [3, 1]},  figsize=(20,10))
  plt.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.15, wspace=0.05)

  axs[0][0].set_ylabel(ylabel)
  axs[1][-1].set_xlabel(xlabel)
  axs[1][0].set_ylabel("%s - %s (%%)"%(y1_label, y2_label))

  for i_col, mx in enumerate(mxs):
    idx = np.where(x1_x2[:,0]==mx)[0]
    axs[0][i_col].scatter(x1_x2[idx,1], y1[idx], label=y1_label, color="tab:blue")
    axs[0][i_col].scatter(x1_x2[idx,1], y2[idx], label=y2_label, color="tab:orange")
    axs[1][i_col].scatter(x1_x2[idx,1], 100*(y1[idx]-y2[idx]), color="tab:blue")

  # if "Low" in savepath:
  #   axs[0][0].legend(loc='center left')
  # else:
  axs[0][0].legend(loc='upper left')

  axs[0][0].set_ylim(0.4, 1.09)
  if "skip" in savepath:
    axs[1][0].set_ylim(-3, 3)
    assert max(abs(100*(y1-y2))) < 3
  else:
    axs[1][0].set_ylim(-5, 5)
    assert max(abs(100*(y1-y2))) < 5
  
  for i, mx in enumerate(mxs):
    axs[0][i].text(0.5, 0.05, r"$m_X=%d\,$GeV"%mx, transform=axs[0][i].transAxes, verticalalignment="bottom", horizontalalignment="center", fontsize=18)
  
  for ax in axs[1]:
    ax.tick_params(axis="x", rotation = 90)    
    
  for i in range(ncol):
    axs[0][i].tick_params(axis="y", which="minor", left=i==0, right=i==ncol-1)
    axs[1][i].tick_params(axis="y", which="minor", left=i==0, right=i==ncol-1)
    axs[0][i].tick_params(axis="y", which="major", left=i==0, right=i==ncol-1)
    axs[1][i].tick_params(axis="y", which="major", left=i==0, right=i==ncol-1)
    
    axs[1][i].axhline(1, color="0.8", linestyle="--")
    axs[1][i].axhline(-1, color="0.8", linestyle="--")
    
    l, r = axs[0][i].get_xlim()
    w = r-l
    axs[0][i].set_xlim(l-0.04*w, r+0.04*w)

  f.savefig(savepath+".png")
  f.savefig(savepath+".pdf")
  print(savepath+".png")
  plt.clf()

def do(indir):
  results = pd.read_csv(os.path.join(indir, "param_test_results.csv"), index_col=0)
  #print(results)

  mx_my = []
  for each in results.index:
    s = each.split("_")
    mx_my.append([int(s[1]), int(s[3])])

  mx_my = np.array(mx_my)

  if (mx_my[:,1] == 125.0).all():
    mx = mx_my[:,0]

    mx=mx[1:]
    results.drop("MX_260_MY_125", inplace=True)

    plotDiff(mx, results["all"], results["only"], "All", "Only", r"$m_X$ [GeV]", "Signal Efficiency", os.path.join(indir, "all_only_new"))
    plotDiff(mx, results["all"], results["skip"], "All", "Skip", r"$m_X$ [GeV]", "Signal Efficiency", os.path.join(indir, "all_skip_new"))
  else:
    plot2DDiff(mx_my, results["all"], results["only"], "All", "Only", r"$m_Y$ [GeV]", "Signal Efficiency", os.path.join(indir, "all_only_new"))
    mx_exclude = [min(mx_my[:,0]), max(mx_my[:,0])]
    my_exclude = [min(mx_my[:,1]), max(mx_my[:,1])]
    s = ~(np.isin(mx_my[:,0],mx_exclude)) & ~(np.isin(mx_my[:,1],my_exclude)) #exclude edge cases
    plot2DDiff(mx_my[s], results["all"][s], results["skip"][s], "All", "Skip", r"$m_Y$ [GeV]", "Signal Efficiency", os.path.join(indir, "all_skip_new"))

if __name__=="__main__":
  dirs = ["Outputs_Oct/Graviton_Param_Tests/", "Outputs_Oct/NMSSM_Y_tautau_Param_Tests/", "Outputs_Oct/NMSSM_Y_gg_Low_Mass_Param_Tests/", "Outputs_Oct/NMSSM_Y_gg_High_Mass_Param_Tests/"]
  
  # for d in dirs:
  #   do(d)
  
  for d in dirs:
    proc = d.split("/")[1].replace("_Param_Tests", "")
    print(f"scp mdk16@lx04.hep.ph.ic.ac.uk:/home/hep/mdk16/PhD/ggtt/ResonantGGTT/{d}/all_only_new.pdf {proc}/all_only.pdf") 
    print(f"scp mdk16@lx04.hep.ph.ic.ac.uk:/home/hep/mdk16/PhD/ggtt/ResonantGGTT/{d}/all_skip_new.pdf {proc}/all_skip.pdf") 

