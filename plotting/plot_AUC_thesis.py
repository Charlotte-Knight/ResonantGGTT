import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
import pickle
import sys
import pandas as pd
import numpy as np
import glob
import json
import matplotlib.ticker as ticker
import matplotlib.colors as colors

def get_MX_MY(sig_proc):
  if "NMSSM" in sig_proc:
    split_name = sig_proc.split("_")
    MX = float(split_name[7])
    MY = float(split_name[9])
  elif "XToHHggTauTau" in sig_proc:
    MX = float(sig_proc.split("_")[1][1:])
    MY = 125.0
  elif "HHTo2G" in sig_proc:
    MX = float(sig_proc.split("-")[1])
    MY = 125.0
  else:
    raise Exception("Unexpected signal process: %s"%sig_proc)
  return MX, MY

def getAUC(indir):
  auc_files = glob.glob(f"{indir}/*/ROC_skimmed.json")
  
  mx = []
  my = []
  train_auc = []
  test_auc = []
  
  for fname in auc_files:
    with open(fname, "r") as f:
      auc = json.load(f)
      
    sig_proc = fname.split("/")[2]
    mx.append(get_MX_MY(sig_proc)[0])
    my.append(get_MX_MY(sig_proc)[1])
    train_auc.append(auc["train_auc"])
    test_auc.append(auc["test_auc"])
      
  return pd.DataFrame({"MX": mx, "MY": my, "train": train_auc, "test": test_auc})
    
def plotAUC1D(indir):
  auc = getAUC(indir)
  print(auc)
  
  #plt.figure(figsize=(10, 4))
  plt.figure(figsize=(10, 10))
  plt.scatter(auc["MX"], auc["train"], label="Train", color="tab:blue")
  plt.scatter(auc["MX"], auc["test"], label="Test", color="tab:orange")
  
  def forward(y):
    return -np.log10(1 - y)  # Transform y-values

  def inverse(t):
      return 1 - 10**(-t)  # Inverse transform

  plt.yscale('function', functions=(forward, inverse))

  # Custom tick locations in data space
  ticks = [0.9, 0.99, 0.999, 0.9999]
  #plt.gca().set_yticks([str(t) for t in ticks])
  plt.gca().set_yticks(ticks)
  plt.gca().set_yticklabels(ticks) 
  minor_ticks = [0.9 + 0.01*i for i in range(9)]
  minor_ticks += [0.99 + 0.001*i for i in range(9)]
  minor_ticks += [0.999 + 0.0001*i for i in range(9)]
  plt.gca().yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
  
  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel("AUC")
  plt.legend(loc="upper left")
  plt.tight_layout()
  plt.savefig(f"{indir}/auc.pdf")
  plt.clf()
  
def plotAUC2D(indir):
  auc = getAUC(indir)
  print(auc)
  print(auc["test"].min())
  assert auc["test"].min() >= 0.9
  assert auc["test"].max() <= 0.9999
  
  mx = np.sort(np.unique(auc["MX"]))
  my = np.sort(np.unique(auc["MY"]))
  mx_edges = np.array([mx[0] - (mx[1]-mx[0])/2] + list(mx[:-1] + (mx[1:] - mx[:-1])/2) + [mx[-1] + (mx[-1]-mx[-2])/2])
  my_edges = np.array([my[0] - (my[1]-my[0])/2] + list(my[:-1] + (my[1:] - my[:-1])/2) + [my[-1] + (my[-1]-my[-2])/2])
  mx_edge_centers = (mx_edges[:-1]+mx_edges[1:])/2
  my_edge_centers = (my_edges[:-1]+my_edges[1:])/2
  
  def forward(y):
    return -np.log10(1 - y)  # Transform y-values

  def inverse(t):
      return 1 - 10**(-t)  # Inverse transform
  
  plt.figure(figsize=(10, 8))
  plt.hist2d(auc["MX"], auc["MY"], [mx_edges, my_edges], weights=forward(auc["test"]), cmin=1e-16, vmin=forward(0.9), vmax=forward(0.9999))
  cbar=plt.colorbar()
  
  ticks = np.array([0.9, 0.99, 0.999, 0.9999])
  cbar.set_ticks(forward(ticks))
  cbar.set_ticklabels(ticks)
  minor_ticks = [0.9 + 0.01*i for i in range(9)]
  minor_ticks += [0.99 + 0.001*i for i in range(9)]
  minor_ticks += [0.999 + 0.0001*i for i in range(9)]
  minor_ticks = np.array(minor_ticks)
  cbar.set_ticks(forward(minor_ticks), minor=True)

  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel(r"$m_Y$ [GeV]")
  cbar.set_label("Test AUC")
  plt.tight_layout()
  plt.savefig(f"{indir}/auc.pdf")
  plt.clf()
  
def plotAUC2DDiff(indir):
  auc = getAUC(indir)
  print(auc)
  print(auc["test"].min())
  assert auc["test"].min() >= 0.9
  assert auc["test"].max() <= 0.9999
  
  mx = np.sort(np.unique(auc["MX"]))
  my = np.sort(np.unique(auc["MY"]))
  mx_edges = np.array([mx[0] - (mx[1]-mx[0])/2] + list(mx[:-1] + (mx[1:] - mx[:-1])/2) + [mx[-1] + (mx[-1]-mx[-2])/2])
  my_edges = np.array([my[0] - (my[1]-my[0])/2] + list(my[:-1] + (my[1:] - my[:-1])/2) + [my[-1] + (my[-1]-my[-2])/2])
  mx_edge_centers = (mx_edges[:-1]+mx_edges[1:])/2
  my_edge_centers = (my_edges[:-1]+my_edges[1:])/2
  
  def forward(y):
    return -np.log10(1 - y)  # Transform y-values

  def inverse(t):
      return 1 - 10**(-t)  # Inverse transform
  
  plt.hist2d(auc["MX"], auc["MY"], [mx_edges, my_edges], weights=abs(auc["train"]-auc["test"]), cmap="afmhot_r", cmin=1e-16)
  cbar=plt.colorbar()
  
  # ticks = np.array([0.9, 0.99, 0.999, 0.9999])
  # cbar.set_ticks(forward(ticks))
  # cbar.set_ticklabels(ticks)
  # minor_ticks = [0.9 + 0.01*i for i in range(9)]
  # minor_ticks += [0.99 + 0.001*i for i in range(9)]
  # minor_ticks += [0.999 + 0.0001*i for i in range(9)]
  # minor_ticks = np.array(minor_ticks)
  # cbar.set_ticks(forward(minor_ticks), minor=True)

  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel(r"$m_Y$ [GeV]")
  cbar.set_label(r"|Train AUC - Train AUC|")
  cbar.formatter.set_powerlimits((0, 0))

  plt.tight_layout()
  plt.savefig(f"{indir}/auc_diff.pdf")
  plt.clf()

plotAUC1D("Outputs/Graviton")
plotAUC1D("Outputs/Radion")
plotAUC2D("Outputs/Y_tautau")
plotAUC2DDiff("Outputs/Y_tautau")
plotAUC2D("Outputs/Y_gg_Low_Mass")
plotAUC2DDiff("Outputs/Y_gg_Low_Mass")
plotAUC2D("Outputs/Y_gg_High_Mass_no_dilep_leadpho_mass")
plotAUC2DDiff("Outputs/Y_gg_High_Mass_no_dilep_leadpho_mass")
