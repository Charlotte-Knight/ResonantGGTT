import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
import matplotlib.patheffects as pe

import pandas as pd
import numpy as np

import argparse
import os

import json
from collections import OrderedDict

from tqdm import tqdm
import common

colour_schemes = {
  4: ['#a6cee3','#1f78b4','#b2df8a','#33a02c'],
  5: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99'],
  6: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c'],
  7: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f'],
  8: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00']
}

poisson_ci = pd.read_csv("poisson_interval_68.csv")
poisson_ci.set_index("N", inplace=True)
poisson_ci = poisson_ci.to_numpy()

def createBkgStack(bkg, column, proc_dict, ordering_cut=0, group=True):
  bkg_stack = []
  bkg_stack_w = []
  bkg_stack_labels = []

  if not group:
    for proc in common.bkg_procs["all"]:
      #if proc == "TT": continue
      bkg_stack.append(bkg[bkg.process_id==proc_dict[proc]][column])
      bkg_stack_w.append(bkg[bkg.process_id==proc_dict[proc]]["weight"])
      bkg_stack_labels.append(proc)
  else:
    for bkg_group in common.bkg_procs.keys():
      if bkg_group == "all": continue
      #if bkg_group == "TT": continue
      proc_ids = [proc_dict[proc] for proc in common.bkg_procs[bkg_group] if proc in proc_dict.keys()]
      bkg_stack.append(bkg[bkg.process_id.isin(proc_ids)][column])
      bkg_stack_w.append(bkg[bkg.process_id.isin(proc_ids)]["weight"])
      bkg_stack_labels.append(common.bkg_procs_latex[bkg_group])

  is_sorted = False
  while not is_sorted:
    is_sorted = True
    for i in range(len(bkg_stack)-1):
      if bkg_stack_w[i][bkg_stack[i]>ordering_cut].sum() > bkg_stack_w[i+1][bkg_stack[i+1]>ordering_cut].sum():
        is_sorted = False
        bkg_stack[i], bkg_stack[i+1] = bkg_stack[i+1], bkg_stack[i]
        bkg_stack_w[i], bkg_stack_w[i+1] = bkg_stack_w[i+1], bkg_stack_w[i]
        bkg_stack_labels[i], bkg_stack_labels[i+1] = bkg_stack_labels[i+1], bkg_stack_labels[i]

  return bkg_stack, bkg_stack_w, bkg_stack_labels

def getBkgError(bkg_stack, bkg_stack_w, edges):
  Ns = []
  sumws = []
  sumw2s = []  
  for i, bkg in enumerate(bkg_stack):
    N, edges = np.histogram(bkg, bins=edges)
    sumw, edges = np.histogram(bkg, bins=edges, weights=bkg_stack_w[i])
    sumw2, edges = np.histogram(bkg, bins=edges, weights=bkg_stack_w[i]**2)
    Ns.append(N)
    sumws.append(sumw)
    sumw2s.append(sumw2)
  Ns = np.array(Ns)
  sumws = np.array(sumws)
  sumw2s = np.array(sumw2s)
  
  N = np.sum(Ns, axis=0)
  sumw = np.sum(sumws, axis=0)
  error = np.sqrt(np.sum(sumw2s, axis=0))
  #error_per_sqrt_sumw = np.sqrt(sumw2s.sum()) / np.sqrt(sumws.sum())
  #error = error_per_sqrt_sumw * np.sqrt(sumw)
  #error_per_sqrt_N = np.sqrt(sumw2s.sum()) / np.sqrt(Ns.sum())
  #error = error_per_sqrt_N * np.sqrt(N)

  return sumw, error

def decayToMath(channel):
  if channel == "gg":
    return r"\gamma\gamma"
  else:
    return r"\tau\tau"

def getSigLabel(sig_proc, sig_sf=None):
  if "NMSSM" in sig_proc:
    split_name = sig_proc.split("_")
    Y_decay = decayToMath(split_name[3])
    H_decay = decayToMath(split_name[5])
    X_mass = int(split_name[7])
    Y_mass = int(split_name[9])
    #label = r"$X_{%d} \rightarrow Y_{%d}(\rightarrow %s)  H(\rightarrow %s)$"%(X_mass, Y_mass, Y_decay, H_decay)
    label = r"$X_{%d} \rightarrow Y_{%d}(\rightarrow %s)  H$"%(X_mass, Y_mass, Y_decay)
  elif "radion" in sig_proc.lower():
    X_mass = int(sig_proc.split("M-")[1].split("_")[0])
    #label = r"$X_{%d}^{(0)} \rightarrow HH \rightarrow \gamma\gamma\tau\tau$"%X_mass
    label = r"$X_{%d}^{(0)} \rightarrow HH$"%X_mass
  elif "graviton" in sig_proc.lower():
    X_mass = int(sig_proc.split("M-")[1].split("_")[0])
    #label = r"$X_{%d}^{(2)} \rightarrow HH \rightarrow \gamma\gamma\tau\tau$"%X_mass
    label = r"$X_{%d}^{(2)} \rightarrow HH$"%X_mass
  elif "M" in sig_proc:
    label = r"$m_X=%d$"%int(sig_proc.split("M")[1])
    #label = sig_proc
  elif sig_proc == "HH":
    label = "SM HH"
  else:
    label = sig_proc

  if (sig_sf is not None) and (sig_sf != 1):
    label += r" $\times$ %d"%sig_sf
  return label

def poissonError(n):
  errors = []
  for i in range(len(n)):
    if n[i] > 1000:
      errors.append([np.sqrt(n[i]), np.sqrt(n[i])])
    else:
      errors.append(poisson_ci[int(n[i])])
  return np.array(errors).T

def plot_feature(data, bkg, sig, proc_dict, sig_procs, column, nbins, feature_range, save_path):
  if type(sig_procs) != list: sig_procs = [sig_procs]

  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  bounds = np.array([0.973610, 0.9891])
  bounds = -np.log10(1-bounds)

  edges = list(np.linspace(feature_range[0], bounds[0], 12)) + [bounds[1], feature_range[1]]

  edges_lin_space = 1 - (1/np.power(10, edges))
  print(edges_lin_space)
  bin_widths = np.diff(edges_lin_space)
  #edges = 20
  #bin_widths = np.ones(edges)

  data_hist, edges = np.histogram(data[column], bins=edges, range=feature_range, weights=data.weight)
  data_error = poissonError(data_hist)

  data_hist /= bin_widths
  data_error /= bin_widths

  bin_centres = (edges[:-1]+edges[1:])/2

  bkg_stack, bkg_stack_w, bkg_stack_labels = createBkgStack(bkg, column, proc_dict, ordering_cut=0.0)
  bkg_stack_ungroup, bkg_stack_w_ungroup, bkg_stack_labels_ungroup = createBkgStack(bkg, column, proc_dict, group=False)
  bkg_sumw, bkg_error = getBkgError(bkg_stack_ungroup, bkg_stack_w_ungroup, edges)

  bkg_sumw /= bin_widths
  bkg_error /= bin_widths

  ratio = data_hist / bkg_sumw
  ratio_err = data_error / bkg_sumw

  axs[0].fill_between(edges, np.append(bkg_sumw-bkg_error, 0), np.append(bkg_sumw+bkg_error, 0), step="post", alpha=0.5, color="grey", zorder=2,
                      label="Background\nuncertainty") #background uncertainty
                      #label="Bkg.\nuncert.") #background uncertainty
  
  bkg_stack_hist = []
  for bkg, bkg_w in zip(bkg_stack, bkg_stack_w):
    hist, edges = np.histogram(bkg, bins=edges, range=feature_range, weights=bkg_w)
    hist /= bin_widths
    bkg_stack_hist.append(hist)

  #axs[0].hist(bkg_stack, edges, weights=bkg_stack_w, label=bkg_stack_labels, stacked=True, color=colour_schemes[len(bkg_stack)], zorder=1) 
  axs[0].hist([edges[:-1] for each in bkg_stack_hist], edges, weights=bkg_stack_hist, label=bkg_stack_labels, stacked=True, color=colour_schemes[len(bkg_stack)], zorder=1) 
  
  axs[0].errorbar(bin_centres, data_hist, data_error, label="Data", fmt='ko', zorder=4) #data
  axs[0].set_ylabel("Events / Bin width")

  axs[1].errorbar(bin_centres, ratio, ratio_err, label="Data", fmt='ko')
  axs[1].fill_between(edges, np.append(1-bkg_error/bkg_sumw, 1), np.append(1+bkg_error/bkg_sumw, 1), step="post", alpha=0.5, color="grey")
  axs[1].axhline(1.0, color="k")

  xlabel = "BDT Score"

  axs[1].set_xlabel(xlabel)
  axs[1].set_ylabel("Data / MC")
  axs[1].set_ylim([0, 2])

  plt.sca(axs[0])
  
  for sig_proc in sig_procs:
    sig_hist, edges = np.histogram(sig[sig.process_id==proc_dict[sig_proc]][column], bins=edges, range=feature_range, weights=sig[sig.process_id==proc_dict[sig_proc]]["weight"])
    sig_hist /= bin_widths
    
    sig_sf = 1
    axs[0].hist(edges[:-1], edges, weights=sig_hist*sig_sf, label=getSigLabel(sig_proc, sig_sf), histtype='step', lw=4, zorder=3, color="k") #signal

  if column == "bdt_score":
    for cut in bounds:
      axs[0].axvline(cut, linestyle=":", zorder=4, linewidth=5, color="gray")
      axs[1].axvline(cut, linestyle=":", zorder=4, linewidth=5, color="gray")

    places = [0, 1, 2, 2.7]
    labels = ["0", "0.9", "0.99", "0.998"]
    axs[0].set_xticks(places)
    axs[0].set_xticklabels(labels)
    minor_places = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
     0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
     0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997])
    axs[0].set_xticks(-np.log10(1-minor_places), minor=True)

    f.text(0.3, 0.3075, "Discarded")
    f.text(0.60, 0.3075, "Cat 1")
    f.text(0.76, 0.3075, "Cat 0")
  
  axs[0].set_yscale("log")
  #axs[0].set_ylim(0.02, 5e5)
  axs[0].set_ylim(1, 1e7)
  axs[0].set_xlim(feature_range)
  
  # legend sorting
  handles, labels = plt.gca().get_legend_handles_labels()
  handles = handles[::-1] # flip
  labels = labels[::-1] # flip
  handles.append(handles.pop(1)) # move bkg uncert line to end
  labels.append(labels.pop(1)) # move bkg uncert line to end
  handles.append(handles.pop(1)) # move HH line to end
  labels.append(labels.pop(1)) # move HH line to end
  #handles.append(handles.pop(0)) # move data line to end
  #labels.append(labels.pop(0)) # move data line to end
  
  axs[0].legend(handles, labels, loc="best", ncol=3, markerfirst=False, frameon=True, framealpha=1)
  #axs[0].legend(loc="best", ncol=3, markerfirst=False, frameon=True, framealpha=1)
  mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig("%s_%s_paper.pdf"%(save_path, sig_proc))
  plt.savefig("%s_%s_paper.png"%(save_path, sig_proc))

  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig("%s_%s_preliminary.pdf"%(save_path, sig_proc))
  plt.savefig("%s_%s_preliminary.png"%(save_path, sig_proc))

  plt.close()

def plot(data, bkg, sig, proc_dict, args, limits=None):
  column = "bdt_score"

  nbins = 20
  feature_range = (0, 2.7)
  save_path = "%s/%s"%(args.output, column)
  plot_feature(data, bkg, sig, proc_dict, args.sig_procs, column, nbins, feature_range, save_path)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', type=str)
  parser.add_argument('--summary', '-s', type=str)
  parser.add_argument('--sig-procs', '-p', type=str, nargs="+")
  parser.add_argument('--output', '-o', type=str, default="plots")
  parser.add_argument('--config', '-c', type=str)
  parser.add_argument('--norm', default=False, action="store_true")
  parser.add_argument('--no-legend', default=False, action="store_true")
  parser.add_argument('--systematic', default=None, type=str, help="Specify a weight to multiply the nominal weight by, i.e apply a systematic variation to the MC")
  parser.add_argument('--columns', default=None, type=str, nargs="+")
  args = parser.parse_args()

  with open(args.summary, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  #sometimes bkg processes won't appear in parquet file because none passed selection
  for bkg_proc in common.bkg_procs["all"]:
    if bkg_proc not in proc_dict.keys():
      proc_dict[bkg_proc] = -9999

  print(">> Loading dataframes")  
  
  columns = ["bdt_score", "weight_central", "process_id"]
  df = pd.read_parquet(args.input, columns=columns)
  
  df.loc[:, "bdt_score"] = -np.log10(1-df["bdt_score"])

  df.rename({"weight_central": "weight"}, axis=1, inplace=True)

  if args.systematic is not None:
    df.loc[:, "weight"] *= df[args.systematic]

  print(">> Splitting into data, background and signal")
  data = df[df.process_id==proc_dict["Data"]]
  bkg_proc_ids = [proc_dict[bkg_proc] for bkg_proc in common.bkg_procs["all"]]
  bkg = df[df.process_id.isin(bkg_proc_ids)]
  sig_proc_ids = [proc_dict[proc] for proc in args.sig_procs]
  sig = df[df.process_id.isin(sig_proc_ids)]
  
  del df

  for proc in proc_dict.keys():
    if proc_dict[proc] in bkg.process_id.unique():
      print(proc, bkg[bkg.process_id==proc_dict[proc]].weight.sum(), len(bkg[bkg.process_id==proc_dict[proc]])) 


  if args.norm:
    bkg.loc[:, "weight"] = bkg.weight * (data.weight.sum() / bkg.weight.sum())

  os.makedirs(args.output, exist_ok=True)
   
  np.seterr(all='ignore')

  plot(data, bkg, sig, proc_dict, args)
  
  