import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
#mplhep.style.use("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
# plt.rcParams.update({
#     "text.usetex": True
# })
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
  1: ['#a6cee3'],
  2: ['#a6cee3','#1f78b4'],
  3: ['#a6cee3','#1f78b4','#b2df8a'],
  4: ['#a6cee3','#1f78b4','#b2df8a','#33a02c'],
  5: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99'],
  #6: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c'],
  6: ['#a6cee3','#e31a1c','#fb9a99','#1f78b4','#33a02c','#b2df8a'],
  7: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f'],
  8: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00']
}

# colour_schemes = {
#   4: ['#66c2a5','#fc8d62','#8da0cb','#e78ac3'],
#   5: ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854'],
#   6: ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f'],
#   7: ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494']
# }

# colour_schemes = {
#   5: ['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6']
# }

def createDefaultConfig(data, bkg, sig, args):
  config = OrderedDict()

  if args.columns is not None:
    columns_to_plot = common.expandColumns(args.columns)
  else:
    columns_to_plot = filter(lambda x: ("weight" not in x), data.columns)
    
  for column in columns_to_plot:
    d = data[column][data[column]!=common.dummy_val]
    b = bkg[column][bkg[column]!=common.dummy_val]
    s = sig[column][sig[column]!=common.dummy_val]
    
    config[column] = {}

    if d.dtype == int:
      low = min([d.min(), b.min(), s.min()])
      high = max([d.max(), b.max(), s.max()])
      config[column]["range"] = (low, high)
      print(column, low, high)
      config[column]["nbins"] = (high-low) 
    else:

      low = min([d.quantile(0.05), b.quantile(0.05), s.quantile(0.05)])
      high = max([d.quantile(0.95), b.quantile(0.95), s.quantile(0.95)])
      
      print(column, low, high)

      #if the absolute minimum and maximums are closeby, then extend to absolute max and/or min
      mmin = min([d.min(), b.min(), s.min()])
      if (low-mmin) / (high-low) < 0.2:
        low = mmin
      mmax = min([d.max(), b.max(), s.max()])
      if (mmax-high) / (high-low) < 0.2:
        high = mmax

      if column == "bdt_score":
        low = 0
        high = 2.7
      elif "score" in column:
        low = 0
        high = 1

      print(column, low, high)

      config[column]["range"] = (float(low),float(high))
      config[column]["nbins"] = 20
      #config[column]["nbins"] = 100

  #exceptions
  #config["Diphoton_mass"] = {"range": [55.0, 180.0]}

  return config

def writeDefaultConfig(data, bkg, sig):
  config = createDefaultConfig(data, bkg, sig)
  with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

def getConfig(data, bkg, sig, args):
  if args.config != None:
    with open(args.config, "r") as f:
      cfg = json.load(f)
  else:
    cfg = createDefaultConfig(data, bkg, sig, args)
    
  return cfg

def createBkgStack(bkg, column, proc_dict, group=True):
  bkg_stack = []
  bkg_stack_w = []
  bkg_stack_labels = []

  if not group:
    for proc in common.bkg_procs["all"]:
      bkg_stack.append(bkg[bkg.process_id==proc_dict[proc]][column])
      bkg_stack_w.append(bkg[bkg.process_id==proc_dict[proc]]["weight"])
      bkg_stack_labels.append(proc)
  else:
    for bkg_group in common.bkg_procs.keys():
      if bkg_group == "all": continue
      proc_ids = [proc_dict[proc] for proc in common.bkg_procs[bkg_group] if proc in proc_dict.keys()]
      bkg_stack.append(bkg[bkg.process_id.isin(proc_ids)][column])
      bkg_stack_w.append(bkg[bkg.process_id.isin(proc_ids)]["weight"])
      bkg_stack_labels.append(common.bkg_procs_latex[bkg_group])

  is_sorted = False
  while not is_sorted:
    is_sorted = True
    for i in range(len(bkg_stack)-1):
      if bkg_stack_w[i].sum() > bkg_stack_w[i+1].sum():
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
    #label = r"$X_{%d}\rightarrow$"%X_mass + "\n" + r"$ Y_{%d}(\rightarrow %s)  H$"%(Y_mass, Y_decay)
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
  else:
    label = sig_proc

  # if (sig_sf is not None) and (sig_sf != 1):
  #   label += r" $\times$ %d"%sig_sf
  return label

def adjustLimits(x, ys, ax):
  data_to_display = ax.transData.transform
  display_to_data = ax.transData.inverted().transform

  tx = lambda x: data_to_display((x,0))[0]
  tx_inv = lambda x: display_to_data((x,0))[0]
  ty = lambda x: data_to_display((0,x))[1]
  ty_inv = lambda x: display_to_data((0,x))[1]

  xlow, xhigh = tx(ax.get_xlim()[0]), tx(ax.get_xlim()[1])
  ylow, yhigh = ty(ax.get_ylim()[0]), ty(ax.get_ylim()[1])
  
  #top side
  ybound = ylow + (yhigh-ylow)*0.60
  max_y = np.nan_to_num(ys).max()
  top_distance_to_move = ty(max_y) - ybound
  
  #right side
  xbound = xlow + (xhigh-xlow)*0.75
  ybound = ylow + (yhigh-ylow)*0.20
  max_y = np.array(ys).T[x>tx_inv(xbound)].max()
  right_distance_to_move = ty(max_y) - ybound

  frameon=True

  if right_distance_to_move <= 0:
    ax.legend(ncol=1, loc="upper right", markerfirst=False, frameon=frameon, framealpha=1.0)
  elif right_distance_to_move < top_distance_to_move:
    ax.legend(ncol=1, loc="upper right", markerfirst=False, frameon=frameon, framealpha=1.0)
    ax.set_ylim(top = ty_inv(yhigh + right_distance_to_move))
  else:
    ax.legend(ncol=3, loc="upper right", markerfirst=False, frameon=frameon, framealpha=1.0)
    ax.set_ylim(top = ty_inv(yhigh + top_distance_to_move))

def plot_feature(data, bkg, sig, proc_dict, sig_procs, column, nbins, feature_range, save_path, no_legend=False, only_legend=False, limits=None, delete_zeros=False):
  if type(sig_procs) != list: sig_procs = [sig_procs]

  if not only_legend: 
    plt.rcParams["figure.figsize"] = (12.5,10)
  else:
    assert no_legend
    plt.rcParams["figure.figsize"] = (12.5*2,10)
  
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  data_hist, edges = np.histogram(data[column], bins=nbins, range=feature_range, weights=data.weight)
  data_error = np.sqrt(data_hist)
  data_error[data_error == 0] = 1.0
  bin_centres = (edges[:-1]+edges[1:])/2

  #bkg.iloc[0, bkg.columns.get_loc(column)] = 0.5
  #bkg.iloc[0, bkg.columns.get_loc("weight")] = 1000

  bkg_stack, bkg_stack_w, bkg_stack_labels = createBkgStack(bkg, column, proc_dict)

  bkg_stack_ungroup, bkg_stack_w_ungroup, bkg_stack_labels_ungroup = createBkgStack(bkg, column, proc_dict, group=False)
  bkg_sumw, bkg_error = getBkgError(bkg_stack_ungroup, bkg_stack_w_ungroup, edges)
  #bkg_sumw, bkg_error = getBkgError(bkg_stack, bkg_stack_w, edges)

  #data_hist *= sum(bkg_sumw) / sum(data_hist) # norm the histograms

  ratio = data_hist / bkg_sumw
  ratio_err = data_error / bkg_sumw

  s = (data_hist != 0) | (not delete_zeros)

  #bkg_unc_label = "Background uncertainty" if common.LOW_MASS_MODE else "Background\nuncertainty"
  bkg_unc_label = "Background\nuncertainty"
  axs[0].fill_between(edges, np.append(bkg_sumw-bkg_error, 0), np.append(bkg_sumw+bkg_error, 0), step="post", alpha=0.5, color="grey", zorder=2, label=bkg_unc_label) #background uncertainty
  axs[0].hist(bkg_stack, edges, weights=bkg_stack_w, label=bkg_stack_labels, stacked=True, color=colour_schemes[len(bkg_stack)], zorder=1) #background
  axs[0].errorbar(bin_centres[s], data_hist[s], data_error[s], label="Data", fmt='ko', zorder=4) #data
  axs[0].set_ylabel("Events")

  axs[1].errorbar(bin_centres[s], ratio[s], ratio_err[s], label="Data", fmt='ko')
  axs[1].fill_between(edges, np.append(1-bkg_error/bkg_sumw, 1), np.append(1+bkg_error/bkg_sumw, 1), step="post", alpha=0.5, color="grey")
  axs[1].axhline(1.0, color="k")

  if column in common.latex_dict:
    xlabel = common.latex_dict[column]
  elif "score" in column:
    mx, my = common.get_MX_MY(column.split("score_")[1])
    if "NMSSM" in column:
      xlabel = r"$f(\vec{x};m_X=%d~GeV, m_Y=%d~GeV)$"%(mx, my)
    else:
      xlabel = r"$f(\vec{x};m_X=%d~GeV)$"%mx
  else:
    xlabel = column
  axs[1].set_xlabel(xlabel)
  axs[1].set_ylabel("Data / MC")
  axs[1].set_ylim([0.5, 1.5])
  #axs[1].set_ylim([0, 2])
  #axs[1].set_ylim([-1, 3])

  plt.sca(axs[0])
  

  # os.makedirs(save_path, exist_ok=True)
  # for sig_proc in sig_procs:
  #   try: _ = [b.remove() for b in bars]
  #   except: pass
  #   sig_hist, edges = np.histogram(sig[sig.process_id==proc_dict[sig_proc]][column], bins=nbins, range=feature_range, weights=sig[sig.process_id==proc_dict[sig_proc]]["weight"])
  #   sig_sf = data_hist.max() / sig_hist.max()
  #   counts, bins, bars = axs[0].hist(edges[:-1], edges, weights=sig_hist*sig_sf, label=getSigLabel(sig_proc), histtype='step', color='r', lw=3, zorder=9) #signal

  #   if not auto_legend: axs[0].legend()
    
  #   axs[0].set_yscale("linear")
  #   axs[0].relim()
  #   axs[0].autoscale()
  #   axs[0].get_ylim()
  #   if auto_legend: adjustLimits(bin_centres, [sig_hist*sig_sf, data_hist], axs[0])
  #   plt.savefig("%s/%s.png"%(save_path, sig_proc))
  #   #plt.savefig("%s.pdf"%save_path)

  #   axs[0].set_yscale("log")
  #   axs[0].relim()
  #   axs[0].autoscale()
  #   axs[0].get_ylim()
  #   if auto_legend: adjustLimits(bin_centres, [sig_hist*sig_sf, data_hist], axs[0])
  #   plt.savefig("%s/%s_log.png"%(save_path, sig_proc))
  #   #plt.savefig("%s_log.pdf"%save_path)

  for sig_proc in sig_procs:
    sig_hist, edges = np.histogram(sig[sig.process_id==proc_dict[sig_proc]][column], bins=nbins, range=feature_range, weights=sig[sig.process_id==proc_dict[sig_proc]]["weight"])
    if limits is None:
      sig_sf = max([bkg_sumw.max(), data_hist.max()]) / sig_hist.max()
      #sig_sf = 25
      #sig_sf = 50      
    else:
      MX, MY = common.get_MX_MY(sig_proc)
      scale = 1000
      sig_sf = limits[(limits.MX==MX)&(limits.MY==MY)].iloc[0]["Expected 95% CL Limit [fb]"] * scale
    
    #axs[0].hist(edges[:-1], edges, weights=sig_hist*sig_sf, label=getSigLabel(sig_proc, sig_sf), histtype='step', lw=4, path_effects=[pe.Stroke(linewidth=6, foreground='w'), pe.Normal()], zorder=3, color="k") #signal
    axs[0].hist(edges[:-1], edges, weights=sig_hist*sig_sf, label=getSigLabel(sig_proc, sig_sf), histtype='step', lw=4, zorder=3, color="k") #signal

  if column == "bdt_score":
    bounds = np.array([0.9891, 0.973610])
    bounds = -np.log10(1-bounds)
    for cut in bounds:
      axs[0].axvline(cut, linestyle=":", zorder=4, linewidth=5, color="gray")
    # places = np.array([0, 0.33, 0.66, 1, 1.33, 1.66, 2.0])
    # labels = ["%.2f"%num for num in (1-np.power(10, -places))] + ["%.3f"%(1-np.power(10, -2.3)), "%.3f"%(1-np.power(10, -2.7))]
    # axs[0].set_xticks([0, 0.33, 0.66, 1, 1.33, 1.66, 2.0, 2.3, 2.7])
    # axs[0].set_xticklabels(labels)
    # axs[0].minorticks_off()

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

  if only_legend: 
    axs[0].legend(ncol=int((2+len(sig_procs)+len(bkg_stack))/3), frameon=True, loc="center", title=r"Signal normalised to %d $\times$ expected limit"%scale)
  
  axs[0].set_yscale("log")
  if not no_legend:
    #axs[0].relim()
    #axs[0].autoscale()
    #axs[0].get_ylim()
    #adjustLimits(bin_centres, [sig_hist*sig_sf, data_hist, bkg_sumw], axs[0])
    
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
    
    plt.legend(handles, labels, ncol=3, loc="upper right", markerfirst=False, frameon=False, framealpha=1.0)

  if common.LOW_MASS_MODE:
    axs[0].set_ylim(1, 5e6)
  else:
    axs[0].set_ylim(1, 5e5)
  #axs[0].set_ylim(0.02, 5e5)
  #axs[0].set_ylim(0, 800)
  axs[0].set_xlim(feature_range)
  mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=0)
  # plt.savefig("%s_%s_paper.pdf"%(save_path, sig_proc))
  # plt.savefig("%s_%s_paper.png"%(save_path, sig_proc))
  
  plt.savefig("%s_%s_log.pdf"%(save_path, sig_proc))
  axs[0].set_yscale("linear")
  plt.savefig("%s_%s_linear.pdf"%(save_path, sig_proc))

  # mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)
  # plt.savefig("%s_%s_preliminary.pdf"%(save_path, sig_proc))
  # plt.savefig("%s_%s_preliminary.png"%(save_path, sig_proc))

  # axs[0].set_yscale("linear")
  # if not no_legend:
  #   axs[0].relim()
  #   axs[0].autoscale()
  #   axs[0].get_ylim()
  #   adjustLimits(bin_centres, [sig_hist*sig_sf, data_hist, bkg_sumw], axs[0])
  # mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=0)
  # plt.savefig("%s_%s_paper.pdf"%(save_path, sig_proc))
  # plt.savefig("%s_%s_paper.png"%(save_path, sig_proc))

  # mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)
  # plt.savefig("%s_%s_preliminary.pdf"%(save_path, sig_proc))
  # plt.savefig("%s_%s_preliminary.png"%(save_path, sig_proc))

  plt.close()

def plot(data, bkg, sig, proc_dict, args, limits=None):
  if args.no_legend:
    data["no_legend"] = np.random.randint(0, 2, size=len(data))
    bkg["no_legend"] = np.random.randint(0, 2, size=len(bkg))
    sig["no_legend"] = np.random.randint(0, 2, size=len(sig))

  cfg = getConfig(data, bkg, sig, args)

  for column in tqdm(cfg.keys()):
    #if ("b_jet" not in column) and ("bjet" not in column): continue
    #if "bdt_score" not in column: continue
    #if "intermediate_transformed_score_GluGluToBulkGravitonToHHTo2G2Tau_M-260" not in column: continue
    print(column)
    nbins = cfg[column]["nbins"]
    feature_range = cfg[column]["range"]
    save_path = "%s/%s"%(args.output, column)
    plot_feature(data, bkg, sig, proc_dict, args.sig_procs, column, nbins, feature_range, save_path, no_legend=args.no_legend, limits=limits)

  if args.no_legend:
    column = "no_legend"
    nbins = cfg[column]["nbins"]
    feature_range = cfg[column]["range"]
    save_path = "%s/%s"%(args.output, column)
    plot_feature(data, bkg, sig, proc_dict, args.sig_procs, column, nbins, feature_range, save_path, no_legend=args.no_legend, only_legend=True, limits=limits)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', type=str)
  parser.add_argument('--summary', '-s', type=str)
  parser.add_argument('--limits', default=None, type=str)
  parser.add_argument('--sig-procs', '-p', type=str, nargs="+")
  parser.add_argument('--output', '-o', type=str, default="plots")
  parser.add_argument('--config', '-c', type=str)
  parser.add_argument('--norm', default=False, action="store_true")
  parser.add_argument('--no-legend', default=False, action="store_true")
  parser.add_argument('--weight', default="weight_central", type=str)
  parser.add_argument('--systematic', default=None, type=str, help="Specify a weight to multiply the nominal weight by, i.e apply a systematic variation to the MC")
  parser.add_argument('--columns', default=None, type=str, nargs="+")
  args = parser.parse_args()

  #args.sig_procs = args.sig_procs[:1]

  with open(args.summary, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  if args.limits != None:
    with open(args.limits, "r") as f:
      limits = pd.read_csv(args.limits, index_col=0)
      print(limits)
  else:
    limits = None

  #sometimes bkg processes won't appear in parquet file because none passed selection
  for bkg_proc in common.bkg_procs["all"]:
    if bkg_proc not in proc_dict.keys():
      proc_dict[bkg_proc] = -9999

  print(">> Loading dataframes")  
  
  if args.columns is not None:
    columns = common.expandColumns(args.columns) + [args.weight, "process_id"]
  else:
    columns = common.getColumns(args.input)
    columns_to_exclude = ["year", "event", "MX", "MY"]
    columns = list(filter(lambda x: ("weight" not in x) and (x not in columns_to_exclude), columns)) + [args.weight]
  
  if args.systematic is not None:
    columns.append(args.systematic)
  columns = list(set(columns))

  df = pd.read_parquet(args.input, columns=columns)
  print(df)
  
  if "bdt_score" in df.columns:
    #print(df["bdt_score"].max())
    df.loc[:, "bdt_score"] = -np.log10(1-df["bdt_score"])

  #df = df[(df.pass_sr_1==0) & (df.pass_sr_0==0)]

  df.rename({args.weight: "weight"}, axis=1, inplace=True)

  if args.systematic is not None:
    df.loc[:, "weight"] *= df[args.systematic]

  print(">> Splitting into data, background and signal")
  data = df[df.process_id==proc_dict["Data"]]
  bkg_proc_ids = [proc_dict[bkg_proc] for bkg_proc in common.bkg_procs["all"]]
  bkg = df[df.process_id.isin(bkg_proc_ids)]
  sig_proc_ids = [proc_dict[proc] for proc in args.sig_procs]
  sig = df[df.process_id.isin(sig_proc_ids)]
  
  del df

  #blinding
  #data = data[(data.Diphoton_mass < 120) | (data.Diphoton_mass>130)]
  #bkg = bkg[(bkg.Diphoton_mass < 120) | (bkg.Diphoton_mass>130)]

  print("Data sumw: %f"%data.weight.sum())
  print("Bkg MC sumw: %f"%bkg.weight.sum())
  #normalise bkg mc to data
  if args.norm:
    bkg.loc[:, "weight"] = bkg.weight * (data.weight.sum() / bkg.weight.sum())

  os.makedirs(args.output, exist_ok=True)
   
  np.seterr(all='ignore')

  #import cProfile
  #cProfile.run('plot(data, bkg, sig, proc_dict, args)', 'restats')
  plot(data, bkg, sig, proc_dict, args, limits)
  
  