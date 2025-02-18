import pandas as pd
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
import numpy as np

df = pd.read_parquet(sys.argv[1])
with open(sys.argv[2], "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

data = df[df.process_id==proc_dict["Data"]]
bkg_proc_ids = [proc_dict[bkg_proc] for bkg_proc in common.bkg_procs["all"]]
bkg = df[df.process_id.isin(bkg_proc_ids)]
sig_proc_ids = [proc_dict["HHggTauTau"] ]
sig = df[df.process_id.isin(sig_proc_ids)]

bkg.loc[:, "weight_central"] *= data["weight_central"].sum() / bkg["weight_central"].sum()

del df
print("\n".join(data.columns))

# up_columns = list(filter(lambda x: ("weight" in x) and ("up" in x), data.columns))
# down_columns = list(filter(lambda x: ("weight" in x) and ("down" in x), data.columns))

up_columns = list(filter(lambda x: ("weight" in x) and ("up" in x) and ("btag" in x), data.columns))
down_columns = list(filter(lambda x: ("weight" in x) and ("down" in x) and ("btag" in x), data.columns))

def getVariations(df, columns):
  nom, edges = np.histogram(df.b_jet_1_btagDeepFlavB, bins=20, range=(0, 1), weights=df.weight_central)

  hists = []

  for upcol in columns:
    hist, edges = np.histogram(df.b_jet_1_btagDeepFlavB, bins=20, range=(0, 1), weights=df.weight_central*df[upcol])
    hists.append(hist / nom)

  return np.array(hists)

# def effectOnSREfficiency(data, bkg, sig):
#   btag_quantiles = [-10, -1] + list(sig[sig.b_jet_1_btagDeepFlavB >= 0].b_jet_1_btagDeepFlavB.quantile(q=[0.1*i for i in range(1, 10)])) + [1]
#   print(btag_quantiles)

#   # data_hist, edges = np.histogram(data.b_jet_1_btagDeepFlavB, bins=20, range=(0, 1), weights=data.weight_central)
#   # bkg_hist, edges = np.histogram(bkg.b_jet_1_btagDeepFlavB, bins=20, range=(0, 1), weights=bkg.weight_central)
#   # sig_hist, edges = np.histogram(sig.b_jet_1_btagDeepFlavB, bins=20, range=(0, 1), weights=sig.weight_central)
#   data_hist, edges = np.histogram(data.b_jet_1_btagDeepFlavB, bins=btag_quantiles, weights=data.weight_central)
#   bkg_hist, edges = np.histogram(bkg.b_jet_1_btagDeepFlavB, bins=btag_quantiles, weights=bkg.weight_central)
#   sig_hist, edges = np.histogram(sig.b_jet_1_btagDeepFlavB, bins=btag_quantiles, weights=sig.weight_central)
#   sig_hist_sr0, edges = np.histogram(sig[sig.pass_sr_0==1].b_jet_1_btagDeepFlavB, bins=btag_quantiles, weights=sig[sig.pass_sr_0==1].weight_central)
   
#   sf = data_hist / bkg_hist
#   print(sf)

#   sig["weight_central_reweight"] = sig["weight_central"]
#   #reweight
#   for i in range(len(edges)-1):
#     s = (sig.b_jet_1_btagDeepFlavB > edges[i]) & (sig.b_jet_1_btagDeepFlavB < edges[i+1]) 
#     sig.loc[s, "weight_central_reweight"] *= sf[i]

#   before_sumw = sig.loc[sig.pass_sr_0==1, "weight_central"].sum()
#   after_sumw = sig.loc[sig.pass_sr_0==1, "weight_central_reweight"].sum()
#   print(before_sumw, after_sumw, (after_sumw-before_sumw)/after_sumw)

#   before_sumw = sig_hist_sr0.sum()
#   after_sumw = (sig_hist_sr0*sf).sum()
#   print(before_sumw, after_sumw, (after_sumw-before_sumw)/after_sumw)
#   # print(sig.weight_central.sum(), sig.weight_central_reweight.sum())

#   exit()
  
#   #sig.loc[:, "weight_central_reweight"] *= sig.loc[:, "weight_central"].sum() / sig.loc[:, "weight_central_reweight"].sum()
#   #print(sig.loc[sig.pass_sr_0==1, "weight_central_reweight"].sum())
  
#   # for col in down_columns:
#   #   print((sig.loc[sig.pass_sr_0==1, "weight_central"]*sig.loc[sig.pass_sr_0==1, col]).sum())

def effectOnSREfficiency(data, bkg, sig):
  bkg.loc[:, "weight_central"] *= data.weight_central.sum() / bkg.weight_central.sum()

  sig_nojet = sig[sig.b_jet_1_btagDeepFlavB >= 0]
  q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

  btag_quantiles = [-10, -1] + list(sig_nojet.b_jet_1_btagDeepFlavB.quantile(q=q)) + [1] # include all
  #btag_quantiles = [-1] + list(sig_nojet.b_jet_1_btagDeepFlavB.quantile(q=q)) + [1] # exclude 0-jet
  print(btag_quantiles)

  data_hist, edges = np.histogram(data.b_jet_1_btagDeepFlavB, bins=btag_quantiles, weights=data.weight_central)
  bkg_hist, edges = np.histogram(bkg.b_jet_1_btagDeepFlavB, bins=btag_quantiles, weights=bkg.weight_central)
  sig_hist_sr0, edges = np.histogram(sig[sig.pass_sr_0==1].b_jet_1_btagDeepFlavB, bins=btag_quantiles, weights=sig[sig.pass_sr_0==1].weight_central)

  bkg_hist *= data_hist.sum() / bkg_hist.sum()
  sf = data_hist / bkg_hist

  #sf = np.concatenate([[1.0], (data_hist / bkg_hist)[1:]])
  
  print(sf)

  before_sumw = sig_hist_sr0.sum()
  after_sumw = (sig_hist_sr0*sf).sum()
  print(before_sumw, after_sumw, (after_sumw-before_sumw)/after_sumw)

def effectOnSREfficiencyNjet(data, bkg, sig):
  bkg.loc[:, "weight_central"] *= data.weight_central.sum() / bkg.weight_central.sum()

  data_hist, edges = np.histogram(data.n_bjets, bins=[0, 1, 2, 3, 4, 5], weights=data.weight_central)
  bkg_hist, edges = np.histogram(bkg.n_bjets,  bins=[0, 1, 2, 3, 4, 5], weights=bkg.weight_central)
  #sig_hist_sr0, edges = np.histogram(sig[sig.pass_sr_0==1].n_bjets, bins=[0, 1, 2, 3, 4, 5], weights=sig[sig.pass_sr_0==1].weight_central)
  sig_hist_sr0, edges = np.histogram(sig[sig.pass_sr_1==1].n_bjets, bins=[0, 1, 2, 3, 4, 5], weights=sig[sig.pass_sr_1==1].weight_central)

  #bkg_hist *= data_hist.sum() / bkg_hist.sum()

  sf = data_hist / bkg_hist
  print(sf)

  before_sumw = sig_hist_sr0.sum()
  after_sumw = (sig_hist_sr0*sf).sum()
  print(before_sumw, after_sumw, (after_sumw-before_sumw)/after_sumw)

effectOnSREfficiency(data, bkg, sig)
effectOnSREfficiencyNjet(data, bkg, sig)


# variations = getVariations(sig, down_columns)

# variations = abs(variations-1)
# print(variations)

# for i in range(len(variations)):
#    print(down_columns[i])
#    print(variations[i])

# first_bins = variations[:,0]
# max_idx = np.argmax(first_bins)
# print(down_columns[max_idx])
# print(variations[max_idx])

# print(variations.shape)
# variations = variations**2
# variations = np.sqrt(np.sum(variations, axis=0))
# print(variations.shape)
# print(variations)