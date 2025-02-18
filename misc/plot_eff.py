import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import numpy as np
import sys
import common
import json
import os

input_dir = sys.argv[1]
parquet_path = os.path.join(input_dir, "merged_nominal.parquet")
summary_path = os.path.join(input_dir, "summary.json")
outdir = os.path.join(input_dir, "preselection_eff")
os.makedirs(outdir, exist_ok=True)

df = pd.read_parquet(parquet_path, columns=["process_id", "weight_central", "year", "weight_central_no_lumi", "weight_tau_idDeepTauVSjet_sf_AnalysisTau_central", "weight_tau_idDeepTauVSjet_sf_AnalysisTau_up", "weight_tau_idDeepTauVSjet_sf_AnalysisTau_down", "category"])

print(df.columns)
with open(summary_path, "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

# sig_ids = [proc_dict[proc] for proc in common.sig_procs["Graviton"]]
# mx = np.array([common.get_MX_MY(proc)[0] for proc in common.sig_procs["Graviton"]])

sig_ids = [proc_dict[proc] for proc in common.sig_procs["Radion"]]
mx = np.array([common.get_MX_MY(proc)[0] for proc in common.sig_procs["Radion"]])

df["weight_up"] = (df["weight_central"] / df["weight_tau_idDeepTauVSjet_sf_AnalysisTau_central"]) * df["weight_tau_idDeepTauVSjet_sf_AnalysisTau_up"]
df["weight_down"] = (df["weight_central"] / df["weight_tau_idDeepTauVSjet_sf_AnalysisTau_central"]) * df["weight_tau_idDeepTauVSjet_sf_AnalysisTau_down"]

for i, year in enumerate([2016, 2017, 2018]):
  eff = [df.loc[(df.process_id==i)&(df.year==year), "weight_central"].sum() / common.lumi_table[year] for i in sig_ids]
  eff_err = [np.sqrt((df.loc[(df.process_id==i)&(df.year==year), "weight_central"]**2).sum()) / common.lumi_table[year] for i in sig_ids]

  # eff_up = np.array([df.loc[(df.process_id==i)&(df.year==year), "weight_up"].sum() / common.lumi_table[year] for i in sig_ids])
  # eff_down = np.array([df.loc[(df.process_id==i)&(df.year==year), "weight_down"].sum() / common.lumi_table[year] for i in sig_ids])
  # eff_err = abs(eff_up-eff_down) / 2

  plt.errorbar(mx+i*0, eff, eff_err, fmt='--o', label=str(year), capsize=5)

tot_lumi = common.lumi_table[2016] + common.lumi_table[2017] + common.lumi_table[2018]
inc_eff = [df.loc[df.process_id==i, "weight_central"].sum() / tot_lumi for i in sig_ids]
inc_eff_err = [np.sqrt((df.loc[df.process_id==i, "weight_central"]**2).sum()) / tot_lumi for i in sig_ids]

tau_eff_up = np.array([df.loc[df.process_id==i, "weight_up"].sum() / tot_lumi for i in sig_ids])
tau_eff_down = np.array([df.loc[df.process_id==i, "weight_down"].sum() / tot_lumi for i in sig_ids])
plt.fill_between(mx, tau_eff_down, tau_eff_up, color="k", alpha=0.15, zorder=-1, label="DeepTauVsJet\nsf uncertainty")

plt.errorbar(mx, inc_eff, inc_eff_err, fmt='k-o', label="Combined")


plt.xlabel(r"$m_X$ [GeV]")
plt.ylabel("Preselection Signal efficiency")
plt.legend()
plt.title(r"(Spin-0) $X \rightarrow HH \rightarrow \gamma\gamma\tau\tau$")
plt.savefig(os.path.join(outdir, "sig_eff.png"))
plt.savefig(os.path.join(outdir, "sig_eff.pdf"))
plt.clf()

# lumi = sum([common.lumi_table[year] for year in common.lumi_table.keys()])
# eff = [df[df.process_id==i].weight_central.sum()/lumi for i in sig_ids]

# N = [sum(df.process_id==i) for i in sig_ids]
# print(N)
# eff_err = eff / np.sqrt(N)

# #plt.scatter(mx, eff, marker='.')
# plt.errorbar(mx, eff, eff_err, fmt='o')
# plt.xlabel(r"$m_X$")
# plt.ylabel("Signal efficiency")
# plt.savefig("sig_eff.png")
# plt.savefig("sig_eff.pdf")
# plt.clf()

# mx=np.array(mx)
# eff=np.array(eff)
# eff_err=np.array(eff_err)
# plt.errorbar(mx[mx<450], eff[mx<450], eff_err[mx<450], fmt='.')
# plt.xlabel("MX")
# plt.ylabel("Signal efficiency")
# plt.savefig("sig_eff_zoom.png")
# plt.clf()

# plt.hist(df[df.process_id==proc_dict['XToHHggTauTau_M260']].weight_central, bins=100)
# plt.savefig("weight_hist.png")

category_map = {
  1: r"$\tau_h \mu$",
  2: r"$\tau_h e$",
  3: r"$\tau_h \tau_h$",
  4: r"$\mu\mu$",
  5: r"$ee$",
  6: r"$\mu e$",
  7: r"$\tau_h +$ Isotrack",
  8: r"$\tau_h$"
}

effs = []
names = []
for category in category_map:
  eff = [df.loc[(df.process_id==i)&(df.category==category), "weight_central"].sum() / tot_lumi for i in sig_ids]
  effs.append(eff)
  names.append(category_map[category])

effs = np.array(effs)
names = np.array(names)

order = np.argsort(effs[:,-1])[::-1]
effs = effs[order]
names = names[order]

effs = np.cumsum(effs, axis=0)

color = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00']
for i, name in enumerate(names):
  if i == 0:
    plt.fill_between(mx, np.zeros_like(effs[i]), effs[i], label=name, color=color[i])
  else:
    plt.fill_between(mx, effs[i-1], effs[i], label=name, color=color[i])

plt.errorbar(mx, inc_eff, inc_eff_err, fmt='k-o', label="Combined")

plt.xlabel(r"$m_X$ [GeV]")
plt.ylabel("Preselection Signal efficiency")
plt.ylim(top=0.6)
plt.legend(loc="upper left")
plt.title(r"(Spin-0) $X \rightarrow HH \rightarrow \gamma\gamma\tau\tau$")
plt.savefig(os.path.join(outdir, "sig_eff_stack.png"))
plt.savefig(os.path.join(outdir, "sig_eff_stack.pdf"))