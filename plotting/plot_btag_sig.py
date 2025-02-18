import pandas as pd
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
import numpy as np

df = pd.read_parquet(sys.argv[1])
print("\n".join(df.columns))

print(df.year)

with open(sys.argv[2], "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

data = df[df.process_id==proc_dict["Data"]]
bkg_proc_ids = [proc_dict[bkg_proc] for bkg_proc in common.bkg_procs["all"]]
bkg = df[df.process_id.isin(bkg_proc_ids)]
sig_proc_ids = [proc_dict["HHggTauTau"] ]
sig = df[df.process_id.isin(sig_proc_ids)]


# plot btag distribution for sr0, sr1, and the rest
sr0 = sig[sig.pass_sr_0==1]
sr1 = sig[sig.pass_sr_1==1]
rest = sig[(sig.pass_sr_0==0)&(sig.pass_sr_1==0)]

plt.hist(sr0.b_jet_1_btagDeepFlavB, bins=20, range=(0, 1), weights=sr0.weight_central, label="SR0", density=True, histtype="step")
plt.hist(sr1.b_jet_1_btagDeepFlavB, bins=20, range=(0, 1), weights=sr1.weight_central, label="SR1", density=True, histtype="step")
plt.hist(rest.b_jet_1_btagDeepFlavB, bins=20, range=(0, 1), weights=rest.weight_central, label="Rest", density=True, histtype="step")
plt.xlabel("b_jet_1_btagDeepFlavB")
plt.legend()
plt.savefig("plotting/plot_btag_sig.pdf")
plt.clf()

# plot bdt score in quantiles of bdt score

btag_quantiles = list(sig[sig.b_jet_1_btagDeepFlavB >= 0].b_jet_1_btagDeepFlavB.quantile(q=[0, 0.25, 0.5, 0.75, 1.0]))
print(btag_quantiles)
for i in range(4):
  sig_s = sig[(sig.b_jet_1_btagDeepFlavB>btag_quantiles[i]) & (sig.b_jet_1_btagDeepFlavB < btag_quantiles[i+1])]
  plt.hist(sig_s.bdt_score, bins=20, range=(0.97, 1), weights=sig_s.weight_central, label=f"Quantile {i}", density=True, histtype="step")
plt.legend(loc="upper left")
plt.xlabel("BDT score")
for cut in [0.9891, 0.973610]:
  plt.axvline(cut, linestyle="--", zorder=9, linewidth=1)
plt.savefig("plotting/plot_btag_sig_quantiles.pdf")


