import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
import sys
import json

from signalModelling.systematics import deriveParquetShapeSystematics, getEffSigma
import signalModelling.systematics as systematics
systematics.Y_gg = True
import common
import numpy as np

inputs_dir = sys.argv[1]
syst = sys.argv[2]
masses = sys.argv[3:]

with open(f"{inputs_dir}/summary.json", "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

example_sig_proc = "XToHHggTauTau_M1000"
#example_sig_proc = "NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_100"

#variables = ["LeadPhoton_pt", "SubleadPhoton_pt", "Diphoton_pt", "Diphoton_mass"]
variables = ["Diphoton_mass"]
columns = ["weight_central", "process_id", "event", "year"] + variables

nominal_f = pd.read_parquet(f"{inputs_dir}/merged_nominal.parquet", columns=columns)
up_f = pd.read_parquet(f"{inputs_dir}/merged_{syst}_up.parquet", columns=columns)
down_f = pd.read_parquet(f"{inputs_dir}/merged_{syst}_down.parquet", columns=columns)

get_2018 = lambda df: df[df.year==2018]
nominal_f = get_2018(nominal_f)
up_f = get_2018(up_f)
down_f = get_2018(down_f)


pd.set_option('mode.chained_assignment',None)

for m in masses:
  mx = int(m)
  my = 125
  print(f"\nMX={mx}, MY={my}")

  sig_proc = common.get_sig_proc(example_sig_proc, mx, my)

  nominal = nominal_f[nominal_f.process_id==proc_dict[sig_proc]]
  up = up_f[up_f.process_id==proc_dict[sig_proc]]
  down = down_f[down_f.process_id==proc_dict[sig_proc]]

  assert sum(np.unique(nominal.event, return_counts=True)[1] > 1) == 0
  assert sum(np.unique(up.event, return_counts=True)[1] > 1) == 0
  assert sum(np.unique(down.event, return_counts=True)[1] > 1) == 0
  ids = set(nominal.event).intersection(up.event).intersection(down.event)

  #print(len(nominal), len(up), len(down))
  nominal = nominal[nominal.event.isin(ids)].set_index("event")
  up = up[up.event.isin(ids)].set_index("event")
  down = down[down.event.isin(ids)].set_index("event")
  #print(len(nominal), len(up), len(down))

  # diff = (up.Diphoton_mass - nominal.Diphoton_mass).astype("float16")
  # print(diff)
  # print(np.unique(diff))
  # print(len(np.unique(diff)))
  diff = (down.Diphoton_mass / nominal.Diphoton_mass).astype("float16")
  print(diff)
  print(np.unique(diff, return_counts=True))
  print(len(np.unique(diff)))

  #up.loc[:, "Diphoton_mass"] += nominal.Diphoton_mass.mean() - up.Diphoton_mass.mean()
  #down.loc[:, "Diphoton_mass"] += nominal.Diphoton_mass.mean() - down.Diphoton_mass.mean()
  #up.loc[:, "Diphoton_mass"] *= nominal.Diphoton_mass.mean() / up.Diphoton_mass.mean()
  #down.loc[:, "Diphoton_mass"] *= nominal.Diphoton_mass.mean() / down.Diphoton_mass.mean()

  dfs = {}
  dfs["nominal"] = nominal
  dfs[f"{syst}_up"] = up
  dfs[f"{syst}_down"] = down

  for k, df in dfs.items():
    df.rename({"weight_central":"weight"}, axis=1, inplace=True)
    df.loc[:, "MX"] = mx
    df.loc[:, "MY"] = my

  #print(dfs)
  consts = deriveParquetShapeSystematics(dfs, syst, f"{mx}_{my}")
  print(json.dumps(consts, indent=2))

  print("EffSigma Up: %.3f"%getEffSigma(dfs[f"{syst}_up"].Diphoton_mass, dfs[f"{syst}_up"].weight, 125))
  print("EffSigma Down: %.3f"%getEffSigma(dfs[f"{syst}_down"].Diphoton_mass, dfs[f"{syst}_down"].weight, 125))
  print("Std Up: %.3f"%dfs[f"{syst}_up"].Diphoton_mass.std())
  print("Std Down: %.3f"%dfs[f"{syst}_down"].Diphoton_mass.std())


# nbins = 50
# for norm in [True, False]:
#   for var in vars:
#     f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

#     l = nominal[var].quantile(0.01)
#     h = nominal[var].quantile(0.99)
#     n_nominal, bins, patches = axs[0].hist(nominal[var], bins=nbins, range=(l,h), weights=nominal.weight_central, label="Central", histtype='step', density=norm)
#     n_up, bins, patches = axs[0].hist(up[var], bins=nbins, range=(l,h), weights=up.weight_central, label="Up", histtype='step', density=norm)
#     n_down, bins, patches = axs[0].hist(down[var], bins=nbins, range=(l,h), weights=down.weight_central, label="Down", histtype='step', density=norm)
#     axs[0].legend()

#     bin_centers = (bins[:-1]+bins[1:])/2
#     axs[1].hist(bin_centers, bins=bins, weights=1 - n_up/n_nominal, label="Up", histtype='step')
#     axs[1].hist(bin_centers, bins=bins, weights=1 - n_down/n_nominal, label="Down", histtype='step')
#     axs[1].legend()
#     axs[1].set_ylabel("1 - Ratio")
#     axs[1].set_xlabel(var)

#     print(n_up/n_nominal)
#     print(n_down/n_nominal)

#     if not norm: name = "plots/smear/%s.png"%var
#     else:        name = "plots/smear/%s_normed.png"%var
#     plt.savefig(name)
#     plt.clf()

import scipy.stats as sps
from scipy.integrate import quad
mean = 124.79
sigma = 1.24
bl = 1.37
ml = 5.65
br = 1.75
mr = 20.0

lint = quad(sps.crystalball(loc=mean, scale=sigma, beta=bl, m=ml).pdf, -np.inf, 124.86)[0]
rint = quad(sps.crystalball(loc=mean, scale=sigma, beta=br, m=mr).pdf, -np.inf, 124.86)[0]
mgg = np.concatenate([
  sps.crystalball(loc=mean, scale=sigma, beta=bl, m=ml).rvs(size=int(100000*lint)),
  -sps.crystalball(loc=-mean, scale=sigma, beta=br, m=mr).rvs(size=int(100000*lint))
])
#plt.hist(mgg, bins=100, range=(112.5, 137.5))
#plt.savefig("test_dcb_toys.png")

print(2*(mgg.mean()-(mgg*1.05).mean())/(mgg.mean()+(1.05*mgg).mean()))

nom_sigma = getEffSigma(pd.Series(mgg), pd.Series(np.ones_like(mgg)), 125)
up_sigma = getEffSigma(pd.Series(mgg*1.05), pd.Series(np.ones_like(mgg)), 125)
print(2*(nom_sigma-up_sigma)/(nom_sigma+up_sigma))