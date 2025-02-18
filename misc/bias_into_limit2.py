import json
import optimisation.limit
import numpy as np
import common
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



basedir = "Outputs/NMSSM_Y_tautau/LimitVsMinNum"
biases = {10: 0.4, 16: 0.35, 20: 0.25}
all_infos = {}

for N in [10, 16, 20]:
  with open(f"{basedir}/{N}/CatOptim/optim_results.json", "r") as f:
    optim_results = json.load(f)
  with open(f"{basedir}/{N}/CatOptim/N_in_sidebands.json", "r") as f:
    N_sidebands = json.load(f)[::-1]
    #N_sidebands = json.load(f)[::-1][-2:]

  bias = biases[N]
  info = []

  for res in optim_results:
    mx, my = common.get_MX_MY(res["sig_proc"])

    s = np.array(res["nsigs"])[1:]
    b = np.array(res["nbkgs"])[1:]
    #s = np.array(res["nsigs"])[-2:]
    #b = np.array(res["nbkgs"])[-2:]
    assert len(N_sidebands) == len(s), print(len(N_sidebands), len(s))
    b_err = 1/np.sqrt(N_sidebands) * b

    nom = optimisation.limit.calculateObservedLimit(s, b, b)
    bp = optimisation.limit.calculateObservedLimit(s, b, b+bias*b_err, rhigh=1000)
    bm = optimisation.limit.calculateObservedLimit(s, b, np.clip(b-bias*b_err, 0.0001, a_max=None), rhigh=1000)
    
    assert optimisation.limit.calculateExpectedLimit_v2(s, b, CLb=0.5) == nom
    nom_p1_sigma = optimisation.limit.calculateExpectedLimit_v2(s, b, CLb=0.16)
    nom_m1_sigma = optimisation.limit.calculateExpectedLimit_v2(s, b, CLb=0.84)
    sigma_width = abs(nom_m1_sigma - nom_p1_sigma) / 2

    bp_rel = (bp - nom) / nom
    bm_rel = (bm - nom) / nom

    bp_pull = (bp - nom) / sigma_width
    bm_pull = (bm - nom) / sigma_width

    info_dict = {"MX": mx, "MY": my, "Nominal Limit (50%)": nom, 
                 "Nominal Limit (16%)": nom_p1_sigma, "Nominal Limit (84%)": nom_m1_sigma,
                  "k += bias*b_err": bp, "k += relative": bp_rel,  "k += pull": bp_pull,
                  "k -= bias*b_err": bm, "k -= relative": bm_rel,  "k -= pull": bm_pull,
                  "b cat0": b[-1], "b_err cat0": b_err[-1]}
    info.append(info_dict)

  df = pd.DataFrame(info)
  print(df)
  print(df.to_string())
  

  all_rel = abs(np.concatenate([df["k += relative"], df["k -= relative"]]))
  plt.hist(all_rel, range=(0, 0.1), bins=10)
  plt.title("All relative changes (k += bias*b_err and k -= bias*berr)")
  plt.xlabel(r"$\Delta$ limit / limit")
  plt.savefig(f"{basedir}/{N}/all_relative.png")
  plt.clf()

  all_pulls = abs(np.concatenate([df["k += pull"], df["k -= pull"]]))
  plt.hist(all_pulls, range=(0, 0.3), bins=15)
  plt.title("All pulls (k += bias*b_err and k -= bias*berr)")
  plt.xlabel(r"Pull = 2 $\times \Delta$ limit / (84% limit - 16% limit)")
  plt.savefig(f"{basedir}/{N}/all_pulls.png")
  plt.clf()

  all_nom = np.concatenate([df["Nominal Limit (50%)"], df["Nominal Limit (50%)"]])
  plt.scatter(all_nom, all_pulls)
  plt.xlabel("Nominal Limit (50%)")
  plt.ylabel("Pull")
  plt.savefig(f"{basedir}/{N}/pull_vs_nom_limit.png")
  plt.clf()

  all_infos[N] = df

for N in [10, 16, 20]:
  df = all_infos[N]
  all_pulls = abs(np.concatenate([df["k += pull"], df["k -= pull"]]))
  plt.hist(all_pulls, range=(0, 0.3), bins=15, alpha=0.5, label=r"$N=%d$"%N)
  
plt.title("All pulls (k += bias*b_err and k -= bias*berr)")
plt.xlabel(r"Pull = 2 $\times \Delta$ limit / (84% limit - 16% limit)")
plt.legend()
plt.savefig(f"{basedir}/all_pulls.png")
plt.clf()