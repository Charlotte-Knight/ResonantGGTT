import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (10,8)

import numpy as np
import sys
import common
import json
import os

with open("Outputs/Graviton/CatOptim_DataFit_New/bootstrap_3/optim_results.json", "r") as f:
  optim_results = json.load(f)

sig_procs = [each["sig_proc"] for each in optim_results]

nsigs = {proc:[] for proc in sig_procs}
nsig_tot = {proc:[] for proc in sig_procs}
bounds = {proc:[] for proc in sig_procs}

for i in range(1, 101):
  if os.path.exists("Outputs/Graviton/CatOptim_DataFit_New/bootstrap_%d/optim_results.json"%i):
    with open("Outputs/Graviton/CatOptim_DataFit_New/bootstrap_%d/optim_results.json"%i, "r") as f:
      optim_results = json.load(f)

    for each in optim_results:
      proc = each["sig_proc"]

      nsig_tot[proc].append(sum(each["nsigs"]))
      nsigs[proc].append(each["nsigs"][::-1][:4])
      bounds[proc].append(each["category_boundaries"][::-1][:4])
  else:
    print("Fail")

mx = []
rel_diff = []
frac_tot = []
for proc in sig_procs:
  print(proc)
  bounds_array = np.array(bounds[proc])

  print(np.mean(bounds_array, axis=0))
  print(np.std(bounds_array, axis=0))

  nsigs_array = np.array(nsigs[proc])

  print(np.mean(nsigs_array, axis=0))
  print(np.std(nsigs_array, axis=0)/np.mean(nsigs_array, axis=0))
  print((np.max(nsigs_array, axis=0)-np.min(nsigs_array, axis=0)) / np.mean(nsigs_array, axis=0))
  print()

  mx.append(common.get_MX_MY(proc)[0])
  rel_diff.append(np.std(nsigs_array, axis=0)/np.mean(nsigs_array, axis=0))

  nsig_tot_array = np.array(nsig_tot[proc])
  frac_tot.append(np.mean(nsigs_array, axis=0) / np.mean(nsig_tot_array))

mx = np.array(mx)
rel_diff = np.array(rel_diff)
frac_tot = np.array(frac_tot)

s = np.argsort(mx)
mx = mx[s]
rel_diff = rel_diff[s]
frac_tot = frac_tot[s]

for i in range(4):
  plt.scatter(mx, rel_diff[:,i])
  plt.xlabel(r"$m_X$")
  plt.ylabel(r"$\sigma(\epsilon) / \bar{\epsilon}$")
  plt.title("Category %d"%i)

  ax = plt.gca()
  ax2=ax.twinx()
  ax2.plot(mx, 1-frac_tot[:,i])
  ax2.set_ylabel("1 - fraction of total signal in category")

  plt.savefig("Outputs/Graviton/CatOptim_DataFit_New/sig_eff_variation_cat%d.png"%i)
  plt.clf()
