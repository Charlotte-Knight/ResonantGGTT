import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")

import numpy as np
import sys
import json
import scipy.interpolate as spi
import os
import common

model_dir = sys.argv[1]

with open(os.path.join(model_dir, "model.json"), "r") as f:
  model = json.load(f)

parameter_names = [r"$\epsilon$", r"$\Delta m_{\gamma\gamma}$", r"$\sigma$", r"$\beta_l$", r"$n_l$", r"$\beta_r$", r"$n_r$"]

masses = []
norm = []
norm_sys = []
params = []
params_err = []

for mass, mass_model in model["2018"]["0"].items():
  if mass_model["this mass"]["closest_mass"] != mass:
    continue
  mx = int(mass.split("_")[0])
  my = int(mass.split("_")[1])
  masses.append([mx, my])
  norm.append(mass_model["this mass"]["norm"])

masses = np.array(masses)
norm = np.array(norm)

bin_edges = []
mx = np.sort(np.unique(masses[:,0]))
my = np.sort(np.unique(masses[:,1]))
mx_edges = np.array([mx[0] - (mx[1]-mx[0])/2] + list(mx[:-1] + (mx[1:] - mx[:-1])/2) + [mx[-1] + (mx[-1]-mx[-2])/2])
my_edges = np.array([my[0] - (my[1]-my[0])/2] + list(my[:-1] + (my[1:] - my[:-1])/2) + [my[-1] + (my[-1]-my[-2])/2])
print(mx_edges)
print(my_edges)

mx_edge_centers = (mx_edges[:-1]+mx_edges[1:])/2
my_edge_centers = (my_edges[:-1]+my_edges[1:])/2
print(mx_edge_centers)
print(my_edge_centers)
interp_masses = []
for mxi in mx_edge_centers:
  for myi in my_edge_centers:
    interp_masses.append([mxi, myi])
interp_masses = np.array(interp_masses)
interp_norm = spi.griddata(masses[:,:2], norm, interp_masses, method="linear", fill_value=0)
plt.hist2d(interp_masses[:,0], interp_masses[:,1], [mx_edges, my_edges], weights=interp_norm, cmap="afmhot_r")

cbar = plt.colorbar()
cbar.set_label(r"$\epsilon$")
plt.xlabel(r"$m_X$ [GeV]")
plt.ylabel(r"$m_Y$ [GeV]")

mplhep.cms.label(llabel="Simulation", loc=0)#, lumi="%.1f"%common.lumi_table[2018])
plt.savefig(os.path.join(model_dir, "signal_norm_vs_mx_my_paper.pdf"))
plt.savefig(os.path.join(model_dir, "signal_norm_vs_mx_my_paper.png"))

mplhep.cms.label(llabel="Simulation Preliminary", loc=0)#, lumi="%.1f"%common.lumi_table[2018])
plt.savefig(os.path.join(model_dir, "signal_norm_vs_mx_my_preliminary.pdf"))
plt.savefig(os.path.join(model_dir, "signal_norm_vs_mx_my_preliminary.png"))