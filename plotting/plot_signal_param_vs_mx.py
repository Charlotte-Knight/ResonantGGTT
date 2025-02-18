import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")

import numpy as np
import sys
import json
import os
import common

model_dir = sys.argv[1]

with open(os.path.join(model_dir, "model.json"), "r") as f:
  model = json.load(f)

parameter_names = [r"$\epsilon$", r"$\Delta m_{\gamma\gamma}$ [GeV]", r"$\sigma$ [GeV]", r"$\beta_l$", r"$n_l$", r"$\beta_r$", r"$n_r$"]

mx = []
norm = []
norm_sys = []
params = []
params_err = []

for mass, mass_model in model["2018"]["0"].items():
  if mass_model["this mass"]["closest_mass"] != mass:
    continue
  mx.append(int(mass.split("_")[0]))
  print(mx[-1])
  norm.append(mass_model["this mass"]["norm"])
  norm_sys.append(mass_model["this mass"]["norm_systematic"])
  params.append(mass_model["this mass"]["parameters"])
  params_err.append(mass_model["this mass"]["parameters_err"])

mx = np.array(mx)
norm = np.array(norm)
norm_sys = np.array(norm_sys)
norm_err = (1-norm_sys) * norm
params = np.array(params)
params_err = np.array(params_err)

s = np.argsort(mx)
mx, norm, params, norm_err, params_err = mx[s], norm[s], params[s], norm_err[s], params_err[s]

# norm_err[:] = 0
# params_err[:,:] = 0

print(norm_err)
print(params_err)

t = lambda x: (x-x.mean()) / x.std()
te = lambda x, y: x / y.std()

t = lambda x: x
te = lambda x, y: x

# plt.errorbar(mx, t(norm), yerr=te(norm_err,norm), fmt='o', label=parameter_names[0])
# #for i in range(len(params[0])):
# for i in range(3):
#   plt.errorbar(mx, t(params[:,i]), yerr=te(params_err[:,i], params[:,i]), fmt='o', label=parameter_names[i+1])

#fig = plt.figure()
#gs = fig.add_gridspec(4, hspace=0)
#axs = gs.subplots(sharex=True, sharey=True)
#axs = gs.subplots(sharex=True, sharey=False)
fig, axs = plt.subplots(3, sharex=True)

plt.sca(axs[0])

axs[0].errorbar(mx, t(norm), yerr=te(norm_err,norm), fmt='o',label=parameter_names[0])
axs[0].set_ylabel(parameter_names[0])

for i in range(1, 3):
  axs[i].errorbar(mx, t(params[:,i]), yerr=te(params_err[:,i], params[:,i]), fmt='o', label=parameter_names[i])
  axs[i].set_ylabel(parameter_names[i])
axs[i].set_xlabel(r"$m_X$ [GeV]")

mplhep.cms.label(llabel="Simulation", loc=0)#, lumi="%.1f"%common.lumi_table[2018])
plt.savefig(os.path.join(model_dir, "signal_param_vs_mx_paper.pdf"))
plt.savefig(os.path.join(model_dir, "signal_param_vs_mx_paper.png"))

mplhep.cms.label(llabel="Simulation Preliminary", loc=0)#, lumi="%.1f"%common.lumi_table[2018])
plt.savefig(os.path.join(model_dir, "signal_param_vs_mx_preliminary.pdf"))
plt.savefig(os.path.join(model_dir, "signal_param_vs_mx_preliminary.png"))
