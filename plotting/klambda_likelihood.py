import matplotlib.patches as mpatches
import matplotlib.lines as lines
import matplotlib.collections as collections

import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (10,8)
import uproot
import sys
import common
import numpy as np
import yaml

import scipy.interpolate as spi
import scipy.optimize as spo

def getInterval(n2ll, threshold=1):
  l = spo.bisect(lambda x: n2ll(x)-threshold, -15, 0)
  r = spo.bisect(lambda x: n2ll(x)-threshold, 0, 20)
  return (l, r)

def getMinimum(n2ll):
  ress = [spo.minimize(n2ll, -15), spo.minimize(n2ll, 20)]
  funcs = [res.fun for res in ress]
  xs = [res.x for res in ress]
  print(funcs)
  return xs[np.argmin(funcs)]

def getKlStr(n2ll):
  minimum = getMinimum(n2ll)
  l, r = getInterval(n2ll)
  m, p = l - minimum, r - minimum

  s = "$\kappa_\lambda = %.1f^{+%.1f}_{%.1f}$"%(minimum, p, m)
  return s

# with open(sys.argv[1], "r") as f:
#   exp = yaml.safe_load(f)
# exp_kl = np.array([each["value"] for each in exp["independent_variables"][0]["values"]])
# exp_2nll = spi.CubicSpline(exp_kl, np.array([each["value"] for each in exp["dependent_variables"][0]["values"]]))

# with open(sys.argv[2], "r") as f:
#   obs = yaml.safe_load(f)
# obs_kl = np.array([each["value"] for each in obs["independent_variables"][0]["values"]])
# obs_2nll = spi.CubicSpline(obs_kl, np.array([each["value"] for each in obs["dependent_variables"][0]["values"]]))

import pickle
with open(sys.argv[1], "rb") as f:
  exp = pickle.load(f)
exp_kl = exp["data"][0]["values"]["kl"]
exp_2nll_points = exp["data"][0]["values"]["dnll2"]
exp_2nll_points -= min(exp_2nll_points)
exp_2nll = spi.CubicSpline(exp_kl, exp_2nll_points)

with open(sys.argv[2], "rb") as f:
  obs = pickle.load(f)
obs_kl = obs["data"][0]["values"]["kl"]
obs_2nll_points = obs["data"][0]["values"]["dnll2"]
obs_2nll_points -= min(obs_2nll_points)
obs_2nll = spi.CubicSpline(obs_kl, obs_2nll_points)

print("68%CL intervals")
print("exp")
print(getInterval(exp_2nll, threshold=0.9889))
print("obs")
print(getInterval(obs_2nll, threshold=0.9889))
print()
print("95%CL intervals")
print("exp")
print(getInterval(exp_2nll, threshold=3.8415))
print("obs")
print(getInterval(obs_2nll, threshold=3.8415))

x = np.linspace(-15, 20, 1000)

plt.axhline(0.9889, color="grey")
plt.axhline(3.8415, color="grey")

plt.xlim(-15, 20)
plt.ylim(0, 10)

plt.plot(x, obs_2nll(x), 'k', label=f"Observed: {getKlStr(obs_2nll)}")
plt.plot(x, exp_2nll(x), 'r--', label=f"Expected: {getKlStr(exp_2nll)}")

# plt.scatter(obs_kl, [obs_2nll(kli) for kli in obs_kl], c='k', marker="o")
# plt.scatter(exp_kl, [exp_2nll(kli) for kli in exp_kl], c='r', marker="o")
# plt.axvline(getMinimum(obs_2nll), color="k", linestyle="--")
# plt.axvline(getMinimum(exp_2nll), color="r", linestyle="--")

plt.xlabel(r"$\kappa_\lambda$")
plt.ylabel(r"$-2~\Delta~\ln~L$")

plt.legend(loc="upper right")

# plt.yscale("log")
# plt.ylim(1e-5, 1e3)

mlabel = mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=2)
plt.savefig("kl_likelihood_paper.pdf")
plt.savefig("kl_likelihood_paper.png")

mlabel[1].set_text("Preliminary")
plt.savefig("kl_likelihood_preliminary.pdf")
plt.savefig("kl_likelihood_preliminary.png")

mlabel[1].set_text("Supplementary")
plt.savefig("kl_likelihood_supplementary.pdf")
plt.savefig("kl_likelihood_supplementary.png")
