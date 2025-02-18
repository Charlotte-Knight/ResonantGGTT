import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import numpy as np
import sys
import tabulate
import pandas as pd
import os
import json
import scipy.interpolate as spi

import common

BR_H_GG = 2.27e-3
BR_H_TT = 6.27e-2
BR_H_BB = 5.84e-1

BR_HH_GGTT = 2 * BR_H_GG * BR_H_TT
BR_HH_GGBB = 2 * BR_H_GG * BR_H_BB

def getAnalysis():
  if "graviton" in sys.argv[1].lower():
    return "Graviton"
  elif "radion" in sys.argv[1].lower():
    return "Radion"
  else:
    raise ValueError("Analysis not recognized")

def getLimits(limits_path, significances_path):
  df_limits = pd.read_csv(limits_path)
  df_significances = pd.read_csv(significances_path)

  limits = df_limits[["exp-2", "exp-1", "exp0", "exp+1", "exp+2", "obs"]]
  significances = df_significances["obs"]
  limits = np.array(pd.concat([limits, significances], axis=1)).T

  limitsErr = df_limits[["exp-2_err", "exp-1_err", "exp0_err", "exp+1_err", "exp+2_err", "obs_err"]]
  significancesErr = df_significances["obs_err"]
  limitsErr = np.array(pd.concat([limitsErr, significancesErr], axis=1)).T

  masses = np.array(df_limits[["mx", "my", "mh"]])

  limits[:6] /= BR_H_GG * 2
  limitsErr[:6] /= BR_H_GG * 2

  mx = masses[:,0]
  masses = masses[np.argsort(mx)]
  limits = limits[:,np.argsort(mx)]
  limitsErr = limitsErr[:,np.argsort(mx)]

  return masses, limits, limitsErr

def parseTheoryPredictions(graviton):
  if graviton:
    predictions = ["plotting/theory_predictions/BulkGraviton_NLO_13TeV_kmpl1.dat"]
  else:
    predictions = ["plotting/theory_predictions/BulkRadion_13TeV_NLO_LR_2TeV.dat", "plotting/theory_predictions/BulkRadion_13TeV_NLO_LR_3TeV.dat"]

  dfs = [pd.read_csv(prediction, sep="\t", names=["MX", "XS"]) for prediction in predictions] 
  splines = [spi.CubicSpline(df.MX, df.XS) for df in dfs]
  return splines

def plotLimits(mX, limits, ylabel, nominal_masses, savename, xlabel=r"$m_X$ [GeV]"):
  obs, = plt.plot(mX, limits[5], 'o-k',  lw=2, zorder=4, label="Observed")
  exp, = plt.plot(mX, limits[2], '--k', lw=2, zorder=3, label="Median expected")
  pm1 = plt.fill_between(mX, limits[1], limits[3], zorder=2, facecolor=(0, 0.8, 0), label="68% expected")
  pm2 = plt.fill_between(mX, limits[0], limits[4], zorder=1, facecolor=(1, 0.8, 0), label="95% expected")
  
  legend1 = plt.gca().legend(handles=[obs, exp, pm1, pm2], loc="upper right", title="95% CL upper limits")
  plt.gca().add_artist(legend1)

  x = np.linspace(260, 1000, 100)
  
  if "graviton" in savename.lower():
    splines = parseTheoryPredictions(graviton=True)
    theory, = plt.plot(x, splines[0](x), 'r-', lw=2, label="Bulk KK Graviton\n($\kappa/\overline{M_{PL}}=1$)")
    plt.gca().legend(handles=[theory], loc=(0.275, 0.8))
  else:
    splines = parseTheoryPredictions(graviton=False)
    theory2, = plt.plot(x, splines[0](x), 'r-', lw=2, label="Bulk Radion\n($\Lambda_R=2$ TeV)")
    theory3, = plt.plot(x, splines[1](x), 'b-', lw=2, label="Bulk Radion\n($\Lambda_R=3$ TeV)")
    plt.gca().legend(handles=[theory2, theory3], loc=(0.3, 0.7))

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.ylim(80, 2e4)
  plt.yscale("log")
  
  mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=2)
  plt.savefig(savename+"_paper.png")
  plt.savefig(savename+"_paper.pdf")
  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=2)
  plt.savefig(savename+"_preliminary.png")
  plt.savefig(savename+"_preliminary.pdf")
  plt.clf()

def tabulateLimits(masses, limits, path):
  df = pd.DataFrame({"MX": masses[:,0], "MY": masses[:,1], "Median expected [fb]": limits[2], "Observed [fb]": limits[5]})
  df.sort_values(["MX", "MY"], inplace=True)

  table = tabulate.tabulate(df, headers='keys', floatfmt=".4f")
  
  with open(os.path.join(path, "param_test_results.txt"), "w") as f:
    f.write(table)
  with open(os.path.join(path, "param_test_results.tex"), "w") as f:
    f.write(df.to_latex(float_format="%.4f"))
  df.to_csv(os.path.join(path, "param_test_results.csv"), float_format="%.4f")

def outputLimits(masses, limits, limitsErr, outdir):
  limits = limits.copy()
  limits /= 1000
  limitsErr = limitsErr.copy()
  limitsErr /= 1000
  df = pd.DataFrame({"mass": masses[:,0]/1000, "exp": limits[2], "obs": limits[5], "obsErr": limitsErr[5],
                     "limits_m2": limits[0], "limits_m1": limits[1],
                     "limits_p1": limits[3], "limits_p2": limits[4]})
  df.to_csv(os.path.join(outdir, "limits.csv"), float_format="%.4f", index=False)

masses, limits, limitsErr = getLimits(sys.argv[1], sys.argv[2])

os.makedirs(os.path.join(sys.argv[3], "Limits_xs"), exist_ok=True)

outputLimits(masses, limits, limitsErr, os.path.join(sys.argv[3], "Limits_xs"))
tabulateLimits(masses, limits, os.path.join(sys.argv[3], "Limits_xs"))

mx = masses[:,0]
nominal_masses = [260,270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900,1000]
  
ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow HH)$ [fb]"
plotLimits(mx, limits, ylabel, nominal_masses, os.path.join(sys.argv[3], "Limits_xs", f"limits_{getAnalysis().lower()}"))