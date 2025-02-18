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

def getLimits(results_path):
  with open(results_path, "r") as f:
    results = f.readlines()

  masses = []
  for line in results:
    m = line.split(".txt")[0].split("_")[-1]
    mx = float(m.split("mx")[1].split("my")[0])
    my = float(m.split("my")[1].split("mh")[0])
    if "mh" in m: mh = float(m.split("mh")[1])
    else:         mh = 125.0
    if [mx, my, mh] not in masses:
      masses.append([mx, my, mh])

  limits = np.zeros((7, len(masses)))
  limits_no_sys = np.zeros((7, len(masses)))
  limits_no_res_bkg = np.zeros((7, len(masses)))
  limits_no_dy_bkg = np.zeros((7, len(masses)))

  for line in results:
    m = line.split(".txt")[0].split("_")[-1]
    mx = float(m.split("mx")[1].split("my")[0])
    my = float(m.split("my")[1].split("mh")[0])
    if "mh" in m: mh = float(m.split("mh")[1])
    else:         mh = 125.0
    idx1 = masses.index([mx, my, mh])

    if "2.5%" in line:
      idx2=0
    elif "16.0%" in line:
      idx2=1
    elif "50.0%" in line:
      idx2=2
    elif "84.0%" in line:
      idx2=3
    elif "97.5%" in line:
      idx2=4
    elif "Observed" in line:
      idx2=5
    elif "Significance" in line:
      idx2=6

    if idx2 < 6:
      limit = float(line.split("r < ")[1])
    else:
      limit = float(line.split("Significance:")[1])

    if "no_sys" in line:
      limits_no_sys[idx2][idx1] = limit
    elif "no_res_bkg" in line:
      limits_no_res_bkg[idx2][idx1] = limit
    elif "no_dy_bkg" in line:
      limits_no_dy_bkg[idx2][idx1] = limit
    elif "no_ggww" in line:
      pass
    else:
      limits[idx2][idx1] = limit
  
  # limits come as xs(pp -> X -> HH) * BR(H->gg) * 2
  # (I divided by BR(H->tautau) or (H->WW) in signal norm already)

  limits[:6] /= BR_H_GG * 2
  limits_no_sys[:6] /= BR_H_GG * 2
  limits_no_res_bkg[:6] /= BR_H_GG * 2
  limits_no_dy_bkg[:6] /= BR_H_GG * 2

  masses = np.array(masses)
  mx = masses[:,0]
  masses = masses[np.argsort(mx)]
  limits = limits[:,np.argsort(mx)]
  limits_no_sys = limits_no_sys[:,np.argsort(mx)]
  limits_no_res_bkg = limits_no_res_bkg[:,np.argsort(mx)]
  limits_no_dy_bkg = limits_no_dy_bkg[:,np.argsort(mx)]

  return masses, limits, limits_no_sys, limits_no_res_bkg, limits_no_dy_bkg

def parseTheoryPredictions():
  predictions = ["plotting/theory_predictions/BulkGraviton_NLO_13TeV_kmpl1.dat"]
  predictions = ["plotting/theory_predictions/BulkRadion_13TeV_NLO_LR_2TeV.dat", "plotting/theory_predictions/BulkRadion_13TeV_NLO_LR_3TeV.dat"]

  dfs = [pd.read_csv(prediction, sep="\t", names=["MX", "XS"]) for prediction in predictions] 
  splines = [spi.CubicSpline(df.MX, df.XS) for df in dfs]
  return splines

def plotLimits(mX, limits, ylabel, nominal_masses, savename, xlabel=r"$m_X$ [GeV]"):
  obs, = plt.plot(mX, limits[5], 'o-k',  lw=2, zorder=4, label="Observed")
  exp, = plt.plot(mX, limits[2], '--k', lw=2, zorder=3, label="Median expected")
  pm1 = plt.fill_between(mX, limits[1], limits[3], zorder=2, facecolor=(0, 0.8, 0), label="68% expected")
  pm2 = plt.fill_between(mX, limits[0], limits[4], zorder=1, facecolor=(1, 0.8, 0), label="95% expected")
  
  #legend1 = plt.gca().legend(handles=[obs, exp, pm1, pm2], loc=(0.6,0.55), title="95% CL upper limits")
  legend1 = plt.gca().legend(handles=[obs, exp, pm1, pm2], loc="upper right", title="95% CL upper limits")
  plt.gca().add_artist(legend1)

  splines = parseTheoryPredictions()
  x = np.linspace(260, 1000, 100)
  
  #theory, = plt.plot(x, spline(x), 'r-', lw=2, label="Bulk KK Graviton\n($\kappa/\overline{M_{PL}}=1$)")
  #plt.gca().legend(handles=[theory], loc=(0.275, 0.8))

  theory2, = plt.plot(x, splines[0](x), 'r-', lw=2, label="Bulk Radion\n($\Lambda_R=2$ TeV)")
  theory3, = plt.plot(x, splines[1](x), 'b-', lw=2, label="Bulk Radion\n($\Lambda_R=3$ TeV)")
  #plt.gca().legend(handles=[theory2, theory3], loc=(0.3, 0.8))
  plt.gca().legend(handles=[theory2, theory3], loc=(0.3, 0.7))
  #plt.gca().legend(handles=[theory], loc=(0.625, 0.85))

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

def plotSignificance(mX, limits, ylabel, nominal_masses, savename=None, xlabel=r"$m_X$ [GeV]"):
  plt.scatter(mX, limits[6], color='k', lw=3, zorder=4)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
    
  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)

  max_idx = np.argmax(limits[6])
  max_mass = mX[max_idx]
  max_sig = limits[6][max_idx]
  print(max_mass, max_sig)
  plt.text(0.5, 0.9, r"Max. Sig. of %.2f at $m_X = %d$"%(max_sig, max_mass), transform=plt.gca().transAxes, fontsize=24, color="black")

  if savename!=None:
    plt.savefig(savename+".png")
    plt.savefig(savename+".pdf")
    plt.clf()

def plotSystematicComparison(mx, limits, limits_no_sys, ylabel, nominal_masses, savename, xlabel=r"$m_X$ [GeV]" ):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  ratio = limits[2]/limits_no_sys[2]
  
  axs[0].plot(mx, limits[2], zorder=3, label="Median expected")
  axs[0].plot(mx, limits_no_sys[2], zorder=3, label="Median expected (no sys)")

  axs[0].set_ylabel(ylabel)
  axs[0].legend()
  axs[0].set_yscale("log")
  
  axs[1].plot(mx, ratio)
  axs[1].scatter(mx[np.isin(mx, nominal_masses)], ratio[np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  axs[1].legend()
  axs[1].set_ylabel("Ratio")
  axs[1].set_xlabel(xlabel)

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")
  plt.clf()
  plt.close(f)

def tabulateLimits(masses, limits, path):
  df = pd.DataFrame({"MX": masses[:,0], "MY": masses[:,1], "Median expected [fb]": limits[2], "Observed [fb]": limits[5]})
  df.sort_values(["MX", "MY"], inplace=True)

  table = tabulate.tabulate(df, headers='keys', floatfmt=".4f")
  
  with open(os.path.join(path, "param_test_results.txt"), "w") as f:
    f.write(table)
  with open(os.path.join(path, "param_test_results.tex"), "w") as f:
    f.write(df.to_latex(float_format="%.4f"))
  df.to_csv(os.path.join(path, "param_test_results.csv"), float_format="%.4f")

def tabulateLimitsAll(masses, limits, limits_no_sys, limits_no_res_bkg, path):
  #df = pd.DataFrame({"MX": masses[:,0], "MY": masses[:,1], "Median expected [fb]": limits[2], "No Systematics": limits_no_sys[2], "No Single SM Higgs": limits_no_res_bkg[2]})
  df = pd.DataFrame({"MX": masses[:,0], "MY": masses[:,1], "Median expected [fb]": limits[2], "No Systematics": limits_no_sys[2], "No Single SM Higgs": limits_no_res_bkg[2], "Observed": limits_no_res_bkg[5]})
  df.sort_values(["MX", "MY"], inplace=True)

  table = tabulate.tabulate(df, headers='keys', floatfmt=".4f")
  
  with open(os.path.join(path, "limits.txt"), "w") as f:
    f.write(table)
  with open(os.path.join(path, "limits.tex"), "w") as f:
    f.write(df.to_latex(float_format="%.4f", index=False))
  df.to_csv(os.path.join(path, "limits.csv"), float_format="%.4f")

def outputLimits(masses, limits, outdir):
  limits = limits.copy()
  limits /= 1000
  df = pd.DataFrame({"mass": masses[:,0]/1000, "exp": limits[2], "obs": limits[5],
                     "limits_m2": limits[0], "limits_m1": limits[1],
                     "limits_p1": limits[3], "limits_p2": limits[4]})
  df.to_csv(os.path.join(outdir, "limits.csv"), float_format="%.4f", index=False)

masses, limits, limits_no_sys, limits_no_res_bkg, limits_no_dy_bkg = getLimits(sys.argv[1])

print(max(limits[6]))

print(min(limits[2]))
print(max(limits[2]))

print(min(limits[5]))
print(max(limits[5]))

os.makedirs(os.path.join(sys.argv[2], "Limits_xs"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_sys"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_res_bkg"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_dy_bkg"), exist_ok=True)

os.makedirs(os.path.join(sys.argv[2], "Limits_systematics_comparison"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_res_bkg_comparison"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_dy_bkg_comparison"), exist_ok=True)

os.makedirs(os.path.join(sys.argv[2], "Significance"), exist_ok=True)

outputLimits(masses, limits, os.path.join(sys.argv[2], "Limits_xs"))
exit()
# tabulateLimits(masses, limits, os.path.join(sys.argv[2], "Limits_xs"))
# tabulateLimits(masses, limits_no_sys, os.path.join(sys.argv[2], "Limits_xs_no_sys"))
# tabulateLimits(masses, limits_no_res_bkg, os.path.join(sys.argv[2], "Limits_xs_no_res_bkg"))

mx = masses[:,0]
nominal_masses = [260,270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900,1000]
  
plotSignificance(mx, limits, "Significance", nominal_masses, os.path.join(sys.argv[2], "Significance", "significance"))

ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow HH)$ [fb]"
plotLimits(mx, limits, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs", "limits"))
plotLimits(mx, limits_no_sys, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs_no_sys", "limits_no_sys"))
plotLimits(mx, limits_no_res_bkg, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs_no_res_bkg", "limits_no_res_bkg"))

plotSystematicComparison(mx, limits, limits_no_sys, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_systematics_comparison", "125_2"))
plotSystematicComparison(mx, limits, limits_no_res_bkg, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_res_bkg_comparison", "125_2"))