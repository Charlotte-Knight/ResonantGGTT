"""
Objective:
Calculate yield of DY (and eventually VV also) in the signal regions
using an ABCD method. Now we are using a fit to extract only the peaking
component. 

Inputs:
1. parquet files with DY events and their pNN scores
  i. Veto applied
  ii. Veto inverted

Outputs:
1. Check whether the electron + pixel veto efficiency is correlated with
    pNN score.
2. Assuming negligible correlation, use ABCD method to estimate DY peaking

A B
C D

x-axis = pnn score\
y-axis = electron + pixel veto

A = least-sensitive (background-rich) signal region + no vetoes
B = chosen signal region + no vetoes
C = least-sensitive (background-rich) signal region + vetoes enabled
D = signal region

The "NoEveto" means that this script runs on a single dataframe that has
an ElectronVeto column rather than two dataframes which represent applied
veto and inverted veto.
"""

import pandas as pd
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
#plt.rcParams["figure.figsize"] = (12.5,5)
plt.rcParams["figure.figsize"] = (12.5,10)

import numpy as np
import scipy.stats as scs
from scipy.optimize import curve_fit
import os
import common
import json

def assignSignalRegions(df, optim_results):
  df["SR"] = -1
  
  boundaries = optim_results["category_boundaries"][::-1] #so that cat0 is most pure
  for i in range(len(boundaries)-1):
    selection = (df[optim_results["score"]] <= boundaries[i]) & (df[optim_results["score"]] >= boundaries[i+1])
    df.loc[selection, "SR"] = i

  assert sum(df.SR==-1) == 0
  return df

def makePlot(df, savepath, title=None):
  l, h, n = 70, 120, 25
  #l, h, n = 0, 200, 100
  bin_spacing = (h-l)/n

  #plt.hist(df.Diphoton_mass, range=(70, 120), bins=20, weights=df.weight)
  y, bin_edges = np.histogram(df.Diphoton_mass, range=(l, h), bins=n, weights=df.weight)
  y_err_2, bin_edges = np.histogram(df.Diphoton_mass, range=(l, h), bins=n, weights=df.weight**2)
  y_err = np.sqrt(y_err_2)

  x = (bin_edges[:-1]+bin_edges[1:])/2
  plt.hist(x, bin_edges, weights=y, histtype="step", color="tab:blue")
  plt.errorbar(x, y, y_err, fmt='_', capsize=2, color="tab:blue")
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.ylabel("Events / %d GeV"%(bin_spacing))
  plt.title(title)
  plt.savefig(savepath)
  plt.clf()

def intExp(a, l, h):
  return -(100/a)*(np.exp(-a*h/100)-np.exp(-a*l/100))

def fitFunction(df, savepath, title=None, plot=True):
  l, h, n = 70, 120, 25

  y, bin_edges = np.histogram(df.Diphoton_mass, range=(l, h), bins=n, weights=df.weight)
  y_err_2, bin_edges = np.histogram(df.Diphoton_mass, range=(l, h), bins=n, weights=df.weight**2)
  
  x = (bin_edges[:-1]+bin_edges[1:])/2
  y_err = np.sqrt(y_err_2)
  #print(y, y_err)
  y_err[y_err==0] = min(y_err[y_err!=0])

  bin_spacing = (h-l)/n
  f = lambda x, a, m, s, f: sum(y)*bin_spacing * ((1-f)*np.exp(-a*x/100)/intExp(a, l, h) + (f/(s*np.sqrt(2*np.pi))) * np.exp(-(x-m)**2/(2*s**2)))

  p0 = [1, 90, 3, 1.0]
  bounds = ([1e-4, 85, 1, 0], [20, 95, 10, 1])
  popt, pcov = curve_fit(f, x, y, p0, y_err, bounds=bounds)
  perr = np.sqrt(np.diag(pcov))

  # calculate number in peak
  N_tot = sum(y)
  N_tot_err = np.sqrt(sum(y_err_2))
  frac_peak = popt[3]
  frac_peak_err = perr[3]

  N_peak = N_tot * popt[3]
  N_peak_err = N_peak * np.sqrt((N_tot_err/N_tot)**2 + (frac_peak_err/frac_peak)**2)

  if plot:
    plt.errorbar(x, y, y_err, fmt='o')
    x_smooth = np.linspace(l, h, 100)
    plt.plot(x_smooth, f(x_smooth, *popt))
    plt.title(title)

    parameter_names = [r"$a$", r"$\mu$", r"$\sigma$", r"$f$"]
    #text = "Exp + Gauss Fit\n" + r"$y=(1-f)e^{-ax/100} + f G(x; \mu, \sigma)$"
    text = "Exp + Gauss Fit"
    text += "\nSumw = %.1f"%sum(y)
    for i, name in enumerate(parameter_names):
      text += "\n" + name + r"$=%.2f \pm %.2f$"%(popt[i], perr[i])
    text += "\n" + r"$N_{peak}$" + r"$=%.0f \pm %.0f$"%(N_peak, N_peak_err)
    plt.text(max(x), max(y+y_err), text, verticalalignment='top', horizontalalignment="right")
    plt.xlabel(r"$m_{\gamma\gamma}$")
    plt.ylabel("Events / %d GeV"%(bin_spacing))
    plt.savefig(savepath)
    plt.clf()

  return N_peak, N_peak_err, popt, perr

def halveDf(df, score, cr_sr, half, split_point):
  if half == 0:
    return df[(df.SR!=cr_sr)|(df[score]<=split_point)]
  elif half == 1:
    return df[(df.SR!=cr_sr)|(df[score]>split_point)]
  else:
      raise Exception("Expected half=0 or 1")

def doFits(df, df_inverted, optim_result, half=None, plot=True, save_dir=None):
  df = assignSignalRegions(df, optim_result)
  df_inverted = assignSignalRegions(df_inverted, optim_result)
  
  cr_sr = max(df.SR) # sr used as control

  # split the control region in two, used for correlation test
  if half != None:
    score = optim_result["score"]
    split_point = df[df.SR==cr_sr][score].median()
    df = halveDf(df, score, cr_sr, half, split_point)
    df_inverted = halveDf(df_inverted, score, cr_sr, half, split_point)

  A = df_inverted[df_inverted.SR==cr_sr]
  C = df[df.SR==cr_sr]

  if save_dir == None:
    save_dir = f"RelicDYEstimation/{optim_result['sig_proc']}/"
  os.makedirs(save_dir, exist_ok=True)
  fitFunction(A, save_dir+"A_fit.png", "Region A", plot=plot)
  C_yield, C_yield_err, C_popt, C_perr = fitFunction(C, save_dir+"C_fit.png", "Region C", plot=plot)
  #C_yield, C_yield_err = C.weight.sum(), np.sqrt((C.weight**2).sum())
  #C_yield, C_yield_err = A_yield * 0.0008, 0

  # choose to take yield for A and B from sum instead of fit
  A_yield, A_yield_err = A.weight.sum(), np.sqrt((A.weight**2).sum())
  
  print(C_yield, A_yield, A[A.Diphoton_mass<70].weight.sum())

  yields = []
  for target_sr in range(cr_sr):
    B = df_inverted[df_inverted.SR==target_sr]
    if len(B) > 0:
      fitFunction(B, save_dir+"B_fit_SR%d.png"%target_sr, "Region B, SR%d"%target_sr, plot=plot)
    B_yield, B_yield_err = B.weight.sum(), np.sqrt((B.weight**2).sum())

    D_yield = B_yield * C_yield / A_yield
    D_yield_err = D_yield * np.sqrt((B_yield_err/B_yield)**2 + (C_yield_err/C_yield)**2 + (A_yield_err/A_yield)**2)
    print(A_yield_err/A_yield, B_yield_err/B_yield, C_yield_err/C_yield)
    yields.append([D_yield, D_yield_err])
  
  mu = [C_popt[1], C_perr[1]]
  sigma = [C_popt[2], C_perr[2]]
  veto_eff = [C_yield / A_yield, (C_yield / A_yield) * np.sqrt((C_yield_err/C_yield)**2 + (A_yield_err/A_yield)**2)]

  return {"yields":yields, "mu":mu, "sigma":sigma, "veto_eff":veto_eff}

def loc(i, nom, step):
  return nom + (i*step)*(1-2*(i%2))

def plotYields(all_info, optim_results, only_90=True):
  #N_sidebands = np.array([10, 10, 10, 10, 10, 10, 40, 160])
  N_sidebands = np.array([20, 20, 20, 20, 40, 80, 320])
  
  # plot relative to N_sidebands
  i = 0
  for proc, info in all_info.items():
    mx, my = common.get_MX_MY(proc)
    if only_90 and my != 90:
      continue
    
    yields = np.array(info["yields"])
    x = np.arange(0, len(yields))
    #plt.errorbar(loc(i, x, 0.025), yields.T[0]/N_sidebands, yields.T[1]/N_sidebands, label=r"$(m_X, m_Y) = (%d, %d)$"%(mx, my), fmt='x')
    plt.errorbar(loc(i, x, 0.025), yields.T[0]/N_sidebands, yields.T[1]/N_sidebands, label=r"$m_X=%d$"%mx, fmt='x')
    i += 1

  plt.legend(ncol=3, frameon=True)
  plt.xlabel("Signal Region")
  plt.ylabel("DY yield in SR / N_sidebands")
  save_dir = f"RelicDYEstimation/"
  plt.savefig(save_dir+"relative_to_sidebands.png")
  plt.clf()

  # plot relative to continuum background in +-1 sigma window around peak
  # signal is has sigma = 0.9 GeV
  frac = 2*(scs.norm.cdf(0.9 / info["sigma"][0])-0.5) # find out what fraction of DY is within the signal +-1 sigma window
  i = 0
  for proc, info in all_info.items():
    mx, my = common.get_MX_MY(proc)
    if only_90 and my != 90:
      continue

    yields = np.array(info["yields"]) * frac
    x = np.arange(0, len(yields))

    for res in optim_results:
      if res["sig_proc"] == proc:
        nbkgs = res["nbkgs"][::-1][:-1]
        nbkgs = N_sidebands * 0.4 / 10 # hard code in because numbers from cat optim are unreliable
        break
      
    print(proc)
    print(yields)
    print(nbkgs)
    print()
    #plt.errorbar(loc(i, x, 0.025), yields.T[0]/nbkgs, yields.T[1]/nbkgs, label=r"$(m_X, m_Y) = (%d, %d)$"%(mx, my), fmt='x')
    plt.errorbar(loc(i, x, 0.025), yields.T[0]/nbkgs, yields.T[1]/nbkgs, label=r"$m_X=%d$"%mx, fmt='x')
    i += 1

  plt.legend(ncol=3, frameon=True)
  plt.xlabel("Signal Region")
  plt.ylabel("DY yield in SR in peak / Continuum in peak")
  plt.ylim(bottom=0)
  save_dir = f"RelicDYEstimation/"
  plt.savefig(save_dir+"relative_to_continuum_in_peak.png")
  plt.clf()

def correlationCheck(df, df_inverted, optim_results, only_90=False):
  info0s = {}
  info1s = {}
  infos_whole = {}

  for i, res in enumerate(optim_results):
    print(res["sig_proc"])
    if only_90 and "MY_90" not in res["sig_proc"]: 
      continue
    df_event = df.event.copy()
    df_inverted_event = df_inverted.event.copy()

    info0s[res["sig_proc"]] = doFits(df, df_inverted, res, half=0, plot=True, save_dir=f"RelicDYEstimation/{res['sig_proc']}/LowerHalf/")
    info1s[res["sig_proc"]] = doFits(df, df_inverted, res, half=1, plot=True, save_dir=f"RelicDYEstimation/{res['sig_proc']}/UpperHalf/")
    infos_whole[res["sig_proc"]] = doFits(df, df_inverted, res, plot=True, save_dir=f"RelicDYEstimation/{res['sig_proc']}/Whole/")

    assert (df_event == df.event).all()
    assert (df_inverted_event == df_inverted.event).all()
    #break

  for i, proc in enumerate(info0s.keys()):
    mx, my = common.get_MX_MY(proc)
    # y = np.array([info0s[proc]["veto_eff"][0], info1s[proc]["veto_eff"][0]]) * 100
    # y_err = np.array([info0s[proc]["veto_eff"][1], info1s[proc]["veto_eff"][1]]) * 100
    # plt.errorbar(loc(i, np.array([0, 1]), 0.005), y, y_err, fmt='x', label=r"$(m_X, m_Y) = (%d, %d)$"%(mx, my))
    y = np.array([info0s[proc]["veto_eff"][0], infos_whole[proc]["veto_eff"][0], info1s[proc]["veto_eff"][0]]) * 100
    y_err = np.array([info0s[proc]["veto_eff"][1], infos_whole[proc]["veto_eff"][1], info1s[proc]["veto_eff"][1]]) * 100
    plt.errorbar(loc(i, np.array([0, 0.5, 1]), 0.005), y, y_err, fmt='x', label=r"$(m_X, m_Y) = (%d, %d)$"%(mx, my))

  plt.legend(ncol=2, frameon=True)
  plt.ylabel("Electron + pixel veto\nefficiency [%]")
  #plt.ylim(0.02, 0.1)
  #plt.xticks([0, 1], ["Lower Half", "Upper Half"])
  plt.xticks([0, 0.5, 1], ["Lower Half", "Whole", "Upper Half"])
  save_dir = f"RelicDYEstimation/"
  plt.savefig(save_dir+"correlation_check.png")
  plt.clf()    

def outputModels(all_info, outdir):
  model = {}
  first_info = next(iter(all_info.items()))[1]
  ncat = len(first_info["yields"])

  for i in range(ncat):
    model[i] = {}
    for proc, info in all_info.items():
      mx, my = common.get_MX_MY(proc)
      model[i]["%d_%d"%(mx, my)] = {}
      model[i]["%d_%d"%(mx, my)]["norm"] = info["yields"][i][0] / (1000*(common.lumi_table[2016]+common.lumi_table[2017]+common.lumi_table[2018]))
      model[i]["%d_%d"%(mx, my)]["norm_err"] = info["yields"][i][1] / (1000*(common.lumi_table[2016]+common.lumi_table[2017]+common.lumi_table[2018]))

      model[i]["%d_%d"%(mx, my)]["parameters"] = [info["mu"][0], info["sigma"][0]]
      model[i]["%d_%d"%(mx, my)]["parameters_err"] = [info["mu"][1], info["sigma"][1]]


  with open(os.path.join(outdir, "model.json"), "w") as f:
    json.dump(model, f, indent=4)

parquet_file = sys.argv[1]
summary_json = sys.argv[2]
cat_optim_file = sys.argv[3]

with open(summary_json, "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

# load in nominal df and apply pixel veto
df = pd.read_parquet(parquet_file)

print(df[df.process_id==proc_dict["Data"]].weight.sum())
print(df[df.process_id!=proc_dict["Data"]].weight.sum())


df = df[df.process_id == proc_dict["Data"]]
#df = df[df.process_id.isin([proc_dict[proc] for proc in ["ZZ", "WW", "WZ"]])]

#df = df[df.year==2018]
#df.loc[:,"weight"] *= 132/55

veto = (df.LeadPhoton_pixelSeed==0)&(df.SubleadPhoton_pixelSeed==0)&(df.LeadPhoton_electronVeto==1)&(df.SubleadPhoton_electronVeto==1)
df, df_inverted = df[veto], df[~veto]

with open(cat_optim_file) as f:
  optim_results = json.load(f) # pick first mass point

makePlot(df, "RelicDYEstimation/all_nominal_preselection.pdf")
makePlot(df_inverted, "RelicDYEstimation/all_inverted.pdf")

#correlationCheck(df, df_inverted, optim_results, only_90=True)

all_info = {}
for res in optim_results:
  if "MY_90" not in res["sig_proc"]: continue
  #if ("MX_300" not in res["sig_proc"]) and ("MX_900" not in res["sig_proc"]) : continue
  print(res["sig_proc"])
  info = doFits(df, df_inverted, res, plot=True)
  all_info[res["sig_proc"]] = info

plotYields(all_info, optim_results, only_90=True)
outputModels(all_info, "RelicDYEstimation/")