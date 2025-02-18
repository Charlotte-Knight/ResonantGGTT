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

x-axis = pnn score
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
import scipy.integrate as spi
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

  #assert sum(df.SR==-1) == 0, print(sum(df.SR==-1), df[df.SR==-1])
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

# def dcb(x, mean, sigma, beta1, m1, beta2, m2, N=1):
#   beta1, m1, beta2, m2 = np.abs(beta1), np.abs(m1), np.abs(beta2), np.abs(m2)

#   with np.errstate(all='ignore'):
#     A1 = np.power(m1/beta1, m1) * np.exp(-beta1**2/2)
#     B1 = m1/beta1 - beta1
#     A2 = np.power(m2/beta2, m2) * np.exp(-beta2**2/2)
#     B2 = m2/beta2 - beta2

#     xs = (x-mean)/sigma

#     middle = N*np.exp(-xs**2/2)*(-beta1 < xs)*(xs < beta2)
#     left = np.nan_to_num(N*A1*np.power(B1-xs, -m1)*(xs<=-beta1), nan=0.0)
#     right = np.nan_to_num(N*A2*np.power(B2+xs, -m2)*(xs>=beta2), nan=0.0)

#   return left + middle + right

def gauss(x, l, h, p):
  m = p[0]
  s = p[1]
  return np.exp(-(x-m)**2/(2*s**2)) / (s*np.sqrt(2*np.pi))

def intExp(l, h, a):
  return -(100/a)*(np.exp(-a*h/100)-np.exp(-a*l/100))

def exp(x, l, h, a):
  return np.exp(-a*x/100) / intExp(l, h, a)

def intDCB(l, h, mean, sigma, beta1, m1, beta2, m2):
  left_cb = scs.crystalball(beta1, m1, scale=sigma, loc=mean)
  right_cb = scs.crystalball(beta2, m2, scale=sigma, loc=mean)
  
  left_int = (left_cb.cdf(mean) - left_cb.cdf(l)) / left_cb.pdf(mean)
  right_int = (right_cb.cdf(mean) - right_cb.cdf(mean - (h-mean))) / right_cb.pdf(mean)
  sum_int = left_int+right_int

  return sum_int

def dcb(x, l, h, p):
  mean, sigma, beta1, m1, beta2, m2 = p
  
  with np.errstate(all='ignore'):
    A1 = np.power(m1/beta1, m1) * np.exp(-beta1**2/2)
    B1 = m1/beta1 - beta1
    A2 = np.power(m2/beta2, m2) * np.exp(-beta2**2/2)
    B2 = m2/beta2 - beta2

    xs = (x-mean)/sigma

    middle = np.exp(-xs**2/2)*(-beta1 < xs)*(xs < beta2)
    left = np.nan_to_num(A1*np.power(B1-xs, -m1)*(xs<=-beta1), nan=0.0)
    right = np.nan_to_num(A2*np.power(B2+xs, -m2)*(xs>=beta2), nan=0.0)

  return (left + middle + right) / intDCB(l, h, mean, sigma, beta1, m1, beta2, m2)

# def intPow(l, h, a):
#   return 100*(1/(-a+1))*(np.power((h/100), -a+1)-np.power((l/100), -a+1))
def intPow(l, h, a):
  return (1/(-a+1))*(np.power((h), -a+1)-np.power((l), -a+1))

def pow(x, l, h, a):
  return np.power((x), -a) / intPow(l, h, a)
# def pow(x, l, h, a):
#   return np.power((x/100), -a) / intPow(l, h, a)

def poly2(x, l, h, a, b):
  xn = (x-100) / 100
  poly = a*xn + b*xn**2
  int_f = lambda x: (100*a/2)*((x-100)/100)**2 + (100*b/3)*((x-100)/100)**3
  integral = int_f(h) - int_f(l)
  c = (1-integral) / (h-l)
  
  return poly + c

def chi2(x, y, y_err, f, p):
  return np.sum((y-f(x, *p))**2 / y_err**2)

def printDeltaChi2(x, y, y_err, f, popt, perr):
  nom_chi2 = chi2(x, y, y_err, f, popt)
  for i, err in enumerate(perr):
    print("parameter %d"%i)
    popt[i] += err
    print("+1 sigma, delta chi2= = %.2f"%(chi2(x, y, y_err, f, popt)-nom_chi2))
    popt[i] += err
    print("+2 sigma, delta chi2= = %.2f"%(chi2(x, y, y_err, f, popt)-nom_chi2))
    popt[i] -= 3*err
    print("-1 sigma, delta chi2= = %.2f"%(chi2(x, y, y_err, f, popt)-nom_chi2))
    popt[i] -= err
    print("-2 sigma, delta chi2= = %.2f"%(chi2(x, y, y_err, f, popt)-nom_chi2))
    popt[i] += err

def checkBoundaries(bounds, popt):
  for i, p in enumerate(popt):
    #print(p, bounds[0][i], bounds[1][i])
    if np.isclose(p, bounds[0][i]) or np.isclose(p, bounds[1][i]):
      print("Warning - parameter %d hit bound"%i)

def fitFunction(df, savepath, title=None, plot=True, bkg_func="exp", sig_func="dcb", sig_p=None):
  l, h, n = 70, 120, 25
  #l, h, n = 80, 110, 15

  y, bin_edges = np.histogram(df.Diphoton_mass, range=(l, h), bins=n, weights=df.weight)
  y_err_2, bin_edges = np.histogram(df.Diphoton_mass, range=(l, h), bins=n, weights=df.weight**2)
  
  x = (bin_edges[:-1]+bin_edges[1:])/2
  y_err = np.sqrt(y_err_2)
  y_err[y_err==0] = min(y_err[y_err!=0])

  bin_spacing = (h-l)/n

  options = {
    "exp": ([1.0], ([1e-4], [20]), [r"$a$"]),
    "gauss": ([90, 3], ([85, 1], [95, 10]), [r"$\mu$", r"$\sigma$"]),
    "dcb": ([90, 3, 1, 2, 1, 2], ([85, 1, 0.5, 1, 0.5, 1], [95, 10, 3, 20, 3, 20]), ["$\mu$", r"$\sigma$", r"$\beta_1$", r"$n_1$", r"$\beta_2$", r"$n_2$"]),
    "pow": ([0.1], ([1e-2], [20]), [r"$a$"]),
    "poly2": ([-0.6, 0.5], ([-10, 0], [0, 10]), [r"$a$", r"$b$"]),
  }
  
  fit_title = f"{bkg_func} + {sig_func} fit"
  
  parameter_names = [r"$f$"] + options[bkg_func][2] + options[sig_func][2]
  lbounds = [0] + options[bkg_func][1][0] + options[sig_func][1][0]
  hbounds = [1] + options[bkg_func][1][1] + options[sig_func][1][1]
  bounds = (lbounds, hbounds)

  if sig_p is None:
    p0 = [1.0] + options[bkg_func][0] + options[sig_func][0]
    f = lambda x, f, a, *args: sum(y)*bin_spacing * ((1-f)*globals()[bkg_func](x, l, h, a) + f*globals()[sig_func](x, l, h, args))
  else:
    p0 = [1.0] + options[bkg_func][0] + list(sig_p)
    f = lambda x, f, a, *args: sum(y)*bin_spacing * ((1-f)*globals()[bkg_func](x, l, h, a) + f*globals()[sig_func](x, l, h, sig_p))

  # if sig_p is None:
  #   p0 = [1.0] + options[bkg_func][0] + options[sig_func][0]
  #   f = lambda x, f, a, b, *args: sum(y)*bin_spacing * ((1-f)*globals()[bkg_func](x, l, h, a, b) + f*globals()[sig_func](x, l, h, args))
  # else:
  #   p0 = [1.0] + options[bkg_func][0] + list(sig_p)
  #   f = lambda x, f, a, b, *args: sum(y)*bin_spacing * ((1-f)*globals()[bkg_func](x, l, h, a, b) + f*globals()[sig_func](x, l, h, sig_p))

  print(p0, bounds)

  popt, pcov = curve_fit(f, x, y, p0, y_err, bounds=bounds)
  checkBoundaries(bounds, popt)
  perr = np.sqrt(np.diag(pcov))
  #printDeltaChi2(x, y, y_err, f, popt, perr)

  # calculate number in peak
  N_tot = sum(y)
  N_tot_err = np.sqrt(sum(y_err_2))
  frac_peak = popt[0]
  frac_peak_err = perr[0]

  N_peak = N_tot * popt[0]
  N_peak_err = N_peak * np.sqrt((N_tot_err/N_tot)**2 + (frac_peak_err/frac_peak)**2)

  # if plot:
  #   plt.errorbar(x, y, y_err, fmt='o')
  #   x_smooth = np.linspace(l, h, 100)
  #   plt.plot(x_smooth, f(x_smooth, *popt), label="S+B")
  #   if frac_peak < 0.9:
  #     popt[3] = 0
  #     plt.plot(x_smooth, f(x_smooth, *popt), '--', label="B only")
  #     popt[3] = frac_peak
  #     plt.legend(loc="upper right")
  #   plt.title(title)
    
  #   parameter_names = [r"$a$", r"$\mu$", r"$\sigma$", r"$f$"]
  #   #text = "Exp + Gauss Fit\n" + r"$y=(1-f)e^{-ax/100} + f G(x; \mu, \sigma)$"
  #   text = "Exp + Gauss Fit"
  #   text += "\nSumw = %.1f"%sum(y)
  #   for i, name in enumerate(parameter_names):
  #     text += "\n" + name + r"$=%.2f \pm %.2f$"%(popt[i], perr[i])
  #   text += "\n" + r"$N_{peak}$" + r"$=%.0f \pm %.0f$"%(N_peak, N_peak_err)
  #   plt.text(max(x), 0.9*max(y+y_err)+0.1*min(y-y_err), text, verticalalignment='top', horizontalalignment="right")
  #   plt.xlabel(r"$m_{\gamma\gamma}$")
  #   plt.ylabel("Events / %d GeV"%(bin_spacing))
  #   plt.savefig(savepath)
  #   plt.clf()

  if plot:
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    axs[0].errorbar(x, y, y_err, fmt='o')
    x_smooth = np.linspace(l, h, 100)
    axs[0].plot(x_smooth, f(x_smooth, *popt), label="S+B")
    
    bkg_popt = popt.copy()
    bkg_popt[0] = 0.0
    bkg_f = lambda x_smooth: (1-frac_peak)*f(x_smooth, *bkg_popt)

    axs[1].errorbar(x, y-bkg_f(x), y_err, fmt='o')
    axs[1].plot(x_smooth, f(x_smooth, *popt)-bkg_f(x_smooth))
    axs[1].plot(x, np.zeros_like(x), 'k--')
    axs[0].set_title(title)
    
    text = fit_title
    #text += "\n" + r"$\chi^2$/dof$=%.2f$"%(chi2(x, y, y_err, f, popt)/(n-len(popt)))
    ndof = n - len(popt)
    gof_pval = 1 - scs.chi2(ndof).cdf(chi2(x, y, y_err, f, popt))
    text += "\n" + "gof pval = %.2f"%gof_pval
    text += "\nSumw = %.1f"%sum(y)
    text += "\n" + parameter_names[0] + r"$=%.3f \pm %.3f$"%(popt[0], perr[0]) # fraction = ...
    text += "\n" + r"($N_{peak}$" + r"$=%.0f \pm %.0f)$"%(N_peak, N_peak_err)
    for i, name in enumerate(parameter_names):
      if i != 0:
        text += "\n" + name + r"$=%.2f \pm %.2f$"%(popt[i], perr[i])
    
    axs[0].text(max(x), max(y+y_err), text, verticalalignment='top', horizontalalignment="right", fontsize="small")
    axs[1].set_xlabel(r"$m_{\gamma\gamma}$")
    axs[0].set_ylabel("Events / %d GeV"%(bin_spacing))
    axs[1].set_ylabel("Bkg. \nsubtracted")

    plt.savefig(savepath)
    plt.clf()
    plt.close()

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
  print("cr frac: %.3f"%(df[df.SR==cr_sr].weight.sum()/df.weight.sum()))

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

  bkg_func = "exp"
  A_yield, A_yield_err, A_popt, A_perr = fitFunction(A, save_dir+"A_fit.png", "Region A", plot=plot, bkg_func=bkg_func, sig_func="dcb")
  A_sig_p = A_popt[2:]
  # choose to take yield for A and B from sum instead of fit
  A_yield, A_yield_err = A.weight.sum(), np.sqrt((A.weight**2).sum())
  C_yield, C_yield_err, C_popt, C_perr = fitFunction(C, save_dir+"C_fit.png", "Region C", plot=plot, bkg_func=bkg_func, sig_func="dcb", sig_p=A_sig_p)
  
  yields = []
  for target_sr in range(cr_sr):
    B = df_inverted[df_inverted.SR==target_sr]
    if len(B) > 0:
      fitFunction(B, save_dir+"B_fit_SR%d.png"%target_sr, "Region B, SR%d"%target_sr, plot=plot, bkg_func=bkg_func, sig_func="gauss")
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
  #N_sidebands = np.array([20, 20, 20, 20, 40, 80, 320])
  N_sidebands = np.array([10, 10, 10, 10, 10, 10, 20, 40, 80, 320])

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
  #frac = 2*(scs.norm.cdf(0.9 / info["sigma"][0])-0.5) # find out what fraction of DY is within the signal +-1 sigma window
  frac = 2*(scs.norm.cdf(0.9 / 3)-0.5) # find out what fraction of DY is within the signal +-1 sigma window
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

def correlationPlots(df, df_inverted, optim_results):
  #df = df[(df.Diphoton_mass>70)&(df.Diphoton_mass<120)]
  #df_inverted = df_inverted[(df_inverted.Diphoton_mass>70)&(df_inverted.Diphoton_mass<120)]

  df = df[(df.Diphoton_mass>85)&(df.Diphoton_mass<95)]
  df_inverted = df_inverted[(df_inverted.Diphoton_mass>85)&(df_inverted.Diphoton_mass<95)]

  for i, res in enumerate(optim_results):
    score = res["score"]
    #y, edges = np.histogram(df[score], range=(0, 1), bins=20, histtype="step", weights=df.weight)

    for j in df_inverted.category.unique():
      plt.hist(df[df.category==j][score], range=(0, 1), bins=20, histtype="step", weights=df[df.category==j].weight, label="Nominal", density=True)
      plt.hist(df_inverted[df_inverted.category==j][score], range=(0, 1), bins=20, histtype="step", weights=df_inverted[df_inverted.category==j].weight, label="Inverted", density=True)

      plt.legend(frameon=True)
      plt.xlabel(score)
      save_dir = f"RelicDYEstimation/"
      plt.savefig(save_dir+f"correlation_check_{score}_cat{j}.png")
      plt.clf()

    # y_tot, edges = np.histogram(df[score], range=(0, 1), bins=20, weights=df.weight)
    # dfs = df[(df.Diphoton_mass<85)|(df.Diphoton_mass>95)]
    # y_sidebands, edges = np.histogram(dfs[score], range=(0, 1), bins=20, weights=dfs.weight)
    # scale = intExp(2.833, 70, 120) / (intExp(2.833, 70, 85)+intExp(2.833, 95, 120))
    # y_sidebands *= scale
    # y_peak = y_tot - y_sidebands
    # plt.hist(edges[:-1], range=(0, 1), bins=20, histtype="step", weights=y_peak, label="Nominal", density=True)
    
    plt.hist(df[score], range=(0, 1), bins=20, histtype="step", weights=df.weight, label="Nominal", density=True)
    plt.hist(df_inverted[score], range=(0, 1), bins=20, histtype="step", weights=df_inverted.weight, label="Inverted", density=True)

    plt.legend(frameon=True)
    plt.xlabel(score)
    save_dir = f"RelicDYEstimation/"
    plt.savefig(save_dir+f"correlation_check_{score}.png")
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

nominal_parquet_file = sys.argv[1]
nominal_summary_json = sys.argv[2]
noveto_parquet_file = sys.argv[3]
noveto_summary_json = sys.argv[4]
cat_optim_file = sys.argv[5]

def loadDataFrame(parquet_file, summary_json, process):
  with open(summary_json, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]
  df = pd.read_parquet(parquet_file)
  return df[df.process_id == proc_dict[process]]

df = loadDataFrame(noveto_parquet_file, noveto_summary_json, "Data")
#veto = (df.LeadPhoton_pixelSeed==0)&(df.SubleadPhoton_pixelSeed==0)&(df.LeadPhoton_electronVeto==1)&(df.SubleadPhoton_electronVeto==1)
#df_inverted = df[~veto]
df_inverted = df

df = loadDataFrame(nominal_parquet_file, nominal_summary_json, "Data")
#assert df.LeadPhoton_pixelSeed.sum() == 0
#assert df.SubleadPhoton_pixelSeed.sum() == 0

#df = df[df.year==2018]
#df.loc[:,"weight"] *= 132/55


with open(cat_optim_file) as f:
  optim_results = json.load(f) # pick first mass point

makePlot(df, "RelicDYEstimation/all_nominal_preselection.pdf")
makePlot(df_inverted, "RelicDYEstimation/all_inverted.pdf")

#correlationPlots(df, df_inverted, optim_results)
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