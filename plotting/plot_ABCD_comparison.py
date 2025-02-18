import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,12.5)

import numpy as np
import sys
import tabulate
import pandas as pd
import os

import scipy.interpolate as spi

import common

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
    else:
      limits[idx2][idx1] = limit

  #print(limits[2])
  #print(limits_no_sys[2])

  masses = np.array(masses)
  #sort out scan over mh (mgg)
  if len(np.unique(np.array(masses)[:,2])) != 1: #if more than 125 in mh
    #find places where mx and mh overlap
    for mx in np.unique(masses[:,0]):
      uniques, counts = np.unique(masses[masses[:,0]==mx, 2], return_counts=True)
      assert sum(counts>2) == 0, str(mx) + " " + str(uniques[counts>2]) #should not have more than 1 overlap
      overlap_mh = uniques[counts==2]

      for mh in overlap_mh:
        idx1, idx2 = np.where( (masses[:,0]==mx) & (masses[:,2]==mh) )[0]
        if limits[2][idx1] < limits[2][idx2]:
          to_delete = idx2
        else:
          to_delete = idx1
        masses = np.delete(masses, to_delete, axis=0)
        limits = np.delete(limits, to_delete, axis=1)
        limits_no_sys = np.delete(limits_no_sys, to_delete, axis=1)
        limits_no_res_bkg = np.delete(limits_no_res_bkg, to_delete, axis=1)
        limits_no_dy_bkg = np.delete(limits_no_dy_bkg, to_delete, axis=1)

    #masses[:,1] = masses[:,2] #set my to be mh
  
  #masses = masses[:,:2]
  #print(masses)

  return masses, limits, limits_no_sys, limits_no_res_bkg, limits_no_dy_bkg

def plotComparison(mx, limits_inc, limits_left, limits_right, ylabel, savename, xlabel=r"$m_X$" ):
  s = np.argsort(mx)
  mx = mx[s]
  limits_inc = limits_inc.T[s].T
  limits_left = limits_left.T[s].T
  limits_right = limits_right.T[s].T
  
  f, axs = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

  axs[0].plot(mx, limits_inc[5], 'ko', zorder=3, label="Observed (default)")
  axs[0].plot(mx, limits_left[5], 'ro', zorder=3, label="Observed (float)")
  axs[0].plot(mx, limits_right[5], 'bo', zorder=3, label="Observed (fixed)")

  axs[0].plot(mx, limits_inc[2], 'k--', zorder=3, label="Expected (default)")
  axs[0].plot(mx, limits_left[2], 'r--', zorder=3, label="Expected (float)")
  axs[0].plot(mx, limits_right[2], 'b--', zorder=3, label="Expected (fixed)")

  axs[0].fill_between(mx, limits_inc[1], limits_inc[3], zorder=2, facecolor="green", label=r"$\pm$ $1\sigma$")
  axs[0].fill_between(mx, limits_inc[0], limits_inc[4], zorder=1, facecolor="yellow", label=r"$\pm$ $2\sigma$")
  
  axs[0].set_ylabel(ylabel)
  axs[0].legend(ncol=3)
  axs[0].set_yscale("log")
  
  ratio_left = limits_left[2]/limits_inc[2]
  ratio_right = limits_right[2]/limits_inc[2]

  axs[1].plot(mx, ratio_left, color="r")
  axs[1].plot(mx, ratio_right, color="b")
  #axs[1].scatter(mx[np.isin(mx, nominal_masses)], ratio[np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  axs[1].set_ylabel("Ratio expected")

  ratio_left = limits_left[5]/limits_inc[5]
  ratio_right = limits_right[5]/limits_inc[5]

  axs[2].plot(mx, ratio_left, color="r")
  axs[2].plot(mx, ratio_right, color="b")
  #axs[1].scatter(mx[np.isin(mx, nominal_masses)], ratio[np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  axs[2].set_ylabel("Ratio observed")

  axs[2].set_xlabel(xlabel)

  axs[0].set_ylim(top=4)

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")
  plt.clf()
  plt.close(f)

def plotSigComparison(mx, limits_inc, limits_left, limits_right, ylabel, savename, xlabel=r"$m_X$" ):
  s = np.argsort(mx)
  mx = mx[s]
  limits_inc = limits_inc.T[s].T
  limits_left = limits_left.T[s].T
  limits_right = limits_right.T[s].T
  
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  axs[0].plot(mx, limits_inc[6], 'ko', zorder=3, label="Observed (default)")
  axs[0].plot(mx, limits_left[6], 'ro', zorder=3, label="Observed (float)")
  axs[0].plot(mx, limits_right[6], 'bo', zorder=3, label="Observed (fixed)")

  axs[0].set_ylabel(ylabel)
  axs[0].legend()
  #axs[0].set_yscale("log")
  
  ratio_left = limits_left[6]/limits_inc[6]
  ratio_right = limits_right[6]/limits_inc[6]

  axs[1].plot(mx, ratio_left, color="r")
  axs[1].plot(mx, ratio_right, color="b")
  #axs[1].scatter(mx[np.isin(mx, nominal_masses)], ratio[np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  axs[1].set_ylabel("Ratio")
  axs[1].set_ylim(bottom=0.5)

  axs[1].set_xlabel(xlabel)

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")
  plt.clf()
  plt.close(f)

masses_inc, limits_inc = getLimits(sys.argv[1])[:2]
masses_left, limits_left = getLimits(sys.argv[2])[:2]
masses_right, limits_right = getLimits(sys.argv[3])[:2]

print(" ".join(["%d"%mx for mx in masses_left[:,0]]))

mx_common = set(masses_inc[:,0]).intersection(masses_left[:,0]).intersection(masses_right[:,0])
mx_common = list(mx_common)
print(mx_common)

print(np.isin(masses_inc[:,0], mx_common))

s = (masses_inc[:,1] == 90) & (masses_inc[:,2] == 90) & (np.isin(masses_inc[:,0], mx_common))
masses_inc = masses_inc[s]
limits_inc = limits_inc.T[s].T

print(masses_inc)

s = np.isin(masses_left[:,0], mx_common)
masses_left = masses_left[s]
limits_left = limits_left.T[s].T

s = np.isin(masses_right[:,0], mx_common)
masses_right = masses_right[s]
limits_right = limits_right.T[s].T

assert (masses_inc[:,0] == masses_left[:,0]).all() and (masses_inc[:,0] == masses_right[:,0]).all()
#assert (masses_inc[:,0] == masses_left[:,0]) and (masses_inc[:,0] == masses_right[:,0])

mx = masses_inc[:,0]
savename = "Outputs/Y_gg_Low_Mass/ABCD_check/limit_comparison_add_bern4"
ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow Y(%d)H \rightarrow \gamma\gamma\tau\tau)$ [fb]"%90
plotComparison(mx, limits_inc, limits_left, limits_right, ylabel, savename, xlabel=r"$m_X$" )
ylabel = "Observed Significance"
plotSigComparison(mx, limits_inc, limits_left, limits_right, ylabel, savename+"_sig", xlabel=r"$m_X$" )


