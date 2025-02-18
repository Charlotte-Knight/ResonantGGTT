import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
plt.rcParams['figure.constrained_layout.use'] = True

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
    m = line.split(".")[0].split("_")[-1]
    mx = int(m.split("mx")[1].split("my")[0])
    my = int(m.split("my")[1].split("mh")[0])
    if "mh" in m: mh = int(m.split("mh")[1])
    else:         mh = 125
    if [mx, my, mh] not in masses:
      masses.append([mx, my, mh])

  limits = np.zeros((5, len(masses)))
  limits_no_dy = np.zeros((5, len(masses)))
  limits_no_dy_sys = np.zeros((5, len(masses)))

  for line in results:
    m = line.split(".")[0].split("_")[-1]
    mx = int(m.split("mx")[1].split("my")[0])
    my = int(m.split("my")[1].split("mh")[0])
    if "mh" in m: mh = int(m.split("mh")[1])
    else:         mh = 125
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
    
    limit = float(line.split("r < ")[1])

    if "no_dy_bkg_mx" in line:
      limits_no_dy[idx2][idx1] = limit
    elif "no_dy_bkg_sys" in line:
      limits_no_dy_sys[idx2][idx1] = limit
    else:
      limits[idx2][idx1] = limit

  
  masses = np.array(masses)
  #sort out scan over mh (mgg)
  if len(np.unique(np.array(masses)[:,2])) != 1: #if more than 125 in mh
    #find places where mx and mh overlap
    for mx in np.unique(masses[:,0]):
      uniques, counts = np.unique(masses[masses[:,0]==mx, 2], return_counts=True)
      assert sum(counts>2) == 0 #should not have more than 1 overlap
      overlap_mh = uniques[counts==2]

      for mh in overlap_mh:
        idx1, idx2 = np.where( (masses[:,0]==mx) & (masses[:,2]==mh) )[0]
        if limits[2][idx1] < limits[2][idx2]:
          to_delete = idx2
        else:
          to_delete = idx1
        masses = np.delete(masses, to_delete, axis=0)
        limits = np.delete(limits, to_delete, axis=1)
        limits_no_dy = np.delete(limits_no_dy, to_delete, axis=1)
        limits_no_dy_sys = np.delete(limits_no_dy_sys, to_delete, axis=1)

    masses[:,1] = masses[:,2] #set my to be mh
  
  #masses = masses[:,:2]
  #print(masses)

  return masses, limits, limits_no_dy, limits_no_dy_sys
    
def plotLimits(mX, limits, ylabel, nominal_masses, savename=None, xlabel=r"$m_X$"):
  plt.scatter(mX, limits[2], zorder=3, facecolors="none", edgecolors="blue")
  plt.scatter(mX[np.isin(mX, nominal_masses)], limits[2][np.isin(mX, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  plt.plot(mX, limits[2], 'b--', zorder=3, label="Expected 95% CL limit")
  plt.fill_between(mX, limits[1], limits[3], zorder=2, facecolor="green", label=r"$\pm$ $1\sigma$")
  plt.fill_between(mX, limits[0], limits[4], zorder=1, facecolor="yellow", label=r"$\pm$ $2\sigma$")
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  
  plt.legend()
  bottom, top = plt.ylim()
  
  mplhep.cms.label(llabel="Work in Progress", data=True, lumi=common.tot_lumi, loc=0)

  if savename!=None:
    plt.savefig(savename+".png")
    plt.savefig(savename+".pdf")
    plt.yscale("log")
    plt.savefig(savename+"_log.png")
    plt.savefig(savename+"_log.pdf")
    plt.clf()

def plotSystematicComparison(mx, limits, limits_no_sys, nominal_masses, savename, xlabel=r"$m_X$"):
  ratio = limits[2]/limits_no_sys[2]
  plt.plot(mx, ratio)
  plt.scatter(mx[np.isin(mx, nominal_masses)], ratio[np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  plt.xlabel(xlabel)
  plt.ylabel("Exp. limit w / wo systematics")

  plt.legend()

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")
  plt.clf()

def plotSystematicComparison2(mx, limits, limits_no_sys, ylabel, nominal_masses, savename, xlabel=r"$m_X$" ):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  ratio = limits[2]/limits_no_sys[2]
  
  axs[0].plot(mx, limits[2], zorder=3, label="Expected 95% CL limit")
  axs[0].plot(mx, limits_no_sys[2], zorder=3, label="Expected 95% CL limit (no dy model)")

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

def plotResBkgComparison2(mx, limits, limits_no_sys, ylabel, nominal_masses, savename, xlabel=r"$m_X$"):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  ratio = limits[2]/limits_no_sys[2]
  
  axs[0].plot(mx, limits[2], zorder=3, label="Expected 95% CL limit")
  axs[0].plot(mx, limits_no_sys[2], zorder=3, label="Expected 95% CL limit (no dy sys)")

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

masses, limits, limits_no_dy, limits_no_dy_sys = getLimits(sys.argv[1])
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_br"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_no_dy_comparison"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_no_dy_sys_comparison"), exist_ok=True)

nominal_mx = [300,400,500,600,700,800,900,1000]
nominal_my = [70,80,90,100,125]
#nominal_my = [70,80,90,100,125,150,200,250,300,400,500,600,700,800]
#nominal_my = [125,150,200,250,300,400,500,600,700,800]

#only grab the nominal points
# s = np.isin(masses[:,0], nominal_mx) & np.isin(masses[:,1], nominal_my)
# limits = limits[:, s]
# limits_no_sys = limits_no_sys[:, s]
# masses = masses[s]

ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow YH \rightarrow \gamma\gamma\tau\tau)$ [fb]"

for mx in np.unique(masses[:,0]):
  my = masses[masses[:,0]==mx,1]
  limits_slice = limits[:,masses[:,0]==mx]
  limits_no_dy_slice = limits_no_dy[:,masses[:,0]==mx]
  limits_no_dy_sys_slice = limits_no_dy_sys[:,masses[:,0]==mx]

  limits_slice = limits_slice[:,np.argsort(my)]
  limits_no_dy_slice = limits_no_dy_slice[:,np.argsort(my)]
  limits_no_dy_sys_slice = limits_no_dy_sys_slice[:,np.argsort(my)]
  my = my[np.argsort(my)]

  if mx in nominal_mx:
    nm = nominal_my
  else:
    nm = []

  print(mx)
  print(my, limits_slice)

  ylabel = r"$\sigma(pp \rightarrow X(%d)) B(X \rightarrow YH \rightarrow \gamma\gamma\tau\tau)$ [fb]"%mx
  plotLimits(my, limits_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br", "limits_mx%d"%mx), xlabel=r"$m_Y$")
  plotSystematicComparison2(my, limits_slice, limits_no_dy_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_no_dy_comparison", "mx%d_2"%mx), xlabel=r"$m_Y$")
  plotResBkgComparison2(my, limits_slice, limits_no_dy_sys_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_no_dy_sys_comparison", "mx%d_2"%mx), xlabel=r"$m_Y$")

for my in np.unique(masses[:,1]):
  mx = masses[masses[:,1]==my,0]
  limits_slice = limits[:,masses[:,1]==my]
  limits_no_dy_slice = limits_no_dy[:,masses[:,1]==my]
  limits_no_dy_sys_slice = limits_no_dy_sys[:,masses[:,1]==my]

  limits_slice = limits_slice[:,np.argsort(mx)]
  limits_no_dy_slice = limits_no_dy_slice[:,np.argsort(mx)]
  limits_no_dy_sys_slice = limits_no_dy_sys_slice[:,np.argsort(mx)]
  mx = mx[np.argsort(mx)]

  if my in nominal_my:
    nm = nominal_mx
  else:
    nm = []

  ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow Y(%d)H \rightarrow \gamma\gamma\tau\tau)$ [fb]"%my
  plotLimits(mx, limits_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br", "limits_my%d"%my))
  plotSystematicComparison2(mx, limits_slice, limits_no_dy_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_no_dy_comparison", "my%d_2"%my))
  plotResBkgComparison2(mx, limits_slice, limits_no_dy_sys_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_no_dy_sys_comparison", "my%d_2"%my))

  

  