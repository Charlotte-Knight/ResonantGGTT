import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import LogFormatter
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import numpy as np
import sys
import tabulate
import pandas as pd
import os

import scipy.interpolate as spi

import common

BR_H_GG = 2.27e-3
BR_H_TT = 6.27e-2
BR_H_BB = 5.84e-1

BR_HH_GGTT = 2 * BR_H_GG * BR_H_TT
BR_HH_GGBB = 2 * BR_H_GG * BR_H_BB

NMSSM_max_allowed_Y_gg = pd.DataFrame({"MX":[410,405,413,500,500,500,600,600,600,700,700,700,  300,300,300],
                                       "MY":[70,100,200,70,100,200,70,100,200,70,100,200,   70,100,200], 
                                       "limit":[4.08,8.85,4.06,0.916,1.62,1.26,0.214,0.365,0.370,0.0580,0.103,0.120,   4.08,8.85,4.06]})


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

    masses[:,1] = masses[:,2] #set my to be mh
  
  # limits come as xs(pp -> X -> HH) * BR(H->gg) * 2
  # (I divided by BR(H->tautau) or (H->WW) in signal norm already)

  # limits[:6] /= BR_H_GG * 2
  # limits_no_sys[:6] /= BR_H_GG * 2
  # limits_no_res_bkg[:6] /= BR_H_GG * 2
  # limits_no_dy_bkg[:6] /= BR_H_GG * 2
  limits[:6] *= BR_H_TT
  limits_no_sys[:6] *= BR_H_TT
  limits_no_res_bkg[:6] *= BR_H_TT
  limits_no_dy_bkg[:6] *= BR_H_TT

  return masses, limits, limits_no_sys, limits_no_res_bkg, limits_no_dy_bkg

def plotLimitsStack(masses, limits, ylabel, nominal_mx, nominal_my, savename, swap=False):
  label2 = "Median expected"
  label3 = "68% expected"
  label4 = "95% expected"
  label5 = "Observed"

  if not swap:
    xlabel = r"$m_Y$ [GeV]"
    mx_idx = 0
    my_idx = 1
    nominal_stack_exceptions = [650]
    #xlims = (None, 1000)
    xlims = (None, 1000)
    #ylims = (None, 5e11)
    ylims = (0.1, 1e9)
  else:
    xlabel = r"$m_X$ [GeV]"
    mx_idx = 1
    my_idx = 0
    nominal_stack_exceptions = [90, 95, 100]
    nominal_mx, nominal_my = nominal_my, nominal_mx
    xlims = (None, 1200)
    #ylims = (None, 1e18)
    #ylims = (0.01, 1e15)
    #ylims = (0.01, 1e7)
    ylims = (0.01, 1e9)
    
  j = -1
  for i, mx in enumerate(np.sort(np.unique(masses[:,mx_idx]))):
    if (mx not in nominal_mx) and (mx not in nominal_stack_exceptions):
      continue
    j += 1

    my = masses[masses[:,mx_idx]==mx,my_idx]
    limits_slice = limits[:,masses[:,mx_idx]==mx]
    limits_slice = limits_slice[:,np.argsort(my)]
    my = my[np.argsort(my)]

    limits_slice *= int(10**j)

    if mx == 650:
      plt.plot(my[my<300], limits_slice[5][my<300], 'ko-', lw=3, zorder=4, label=label5)
      plt.plot(my[my>300], limits_slice[5][my>300], 'ko-', lw=3, zorder=4)
    else:
      plt.plot(my, limits_slice[5], 'ko-', lw=3, zorder=4, label=label5)

    plt.plot(my, limits_slice[2], 'k--', zorder=3, label=label2)
    if len(my) > 1:
      plt.fill_between(my, limits_slice[1], limits_slice[3], zorder=2, facecolor=(0, 0.8, 0), label=label3)
      plt.fill_between(my, limits_slice[0], limits_slice[4], zorder=1, facecolor=(1, 0.8, 0), label=label4)
    else:
      plt.fill_between([my[0]-10]+list(my), list(limits_slice[1])*2, list(limits_slice[3])*2, zorder=2, facecolor=(0, 0.8, 0), label=label3)
      plt.fill_between([my[0]-10]+list(my), list(limits_slice[0])*2, list(limits_slice[4])*2, zorder=1, facecolor=(1, 0.8, 0), label=label4)
    label1 = label2 = label3 = label4 = label5 = None

    if not swap:
      plt.text(my[-1]+10, limits_slice[2][-1], r"$m_X=%d$ GeV $(\times 10^{%d})$"%(mx, j), fontsize=12, verticalalignment="center")
    else:
      plt.text(my[-1]+10, limits_slice[2][-1], r"$m_Y=%d$ GeV $(\times 10^{%d})$"%(mx, j), fontsize=12, verticalalignment="center")

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)  
  plt.legend(ncol=2, loc="upper left")
  bottom, top = plt.ylim()
  #plt.ylim(limits[:6].min(), limits.max()*10**(j+1))
  #plt.ylim(10, limits.max()*10**(j+1))
  plt.ylim(ylims)
  plt.xlim(xlims)
  
  plt.yscale("log")

  mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_paper.png")
  plt.savefig(savename+"_paper.pdf")
  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_preliminary.png")
  plt.savefig(savename+"_preliminary.pdf")
  plt.clf()

def plotLimits2D(masses, limits, ylabel, savename, observed=False):
  bin_edges = []
  mx = np.sort(np.unique(masses[:,0]))
  my = np.sort(np.unique(masses[:,1]))
  mx_edges = np.array([mx[0] - (mx[1]-mx[0])/2] + list(mx[:-1] + (mx[1:] - mx[:-1])/2) + [mx[-1] + (mx[-1]-mx[-2])/2])
  my_edges = np.array([my[0] - (my[1]-my[0])/2] + list(my[:-1] + (my[1:] - my[:-1])/2) + [my[-1] + (my[-1]-my[-2])/2])

  spline = spi.interp2d(NMSSM_max_allowed_Y_gg.MX, NMSSM_max_allowed_Y_gg.MY, NMSSM_max_allowed_Y_gg.limit, kind='linear', fill_value=0)
  max_allowed_values = [spline(m[0], m[1])[0] for m in masses]
  for i, m in enumerate(masses):
    print(m, max_allowed_values[i])
  
  mx_edge_centers = (mx_edges[:-1]+mx_edges[1:])/2
  my_edge_centers = (my_edges[:-1]+my_edges[1:])/2
  interp_masses = []
  interp_limits = []
  for mxi in mx_edge_centers:
    for myi in my_edge_centers:
      interp_masses.append([mxi, myi])
  interp_masses = np.array(interp_masses)
  
  if not observed:
    limits_idx = 2
  else:
    limits_idx = 5
  interp_limits = spi.griddata(masses[:,:2], limits[limits_idx], interp_masses, method="linear", fill_value=0)
  plt.hist2d(interp_masses[:,0], interp_masses[:,1], [mx_edges, my_edges], weights=interp_limits, norm=matplotlib.colors.LogNorm())
  
  #formatter = LogFormatter(10, labelOnlyBase=False) 
  #cbar = plt.colorbar(ticks=[1e3, 2e3], format=formatter)
  cbar = plt.colorbar()
  #cbar.set_ticks([1e3, 2e3, 3e3])
  #cbar.set_ticklabels(["1e3", "2e3", "3e3"])
  cbar.set_label(ylabel)
  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel(r"$m_Y$ [GeV]")

  #plt.text(0.05, 0.9, tag, transform=plt.gca().transAxes, fontsize=32)
  #plt.text(0.05, 0.9, r"$Y\rightarrow\tau\tau$", transform=plt.gca().transAxes, fontsize=32)
  #plt.text(0.05, 0.9, r"$Y\rightarrow\gamma\gamma$", transform=plt.gca().transAxes, fontsize=32)

  mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_paper.png")
  plt.savefig(savename+"_paper.pdf")
  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_preliminary.png")
  plt.savefig(savename+"_preliminary.pdf")
  
  # plt.fill_between([250,650],[65,65],[my_edges[-1],my_edges[-1]],facecolor="none",hatch="x",edgecolor="red", label="Limit below maximally\nallowed in NMSSM")
  # plt.legend(frameon=True)
  # plt.savefig(savename+"_exclude.png")
  # plt.savefig(savename+"_exclude.pdf")
  # s = limits[limits_idx] < max_allowed_values
  # plt.scatter(masses[s,0], masses[s,1], marker='x', color="r", label="Limit below maximally allowed in NMSSM") 
  # plt.savefig(savename+"_exclude_points.png")
  # plt.savefig(savename+"_exclude_points.pdf")

  plt.clf()

def plotSignificance2D(masses, limits, ylabel, savename):
  bin_edges = []
  mx = np.sort(np.unique(masses[:,0]))
  my = np.sort(np.unique(masses[:,1]))
  mx_edges = np.array([mx[0] - (mx[1]-mx[0])/2] + list(mx[:-1] + (mx[1:] - mx[:-1])/2) + [mx[-1] + (mx[-1]-mx[-2])/2])
  my_edges = np.array([my[0] - (my[1]-my[0])/2] + list(my[:-1] + (my[1:] - my[:-1])/2) + [my[-1] + (my[-1]-my[-2])/2])
  
  #plt.hist2d(masses[:,0], masses[:,1], [mx_edges, my_edges], weights=limits[2], norm=matplotlib.colors.LogNorm())
  mx_edge_centers = (mx_edges[:-1]+mx_edges[1:])/2
  my_edge_centers = (my_edges[:-1]+my_edges[1:])/2
  interp_masses = []
  interp_limits = []
  for mxi in mx_edge_centers:
    for myi in my_edge_centers:
      interp_masses.append([mxi, myi])
  interp_masses = np.array(interp_masses)
  interp_limits = spi.griddata(masses[:,:2], limits[6], interp_masses, method="linear", fill_value=0)
  plt.hist2d(interp_masses[:,0], interp_masses[:,1], [mx_edges, my_edges], weights=interp_limits, cmap="afmhot_r")
  
  cbar = plt.colorbar()
  cbar.set_label(ylabel)
  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel(r"$m_Y$ [GeV]")

  max_idx = np.argmax(limits[6])
  max_mass = masses[max_idx]
  max_sig = limits[6][max_idx]
  print(max_mass, max_sig)
  
  max_sig_str = "Max. Sig. of %.2f at\n"%max_sig + r"$(m_X, m_Y) = (%d, %d)$"%(max_mass[0], max_mass[1])
  plt.text(0.55, 0.85, max_sig_str, transform=plt.gca().transAxes, fontsize=24, color="black")
  #plt.text(0.05, 0.9, r"$Y\rightarrow\tau\tau$", transform=plt.gca().transAxes, fontsize=32)
  #plt.text(0.05, 0.9, r"$Y\rightarrow\gamma\gamma$", transform=plt.gca().transAxes, fontsize=32)

  mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_paper.png")
  plt.savefig(savename+"_paper.pdf")
  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_preliminary.png")
  plt.savefig(savename+"_preliminary.pdf")
  plt.clf()

masses, limits, limits_no_sys, limits_no_res_bkg, limits_no_dy_bkg = getLimits(sys.argv[1])
print(limits)
s = (masses[:,0] == 1000) & (masses[:,1] == 750)
masses = masses[~s]
limits = limits.T[~s].T
limits_no_sys = limits_no_sys.T[~s].T
limits_no_res_bkg = limits_no_res_bkg.T[~s].T
limits_no_dy_bkg = limits_no_dy_bkg.T[~s].T

os.makedirs(os.path.join(sys.argv[2], "Limits_xs"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_sys"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_res_bkg"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_dy_bkg"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Significance"), exist_ok=True)

nominal_mx = [300,400,500,600,700,800,900,1000]

#low mass
#nominal_my = [70,80,90,100,125]
#high mass
nominal_my = [125,150,200,250,300,400,500,600,700,800]
nominal_my = [125,150,200,300,400,500,600,700,800]
#y_tautau
#nominal_my = [50,70,80,90,100,125,150,200,250,300,400,500,600,700,800]

plotSignificance2D(masses, limits, "Significance", os.path.join(sys.argv[2], "Significance", "significance"))

#ylabel = r"95% CL limit on $\sigma(pp \rightarrow X \rightarrow YH) B(Y \rightarrow \gamma\gamma)$ [fb]"
ylabel = r"95% CL limit on $\sigma(pp \rightarrow X \rightarrow YH) B(Y \rightarrow \tau\tau)$ [fb]"
plotLimitsStack(masses, limits,             ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs", "limits_stack_mx"))
plotLimitsStack(masses, limits_no_sys,      ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_no_sys", "limits_stack_mx_no_sys"))
plotLimitsStack(masses, limits_no_res_bkg,  ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_no_res_bkg", "limits_stack_mx_no_res_bkg"))

plotLimitsStack(masses, limits,             ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs", "limits_stack_my"), swap=True)
plotLimitsStack(masses, limits_no_sys,      ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_no_sys", "limits_stack_my_no_sys"), swap=True)
plotLimitsStack(masses, limits_no_res_bkg,  ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_no_res_bkg", "limits_stack_my_no_res_bkg"), swap=True)

plotLimits2D(masses, limits,        ylabel, os.path.join(sys.argv[2], "Limits_xs", "limits_2d"))
plotLimits2D(masses, limits,        ylabel, os.path.join(sys.argv[2], "Limits_xs", "limits_2d_obs"), observed=True)
plotLimits2D(masses, limits_no_sys, ylabel, os.path.join(sys.argv[2], "Limits_xs_no_sys", "limits_2d_no_sys"))
plotLimits2D(masses, limits_no_res_bkg, ylabel, os.path.join(sys.argv[2], "Limits_xs_no_res_bkg", "limits_2d_no_res_bkg"))