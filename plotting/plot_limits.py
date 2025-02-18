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
  
  #masses = masses[:,:2]
  #print(masses)

  return masses, limits, limits_no_sys, limits_no_res_bkg, limits_no_dy_bkg

# def parseTheoryPredictions():
#   dfs = []
#   for kl in ["0p1", "0p5", "1"]:
#     df = pd.read_csv("plotting/theory_predictions/BulkGraviton_NLO_13TeV_kmpl%s.dat"%kl, sep="\t", names=["MX", "XS"])
#     dfs.append(df)

#   return {kl:dfs[i] for i, kl in enumerate([0.1, 0.5, 1.0])}

def parseTheoryPredictions():
  df = pd.read_csv("plotting/theory_predictions/BulkGraviton_NLO_13TeV_kmpl1.dat", sep="\t", names=["MX", "XS"])
  #df = pd.read_csv("plotting/theory_predictions/BulkRadion_13TeV_NLO_LR_2TeV.dat", sep="\t", names=["MX", "XS"])
  spline = spi.CubicSpline(df.MX, df.XS)
  return spline

def plotLimits(mX, limits, ylabel, nominal_masses, savename=None, xlabel=r"$m_X$ [GeV]"):
  obs, = plt.plot(mX, limits[5], 'o-k',  lw=2, zorder=4, label="Observed")
  exp, = plt.plot(mX, limits[2], '--k', lw=2, zorder=3, label="Median expected")
  pm1 = plt.fill_between(mX, limits[1], limits[3], zorder=2, facecolor=(0, 0.8, 0), label="68% expected")
  pm2 = plt.fill_between(mX, limits[0], limits[4], zorder=1, facecolor=(1, 0.8, 0), label="95% expected")
  
  #legend1 = plt.gca().legend(handles=[obs, exp, pm1, pm2], loc=(0.6,0.55), title="95% CL upper limits")
  legend1 = plt.gca().legend(handles=[obs, exp, pm1, pm2], loc="upper right", title="95% CL upper limits")
  plt.gca().add_artist(legend1)

  spline = parseTheoryPredictions()
  x = np.linspace(260, 1000, 100)
  theory, = plt.plot(x, spline(x), 'r-', lw=2, label="Bulk KK Graviton\n($\kappa/\overline{M_{PL}}=1$)")
  plt.gca().legend(handles=[theory], loc=(0.275, 0.8))

  #theory, = plt.plot(x, spline(x), 'r-', lw=2, label="Bulk Radion\n($\Lambda_R=2$ TeV)")
  #plt.gca().legend(handles=[theory], loc=(0.3, 0.8))
  #plt.gca().legend(handles=[theory], loc=(0.625, 0.85))

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  
  #mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=2)
  mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=2)

  if savename!=None:
    plt.savefig(savename+".png")
    plt.savefig(savename+".pdf")
    plt.yscale("log")
    mini, maxi = np.log10(min(limits[0])), np.log10(max(limits[4]))
    width = maxi - mini
    bottom = mini - 0.1*width
    top = maxi + 0.3*width
    plt.ylim(np.power(10, bottom), np.power(10, top))
    plt.savefig(savename+"_log.png")
    plt.savefig(savename+"_log.pdf")
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

def plotLimitsStackMX(masses, limits, ylabel, nominal_mx, nominal_my, savename):
  label1 = "Nominal masses"
  label2 = "Median expected"
  label3 = "68% expected"
  label4 = "95% expected"
  label5 = "Observed"

  j = -1
  for i, mx in enumerate(np.sort(np.unique(masses[:,0]))):
    if (mx not in nominal_mx) and (mx != 650):
      continue
    j += 1

    my = masses[masses[:,0]==mx,1]
    limits_slice = limits[:,masses[:,0]==mx]
    
    limits_slice = limits_slice[:,np.argsort(my)]
    my = my[np.argsort(my)]

    limits_slice *= int(10**j)

    plt.scatter(my, limits_slice[2], zorder=3, facecolors="none", edgecolors="blue")
    if mx in nominal_mx:
      plt.scatter(my[np.isin(my, nominal_my)], limits_slice[2][np.isin(my, nominal_my)], zorder=4, facecolors="none", edgecolors="red", label=label1)
    plt.plot(my, limits_slice[2], 'k--', zorder=3, label=label2)
    plt.plot(my, limits_slice[5], 'ko', lw=3, zorder=4, label=label5)
    #plt.scatter(my, limits_slice[5], color='k', zorder=4, label=label5)
    if len(my) > 1:
      plt.fill_between(my, limits_slice[1], limits_slice[3], zorder=2, facecolor=(0, 0.6, 0), label=label3)
      plt.fill_between(my, limits_slice[0], limits_slice[4], zorder=1, facecolor=(1, 0.8, 0.2), label=label4)
    else:
      plt.fill_between([my[0]-10]+list(my), list(limits_slice[1])*2, list(limits_slice[3])*2, zorder=2, facecolor=(0, 0.6, 0), label=label3)
      plt.fill_between([my[0]-10]+list(my), list(limits_slice[0])*2, list(limits_slice[4])*2, zorder=1, facecolor=(1, 0.8, 0.2), label=label4)
    label1 = label2 = label3 = label4 = label5 = None

    plt.text(my[-1]+2.5, limits_slice[2][-1], r"$m_X=%d$ GeV $(\times 10^{%d})$"%(mx, j), fontsize=12, verticalalignment="center")

  plt.xlabel(r"$m_Y$ [GeV]")
  plt.ylabel(ylabel)  
  plt.legend(ncol=2, loc="upper left")
  bottom, top = plt.ylim()
  plt.ylim(limits[:6].min(), limits.max()*10**(j+2))
  left, right = plt.xlim()
  plt.xlim(left, my.max()*1.15)
  #plt.xlim(left, my.max()*1.2)
  
  plt.yscale("log")
  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)

  if savename!=None:
    plt.savefig(savename+"_log.png")
    plt.savefig(savename+"_log.pdf")
    plt.yscale("linear")
    plt.savefig(savename+".png")
    plt.savefig(savename+".pdf")
    plt.clf()

def plotLimitsStackMY(masses, limits, ylabel, nominal_mx, nominal_my, savename):
  label1 = "Nominal masses"
  label2 = "Median expected"
  label3 = "68% expected"
  label4 = "95% expected"
  label5 = "Observed"

  all_my = np.unique(masses[:,1])

  j = -1
  for i, my in enumerate(all_my):
    if (my not in nominal_my) and (my != 95):
      continue
    j += 1

    mx = masses[masses[:,1]==my,0]
    limits_slice = limits[:,masses[:,1]==my]
    
    limits_slice = limits_slice[:,np.argsort(mx)]
    mx = mx[np.argsort(mx)]

    print(limits_slice)
    print(limits_slice.dtype)
    limits_slice *= int(10**j)

    plt.scatter(mx, limits_slice[2], zorder=3, facecolors="none", edgecolors="blue")
    if my in nominal_my:
      plt.scatter(mx[np.isin(mx, nominal_mx)], limits_slice[2][np.isin(mx, nominal_mx)], zorder=4, facecolors="none", edgecolors="red", label=label1)
    plt.plot(mx, limits_slice[2], 'k--', zorder=3, label=label2)
    
    if len(mx) > 1:
      plt.fill_between(mx, limits_slice[1], limits_slice[3], zorder=2, facecolor=(0, 0.6, 0), label=label3)
      plt.fill_between(mx, limits_slice[0], limits_slice[4], zorder=1, facecolor=(1, 0.8, 0.2), label=label4)
      plt.plot(mx, limits_slice[5], 'ko', lw=3, zorder=4, label=label5)
      #plt.scatter(mx, limits_slice[5], color='k', zorder=4, label=label5)
    else:
      plt.fill_between([mx[0]-10]+list(mx), list(limits_slice[1])*2, list(limits_slice[3])*2, zorder=2, facecolor=(0, 0.6, 0), label=label3)
      plt.fill_between([mx[0]-10]+list(mx), list(limits_slice[0])*2, list(limits_slice[4])*2, zorder=1, facecolor=(1, 0.8, 0.2), label=label4)
      plt.plot([mx[0]-0]+list(mx), list(limits_slice[5])*2, 'ko', lw=3, zorder=4, label=label5)
      #plt.scatter(mx[0], list(limits_slice[5]), color='k', zorder=4, label=label5)
    label1 = label2 = label3 = label4 = label5 = None

    plt.text(mx[-1]+10, limits_slice[2][-1], r"$m_Y=%d$ GeV $(\times 10^{%d})$"%(my, j), fontsize=12, verticalalignment="center")

  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel(ylabel)  
  plt.legend(ncol=2, loc="upper left")
  bottom, top = plt.ylim()
  plt.ylim(limits[:6].min(), limits.max()*10**(j+1))
  left, right = plt.xlim()
  plt.xlim(left, 1175)

  plt.yscale("log")
  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)

  if savename!=None:
    plt.savefig(savename+"_log.png")
    plt.savefig(savename+"_log.pdf")
    plt.yscale("linear")
    plt.savefig(savename+".png")
    plt.savefig(savename+".pdf")
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
  
  #plt.hist2d(masses[:,0], masses[:,1], [mx_edges, my_edges], weights=limits[2], norm=matplotlib.colors.LogNorm())
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
  
  cbar = plt.colorbar()
  cbar.set_label(ylabel)
  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel(r"$m_Y$ [GeV]")

  #plt.text(0.05, 0.9, r"$Y\rightarrow\tau\tau$", transform=plt.gca().transAxes, fontsize=32)
  plt.text(0.05, 0.9, r"$Y\rightarrow\gamma\gamma$", transform=plt.gca().transAxes, fontsize=32)

  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")
  
  plt.fill_between([250,650],[65,65],[my_edges[-1],my_edges[-1]],facecolor="none",hatch="/",edgecolor="red", label="Limit below maximally\nallowed in NMSSM")
  plt.legend(frameon=True)
  plt.savefig(savename+"_exclude.png")
  plt.savefig(savename+"_exclude.pdf")
  s = limits[limits_idx] < max_allowed_values
  plt.scatter(masses[s,0], masses[s,1], marker='x', color="r", label="Limit below maximally allowed in NMSSM") 
  plt.savefig(savename+"_exclude_points.png")
  plt.savefig(savename+"_exclude_points.pdf")

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
  plt.text(0.05, 0.9, r"Max. Sig. of %.2f at $(m_X, m_Y) = (%d, %d)$"%(max_sig, max_mass[0], max_mass[1]), transform=plt.gca().transAxes, fontsize=24, color="black")

  #plt.text(0.05, 0.9, r"$Y\rightarrow\tau\tau$", transform=plt.gca().transAxes, fontsize=32)
  #plt.text(0.05, 0.9, r"$Y\rightarrow\gamma\gamma$", transform=plt.gca().transAxes, fontsize=32)

  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")

  plt.clf()

def plotSystematicComparison(mx, limits, limits_no_sys, nominal_masses, savename, xlabel=r"$m_X$ [GeV]"):
  ratio = limits[2]/limits_no_sys[2]
  plt.plot(mx, ratio)
  plt.scatter(mx[np.isin(mx, nominal_masses)], ratio[np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  plt.xlabel(xlabel)
  plt.ylabel("Exp. limit w / wo systematics")

  plt.legend()

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")
  plt.clf()

def plotSystematicComparison2(mx, limits, limits_no_sys, ylabel, nominal_masses, savename, xlabel=r"$m_X$ [GeV]" ):
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

def plotResBkgComparison2(mx, limits, limits_no_sys, ylabel, nominal_masses, savename, xlabel=r"$m_X$ [GeV]"):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  ratio = limits[2]/limits_no_sys[2]
  
  axs[0].plot(mx, limits[2], zorder=3, label="Median expected")
  axs[0].plot(mx, limits_no_sys[2], zorder=3, label="Median expected (no res bkg)")

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

def plotDYBkgComparison2(mx, limits, limits_no_sys, ylabel, nominal_masses, savename, xlabel=r"$m_X$ [GeV]"):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  ratio = limits[2]/limits_no_sys[2]
  
  axs[0].plot(mx, limits[2], zorder=3, label="Median expected")
  axs[0].plot(mx, limits_no_sys[2], zorder=3, label="Median expected (no dy bkg)")

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
  df = pd.DataFrame({"MX": masses[:,0], "MY": masses[:,1], "Median expected [fb]": limits[2]})
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

def checkForExcess(masses, limits):
  for i, limit_set in enumerate(limits.T):
    if limit_set[5] > limit_set[4]:
      pseudo_significance = (limit_set[5]-limit_set[2]) / (limit_set[3]-limit_set[2])
      print(f"Found excess at {masses[i]} with pseudo significance of {pseudo_significance}")
      #print(limit_set)

masses, limits, limits_no_sys, limits_no_res_bkg, limits_no_dy_bkg = getLimits(sys.argv[1])
print(masses)
print(limits.shape)
print(limits)
print(limits.T[:10])
print(list(limits[6]))
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_br"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_br_nominal"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_br_no_sys"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_sys"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_br_no_res_bkg"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_res_bkg"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_br_no_dy_bkg"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_dy_bkg"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_systematics_comparison"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_res_bkg_comparison"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_dy_bkg_comparison"), exist_ok=True)

os.makedirs(os.path.join(sys.argv[2], "Significance"), exist_ok=True)

checkForExcess(masses, limits)

tabulateLimits(masses, limits, os.path.join(sys.argv[2], "Limits_xs_br"))
tabulateLimitsAll(masses, limits, limits_no_sys, limits_no_res_bkg, os.path.join(sys.argv[2], "Limits_xs_br"))
tabulateLimits(masses, limits / BR_HH_GGTT, os.path.join(sys.argv[2], "Limits_xs"))
tabulateLimitsAll(masses, limits/BR_HH_GGTT, limits_no_sys/BR_HH_GGTT, limits_no_res_bkg/BR_HH_GGTT, os.path.join(sys.argv[2], "Limits_xs"))

tabulateLimits(masses, limits_no_sys, os.path.join(sys.argv[2], "Limits_xs_br_no_sys"))
tabulateLimits(masses, limits_no_sys / BR_HH_GGTT, os.path.join(sys.argv[2], "Limits_xs_no_sys"))

tabulateLimits(masses, limits_no_res_bkg, os.path.join(sys.argv[2], "Limits_xs_br_no_res_bkg"))
tabulateLimits(masses, limits_no_res_bkg / BR_HH_GGTT, os.path.join(sys.argv[2], "Limits_xs_no_res_bkg"))

if len(np.unique(masses[:,1])) == 1: #if 1D (graviton or radion)
  mx = masses[:,0]
  limits = limits[:,np.argsort(mx)]
  limits_no_sys = limits_no_sys[:,np.argsort(mx)]
  limits_no_res_bkg = limits_no_res_bkg[:,np.argsort(mx)]
  limits_no_dy_bkg = limits_no_dy_bkg[:,np.argsort(mx)]

  mx = mx[np.argsort(mx)]

  nominal_masses = [260,270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900,1000]
  
  plotSignificance(mx, limits, "Significance", nominal_masses, os.path.join(sys.argv[2], "Significance", "significance"))

  #ylabel = r"95% CL limit on $\sigma(pp \rightarrow X) B(X \rightarrow HH \rightarrow \gamma\gamma\tau\tau)$ [fb]"
  ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow HH \rightarrow \gamma\gamma\tau\tau)$ [fb]"
  plotLimits(mx, limits, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs_br", "limits"))
  plotLimits(mx, limits_no_sys, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs_br_no_sys", "limits_no_sys"))
  plotLimits(mx, limits_no_res_bkg, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs_br_no_res_bkg", "limits_no_res_bkg"))

  #ylabel = r"95% CL limit on $\sigma(pp \rightarrow X) B(X \rightarrow HH)$ [fb]"
  ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow HH)$ [fb]"
  plotLimits(mx, limits / BR_HH_GGTT, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs", "limits"))
  plotLimits(mx, limits_no_sys / BR_HH_GGTT, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs_no_sys", "limits_no_sys"))
  plotLimits(mx, limits_no_res_bkg / BR_HH_GGTT, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_xs_no_res_bkg", "limits_no_res_bkg"))

  plotSystematicComparison(mx, limits, limits_no_sys, nominal_masses, os.path.join(sys.argv[2], "Limits_systematics_comparison", "125"))
  ylabel = r"95% CL limit on $\sigma(pp \rightarrow X) B(X \rightarrow HH \rightarrow \gamma\gamma\tau\tau)$ [fb]"
  plotSystematicComparison2(mx, limits, limits_no_sys, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_systematics_comparison", "125_2"))
  plotResBkgComparison2(mx, limits, limits_no_res_bkg, ylabel, nominal_masses, os.path.join(sys.argv[2], "Limits_res_bkg_comparison", "125_2"))

else:
  nominal_mx = [300,400,500,600,700,800,900,1000]
  nominal_my = [70,80,90,100,125]
  #nominal_my = [50,70,80,90,100,125,150,200,250,300,400,500,600,700,800]
  #nominal_my = [125,150,200,250,300,400,500,600,700,800]

  #only grab the nominal points
  # s = np.isin(masses[:,0], nominal_mx) & np.isin(masses[:,1], nominal_my)
  # limits = limits[:, s]
  # limits_no_sys = limits_no_sys[:, s]
  # limits_no_res_bkg = limits_no_res_bkg[:, s]
  # masses = masses[s]
  # tabulateLimitsAll(masses, limits, limits_no_sys, limits_no_res_bkg, os.path.join(sys.argv[2], "Limits_xs_br"))

  plotSignificance2D(masses, limits, "Significance", os.path.join(sys.argv[2], "Significance", "significance"))

  ylabel = r"95% CL limit on $\sigma(pp \rightarrow X) B(X \rightarrow YH \rightarrow \gamma\gamma\tau\tau)$ [fb]"
  plotLimitsStackMX(masses, limits,             ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_br", "limits_stack_mx"))
  plotLimitsStackMX(masses, limits_no_sys,      ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_br_no_sys", "limits_stack_mx_no_sys"))
  plotLimitsStackMX(masses, limits_no_res_bkg,  ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_br_no_res_bkg", "limits_stack_mx_no_res_bkg"))
  plotLimitsStackMY(masses, limits,             ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_br", "limits_stack_my"))
  plotLimitsStackMY(masses, limits_no_sys,      ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_br_no_sys", "limits_stack_my_no_sys"))
  plotLimitsStackMY(masses, limits_no_res_bkg,  ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[2], "Limits_xs_br_no_res_bkg", "limits_stack_my_no_res_bkg"))
  plotLimits2D(masses, limits,        ylabel, os.path.join(sys.argv[2], "Limits_xs_br", "limits_2d"))
  plotLimits2D(masses, limits,        ylabel, os.path.join(sys.argv[2], "Limits_xs_br", "limits_2d_obs"), observed=True)
  plotLimits2D(masses, limits_no_sys, ylabel, os.path.join(sys.argv[2], "Limits_xs_br_no_sys", "limits_2d_no_sys"))
  plotLimits2D(masses, limits_no_res_bkg, ylabel, os.path.join(sys.argv[2], "Limits_xs_br_no_res_bkg", "limits_2d_no_res_bkg"))

  for mx in nominal_mx:
    my = masses[masses[:,0]==mx,1]
    limits_slice = limits[:,masses[:,0]==mx]
    limits_no_sys_slice = limits_no_sys[:,masses[:,0]==mx]
    limits_no_res_bkg_slice = limits_no_res_bkg[:,masses[:,0]==mx]
    limits_no_dy_bkg_slice = limits_no_dy_bkg[:,masses[:,0]==mx]

    limits_slice = limits_slice[:,np.argsort(my)]
    limits_no_sys_slice = limits_no_sys_slice[:,np.argsort(my)]
    limits_no_res_bkg_slice = limits_no_res_bkg_slice[:,np.argsort(my)]
    limits_no_dy_bkg_slice = limits_no_dy_bkg_slice[:,np.argsort(my)]
    my = my[np.argsort(my)]

    if mx in nominal_mx:
      nm = nominal_my
    else:
      nm = []

    print(mx)
    print(my, limits_slice)

    ylabel = r"$\sigma(pp \rightarrow X(%d)) B(X \rightarrow YH \rightarrow \gamma\gamma\tau\tau)$ [fb]"%mx
    plotLimits(my, limits_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br", "limits_mx%d"%mx), xlabel=r"$m_Y$ [GeV]")
    plotLimits(my, limits_no_sys_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br_no_sys", "limits_mx%d_no_sys"%mx), xlabel=r"$m_Y$ [GeV]")
    plotLimits(my, limits_no_res_bkg_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br_no_res_bkg", "limits_mx%d_no_res_bkg"%mx), xlabel=r"$m_Y$ [GeV]")
    plotLimits(my, limits_no_dy_bkg_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br_no_dy_bkg", "limits_mx%d_no_dy_bkg"%mx), xlabel=r"$m_Y$ [GeV]")
    plotSystematicComparison(my, limits_slice, limits_no_sys_slice, nm, os.path.join(sys.argv[2], "Limits_systematics_comparison", "mx%d"%mx), xlabel=r"$m_Y$ [GeV]")
    plotSystematicComparison2(my, limits_slice, limits_no_sys_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_systematics_comparison", "mx%d_2"%mx), xlabel=r"$m_Y$ [GeV]")
    plotResBkgComparison2(my, limits_slice, limits_no_res_bkg_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_res_bkg_comparison", "mx%d_2"%mx), xlabel=r"$m_Y$ [GeV]")
    plotDYBkgComparison2(my, limits_slice, limits_no_dy_bkg_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_dy_bkg_comparison", "mx%d_2"%mx), xlabel=r"$m_Y$ [GeV]")

  for my in nominal_my:
    mx = masses[masses[:,1]==my,0]
    limits_slice = limits[:,masses[:,1]==my]
    limits_no_sys_slice = limits_no_sys[:,masses[:,1]==my]
    limits_no_res_bkg_slice = limits_no_res_bkg[:,masses[:,1]==my]
    limits_no_dy_bkg_slice = limits_no_dy_bkg[:,masses[:,1]==my]


    limits_slice = limits_slice[:,np.argsort(mx)]
    limits_no_sys_slice = limits_no_sys_slice[:,np.argsort(mx)]
    limits_no_res_bkg_slice = limits_no_res_bkg_slice[:,np.argsort(mx)]
    limits_no_dy_bkg_slice = limits_no_dy_bkg_slice[:,np.argsort(mx)]
    mx = mx[np.argsort(mx)]

    if my in nominal_my:
      nm = nominal_mx
    else:
      nm = []

    ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow Y(%d)H \rightarrow \gamma\gamma\tau\tau)$ [fb]"%my
    plotLimits(mx, limits_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br", "limits_my%d"%my))
    plotLimits(mx, limits_no_sys_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br_no_sys", "limits_my%d_no_sys"%my))
    plotLimits(mx, limits_no_res_bkg_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br_no_res_bkg", "limits_my%d_no_res_bkg"%my))
    plotLimits(mx, limits_no_dy_bkg_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_xs_br_no_dy_bkg", "limits_my%d_no_dy_bkg"%my))
    plotSystematicComparison(mx, limits_slice, limits_no_sys_slice, nm, os.path.join(sys.argv[2], "Limits_systematics_comparison", "my%d"%my))
    plotSystematicComparison2(mx, limits_slice, limits_no_sys_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_systematics_comparison", "my%d_2"%my))
    plotResBkgComparison2(mx, limits_slice, limits_no_res_bkg_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_res_bkg_comparison", "my%d_2"%my))
    plotDYBkgComparison2(mx, limits_slice, limits_no_dy_bkg_slice, ylabel, nm, os.path.join(sys.argv[2], "Limits_dy_bkg_comparison", "my%d_2"%my))

  