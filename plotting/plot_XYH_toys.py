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
NMSSM_max_allowed_Y_gg.loc[:, "limit"] /= BR_H_TT

def getAnalysis():
  if "y_tautau" in sys.argv[1].lower():
    return "Y_tautau"
  elif "low_mass" in sys.argv[1].lower():
    return "Y_gg_Low_Mass"
  elif "high_mass" in sys.argv[1].lower():
    return "Y_gg_High_Mass"
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

  if getAnalysis() == "Y_tautau":
    limits[:6] *= BR_H_TT
    limitsErr[:6] *= BR_H_TT

  # remove overlapping MH points
  if getAnalysis() in ["Y_gg_High_Mass", "Y_gg_Low_Mass"]:
    for mx in np.unique(masses[:,0]):
      uniques, counts = np.unique(masses[masses[:,0]==mx, 2], return_counts=True)
      assert sum(counts>2) == 0, str(mx) + " " + str(uniques[counts>2]) #should not have more than 1 overlap
      overlap_mh = uniques[counts==2]

      for mh in overlap_mh:
        idx1, idx2 = np.where( (masses[:,0]==mx) & (masses[:,2]==mh) )[0]
        # delete point with worse expected limit
        if limits[2][idx1] < limits[2][idx2]:
          to_delete = idx2
        else:
          to_delete = idx1
        masses = np.delete(masses, to_delete, axis=0)
        limits = np.delete(limits, to_delete, axis=1)
        limitsErr = np.delete(limitsErr, to_delete, axis=1)

    masses[:,1] = masses[:,2] #set my to be mh

  return masses, limits, limitsErr


def plotLimitsStack(masses, limits, ylabel, nominal_mx, nominal_my, savename, swap=False):
  label2 = "Median expected"
  label3 = "68% expected"
  label4 = "95% expected"
  label5 = "Observed"

  if not swap:
    xlabel = r"$m_Y$ [GeV]"
    mx_idx = 0
    my_idx = 1
    #nominal_stack_exceptions = [650]
    nominal_stack_exceptions = []
    xlims = (None, 1000)
    
    if getAnalysis() == "Y_tautau":
      xlims = (None, 1000)
      ylims = (0.1, 1e8)
      label_dist = 10
    elif getAnalysis() == "Y_gg_High_Mass":
      xlims = (None, 1000)
      ylims = (1, 1e9)
      label_dist = 10
    elif getAnalysis() == "Y_gg_Low_Mass":
      xlims = (None, 145)
      ylims = (1, 1e9)
      label_dist = 2.5
    else:
      raise Exception()
  else:
    xlabel = r"$m_X$ [GeV]"
    mx_idx = 1
    my_idx = 0
    nominal_stack_exceptions = [90, 95, 100]
    nominal_mx, nominal_my = nominal_my, nominal_mx
    xlims = (None, 1200)
    label_dist = 10
    if getAnalysis() == "Y_tautau":
      ylims = (0.01, 1e15)
    elif getAnalysis() == "Y_gg_High_Mass":
      ylims = (0.1, 1e10)
    elif getAnalysis() == "Y_gg_Low_Mass":
      ylims = (0.1, 1e8)
    else:
      raise Exception()
    
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

    print(mx)

    if mx == 650:
      ss = [my<300, my>300]
    else:
      ss = [my > 0]

    if getAnalysis() == "Y_gg_High_Mass" and mx == 650:
      print("haha")
      ss = [my<300, my>300]
    else:
      ss = [my > 0]

    for s in ss:
      plt.plot(my[s], limits_slice[5][s], 'ko-', lw=3, zorder=4, label=label5)
      plt.plot(my[s], limits_slice[2][s], 'k--', zorder=3, label=label2)
      if len(my) > 1:
        plt.fill_between(my[s], limits_slice[1][s], limits_slice[3][s], zorder=2, facecolor=(0, 0.8, 0), label=label3)
        plt.fill_between(my[s], limits_slice[0][s], limits_slice[4][s], zorder=1, facecolor=(1, 0.8, 0), label=label4)
      else:
        plt.fill_between([my[0]-10]+list(my), list(limits_slice[1])*2, list(limits_slice[3])*2, zorder=2, facecolor=(0, 0.8, 0), label=label3)
        plt.fill_between([my[0]-10]+list(my), list(limits_slice[0])*2, list(limits_slice[4])*2, zorder=1, facecolor=(1, 0.8, 0), label=label4)
      label1 = label2 = label3 = label4 = label5 = None

    if not swap:
      plt.text(my[-1]+label_dist, limits_slice[2][-1], r"$m_X=%d$ GeV $(\times 10^{%d})$"%(mx, j), fontsize=12, verticalalignment="center")
    else:
      plt.text(my[-1]+label_dist, limits_slice[2][-1], r"$m_Y=%d$ GeV $(\times 10^{%d})$"%(mx, j), fontsize=12, verticalalignment="center")

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)  
  plt.legend(ncol=2, loc="upper left")
  bottom, top = plt.ylim()
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

def getMaxAllowedValues(masses):
  # spline_log = spi.interp2d(NMSSM_max_allowed_Y_gg.MX, NMSSM_max_allowed_Y_gg.MY, np.log(NMSSM_max_allowed_Y_gg.limit), kind='linear', fill_value=-np.inf)
  # spline = lambda m: np.exp(spline_log(m[0], m[1])[0])
  # max_allowed_values = np.array([spline(m) for m in masses])

  max_allowed_x_y = np.array([NMSSM_max_allowed_Y_gg.MX, NMSSM_max_allowed_Y_gg.MY]).T
  max_allowed_values = np.exp(spi.griddata(max_allowed_x_y, np.log(NMSSM_max_allowed_Y_gg.limit), masses[:,:2], method="linear", fill_value=-np.inf))
  return max_allowed_values

def getInterpMassesEdges(masses, n_mx=None, n_my=None):
  mx = np.sort(np.unique(masses[:,0]))
  my = np.sort(np.unique(masses[:,1]))

  # if n_mx is None:
  #   mx_spacing = np.min(np.diff(mx))
  # else:
  #   mx_spacing = (np.max(mx) - np.min(mx)) / n_mx

  # if n_my is None:
  #   my_spacing = np.min(np.diff(my))
  # else:
  #   my_spacing = (np.max(my) - np.min(my)) / n_my
  
  # mx_edges = np.arange(min(mx)-mx_spacing/2, max(mx)+1.5*mx_spacing, mx_spacing)
  # my_edges = np.arange(min(my)-my_spacing/2, max(my)+1.5*my_spacing, my_spacing)

  if n_mx is None:
    mx_edges = np.array([mx[0] - (mx[1]-mx[0])/2] + list(mx[:-1] + (mx[1:] - mx[:-1])/2) + [mx[-1] + (mx[-1]-mx[-2])/2])
  else:
    #mx_spacing = np.min(np.diff(mx))
    mx_spacing = (np.max(mx) - np.min(mx)) / n_mx
    mx_edges = np.arange(min(mx)-mx_spacing/2, max(mx)+1.5*mx_spacing, mx_spacing)

  if n_my is None:
    my_edges = np.array([my[0] - (my[1]-my[0])/2] + list(my[:-1] + (my[1:] - my[:-1])/2) + [my[-1] + (my[-1]-my[-2])/2])
  else:
    #my_spacing = np.min(np.diff(my))
    my_spacing = (np.max(my) - np.min(my)) / n_my
    my_edges = np.arange(min(my)-my_spacing/2, max(my)+1.5*my_spacing, my_spacing)

  mx_edge_centers = (mx_edges[:-1]+mx_edges[1:])/2
  my_edge_centers = (my_edges[:-1]+my_edges[1:])/2
  interp_masses = []
  for mxi in mx_edge_centers:
    for myi in my_edge_centers:
      interp_masses.append([mxi, myi])
  interp_masses = np.array(interp_masses)
  return interp_masses, mx_edges, my_edges

def getPolygonCoords(masses, is_below):
  mxs = np.sort(np.unique(masses[:,0]))
  mys = np.sort(np.unique(masses[:,1]))

  mx_norm = max(mxs) - min(mxs)
  my_norm = max(mys) - min(mys)
  
  print(mxs)
  print(mys)
  print(is_below)

  def is_below_m(i, j):
    return is_below[(masses[:,0]==mxs[i]) & (masses[:,1]==mys[j])][0]

  coords = []

  for i, mx in enumerate(mxs):
    for j, my in enumerate(mys):
      if not is_below_m(i, j):
        continue
      elif (i==0 or i==len(mxs)-1 or j==0 or j==len(mys)-1):
        print([mx, my], "is on edge")
        coords.append([mx, my])
      elif not is_below_m(i-1, j) or not is_below_m(i+1, j) or not is_below_m(i, j-1) or not is_below_m(i, j+1):
        print([mx, my], "borders above limit")
        coords.append([mx, my])

  sorted_coords = []

  for i in range(len(coords)-1):
    if i == 0:
      next_idx = 0
    else:
      coords_np = np.array(coords)
      previous_coord = sorted_coords[-1]
      dist = np.sqrt( ((previous_coord[0]-coords_np[:,0]) / mx_norm)**2 + ((previous_coord[1]-coords_np[:,1]) / my_norm)**2 )
      next_idx = np.argmin(dist)

    sorted_coords.append(coords[next_idx])
    del coords[next_idx]

  return np.array(sorted_coords)

def plotLimits2D(masses, limits, ylabel, savename, observed=False):
  MX_norm = max(masses[:,0]) - min(masses[:,0])
  MY_norm = max(masses[:,1]) - min(masses[:,1])
  interp_masses, mx_edges, my_edges = getInterpMassesEdges(masses)
  
  if not observed:
    limits_idx = 2
  else:
    limits_idx = 5
  
  masses_normed = masses[:,:2] / [MX_norm, MY_norm]
  interp_masses_normed = interp_masses / [MX_norm, MY_norm]

  interp_limits = spi.griddata(masses_normed, limits[limits_idx], interp_masses_normed, method="linear", fill_value=0)

  for i, m in enumerate(interp_masses):
    # closest_mx = masses[np.argmin(abs(masses[:,0] - m[0])),0]
    # if m[1] > np.max(masses[masses[:,0]==closest_mx,1]):
    #   interp_limits[i] = 0
    if m[0] - m[1] < 125:
      interp_limits[i] = 0
  plt.hist2d(interp_masses[:,0], interp_masses[:,1], [mx_edges, my_edges], weights=interp_limits, norm=matplotlib.colors.LogNorm())  
  
  df = pd.DataFrame({"mx":interp_masses[:,0], "my":interp_masses[:,1], "limit":interp_limits})
  df.to_csv("test.csv")

  cbar = plt.colorbar()
  cbar.set_label(ylabel)
  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel(r"$m_Y$ [GeV]")

  plt.xlim(min(masses[:,0]), max(masses[:,0]))
  plt.ylim(min(masses[:,1]), max(masses[:,1]))

  if getAnalysis() == "Y_gg_Low_Mass":
    contour_masses = getInterpMassesEdges(masses, n_mx=70, n_my=70)[0]
    contour_masses_normed = contour_masses / [MX_norm, MY_norm]
    contour_limits = spi.griddata(masses_normed, limits[limits_idx], contour_masses_normed, method="nearest", fill_value=0)
    s = contour_limits < getMaxAllowedValues(contour_masses)

    coords = getPolygonCoords(contour_masses, s)
    poly = matplotlib.patches.Polygon(coords, edgecolor="r", hatch="x", facecolor="none", label="Limit below maximally\nallowed in NMSSM")
    plt.gca().add_patch(poly)
    plt.legend(frameon=True)

    #plt.scatter(contour_masses[s,0], contour_masses[s,1], marker='.', color="r", label="Limit below maximally allowed in NMSSM") 
    #plt.scatter(contour_masses[~s,0], contour_masses[~s,1], marker='.', color="b", label="Limit below maximally allowed in NMSSM") 


  mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_paper.png")
  plt.savefig(savename+"_paper.pdf")
  mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_preliminary.png")
  plt.savefig(savename+"_preliminary.pdf")
  
  plt.clf()

def plotMaxLimitsSpline(masses, limits, ylabel, savename, observed=False):
  interp_masses, mx_edges, my_edges = getInterpMassesEdges(masses)

  max_allowed_values = getMaxAllowedValues(interp_masses)

  plt.hist2d(interp_masses[:,0], interp_masses[:,1], [mx_edges, my_edges], weights=max_allowed_values, norm=matplotlib.colors.LogNorm())
  cbar = plt.colorbar()
  plt.savefig(savename+"_max_limits_spline.png")
  plt.savefig(savename+"_max_limits_spline.pdf")
  plt.clf()


def plotSignificance2D(masses, limits, ylabel, savename):
  mx = np.sort(np.unique(masses[:,0]))
  my = np.sort(np.unique(masses[:,1]))
  mx_edges = np.array([mx[0] - (mx[1]-mx[0])/2] + list(mx[:-1] + (mx[1:] - mx[:-1])/2) + [mx[-1] + (mx[-1]-mx[-2])/2])
  my_edges = np.array([my[0] - (my[1]-my[0])/2] + list(my[:-1] + (my[1:] - my[:-1])/2) + [my[-1] + (my[-1]-my[-2])/2])
  
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
  
  #max_sig_str = "Max. Sig. of %.2f at\n"%max_sig + r"$(m_X, m_Y) = (%d, %d)$"%(max_mass[0], max_mass[1])
  #plt.text(0.55, 0.85, max_sig_str, transform=plt.gca().transAxes, fontsize=24, color="black")
  
  mplhep.cms.label(llabel="Supplementary", data=True, lumi=common.tot_lumi, loc=0)
  plt.savefig(savename+"_supplementary.png")
  plt.savefig(savename+"_supplementary.pdf")
  # mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=0)
  # plt.savefig(savename+"_preliminary.png")
  # plt.savefig(savename+"_preliminary.pdf")
  plt.clf()

masses, limits, limitsErr = getLimits(sys.argv[1], sys.argv[2])

if getAnalysis() == "Y_tautau":
  s = (masses[:,0] == 1000) & (masses[:,1] == 750)
  masses = masses[~s]
  limits = limits.T[~s].T
  limitsErr = limitsErr.T[~s].T


os.makedirs(os.path.join(sys.argv[3], "Limits_xs"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[3], "Significance"), exist_ok=True)

nominal_mx = [300,400,500,600,700,800,900,1000]

if getAnalysis() == "Y_tautau":
  nominal_my = [50,70,80,90,100,125,150,200,250,300,400,500,600,700,800]
elif getAnalysis() == "Y_gg_Low_Mass":
  nominal_my = [70,80,90,100,125]
elif getAnalysis() == "Y_gg_High_Mass":
  nominal_my = [125,150,200,250,300,400,500,600,700,800]
else:
  raise Exception()

plotSignificance2D(masses, limits, "Significance", os.path.join(sys.argv[3], "Significance", f"significance_{getAnalysis().lower()}"))

def outputLimits(masses, limits, limitsErr, outdir):
  df = pd.DataFrame({"MX": masses[:,0], "MY": masses[:,1], "exp": limits[2], "obs": limits[5], "obsErr": limitsErr[5],
                     "limits_m2": limits[0], "limits_m1": limits[1],
                     "limits_p1": limits[3], "limits_p2": limits[4]})
  df.to_csv(os.path.join(outdir, "limits.csv"), float_format="%.4f", index=False)

outputLimits(masses, limits, limitsErr, os.path.join(sys.argv[3], "Limits_xs"))

if getAnalysis() == "Y_tautau":
  ylabel = r"95% CL limit on $\sigma(pp \rightarrow X)B(X \rightarrow YH \rightarrow \gamma\gamma\tau\tau)$ [fb]"
else:
  ylabel = r"95% CL limit on $\sigma(pp \rightarrow X \rightarrow YH) B(Y \rightarrow \gamma\gamma)$ [fb]"

plotLimitsStack(masses, limits, ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[3], "Limits_xs", f"limits_stack_mx_{getAnalysis().lower()}"))
plotLimitsStack(masses, limits, ylabel, nominal_mx, nominal_my, os.path.join(sys.argv[3], "Limits_xs", f"limits_stack_my_{getAnalysis().lower()}"), swap=True)

if getAnalysis() == "Y_gg_Low_Mass":
  plotMaxLimitsSpline(masses, limits, ylabel, os.path.join(sys.argv[3], "Limits_xs", "limits_2d"))

plotLimits2D(masses, limits,  ylabel, os.path.join(sys.argv[3], "Limits_xs", f"limits_2d_{getAnalysis().lower()}"))
plotLimits2D(masses, limits,  ylabel, os.path.join(sys.argv[3], "Limits_xs", f"limits_2d_obs_{getAnalysis().lower()}"), observed=True)