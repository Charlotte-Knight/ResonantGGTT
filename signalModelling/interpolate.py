import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

import scipy.interpolate as spi

from signalModelling.signal_fit import fitDCB
import signalModelling.signal_fit as signal_fit

import argparse
import os
import json
import common
import sys
import copy

mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

class interp1d(spi.interp1d):
  def __init__(self, x, y, kind='cubic', bounds_error=True):
    if len(x) < 4:
      super().__init__(x, y, kind='linear', bounds_error=bounds_error)
    else:
      super().__init__(x, y, kind, bounds_error=bounds_error)
  def __call__(self, x, y=None):
    return super().__call__(x)

class interp2d():
  def __init__(self, x, y, z, kind='cubic', bounds_error=True):
    self.points = np.array([[x[i],y[i]] for i in range(len(x))])
    self.values = np.array(z)
    self.kind = kind

  def __call__(self, x, y):
    if (type(x) != np.ndarray) and (type(y) != np.ndarray):
      return spi.griddata(self.points, self.values, np.array([[x,y]]), method=self.kind)[0]
    elif type(x) != np.ndarray:
      return spi.griddata(self.points, self.values, np.array([[x,y[i]] for i in range(len(y))]), method=self.kind)
    elif type(y) != np.ndarray:
      return spi.griddata(self.points, self.values, np.array([[x[i],y] for i in range(len(x))]), method=self.kind)
    else:
      return spi.griddata(self.points, self.values, np.array([[x[i],y[i]] for i in range(len(y))]), method=self.kind)

def interp(x, y, z, kind='cubic', bounds_error=True):
  if len(np.unique(x)) == 1:
    return interp1d(y, z, kind, bounds_error)
  elif len(np.unique(y)) == 1:
    return interp1d(x, z, kind, bounds_error)
  else:
    return interp2d(x, y, z, kind, bounds_error)

def fitSignalModel(df, mx, outdir, my=125, fit_range_my=False, make_plots=False):  
  if sum((df.MX==mx)&(df.MY==my)) < 100:
    #find closest mass with > 100 events
    #print(mx, my)
    #print(df)
    #print(df[df.MY==my])
    possible_masses, counts = np.unique(df[df.MY==my].MX, return_counts=True)
    possible_masses_100 = possible_masses[counts>=100]
    if len(possible_masses_100) > 0:
      mx = possible_masses_100[np.argmin(abs(possible_masses_100-mx))]
    else:
      mx = possible_masses[np.argmax(counts)]
    print("Using mx=%d as substitute"%mx)

  if fit_range_my:
    fit_range = [my-10*(my/100),my+10*(my/100)] #if y->gg
  else:
    fit_range = [112.5, 137.5] #everything else

  if make_plots: 
    os.makedirs(outdir, exist_ok=True)
    popt, perr = fitDCB(df[(df.MX==mx)&(df.MY==my)], fit_range=fit_range, savepath=os.path.join(outdir, "mx_%d_my_%d.png"%(mx,my)))
  else:          
    popt, perr = fitDCB(df[(df.MX==mx)&(df.MY==my)], fit_range=fit_range, savepath=None)

  if fit_range_my:
    popt[1] = popt[1] - my #change mean to a delta (dm)
  else:
    popt[1] = popt[1] - 125.0

  return popt, perr

def doSkipClosest(MX, MY, to_fit_masses, norms_nominal):
  unique_mx = np.unique(to_fit_masses[:, 0])
  unique_my = np.unique(to_fit_masses[:, 1])
  argx = interp1d(unique_mx, np.arange(len(unique_mx)), kind='linear')
  argy = lambda x: interp1d(unique_my, np.arange(len(unique_my)), kind='linear')(x) if len(unique_my)>1 else 0

  #find closest points on the grid
  dist = np.sqrt((MX-to_fit_masses[:,0])**2 + (MY-to_fit_masses[:,1])**2)
  dist_grid = np.sqrt((argx(MX) - argx(to_fit_masses[:,0]))**2 + (argy(MY) - argy(to_fit_masses[:,1]))**2)
  idxs = list(np.argsort(dist_grid)[::-1])
  
  #only keep those which are not on edge of grid
  if len(np.unique(to_fit_masses[:,1])) == 1: #if 1d interp:
    not_okay = lambda idx: (to_fit_masses[idx,0] == min(to_fit_masses[:,0])) or (to_fit_masses[idx,0] == max(to_fit_masses[:,0]))
  else:
    #is corner piece in grid
    is_edge = lambda idx: ((to_fit_masses[idx,0] == min(to_fit_masses[:,0])) or (to_fit_masses[idx,0] == max(to_fit_masses[:,0])))   and   ((to_fit_masses[idx,1] == min(to_fit_masses[:,1])) or (to_fit_masses[idx,1] == max(to_fit_masses[:,1])))
    is_corner = lambda idx: (to_fit_masses[idx,1]) == max(to_fit_masses[to_fit_masses[:,0]==to_fit_masses[idx,0],1])
    not_okay = lambda idx: is_edge(idx) or is_corner(idx)
  idx = idxs.pop()
  while not_okay(idx):
    idx = idxs.pop()

  not_idx = lambda arr, idx: arr[np.arange(len(arr))!=idx]
  m = not_idx(to_fit_masses, idx)
  n = not_idx(norms_nominal, idx)

  skipped_mass = to_fit_masses[idx]

  return m, n, skipped_mass

def doInterpolation(MX, MY, to_fit_masses, norms_nominal, skip_closest=False):
  if skip_closest:
    m, n, skipped_mass = doSkipClosest(MX, MY, to_fit_masses, norms_nominal)
  else:
    m = to_fit_masses
    n = norms_nominal
    skipped_mass = None

  norm_cubic_spline = interp(m[:,0], m[:,1], n, kind='cubic')
  norm_linear_spline = interp(m[:,0], m[:,1], n, kind='linear')

  norm_cubic = float(norm_cubic_spline(MX, MY))
  norm_linear = float(norm_linear_spline(MX, MY))

  return norm_cubic, norm_linear, skipped_mass

def getToFitMasses(MX, MY, optim_results, proc_dict, only_nominal):
  all_masses = [common.get_MX_MY(entry["sig_proc"]) for entry in optim_results]
  nominal_masses = [common.get_MX_MY(entry["sig_proc"]) for entry in optim_results if entry["sig_proc"] in proc_dict.keys()]
  if ((MX, MY) in nominal_masses) and (only_nominal):
    return [[MX, MY]]
  
  nominal_masses = np.array(nominal_masses)

  nominal_mxs = np.sort(np.array(list(set(nominal_masses[:,0]))))
  nominal_mys = np.sort(np.array(list(set(nominal_masses[:,1]))))
  gridx = np.arange(len(nominal_mxs))
  gridy = np.arange(len(nominal_mys))
  #argx and argy tell us where we are in grid coordinates as function of mx and my
  argx = lambda x: interp1d(nominal_mxs, gridx, kind='linear')(x) if len(nominal_mxs)>1 else 0
  argy = lambda x: interp1d(nominal_mys, gridy, kind='linear')(x) if len(nominal_mys)>1 else 0

  to_fit_mxs = nominal_mxs[gridx[(gridx>=argx(MX)-3)&(gridx<=argx(MX)+3)]] #closest mx within +-r points
  to_fit_mys = nominal_mys[gridy[(gridy>=argy(MY)-3)&(gridy<=argy(MY)+3)]] #closest mx within +-r points
  to_fit_masses = [[mx, my] for mx in to_fit_mxs for my in to_fit_mys if (mx, my) in all_masses]

  return to_fit_masses

def deriveModels(original_df, proc_dict, optim_results, original_outdir, make_plots=False, do_same_score_interp=False, fit_range_my=False, masses_to_do=None):
  #derive model for every mass point which you can find from optim_results

  all_masses = [common.get_MX_MY(entry["sig_proc"]) for entry in optim_results]
  nominal_masses = np.array([common.get_MX_MY(entry["sig_proc"]) for entry in optim_results if entry["sig_proc"] in proc_dict.keys()])

  nominal_mxs = np.sort(np.array(list(set(nominal_masses[:,0]))))
  nominal_mys = np.sort(np.array(list(set(nominal_masses[:,1]))))
  gridx = np.arange(len(nominal_mxs))
  gridy = np.arange(len(nominal_mys))
  #argx and argy tell us where we are in grid coordinates as function of mx and my
  argx = lambda x: interp1d(nominal_mxs, gridx, kind='linear')(x) if len(nominal_mxs)>1 else 0
  argy = lambda x: interp1d(nominal_mys, gridy, kind='linear')(x) if len(nominal_mys)>1 else 0

  nSR = len(optim_results[0]["category_boundaries"]) - 1
  models = {str(year):{str(SR):{} for SR in range(nSR)} for year in np.unique(original_df.year)}

  for year in np.unique(original_df.year):
    print(year)
    df_year = original_df[original_df.year==year]
    
    for entry in optim_results[:]:
      MX, MY = common.get_MX_MY(entry["sig_proc"])
      if (masses_to_do != None) and ([MX, MY] not in masses_to_do): 
        continue
      print(MX, MY)

      closest_mx = nominal_mxs[gridx[np.argmin(abs(gridx-argx(MX)))]]
      closest_my = nominal_mys[gridy[np.argmin(abs(gridy-argy(MY)))]]

      is_nominal_mass = (MX==closest_mx) and (MY==closest_my)
      if is_nominal_mass: r = 1
      else:               r = 3

      to_fit_mxs = nominal_mxs[gridx[(gridx>=argx(MX)-r)&(gridx<=argx(MX)+r)]] #closest mx within +-r points
      to_fit_mys = nominal_mys[gridy[(gridy>=argy(MY)-r)&(gridy<=argy(MY)+r)]] #closest mx within +-r points

      to_fit_masses = [[mx, my] for mx in to_fit_mxs for my in to_fit_mys if (mx, my) in all_masses] #convenient to leave as list for tagSignals
      df_tagged = tagSignals(df_year, entry, proc_dict, to_fit_masses)
      to_fit_masses = np.array(to_fit_masses)

      #for interpolation at combine level in MH (MY)
      if do_same_score_interp: 
        to_fit_mxs_same_score = [closest_mx]
        to_fit_mys_same_score = nominal_mys[gridy[(gridy>=argy(closest_my)-r)&(gridy<=argy(closest_my)+r)]] #closest my and +- 1 points
        to_fit_masses_same_score = [[mx, my] for mx in to_fit_mxs_same_score for my in to_fit_mys_same_score if (mx, my) in all_masses]
        df_tagged_same_score = tagSignals(df_year, entry, proc_dict, to_fit_masses_same_score, use_same_score=True)
        to_fit_masses_same_score = np.array(to_fit_masses_same_score)
        print(to_fit_mys)
        print(to_fit_masses_same_score)

      for SR in range(len(entry["category_boundaries"])-1):
        print(SR, flush=True)
        df = df_tagged[df_tagged.SR==SR]
        if do_same_score_interp: 
          df_same_score = df_tagged_same_score[df_tagged_same_score.SR==SR]
          assert len(df_same_score) != 0, print(MX, MY, SR)

        outdir = os.path.join(original_outdir, str(year), str(SR), "%d_%d"%(MX,MY))
        
        models[str(year)][str(SR)]["%d_%d"%(MX, MY)] = {}

        #set tiny norms to zero
        norm_closest = df.loc[(df.MX==closest_mx)&(df.MY==closest_my), "weight"].sum()/common.lumi_table[year]
        print(norm_closest)
        if norm_closest < 0.001:
          popt = np.array([1, 0.0, 2, 1.2, 10, 1.2, 10]) #doesn't matter but give sensible numbers anyway
          #models[str(year)][str(SR)]["%d_%d"%(MX, MY)] = {"%d_%d"%tuple(to_fit_masses[i]):{"norm": 0.0, "norm_err": 0.0} for i in range(len(to_fit_masses))}
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"] = {"norm": 0.0, "norm_systematic": 1.0, "norm_spline":"linear", "parameters":list(popt), "skipped_mass":"%d_%d"%(closest_mx,closest_my), "closest_mass":"%d_%d"%(closest_mx,closest_my)}
          if do_same_score_interp:
            models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["same_score"] = {"grad_norm_pos": 0.0, "grad_norm_neg": 0.0, "grad_dm": 0.0, "grad_sigma": 0.0}
          continue

        #handle normalisation
        if is_nominal_mass:
          norm = df.loc[(df.MX==MX)&(df.MY==MY), "weight"].sum()/common.lumi_table[year]
          n_sig_events = sum((df.MX==MX)&(df.MY==MY))
          #norm_err = norm / np.sqrt(n_sig_events)
          norm_err = np.sqrt((df.loc[(df.MX==MX)&(df.MY==MY), "weight"]**2).sum())/common.lumi_table[year]
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"] = {"norm": norm, "norm_err": norm_err, "norm_systematic": 1.0, "n_sig_events":n_sig_events}
        else:
          norms_nominal = np.array([df.loc[(df.MX==mx)&(df.MY==my), "weight"].sum()/common.lumi_table[year] for mx,my in to_fit_masses])
          n_sig_events_nominal = [sum((df.MX==mx)&(df.MY==my)) for mx,my in to_fit_masses]
          #norms_nominal_err = norms_nominal / np.sqrt(n_sig_events_nominal)
          norms_nominal_err = np.array([np.sqrt((df.loc[(df.MX==mx)&(df.MY==my), "weight"]**2).sum())/common.lumi_table[year] for mx,my in to_fit_masses])

          models[str(year)][str(SR)]["%d_%d"%(MX, MY)] = {"%d_%d"%tuple(to_fit_masses[i]):{"norm": float(norms_nominal[i]), "norm_err": float(norms_nominal_err[i]), "n_sig_events":n_sig_events_nominal[i]} for i in range(len(to_fit_masses))}

          norm_cubic, norm_linear, skipped_mass = doInterpolation(MX, MY, to_fit_masses, norms_nominal)
          norm_cubic_skip, norm_linear_skip, skipped_mass = doInterpolation(MX, MY, to_fit_masses, norms_nominal, skip_closest=True)
          
          norm_cubic_systematic = 1 + abs((norm_cubic-norm_cubic_skip)/norm_cubic)
          norm_linear_systematic = 1 + abs((norm_linear-norm_linear_skip)/norm_linear)
                    
          if norm_cubic_systematic < norm_linear_systematic:
            models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"] = {"norm": norm_cubic, "norm_systematic": norm_cubic_systematic, "norm_spline":"cubic"}
          else:
            models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"] = {"norm": norm_linear, "norm_systematic": norm_linear_systematic, "norm_spline":"linear"}
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"]["skipped_mass"] = "%d_%d"%tuple(skipped_mass)
        
        models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"]["closest_mass"] = "%d_%d"%(closest_mx, closest_my)

        #handle shape
        if is_nominal_mass:
          popt, perr = fitSignalModel(df, MX, outdir, MY, fit_range_my, make_plots)
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"]["parameters"] = list(popt)
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"]["parameters_err"] = list(perr)
        else:
          #only need the +-1 points since using linear interpolation for shape
          to_fit_mxs_shape = nominal_mxs[gridx[(gridx>=argx(MX)-1)&(gridx<=argx(MX)+1)]] #closest mx within +-r points
          to_fit_mys_shape = nominal_mys[gridy[(gridy>=argy(MY)-1)&(gridy<=argy(MY)+1)]] #closest mx within +-r points
          to_fit_masses_shape = np.array([[mx, my] for mx in to_fit_mxs_shape for my in to_fit_mys_shape if (mx, my) in all_masses])

          dms = []
          sigmas = []
          for (mx, my) in to_fit_masses_shape:
            popt, perr = fitSignalModel(df, mx, outdir, my, fit_range_my, make_plots)
            models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["%d_%d"%(mx,my)]["parameters"] = list(popt)
            models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["%d_%d"%(mx,my)]["parameters_err"] = list(perr)
            dms.append(popt[1])
            sigmas.append(popt[2])

          dm_spline = interp(to_fit_masses_shape[:,0], to_fit_masses_shape[:,1], dms, kind='linear')
          sigma_spline = interp(to_fit_masses_shape[:,0], to_fit_masses_shape[:,1], sigmas, kind='linear')

          closest_popt = models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["%d_%d"%(closest_mx, closest_my)]["parameters"].copy()
          closest_popt[1] = dm_spline(MX, MY)
          closest_popt[2] = sigma_spline(MX, MY)

          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"]["parameters"] = closest_popt

        #for interpolation at combine level in MH (MY)
        if do_same_score_interp:
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["same_score"] = {}
          for (mx, my) in to_fit_masses_same_score:
            norm = df_same_score.loc[(df_same_score.MX==mx)&(df_same_score.MY==my), "weight"].sum()/common.lumi_table[year]
            n_sig_events = sum((df_same_score.MX==mx)&(df_same_score.MY==my))
            #norm_err = norm / np.sqrt(n_sig_events)
            norm_err = np.sqrt((df_same_score.loc[(df_same_score.MX==mx)&(df_same_score.MY==my), "weight"]**2).sum())/common.lumi_table[year]

            models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["same_score"]["%d_%d"%(mx,my)] = {"norm": norm, "norm_err": norm_err, "n_sig_events":n_sig_events}
            
            popt, perr = fitSignalModel(df_same_score, mx, os.path.join(outdir, "same_score"), my, fit_range_my, make_plots)
            models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["same_score"]["%d_%d"%(mx,my)]["parameters"] = list(popt)
            models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["same_score"]["%d_%d"%(mx,my)]["parameters_err"] = list(perr)

          #find the gradients
          my_low, my_high = min(to_fit_masses_same_score[:,1]), max(to_fit_masses_same_score[:,1])
          my_mid = closest_my
          print(to_fit_masses_same_score)
          print(closest_my)
          print(my_low, my_mid, my_high)
          sc_dict = models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["same_score"]
          #scale norm variations from MC used to calculate gradients to actual norm for the mass point being probed
          norm_sf = sc_dict["%d_%d"%(mx,my_mid)]["norm"] / models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"]["norm"] 
          if (my_high-my_mid) != 0: sc_dict["grad_norm_pos"] = norm_sf * ((sc_dict["%d_%d"%(mx,my_high)]["norm"] - sc_dict["%d_%d"%(mx,my_mid)]["norm"]) / (my_high-my_mid))
          else:                     sc_dict["grad_norm_pos"] = 0
          if (my_mid-my_low) != 0: sc_dict["grad_norm_neg"] = norm_sf * ((sc_dict["%d_%d"%(mx,my_mid)]["norm"] - sc_dict["%d_%d"%(mx,my_low)]["norm"]) / (my_mid-my_low))
          else:                    sc_dict["grad_norm_neg"] = 0
          
          if sc_dict["grad_norm_pos"] == 0:
            sc_dict["grad_norm_pos"] = -sc_dict["grad_norm_neg"]
          if sc_dict["grad_norm_neg"] == 0:
            sc_dict["grad_norm_neg"] = -sc_dict["grad_norm_pos"]
          sc_dict["grad_dm"] = (sc_dict["%d_%d"%(mx,my_high)]["parameters"][1] - sc_dict["%d_%d"%(mx,my_low)]["parameters"][1]) / (my_high-my_low)
          sc_dict["grad_sigma"] = (sc_dict["%d_%d"%(mx,my_high)]["parameters"][2] - sc_dict["%d_%d"%(mx,my_low)]["parameters"][2]) / (my_high-my_low)

  with open(os.path.join(original_outdir, "model.json"), "w") as f:
    json.dump(models, f, indent=4, sort_keys=True, cls=common.NumpyEncoder)

def checkInterpolation(original_df, proc_dict, optim_results, original_outdir, make_plots=False, do_same_score_interp=False, fit_range_my=False):
  nominal_masses = np.array([common.get_MX_MY(entry["sig_proc"]) for entry in optim_results if entry["sig_proc"] in proc_dict.keys()])
  print(nominal_masses)

  for mx, my in nominal_masses:
    print(mx, my)
    min_mx, max_mx = nominal_masses[:,0].min(), nominal_masses[:,0].max()
    if (mx == min_mx) or (mx == max_mx):
      continue

    sig_proc = common.get_sig_proc(optim_results[0]["sig_proc"], mx, my)
    pruned_proc_dict = proc_dict.copy()
    del pruned_proc_dict[sig_proc]

    masses_to_do = [[mx, my]]

    new_dir = os.path.join(original_outdir, "Interpolation_Check", sig_proc)
    os.makedirs(new_dir, exist_ok=True)
    deriveModels(original_df, pruned_proc_dict, optim_results, new_dir, make_plots=make_plots, fit_range_my=fit_range_my, masses_to_do=masses_to_do)

def tagSignals(df, entry, proc_dict, to_fit_masses, use_same_score=False):
  pd.options.mode.chained_assignment = None

  to_fit_ids = [i for i in df.process_id.unique() if list(df[df.process_id==i].iloc[0][["MX", "MY"]]) in to_fit_masses]
  df = df[df.process_id.isin(to_fit_ids)]
  
  df.loc[:, "SR"] = -1

  boundaries = entry["category_boundaries"][::-1]
  scores = list(filter(lambda x: "intermediate" in x, df.columns))

  if not use_same_score:
    for score in scores:
      proc = score.split("intermediate_transformed_score_")[1]
      if list(common.get_MX_MY(proc)) in to_fit_masses:

        if proc in proc_dict.keys():
          for i in range(len(boundaries)-1):
            selection = (df[score] <= boundaries[i]) & (df[score] > boundaries[i+1]) & (df.process_id==proc_dict[proc])
            df.loc[selection, "SR"] = i
  else:
    score = entry["score"]
    for i in range(len(boundaries)-1):
      selection = (df[score] <= boundaries[i]) & (df[score] > boundaries[i+1])
      df.loc[selection, "SR"] = i

  pd.options.mode.chained_assignment = "warn"
  return df[df.SR!=-1]

def filterScores(score, masses):
  if "intermediate" in score:
    sig_proc = "_".join(score.split("_")[3:])
    return common.get_MX_MY(sig_proc) in masses
  else:
    return True 

def getColumns(path, masses):
  columns = common.getColumns(path)
  columns = list(filter(lambda x: ("intermediate_transformed_score" in x), columns)) + ["Diphoton_mass", "process_id", "weight", "y", "year"]
  if masses is not None:
    columns = list(filter(lambda x: filterScores(x, masses), columns))
  return columns

def mergeBatchSplit(outdir, mass_points):
  expected_dirs = [mass.replace(",","_") for mass in mass_points]
  missing_mass_points = set(expected_dirs).difference(os.listdir(os.path.join(outdir, "batch_split")))
  assert len(missing_mass_points) == 0, print("Some jobs must have failed, these mass points are missing: %s"%str(missing_mass_points))

  merged_model = None
  for mass in mass_points:
    mass = mass.replace(",", "_")

    with open(os.path.join(outdir, "batch_split", mass, "model.json"), "r") as f:
      model = json.load(f)
    if merged_model is None:
      merged_model = model
    else:
      assert merged_model.keys() == model.keys()
      for year in merged_model.keys():
        assert merged_model[year].keys() == model[year].keys()
        os.makedirs(os.path.join(outdir, year), exist_ok=True)
        for cat in merged_model[year]:
          os.makedirs(os.path.join(outdir, year, cat, mass), exist_ok=True)
          #os.system("cp %s/* %s"%(os.path.join(outdir, "batch_split", mass, year, cat, mass), os.path.join(outdir, year, cat, mass)))

          merged_model[year][cat][mass] = model[year][cat][mass]
      
    if "Interpolation_Check" in os.listdir(os.path.join(outdir, "batch_split", mass)):
      os.makedirs(os.path.join(outdir, "Interpolation_Check"), exist_ok=True)
      example_sig_proc = os.listdir(os.path.join(outdir, "batch_split", mass, "Interpolation_Check"))[0]
      sig_proc = common.get_sig_proc(example_sig_proc, int(mass.split("_")[0]), int(mass.split("_")[1]))
      #os.system("cp -r %s %s"%(os.path.join(outdir, "batch_split", mass, "Interpolation_Check", sig_proc), os.path.join(outdir, "Interpolation_Check", sig_proc)))
  
  with open(os.path.join(outdir, "model.json"), "w") as f:
    json.dump(merged_model, f, indent=4, sort_keys=True, cls=common.NumpyEncoder)

def main(args):
  with open(args.optim_results) as f:
     optim_results = json.load(f)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  if args.mass_points is None:
    all_masses = common.getAllMasses(optim_results)
    args.mass_points = ["%d,%d"%(m[0], m[1]) for m in all_masses]

  if args.merge_batch_split:
    mergeBatchSplit(args.outdir, args.mass_points)
    return True

  if args.batch:
    if not args.batch_split:
      common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
      return True
    else:
      mass_points_copy = args.mass_points.copy()
      outdir_copy = copy.copy(args.outdir)
      for mass in mass_points_copy:
        args.mass_points = [mass]
        args.outdir = os.path.join(outdir_copy, "batch_split", mass.replace(",","_"))
        common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
      return True        

  if args.mass_points is not None:
    for MX, MY in [pair.split(",") for pair in args.mass_points]:
      to_fit_masses = getToFitMasses(float(MX), float(MY), optim_results, proc_dict, only_nominal=(not args.interp_checks))
      args.mass_points.extend(["%d,%d"%(m[0],m[1]) for m in to_fit_masses])
    args.mass_points = list(set(args.mass_points))
    print(args.mass_points)

  masses, optim_results = common.getMassesToRun(args.mass_points, optim_results)

  df = pd.read_parquet(args.parquet_input, columns=getColumns(args.parquet_input, masses))
  example_sig_proc = optim_results[0]["sig_proc"]
  keep_proc_ids = [proc_dict[common.get_sig_proc(example_sig_proc, MX, MY)] for MX, MY in masses if common.get_sig_proc(example_sig_proc, MX, MY) in proc_dict.keys()]
  df = df[df.process_id.isin(keep_proc_ids)]

  common.add_MX_MY(df, proc_dict)

  deriveModels(df, proc_dict, optim_results, args.outdir, args.make_plots, args.same_score_interp, args.fit_range_my)
  if args.interp_checks:
    checkInterpolation(df, proc_dict, optim_results, args.outdir, args.make_plots, args.same_score_interp, args.fit_range_my)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--step', type=float, default=10.0)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)
  parser.add_argument('--batch-split', action="store_true")
  parser.add_argument('--merge-batch-split', action="store_true")
  parser.add_argument('--make-plots', action="store_true")
  parser.add_argument('--interp-checks', action="store_true")
  parser.add_argument('--same-score-interp', action="store_true", help="Fit models at different MC but within same category. Needed for interpolation at combine level.")
  parser.add_argument('--fit-range-my', action="store_true", help="Change fit range depending on my (for Y->gg)")
  parser.add_argument('--mass-points', nargs="+", default=None, help="Only run these mass points. Provide a list of MX,MY like 300,125 400,150...")
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)