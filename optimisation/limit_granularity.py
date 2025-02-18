import pandas as pd
import numpy as np
import argparse
import os
import json
import common
import sys
import copy

from optimisation.limit import getBoundariesPerformance
from optimisation.tools import get_pres,get_sr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
#plt.rcParams["figure.figsize"] = (12.5,3)
#plt.rcParams['figure.constrained_layout.use'] = True

def loadDataFrame(args, scores):
  columns = common.getColumns(args.parquet_input)
  columns_load = ["Diphoton_mass", "weight", "year", "process_id"]
  columns_load += scores
  columns_load = list(set(columns).intersection(columns_load))

  df = pd.read_parquet(args.parquet_input, columns=columns_load)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  data = df[df.process_id == proc_dict["Data"]]
  sigs = {sig_proc: df[df.process_id == proc_dict[sig_proc]] for sig_proc in args.sig_procs}

  if args.bkg_source == "data":
    bkg = data
  elif args.bkg_source == "MC":
    bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"] if proc in proc_dict.keys()]
    bkg = df[df.process_id.isin(bkg_ids)]
  else:
    raise AttributeError("Background source must be 'data' or 'MC'")
  
  return bkg, sigs, proc_dict

def getLimit(sigs, bkg, optim_results_dict, m_MC, m_cats):
  """
  Expected limit found for a singal (m_MC) when using the categories that target
  a different mass point (m_cats).
  """

  proc_MC = common.get_sig_proc(list(optim_results_dict.keys())[0], m_MC[0], m_MC[1])
  proc_cats = common.get_sig_proc(list(optim_results_dict.keys())[0], m_cats[0], m_cats[1])
  
  score = optim_results_dict[proc_cats]["score"]
  boundaries = optim_results_dict[proc_cats]["category_boundaries"]

  pres = get_pres(proc_MC)
  sr = get_sr(proc_MC)

  bm = bkg.Diphoton_mass
  bkg_no_sr = bkg[(bm <= sr[0]) | (bm >= sr[1])]

  bkg_package = pd.DataFrame({"year":bkg_no_sr.year, "mass":bkg_no_sr.Diphoton_mass, "weight":bkg_no_sr.weight, "score":bkg_no_sr[score]})
  sig = sigs[proc_MC]
  sig_package = pd.DataFrame({"year":sig.year, "mass":sig.Diphoton_mass, "weight":sig.weight, "score":sig[score]})
  
  return getBoundariesPerformance(bkg_package, sig_package, pres, sr, boundaries)[0]

def getFactors(n):
  factors = [n]
  for i in range(1,int(n//2)+1):
    if n%i == 0:
      factors.append(i)
  return np.array(factors)

def getNominalMasses(optim_results):
  masses = []
  for entry in optim_results:
    if "optimal_limit" not in entry.keys(): 
      continue
    mx, my = common.get_MX_MY(entry["sig_proc"])
    masses.append([mx, my])
  return np.array(masses)

def getPoints(f, grad, m1, m2):
  #find division of interval that ensures only f level of loss in limit
  spacing = (f / grad) * 2
  n_points = int(np.ceil((m2-m1) / spacing))
  spacing = (m2-m1) / n_points
  assert spacing >= 1

  points = [int(m1+spacing*i) for i in range(n_points)]

  return points

def main(args):
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
    return True

  with open(args.optim_results, "r") as f:
    optim_results = json.load(f)
  optim_results_dict = {each["sig_proc"]:each for each in optim_results}
  scores = [entry["score"] for entry in optim_results]

  args.sig_procs = common.expandSigProcs(args.sig_procs)

  bkg, sigs, proc_dict = loadDataFrame(args, scores)

  pres = get_pres(list(sigs.keys())[0])
  bm = bkg.Diphoton_mass
  bkg = bkg[((bm >= pres[0]) & (bm <= pres[1]))]

  os.makedirs(args.outdir, exist_ok=True)

  nominal_masses = getNominalMasses(optim_results)
  
  interpoints_my = {}

  for my in np.unique(nominal_masses[:,1]):
    mxs = nominal_masses[:,0][nominal_masses[:,1]==my]
    mxs = np.sort(mxs)
    interpoints = []

    for i, mx in enumerate(mxs):
      to_plot_mx = mxs[np.argsort(abs(mxs-mx))][:5]
      # MC for [mx, my] in categories targetting [plot_mx, my]
      limits = np.array([getLimit(sigs, bkg, optim_results_dict, [mx, my], [plot_mx, my]) for plot_mx in to_plot_mx])

      rel_change = limits/limits[0] - 1
      #plt.scatter(to_plot_mx, rel_change)
      plt.plot(np.sort(to_plot_mx), rel_change[np.argsort(to_plot_mx)], marker="o")
      plt.xlabel(r"$m_X$ that categories target")
      plt.ylabel("Relative change in limit")
      plt.title(r"MC for $m_X=%d, m_Y=%d$"%(mx, my))
      plt.savefig(os.path.join(args.outdir, "mx_%d_my_%d_mx.png"%(mx, my)))
      plt.clf()

      if i == len(mxs)-1: #if reached last one
        continue 
      above_mx = mxs[i+1]
      if i == 0: 
        below_mx = mxs[i+1]
      else:      
        below_mx = mxs[i-1]
      nominal_limit = limits[to_plot_mx==mx]
      above_limit = limits[to_plot_mx==above_mx]
      below_limit = limits[to_plot_mx==below_mx]
      above_grad = (2 * abs(above_limit-nominal_limit) / (above_limit+nominal_limit)) / abs(above_mx-mx)
      below_grad = (2 * abs(below_limit-nominal_limit) / (below_limit+nominal_limit)) / abs(below_mx-mx)
      grad = max([above_grad, below_grad, 1e-8])

      interpoints += getPoints(args.max_loss, grad, mx, above_mx)
    interpoints.append(mx) #add last mx
    interpoints_my[my] = interpoints
      
  interpoints_mx = {}
  print()

  for mx in np.unique(nominal_masses[:,0]):
    mys = nominal_masses[:,1][nominal_masses[:,0]==mx]
    if len(mys) == 1: 
      continue
    mys = np.sort(mys)
    interpoints = []
    for i, my in enumerate(mys):
      to_plot_my = mys[np.argsort(abs(mys-my))][:5]
      limits = np.array([getLimit(sigs, bkg, optim_results_dict, [mx, my], [mx, plot_my]) for plot_my in to_plot_my])

      rel_change = limits/limits[0] - 1
      plt.scatter(to_plot_my, rel_change)
      plt.xlabel(r"$m_Y$")
      plt.ylabel("Relative change in limit")
      plt.title(r"$m_X=%d, m_Y=%d$"%(mx, my))
      plt.savefig(os.path.join(args.outdir, "mx_%d_my_%d_my.png"%(mx, my)))
      plt.clf()

      if i == len(mys)-1: 
        continue #if reached last one
      above_my = mys[i+1]
      if i == 0: 
        below_my = mys[i+1]
      else:      
        below_my = mys[i-1]
      nominal_limit = limits[to_plot_my==my]
      above_limit = limits[to_plot_my==above_my]
      below_limit = limits[to_plot_my==below_my]
      above_grad = abs((2 * (above_limit-nominal_limit) / (above_limit+nominal_limit)) / (above_my-my))
      below_grad = abs((2 * (below_limit-nominal_limit) / (below_limit+nominal_limit)) / (below_my-my))
      grad = max([above_grad, below_grad, 1e-8])
      
      interpoints += getPoints(args.max_loss, grad, my, above_my)
    interpoints.append(my) #add last mx
    interpoints_mx[mx] = interpoints

  for my in interpoints_my.keys():
    print(my, interpoints_my[my])
  print()
  for mx in interpoints_mx.keys():
    print(mx, interpoints_mx[mx])

  all_masses = [(mx, my) for mx in interpoints_mx.keys() for my in interpoints_mx[mx]]
  all_masses += [(mx, my) for my in interpoints_my.keys() for mx in interpoints_my[my]]
  all_masses = list(set(all_masses)) #remove duplicates

  plt.scatter([m[0] for m in nominal_masses], [m[1] for m in nominal_masses], marker='.')
  plt.scatter([m[0] for m in nominal_masses], [m[1] for m in nominal_masses], marker='.', label="Nominal masses (N=%d)"%len(nominal_masses))
  plt.ylim(top=(plt.ylim()[0] + 1.2*(plt.ylim()[1] - plt.ylim()[0])))
  plt.legend(ncol=1)
  plt.xlabel(r"$m_X$")
  plt.ylabel(r"$m_Y$")
  plt.savefig(os.path.join(args.outdir, "final_granularity_%s_nominal.pdf"%str(args.max_loss).replace(".","p")))
  plt.clf()

  plt.scatter([m[0] for m in all_masses], [m[1] for m in all_masses], marker='.', label="All masses (N=%d)"%len(all_masses))
  plt.scatter([m[0] for m in nominal_masses], [m[1] for m in nominal_masses], marker='.', label="Nominal masses (N=%d)"%len(nominal_masses))
  plt.ylim(top=(plt.ylim()[0] + 1.2*(plt.ylim()[1] - plt.ylim()[0])))
  plt.legend(ncol=1)
  plt.xlabel(r"$m_X$")
  plt.ylabel(r"$m_Y$")
  plt.savefig(os.path.join(args.outdir, "final_granularity_%s.pdf"%str(args.max_loss).replace(".","p")))
  plt.clf()

  nominal_mx = np.sort(np.unique(nominal_masses[:,0]))
  nominal_my = np.sort(np.unique(nominal_masses[:,1]))

  for mx, my in nominal_masses:
    print(mx, my)
    if max(nominal_masses[nominal_masses[:,1]==my][:,0]) == mx: continue #if at far right of grid
    if max(nominal_masses[nominal_masses[:,0]==mx][:,1]) == my: continue #if at top of grid

    #find my that are in between this my and the next nominal
    poss_my = np.array(interpoints_mx[mx])
    next_nominal_my = min(nominal_my[nominal_my>my])
    between_mys = poss_my[(poss_my>my)&(poss_my<next_nominal_my)]

    #find mx to fill grid with
    poss_mx = np.array(interpoints_my[my])
    next_nominal_mx = min(nominal_mx[nominal_mx>mx])
    between_mxs = poss_mx[(poss_mx>mx)&(poss_mx<next_nominal_mx)]
    print(between_mxs)

    for between_my in between_mys:
      print(between_my)
      if between_my not in interpoints_my.keys(): interpoints_my[between_my] = []
      interpoints_my[between_my] += list(between_mxs)

  all_masses = [(mx, my) for mx in interpoints_mx.keys() for my in interpoints_mx[mx]]
  all_masses += [(mx, my) for my in interpoints_my.keys() for mx in interpoints_my[my]]
  all_masses = list(set(all_masses)) #remove duplicates

  plt.scatter([m[0] for m in all_masses], [m[1] for m in all_masses], marker='.', label="All masses (N=%d)"%len(all_masses))
  plt.scatter([m[0] for m in nominal_masses], [m[1] for m in nominal_masses], marker='.', label="Nominal masses (N=%d)"%len(nominal_masses))
  plt.ylim(top=(plt.ylim()[0] + 1.2*(plt.ylim()[1] - plt.ylim()[0])))
  plt.legend(ncol=1)
  plt.xlabel(r"$m_X$")
  plt.ylabel(r"$m_Y$")
  plt.savefig(os.path.join(args.outdir, "final_granularity_%s_filled.pdf"%str(args.max_loss).replace(".","p")))
  plt.clf()

  with open(os.path.join(args.outdir, "all_masses_%s.json"%str(args.max_loss).replace(".","p")), "w") as f:
    json.dump(all_masses, f)

  extra_masses = []
  for mx, my in all_masses:
    if [float(mx), float(my)] not in nominal_masses.tolist():
      extra_masses.append([float(mx), float(my)])
  with open(os.path.join(args.outdir, "extra_masses_%s.json"%str(args.max_loss).replace(".","p")), "w") as f:
    json.dump(extra_masses, f)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str)
  parser.add_argument('--sig-procs', '-p', type=str, required=True, nargs="+")
  
  parser.add_argument('--bkg-source', '-b', type=str, default="data")

  parser.add_argument('--optim-results', '-r', type=str, required=True)

  parser.add_argument('--max-loss', type=float, default=0.1)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)


  args = parser.parse_args()

  df = main(args)




