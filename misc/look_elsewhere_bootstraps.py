import pandas as pd
import numpy as np
import argparse
import json
import common
import uproot
from tqdm import tqdm
import os
import sys

def get_pres(sig_proc):
  """Return the correct preselection range for this sig proc"""
  if ("XToHHggTauTau" in sig_proc) or ("HHTo2G2Tau" in sig_proc): #if graviton
    return (100, 180)
  elif "NMSSM_XYH_Y_gg_H_tautau" in sig_proc: #if Y_gg
    if common.LOW_MASS_MODE:
      return (65, 150)
    else:
      return (100, 1000)
  elif "NMSSM_XYH_Y_tautau_H_gg" in sig_proc: #if Y_tautau
    return (100, 180)

def assignSignalRegions(df, optim_results, sig_procs):
  for entry in optim_results:
    if entry["sig_proc"] not in sig_procs:
      continue
    else:
      sig_proc = entry["sig_proc"]
      score_name = entry["score"]

    df[sig_proc+"_SR"] = -1
    
    boundaries = entry["category_boundaries"][::-1] #so that cat0 is most pure
    for i in range(len(boundaries)-1):
      selection = (df[score_name] <= boundaries[i]) & (df[score_name] >= boundaries[i+1])
      df.loc[selection, sig_proc+"_SR"] = i

    assert (df[sig_proc+"_SR"] == -1).sum() == 0, print(df[df[sig_proc+"_SR"] == -1])

  return df, len(boundaries)-1

def main(args):
  with open(args.optim_results) as f:
    optim_results = json.load(f)
  if args.sig_procs is None:
    args.sig_procs = [entry["sig_proc"] for entry in optim_results]

  if args.batch:
    if not args.batch_split:
      common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
      return True
    else:
      sig_procs_copy = args.sig_procs.copy()
      for sig_proc in sig_procs_copy:
        args.sig_procs = [sig_proc]
        common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
      return True     

  os.makedirs(args.outdir, exist_ok=True)
  
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  columns = ["Diphoton_mass", "process_id", "event", "weight"]
  columns += [f"intermediate_transformed_score_{proc}" for proc in args.sig_procs]  
  
  df = pd.read_parquet(args.parquet_input, columns=columns)
  pres = get_pres(args.sig_procs[0])
  df = df[(df.Diphoton_mass >= pres[0]) & (df.Diphoton_mass <= pres[1])]

  df.rename({"Diphoton_mass":"CMS_hgg_mass"}, axis=1, inplace=True)
  
  n_data = (df.process_id == proc_dict["Data"]).sum()
  diphoton_labels = ["DiPhoton"]
  if "DiPhoton_Low" in proc_dict:
    diphoton_labels.append("DiPhoton_Low")
  diphoton_ids = [proc_dict[label] for label in diphoton_labels]

  MC = df[df.process_id.isin(diphoton_ids)]

  del df
  MC.drop(columns="process_id", inplace=True) 

  MC = MC[MC.weight > 0]

  MC, n_SR = assignSignalRegions(MC, optim_results, args.sig_procs)
  
  def file_name(sig_proc):
    MX, MY = common.get_MX_MY(sig_proc)
    return f"mx{int(MX)}my{int(MY)}"
  
  #files = {sig_proc: uproot.recreate(f"{args.outdir}/{sig_proc}.root") for sig_proc in args.sig_procs}
  files = {sig_proc: uproot.recreate(f"{args.outdir}/{file_name(sig_proc)}.root") for sig_proc in args.sig_procs}

  np.random.seed(0)
  n_data_poisson = np.random.poisson(lam=n_data, size=args.n_bootstraps)
  for i in tqdm(range(args.n_bootstraps)):
    MC_bootstrap = MC.sample(n_data_poisson[i], replace=True, random_state=i, weights=MC.weight)

    for sig_proc in args.sig_procs:    
      MX, MY = common.get_MX_MY(sig_proc)  
      for SR in range(n_SR-1):
        cat_name = f"ggttresmx{int(MX)}my{int(MY)}cat{SR}"
        files[sig_proc][f"toy_{i+1}/{cat_name}"] = MC_bootstrap[MC_bootstrap[sig_proc+"_SR"] == SR][["CMS_hgg_mass", "event"]]

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--sig-procs', '-p', type=str, nargs="+", default=None)
  parser.add_argument('--n-bootstraps', '-n', type=int, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)
  parser.add_argument('--batch-split', action="store_true")

  args = parser.parse_args()
  
  df = main(args)