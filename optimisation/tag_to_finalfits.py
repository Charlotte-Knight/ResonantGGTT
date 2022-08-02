import pandas as pd
import numpy as np
import json 
import argparse
import os
import uproot
import common

def assignSignalRegions(df, optim_results, score_name):
  df["SR"] = -1
  
  boundaries = optim_results["category_boundaries"][::-1] #so that cat0 is most pure
  for i in range(len(boundaries)-1):
    selection = (df[score_name] <= boundaries[i]) & (df[score_name] > boundaries[i+1])
    df.loc[selection, "SR"] = i
  return df[df.SR!=-1]

def writeOutputTree(mass, weight, process, cat_name, year, undo_lumi_scaling=False, scale_signal=False):
  df = pd.DataFrame({"dZ": np.zeros(len(mass)), "CMS_hgg_mass": mass, "weight": weight})
  
  if undo_lumi_scaling:
    df.loc[:,"weight"] /= common.lumi_table[year]
  if scale_signal:
    df.loc[:,"weight"] /= 1000

  print(process, cat_name, year)
  print("  Sumw full mgg range: %d"%df.weight.sum())
  print("  Sumw 100 < mgg < 180: %d"%df[(df.CMS_hgg_mass>=100)&(df.CMS_hgg_mass<=180)].weight.sum())
  print("  Sumw 65 < mgg < 150: %d"%df[(df.CMS_hgg_mass>=65)&(df.CMS_hgg_mass<=150)].weight.sum())
  #assert df[(df.CMS_hgg_mass>=65)&(df.CMS_hgg_mass<=150)].weight.sum() >= 10

  path = os.path.join(args.outdir, "outputTrees", str(year))
  os.makedirs(path, exist_ok=True)
  with uproot.recreate(os.path.join(path, "%s_13TeV_%s_%s.root"%(process, cat_name, year))) as f:
    f["%s_13TeV_%s"%(process, cat_name)] = df

def main(args):
  df = pd.read_parquet(args.parquet_input)
  with open(args.optim_results) as f:
    optim_results = json.load(f)
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  #only have data, may need to change if doing res bkg
  df = df[df.process_id == proc_dict["Data"]]

  for entry in optim_results:
    MX, MY = common.get_MX_MY(entry["sig_proc"])

    tagged_df = assignSignalRegions(df, entry, entry["score"])
    data = tagged_df[tagged_df.process_id == proc_dict["Data"]]

    proc_name = "ggttresmx%dmy%d"%(MX, MY)
    years = data.year.unique()

    for i, year in enumerate(years):
      SRs = np.sort(tagged_df.SR.unique())
      if args.dropLastCat: SRs = SRs[:-1]
      for SR in SRs:
        if args.combineYears and (i==0):
          writeOutputTree(data[(data.SR==SR)].Diphoton_mass, data[(data.SR==SR)].weight, "Data", "%scat%d"%(proc_name, SR), "combined")
        elif not args.combineYears:
          writeOutputTree(data[(data.SR==SR)&(data.year==year)].Diphoton_mass, data[(data.SR==SR)&(data.year==year)].weight, "Data", "%scat%d"%(proc_name, SR), year)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--injectSignal', type=str, default="")
  parser.add_argument('--combineYears', action="store_true", help="Output data merged across years")
  parser.add_argument('--dropLastCat', action="store_true")
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)