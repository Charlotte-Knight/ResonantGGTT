import pandas as pd
import numpy as np
import json 
import argparse
import os
import uproot
import common
import tabulate
import sys
import optimisation.tools as tools

def assignSignalRegions(df, optim_results, score_name):
  df["SR"] = -1
  
  boundaries = optim_results["category_boundaries"][::-1] #so that cat0 is most pure
  for i in range(len(boundaries)-1):
    selection = (df[score_name] <= boundaries[i]) & (df[score_name] > boundaries[i+1])
    df.loc[selection, "SR"] = i
  return df[df.SR!=-1]

def writeOutputTree(mass, weight, process, cat_name, year, undo_lumi_scaling=False, scale_signal=False, event=None):
  df = pd.DataFrame({"dZ": np.zeros(len(mass)), "CMS_hgg_mass": mass, "weight": weight})
  if event is not None:
    df["event"] = event
  #print(df)
  #assert len(df) >= 4, len(df)

  if undo_lumi_scaling:
    df.loc[:,"weight"] /= common.lumi_table[year]
  if scale_signal:
    df.loc[:,"weight"] /= 1000

  #s = ((df.CMS_hgg_mass>=100)&(df.CMS_hgg_mass<=115)) | ((df.CMS_hgg_mass>=135)&(df.CMS_hgg_mass<=180))
  #s = (df.CMS_hgg_mass >= 65)
  #s = (df.CMS_hgg_mass >= 100)
  #print(sum(s), flush=True)
  #print("  Sumw full mgg range: %d"%df.weight.sum())
  #print("  Sumw 100 < mgg < 180: %d"%df[(df.CMS_hgg_mass>=100)&(df.CMS_hgg_mass<=180)].weight.sum())
  #print("  Sumw 65 < mgg < 150: %d"%df[(df.CMS_hgg_mass>=65)&(df.CMS_hgg_mass<=150)].weight.sum())
  #assert df[(df.CMS_hgg_mass>=65)&(df.CMS_hgg_mass<=150)].weight.sum() >= 10

  path = os.path.join(args.outdir, "outputTrees", str(year))
  os.makedirs(path, exist_ok=True)
  with uproot.recreate(os.path.join(path, "%s_13TeV_%s_%s.root"%(process, cat_name, year))) as f:
    f["%s_13TeV_%s"%(process, cat_name)] = df

def getYieldTable(df, proc_dict, path):
  reverse_proc_dict = {value:key for (key, value) in proc_dict.items()}
  procs = [reverse_proc_dict[proc_id] for proc_id in df.process_id.unique()]
  #procs.remove("Data") #blind yield table

  #procs=["ZZ", "WW", "WZ", "DY"]

  yield_table = {}
  for SR in df.SR.unique():
    yield_table[SR] = [df.loc[(df.process_id==proc_dict[proc])&(df.SR==SR), "weight"].sum() for proc in procs]
    #yield_table[SR] = [((df.process_id==proc_dict[proc])&(df.SR==SR)).sum() for proc in procs]
  yield_table = pd.DataFrame(yield_table)
  yield_table = yield_table[sorted(yield_table.columns)]
  yield_table.index = procs

  #convert signal procs into efficiencies
  #s = ["_M" in proc for proc in procs]
  #yield_table[s] /= sum([common.lumi_table[year] for year in df.year.unique()])
  # for proc in procs:
  #   if ("MX" not in proc) and ("XToHHggTauTau" not in proc): continue
  #   yield_table.loc[proc] /= df.loc[df.process_id==proc_dict[proc], "weight"].sum()

  table = tabulate.tabulate(yield_table, headers='keys', floatfmt=".4f")
  with open(path+".txt", "w") as f:
    f.write(table)
  with open(path+".tex", "w") as f:
    f.write(yield_table.to_latex(float_format="%.2f", column_format="l"+"|r"*(len(yield_table.columns))))
  yield_table.to_csv(path+".csv", float_format="%.2f")

def printDuplications(df):
  for process_id in df.process_id.unique():
    for year in df[df.process_id==process_id].year.unique():
      uniques, counts = np.unique(df[(df.process_id==process_id)&(df.year==year)].event, return_counts=True)
      duplicates = uniques[counts > 1]

      print(year, process_id)
      print(df[(df.process_id==process_id)&(df.year==year)&(df.event.isin(duplicates))].sort_values(by="event")[["Diphoton_mass", "event"]])

def halveLastSR(optim_results, keep_left=True):
  for entry in optim_results:
    if keep_left:
      entry["category_boundaries"][1] = entry["category_boundaries"][1] / 2
    else:
      entry["category_boundaries"][0] = entry["category_boundaries"][1] / 2

def main(args):
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
    return True

  with open(args.optim_results) as f:
    optim_results = json.load(f)
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  #halveLastSR(optim_results, False)
  print(optim_results[0])

  columns = common.getColumns(args.parquet_input)
  columns = list(filter(lambda x: x[:5] != "score", columns))
  df = pd.read_parquet(args.parquet_input, columns=columns)

  #only have data, may need to change if doing res bkg
  df = df[df.process_id == proc_dict["Data"]]

  printDuplications(df)
  df.drop_duplicates(inplace=True)
  printDuplications(df)

  sig_proc_example = optim_results[0]["sig_proc"]
  pres = tools.get_pres(sig_proc_example)
  df = df[(df.Diphoton_mass >= pres[0]) & (df.Diphoton_mass <= pres[1])]
  
  yield_tables = []
  yield_tables_path = os.path.join(args.outdir, "yieldTables")
  os.makedirs(yield_tables_path, exist_ok=True)

  for entry in optim_results:
    MX, MY = common.get_MX_MY(entry["sig_proc"])
    #if (MX != 600) or (MY != 90): continue
    #print(MX)

    print("Tagging")
    tagged_df = assignSignalRegions(df, entry, entry["score"])

    print("Doing yields")
    getYieldTable(tagged_df, proc_dict, os.path.join(yield_tables_path, entry["sig_proc"]))

    if not args.justYields:
      data = tagged_df[tagged_df.process_id == proc_dict["Data"]]
      #sig = tagged_df[tagged_df.process_id == proc_dict[entry["sig_proc"]]]

      proc_name = "ggttresmx%dmy%d"%(MX, MY)
      years = data.year.unique()

      for i, year in enumerate(years):
        SRs = np.sort(tagged_df.SR.unique())
        if args.dropLastCat: 
          SRs = SRs[:-1]

        if (SRs != [i for i in range(len(SRs))]).any():
          print(f"Missing categories for MX={MX}, MY={MY}")
          print("Will drop this mass point")
          print(np.unique(data.SR, return_counts=True))
          continue
        
        for SR in SRs:
          cat_name = "%scat%d"%(proc_name, SR)
          if args.controlRegions:
            cat_name += "cr"
          if args.combineYears and (i==0):
            mgg = data[(data.SR==SR)].Diphoton_mass
            w = data[(data.SR==SR)].weight
            event = data[(data.SR==SR)].event
            sr = tools.get_sr(entry["sig_proc"])
            print("Data", cat_name, year)
            print(sum((mgg <= sr[0]) | (mgg >= sr[1])), flush=True)

            writeOutputTree(mgg, w, "Data", cat_name, "combined", event=event)
          elif not args.combineYears:
            writeOutputTree(data[(data.SR==SR)&(data.year==year)].Diphoton_mass, data[(data.SR==SR)&(data.year==year)].weight, "Data", cat_name, year)

          #writeOutputTree(sig[(sig.SR==SR)&(sig.year==year)].Diphoton_mass, sig[(sig.SR==SR)&(sig.year==year)].weight, proc_name, "%scat%d"%(proc_name, SR), year, undo_lumi_scaling=True)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--combineYears', action="store_true", help="Output data merged across years")
  parser.add_argument('--dropLastCat', action="store_true")
  parser.add_argument('--justYields', action="store_true")
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)
  parser.add_argument('--controlRegions', action="store_true")
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  df = main(args)