import pandas as pd
import json
import argparse
import os

def loadSummaries(args):
  summaries = []
  for summary_path in args.summary_input:
    print(summary_path)
    with open(summary_path, "r") as f:
      summaries.append(json.load(f)["sample_id_map"])

  return summaries

def loadDataFrames(args):
  dfs = []
  for parquet_path in args.parquet_input:
    print(parquet_path)
    dfs.append(pd.read_parquet(parquet_path))

  return dfs

def invertSummary(summary):
  inverted_summary = {}
  for proc_name in summary.keys():
    inverted_summary[summary[proc_name]] = proc_name
  return inverted_summary

def idToName(dfs, summaries):
  for i, df in enumerate(dfs):
    inverted_summary = invertSummary(summaries[i])
    df.process_id.replace(inverted_summary, inplace=True)  
  return dfs

def nameToId(df, summary):
  df.process_id.replace(summary, inplace=True)
  return df

def removeExcludedProcesses(dfs, summaries, args):
  for i, summary in enumerate(summaries):
    for process in list(summary.keys()):
      if process in args.exclude_procs:
        dfs[i] = dfs[i][dfs[i].process_id != summary[process]]
        del summary[process]        

  return dfs, summaries

def scaleDataFrames(dfs, args):
  print("Scaling")
  for i, df in enumerate(dfs):
    df.loc[:, "weight_central"] *= args.scale[i]

def mergeDataFrames(args):
  summaries = loadSummaries(args)
  dfs = loadDataFrames(args)
  scaleDataFrames(dfs, args)

  dfs, summaries = removeExcludedProcesses(dfs, summaries, args)
  #dfs = idToName(dfs, summaries)

  merged_summary = {}
  for summary in summaries: merged_summary.update(summary)

  # if len(merged_summary.keys()) == sum([len(summary.keys()) for summary in summaries]): #no conflict
  #   print(">> No overlapping process names")
  #   print("Merging")
  #   merged_df = pd.concat(dfs, ignore_index=True)
    
  #   all_procs = list(merged_df.process_id.unique())
  #   if "Data" in all_procs:
  #     all_procs.remove("Data")
  #   all_procs = sorted(all_procs)
  #   merged_summary = {process:i+1 for i, process in enumerate(all_procs)}
  #   if "Data" in merged_df.process_id.unique():
  #     merged_summary["Data"] = 0 
  # else:
  #   print("Have not implemented a way to deal with conflicting process names yet. Script will now exit.")
  #   exit()

  merged_df = pd.concat(dfs, ignore_index=True)

  #merged_df = nameToId(merged_df, merged_summary)

  merged_df.to_parquet(args.parquet_output)
  with open(args.summary_output, "w") as f:
    json.dump({"sample_id_map": merged_summary}, f, indent=4)  

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-p', type=str, nargs='+', help="List of parquet files to merge, e.g. low_mass.parquet sm_trigger.parquet ...")
  parser.add_argument('--summary-input', '-s', type=str, nargs='+', help="List of summary json files to merge, e.g. low_mass_summary.json sm_trigger_summary.json ...")

  parser.add_argument('--parquet-output', type=str, default="merged_nominal.parquet")
  parser.add_argument('--summary-output', type=str, default="summary.json")

  parser.add_argument('--exclude-procs', '-e', type=str, nargs='+', default=[], help="List of processes to not include in the merging, e.g. Diphoton TTGamma ...")

  parser.add_argument('--batch', action="store_true")

  parser.add_argument('--scale', type=float, nargs='+', default=None)

  args = parser.parse_args()

  if args.scale is None:
    args.scale = [1.0 for each in args.parquet_input]

  assert len(args.scale) == len(args.parquet_input) == len(args.summary_input), print(args.parquet_input, args.summary_input)

  if args.batch:
    import common
    import sys
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=2)
  else:
    mergeDataFrames(args)

