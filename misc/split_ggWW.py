import os
import pandas as pd
import json
import common
import argparse
import sys
import numpy as np

def main(args):
  os.makedirs(os.path.join(args.dir, "GGTT"), exist_ok=True)
  os.makedirs(os.path.join(args.dir, "GGWWdi"), exist_ok=True)
  os.makedirs(os.path.join(args.dir, "GGWWsemi"), exist_ok=True)

  with open(args.summary, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  df = pd.read_parquet(args.parquet)
  #df = df[df.process_id != -181]

  for proc_id in df.process_id.unique():
    num = (df.process_id==proc_id).sum()

    if num > 150_000:
      sumw = df.loc[df.process_id==proc_id, "weight_central"].sum()

      ind = df[df.process_id==proc_id].index.to_numpy()
      np.random.shuffle(ind)
      ind = ind[150_000:]
      df.drop(ind, inplace=True)
      df.loc[df.process_id==proc_id, "weight_central"] *= (sumw / df.loc[df.process_id==proc_id, "weight_central"].sum())
      #assert df.loc[df.process_id==proc_id, "weight_central"].sum() == sumw, print(df.loc[df.process_id==proc_id, "weight_central"].sum(), sumw)

    assert (df.process_id==proc_id).sum() <= 150_000

  ggtt_ids = [item for key,item in proc_dict.items() if "To2G2Tau" in key or "tautau" in key]
  ggww_di_ids = [item for key,item in proc_dict.items() if "To2G2WTo2G2L2Nu" in key or "lnulnu" in key]
  ggww_semi_ids = [item for key,item in proc_dict.items() if "To2G2WTo2G2Q1L1Nu" in key or "2qlnu" in key]

  sm_higgs_ids = [item for key,item in proc_dict.items() if "M125" in key]

  ggtt_ids = ggtt_ids + sm_higgs_ids

  filename = args.parquet.split("/")[-1]

  df.loc[df.process_id.isin(ggtt_ids), :].to_parquet(os.path.join(args.dir, "GGTT", filename))
  df.loc[df.process_id.isin(ggww_di_ids), :].to_parquet(os.path.join(args.dir, "GGWWdi", filename))
  df.loc[df.process_id.isin(ggww_semi_ids), :].to_parquet(os.path.join(args.dir, "GGWWsemi", filename))

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--summary', '-s', type=str, required=True)
  parser.add_argument('--parquet', '-i', type=str, required=True)
  parser.add_argument('--dir', '-d', type=str, required=True)
  parser.add_argument('--batch', action="store_true")
  
  args = parser.parse_args()

  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=4)
  else:
    main(args)