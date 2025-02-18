import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import common
import json

fb = sys.argv[1] # file for before (with sculpting)
fa = sys.argv[2] # file for after (without sculpting)

with open(sys.argv[3], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

fb_columns = common.getColumns(fb)
fa_columns = common.getColumns(fa)
columns = set(fb_columns).intersection(fa_columns)
columns = list(filter(lambda x: "intermediate" in x, columns)) + ["process_id", "weight", "category"] 

dfb = pd.read_parquet(fb, columns = columns)
dfa = pd.read_parquet(fa, columns = columns)

reco_MX_mgg = pd.read_parquet(sys.argv[4], columns=["reco_MX_mgg", "event"])
reco_MX_mgg.rename(columns={"event": "event2"}, inplace=True)

dfa = dfa[dfa.category==4]
dfb = dfb[dfb.category==4]


NMSSM_procs = list(filter(lambda x: "NMSSM" in x, proc_dict.keys()))
NMSSM_ids = [proc_dict[proc] for proc in NMSSM_procs]
dfb = dfb[dfb.process_id.isin(NMSSM_ids)]
dfa = dfa[dfa.process_id.isin(NMSSM_ids)]

def getEff(df, proc, score, threshold=0.999):
  #return "%.1f" % (100 * df.loc[(df.process_id==proc_dict[proc])&(df[score]>threshold), "weight"].sum() / 138)
  return 100 * df.loc[(df.process_id==proc_dict[proc])&(df[score]>threshold), "weight"].sum() / 138

def getRecoMX(df, proc, threshold=0.999):
  score = f"intermediate_transformed_score_{proc}"
  return df.loc[(df.process_id==proc_dict[proc])&(df[score]>threshold), "reco_MX_mgg"]

effs = {}

for proc in NMSSM_procs:
  proc_MX, proc_MY = common.get_MX_MY(proc)

  effs[proc] = {}

  for column in columns:
    if "NMSSM" not in column:
      continue
    
    column_proc = "_".join(column.split("_")[3:])
    MX, MY = common.get_MX_MY(column_proc)
    if MY != proc_MY:
      continue

    effs[proc][MX] = (getEff(dfb, proc, column), getEff(dfa, proc, column))

  print(proc)
  print(effs[proc])

  mxs = sorted(effs[proc].keys())
  dfb_eff = np.array([effs[proc][mx][0] for mx in mxs])
  dfa_eff = np.array([effs[proc][mx][1] for mx in mxs])

  if (len(dfb_eff) == 0) or (len(dfa_eff) == 0):
    continue

  dfa_eff *= max(dfb_eff) / max(dfa_eff)

  plt.plot(mxs, dfb_eff, label="Before (with sculpting)")
  plt.plot(mxs, dfa_eff, label="After (without sculpting)")
  plt.legend()
  plt.title(f"MC for {proc}")
  plt.ylabel(r"Signal efficiency in category designed for $m_X$ [%]")
  plt.xlabel(r"$m_X$")
  plt.savefig(f"mgg_sculpting_eff_spread/{proc}.png")
  plt.clf()

# for proc in NMSSM_procs:
#   MX, MY = common.get_MX_MY(proc)
#   if MY == 100:
#     continue
#   plt.hist(getRecoMX(dfb, proc), alpha=0.5, density=True, label="Before (with sculpting)")
#   plt.hist(getRecoMX(dfb, proc), alpha=0.5, density=True, label="Before (with sculpting)")
#   #plt.hist(getRecoMX(dfa, proc), alpha=0.5, label="After (without sculpting)")
#   plt.savefig(f"mgg_sculpting_eff_spread/{proc}_mx")

  