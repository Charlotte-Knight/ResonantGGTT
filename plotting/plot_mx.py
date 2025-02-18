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
columns = list(filter(lambda x: "intermediate" in x, columns)) + ["process_id", "weight", "category", "event"] 

dfb = pd.read_parquet(fb, columns = columns)
dfb = dfb[dfb.process_id==proc_dict["DiPhoton"]]
dfa = pd.read_parquet(fa, columns = columns)
dfa = dfa[dfa.process_id==proc_dict["DiPhoton"]]

reco_MX_mgg = pd.read_parquet(sys.argv[4], columns=["reco_MX_mgg", "event", "process_id"])
reco_MX_mgg.rename(columns={"event": "event2"}, inplace=True)
reco_MX_mgg = reco_MX_mgg[reco_MX_mgg.process_id==proc_dict["DiPhoton"]]

print("merging")
dfb = pd.concat([dfb, reco_MX_mgg], axis=1)
dfa = pd.concat([dfa, reco_MX_mgg], axis=1)
del reco_MX_mgg

dfb = dfb[dfb.reco_MX_mgg != -9]
dfa = dfa[dfa.reco_MX_mgg != -9]

for column in columns:
  if "score" in column:
    proc = "_".join(column.split("_")[3:])
    print(proc)

    dfb_s = dfb.loc[dfb[column] > 0.999, "reco_MX_mgg"]
    dfa_s = dfa.loc[dfa[column] > 0.999, "reco_MX_mgg"]

    plt.hist(dfa.reco_MX_mgg, alpha=0.5, density=True, label="No cut", range=(1, 5), bins=10, histtype="step")
    plt.hist(dfb_s, alpha=0.5, density=True, label="Before", range=(1, 5), bins=10, histtype="step")
    plt.hist(dfa_s, alpha=0.5, density=True, label="After", range=(1, 5), bins=10, histtype="step")
    plt.title(proc)
    plt.legend()
    plt.xlabel("reco_MX_mgg")
    plt.savefig(f"mgg_sculpting_eff_spread/{proc}_mx.png")
    plt.clf()