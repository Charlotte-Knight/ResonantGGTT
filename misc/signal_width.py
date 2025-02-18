import pandas as pd
import sys
import json
import common
import numpy as np

df = pd.read_parquet(sys.argv[1], columns=["process_id", "Diphoton_mass", "weight_central"])
df.rename({"weight_central":"weight", "Diphoton_mass":"mgg"}, axis=1, inplace=True)

print(df)

with open(sys.argv[2], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

unique_proc_ids = df.process_id.unique()
for proc in proc_dict.keys():
  if ("NMSSM" in proc) and (proc_dict[proc] in unique_proc_ids):
    
    df_sig = df[df.process_id==proc_dict[proc]]
    
    mx, my = common.get_MX_MY(proc)
    #if my != 125: continue 

    N = 10
    df_sr = df_sig[(df_sig.mgg>my-N*(my/125))&(df_sig.mgg<my+N*(my/125))]
    print()
    print(mx, my)
    print(N*(my/125))
    print(df_sr.weight.sum()/df_sig.weight.sum())
    print(df_sig[(df_sig.mgg<my-N*(my/125))].weight.sum() / df_sig.weight.sum())
    print(df_sig[(df_sig.mgg>my+N*(my/125))].weight.sum() / df_sig.weight.sum())
    print(df_sig[(df_sig.mgg>65)&(df_sig.mgg<66)].weight.sum() / df_sig.weight.sum())

    df_sig = df_sig.iloc[np.argsort(abs(df_sig.mgg-my))]

    cdf = np.cumsum(df_sig.weight) / sum(df_sig.weight)
    idx_99 = np.argmin(abs(cdf-0.95))

    width = abs(df_sig.iloc[idx_99].mgg - my)

    print(width, width*(125/my))