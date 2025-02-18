import pandas as pd
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12.5,8)

import common
import numpy as np

df = pd.read_parquet(sys.argv[1], columns=["process_id", "year", "weight_central", "category"])
with open(sys.argv[2], "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

df.loc[df.year==b"2016UL_pre", "year"] = "2016"
df.loc[df.year==b"2016UL_pos", "year"] = "2016"
df["year"] = df["year"].astype("uint16")

for year in df.year.unique():
  df.loc[df.year==year, "weight_central"] /= common.lumi_table[year]

#mx = [260, 270, 280, 290, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]
#proc_ids = [proc_dict[f"GluGluToBulkGravitonToHHTo2G2Tau_M-{mxi}"] for mxi in mx]
#proc_ids = [proc_dict[f"GluGluToRadionToHHTo2G2Tau_M-{mxi}"] for mxi in mx]
mx = [300, 400, 500, 600, 700, 800, 900, 1000]
proc_ids = [proc_dict[f"NMSSM_XYH_Y_tautau_H_gg_MX_{mxi}_MY_125"] for mxi in mx]

for year in df.year.unique():
  s = df.year==year
  sumw = [df.loc[s & (df.process_id==proc_id),  "weight_central"].sum() for proc_id in proc_ids]
  sumw2 = [(df.loc[s & (df.process_id==proc_id),  "weight_central"]**2).sum() for proc_id in proc_ids]
  plt.errorbar(mx, sumw, np.sqrt(sumw2), label=str(year))
plt.title("Inclusive")
plt.legend()
plt.savefig("plotting/inclusive.png")
plt.clf()

fig, axs = plt.subplots(2, 4, sharex=True)
axs = axs.flatten()

for i, category in enumerate(range(1,9)):
  for year in df.year.unique():
    s = (df.category==category)&(df.year==year)
    
    sumw = [df.loc[s & (df.process_id==proc_id),  "weight_central"].sum() for proc_id in proc_ids]
    sumw2 = [(df.loc[s & (df.process_id==proc_id),  "weight_central"]**2).sum() for proc_id in proc_ids]

    axs[i].errorbar(mx, sumw, np.sqrt(sumw2), label=str(year))
  axs[i].set_title(common.category_map[category])

axs[0].legend()
fig.savefig(f"plotting/cat_all.png")
