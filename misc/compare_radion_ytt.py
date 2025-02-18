import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import common

#cols = list(filter(lambda x: "weight" not in x, common.getColumns("Inputs/Radion/merged_nominal.parquet")))
cols = list(filter(lambda x: "weight" not in x.lower(), common.getColumns("/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/Radion/Feb23/merged_nominal.parquet")))
cols += ["weight_central"]

# #rad = pd.read_parquet("Inputs/Radion/merged_nominal.parquet", columns=cols)
# rad = pd.read_parquet("/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/Radion/Feb23/merged_nominal.parquet", columns=cols)
# rad = rad[rad.process_id == -195]
# #ytt = pd.read_parquet("Inputs/Y_tautau/merged_nominal.parquet", columns=cols)
# ytt = pd.read_parquet("/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/NMSSM_Y_tautau/Feb23/merged_nominal.parquet", columns=cols)
# ytt = ytt[ytt.process_id == -274]

for col in cols:
  if "weight" not in col:
    if col == "year": 
      continue
    print(col)
    read_cols = list(set([col, "process_id", "weight_central"]))
    #rad = pd.read_parquet("Inputs/Radion/merged_nominal.parquet", columns=cols)
    rad = pd.read_parquet("/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/Radion/Feb23/merged_nominal.parquet", columns=read_cols)
    rad = rad[rad.process_id == -195]
    #ytt = pd.read_parquet("Inputs/Y_tautau/merged_nominal.parquet", columns=cols)
    ytt = pd.read_parquet("/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/NMSSM_Y_tautau/Feb23/merged_nominal.parquet", columns=read_cols)
    ytt = ytt[ytt.process_id == -274]

    rl, rh = rad[rad[col]!=common.dummy_val][col].quantile(0.01), rad[rad[col]!=common.dummy_val][col].quantile(0.99)
    yl, yh = ytt[ytt[col]!=common.dummy_val][col].quantile(0.01), ytt[ytt[col]!=common.dummy_val][col].quantile(0.99)
    l, h = min([rl, yl]), max([rh, yh])
    if (h-l) == 0:
      continue

    if (abs(l/(h-l)) < 0.2) and (l > 0):
      l = 0

    n, bins, patches = plt.hist(rad[rad[col]!=common.dummy_val][col], bins=50, range=(l, h), density=True, weights=rad[rad[col]!=common.dummy_val]["weight_central"], histtype="step", label="Radion")
    plt.hist(ytt[ytt[col]!=common.dummy_val][col], bins=bins, density=True, weights=ytt[ytt[col]!=common.dummy_val]["weight_central"], histtype="step", label="Ytt")
    plt.xlabel(col)
    plt.legend()
    plt.savefig("plots/radion_ytt_compare/%s.png"%col)
    plt.clf()