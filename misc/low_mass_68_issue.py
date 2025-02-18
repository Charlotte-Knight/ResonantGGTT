import pandas as pd
import json 
import sys
import common

sig_proc = "NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_125"
#sig_proc = "XToHHggTauTau_M1000"
MX, MY = common.get_MX_MY(sig_proc)
cat_num = 1

score = "intermediate_transformed_score_%s"%sig_proc
df = pd.read_parquet(sys.argv[1], columns=["process_id", "weight", "Diphoton_mass", score])
df.rename({score:"score"}, axis=1, inplace=True)

with open(sys.argv[2], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

with open(sys.argv[3], "r") as f:
  optim_results = json.load(f)

for result in optim_results:
  if result["sig_proc"] == sig_proc:
    boundaries = result["category_boundaries"]

selection = (df.process_id==proc_dict[sig_proc]) & (df.score>boundaries[-cat_num-2]) & (df.score<=boundaries[-cat_num-1])
df = df[selection]

print(df.weight.sum() / 138)

df.loc[:, "weight"] /= df.weight.sum()
print_yield = lambda l, h: print(df.loc[(df.Diphoton_mass>l)&(df.Diphoton_mass<h), "weight"].sum())

print(df.weight.sum())
print_yield(50, 500)
print_yield(64.4, 75.6)

print(df.Diphoton_mass.min())
print(df.Diphoton_mass.max())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.hist(df.Diphoton_mass, bins=20)
plt.title(r"$m_X=%d, m_Y=%d$, cat%d"%(MX, MY, cat_num))
plt.xlabel(r"$m_{\gamma\gamma}$")
plt.savefig("low_mass_68.png")