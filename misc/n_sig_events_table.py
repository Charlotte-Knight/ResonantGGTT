import json
import pandas as pd

mx = []
my = []
cat = []
norm = []
norm_err = []
n_sig_events = []
N_sidebands = []

for N in range(2, 22, 2):
  with open("Outputs/Y_gg_High_Mass/LimitVsMinNum_TestEffErr/%d/Interpolation/model.json"%N, "r") as f:
    model = json.load(f)["2018"]

  for cat_i, catd in model.items():
    for mass_i, massd in catd.items():
      mx.append(int(mass_i.split("_")[0]))
      my.append(int(mass_i.split("_")[1]))
      cat.append(int(cat_i))
      norm.append(massd["this mass"]["norm"])
      if "norm_err" in massd["this mass"].keys():
        norm_err.append(massd["this mass"]["norm_err"])
        n_sig_events.append(massd["this mass"]["n_sig_events"])
      else:
        norm_err.append(0)
        n_sig_events.append(0)
      N_sidebands.append(N)

df = pd.DataFrame({"mx":mx, "my":my, "cat":cat, "norm":norm, "norm_err":norm_err, "n_sig_events":n_sig_events, "N_sidebands":N_sidebands})


