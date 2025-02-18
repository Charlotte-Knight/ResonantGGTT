import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

analysis = sys.argv[1]

outputs_dir = f"/home/hep/mdk16/PhD/ggtt/ResonantGGTT/Outputs/{analysis}_Feb24/"

if analysis == "Radion":
	asymp_path = os.path.join(outputs_dir, "LimitPlots_fixfnuf_new_theory/Limits_xs/limits.csv")
elif analysis in ["Y_tautau", "Y_gg_High_Mass"]:
  asymp_path = os.path.join(outputs_dir, "LimitPlots_supp/Limits_xs/limits.csv")
else:
	asymp_path = os.path.join(outputs_dir, "LimitPlots_fixfnuf/Limits_xs/limits.csv")

toy_path = os.path.join(outputs_dir, "LimitPlots_Toys_May30_filtered/Limits_xs/limits.csv")

asymp = pd.read_csv(asymp_path)
toy = pd.read_csv(toy_path)


assert len(asymp) == len(toy), f"Lengths don't match: {len(asymp)} vs {len(toy)}"
if analysis in ["Graviton", "Radion"]:
  asymp.sort_values(by=["mass"], inplace=True, ignore_index=True)
  toy.sort_values(by=["mass"], inplace=True, ignore_index=True)
  assert all(asymp.mass == toy.mass)
else:
  asymp.sort_values(by=["MX", "MY"], inplace=True, ignore_index=True)
  toy.sort_values(by=["MX", "MY"], inplace=True, ignore_index=True)
  assert all(asymp.MX == toy.MX) and all(asymp.MY == toy.MY)

rel_diff = ((toy.obs - asymp.obs)) / asymp.obs
pull = abs(asymp.obs - toy.obs) / toy.obsErr

plt.hist(rel_diff, bins=20)
plt.title(analysis)
plt.xlabel("Toys - Asymptotic / Asymptotic")
plt.ylabel("Count")
path = os.path.join(outputs_dir, f"LimitPlots_Toys_May30_filtered/rel_diff_{analysis}.png")
plt.savefig(path, dpi=200)
print(f"Saved to {path}")
plt.clf()

plt.hist(pull, bins=20)
plt.title(analysis)
plt.xlabel("Pull")
plt.ylabel("Count")
path = os.path.join(outputs_dir, f"LimitPlots_Toys_May30_filtered/pull_{analysis}.png")
plt.savefig(path)
print(f"Saved to {path}")
plt.clf()