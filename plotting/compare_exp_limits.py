import matplotlib.pyplot as plt
import sys
import json
import common

def convert(optim_results):
  new_optim_results = {}
  for results in optim_results:
    new_optim_results[results["sig_proc"]] = results
  return new_optim_results

with open(sys.argv[1]) as f:
  optim_results_1 = json.load(f)
with open(sys.argv[2]) as f:
  optim_results_2 = json.load(f)

optim_results_1 = convert(optim_results_1)
optim_results_2 = convert(optim_results_2)

mx = []
my = []
diff = []

for key in optim_results_1.keys():
  if key in optim_results_2:
    results1 = optim_results_1[key]
    results2 = optim_results_2[key]

    MX, MY = common.get_MX_MY(key)
    mx.append(MX)
    my.append(MY)
    diff.append((results1["optimal_limit"] - results2["optimal_limit"]) / results1["optimal_limit"])

print(diff)

plt.hist2d(mx, my, bins=(16, 31), range=((300, 1100), (125, 900)), weights=diff)
cbar = plt.colorbar(label="Original limit - new limit / original limit (positive is better)")
plt.xlabel("MX")
plt.ylabel("MY")
plt.savefig("compare_exp_limits.pdf")