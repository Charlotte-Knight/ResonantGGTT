import json
import sys

with open(f"batch/signalModelling_systematics_py_{sys.argv[1]}.sh", "r") as f:
  bash = f.read().split("\n")
python_line = bash[5]
N_sidebands = python_line.split("/")[4]
print(N_sidebands)

with open(f"batch/signalModelling_systematics_py_{sys.argv[1]}.sh.out", "r") as f:
  out = f.read().split("\n")
d = eval(out[-2])

with open("Outputs/Y_gg_Low_Mass/LimitVsMinNum/{N_sidebands}/Interpolation/systematics.json", "w") as f:
  json.dump(d, f, indent=4, sort_keys=True)
