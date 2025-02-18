import uproot
import sys
import numpy as np
import os
from tqdm import tqdm
import math

def get_sr(my):
  width = math.ceil(5 * (my/125.))
  low, high = my-width, my+width
  low = max([low, 68]) #don't let lower bound go lower than 68 GeV
  return (low, high)

failures = {}

Ygg = True

names = os.listdir(sys.argv[1])
names = list(filter(lambda x: "my125" not in x, names))
names = list(filter(lambda x: "cr" not in x, names))

for fname in tqdm(names):
  if ".root" not in fname:
    continue

  mx = int(fname.split("mx")[1].split("my")[0])
  my = int(fname.split("my")[1].split("cat")[0])

  f = uproot.open(os.path.join(sys.argv[1], fname))
  key = list(f.keys())[0]
  events = f[key]["CMS_hgg_mass"].array()
  #print(int(max(events)))

  if not Ygg:
    threshold = 130
  else:
    threshold = get_sr(my)[1]

  if max(events) < threshold:
    failures[fname] = max(events)

  # if max(events) < 120:
  #   print(fname, "gt 120")
  # elif max(events) < 130:
  #   print(fname, "gt 130")

print(failures)
print("\n".join(sorted(failures.keys())))

for name in failures.keys():
  cat = name.split("cat")[1].split("_")[0]
  #print(cat)
  if cat == "0":
    print(name)

mass_points = [name.split("cat")[0] for name in failures.keys()]
print(len(set(mass_points)))