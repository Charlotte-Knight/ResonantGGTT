import json
import sys
import numpy as np

def get_mx_my(mass):
  mx, my = mass.split("_")
  mx = int(mx)
  my = int(my)
  return mx, my

def greatestDistance(all_masses, mass):
  mx, my = get_mx_my(mass)
  same_mx_masses = np.array([m for m in all_masses if m[0]==mx])
  same_mys = np.sort(same_mx_masses[:,1])

  idx = np.searchsorted(same_mys, my)
  l_idx = idx-1
  h_idx = idx+1

  if l_idx == -1:
    l_idx = idx
  if h_idx == len(same_mys):
    h_idx = idx

  neighbouring_my = np.array([
    same_mys[l_idx],
    same_mys[h_idx]
  ])
  
  distances = abs(neighbouring_my - my)
  
  return np.max(distances) / 2

with open(sys.argv[1], "r") as f:
  model = json.load(f)

grads = []
displacements = []
grad_names = ["grad_dm", "grad_norm_neg", "grad_norm_pos", "grad_sigma"]

all_masses = model["2016"]["0"]
all_masses = [get_mx_my(mass) for mass in all_masses]

meta = []

for year in model.keys():
  for cat in model[year].keys():
    for mass in model[year][cat].keys():
      eff = model[year][cat][mass]["this mass"]["norm"]
      if (eff > 0.01) and (cat not in ["6", "7"]):
        meta.append([year, cat, mass])
        d = model[year][cat][mass]["same_score"]
        grads.append(np.array([d[n] for n in grad_names]))
        
        my_dist = greatestDistance(all_masses, mass)
        displacements.append(my_dist * grads[-1])

grads = abs(np.array(grads))
displacements = abs(np.array(displacements))

# s = grads.sum(axis=1) != 0
# grads = grads[s]
# displacements = displacements[s]

print(grads.mean(axis=0))
print(grads.min(axis=0))
print(grads.max(axis=0))
print(grads.std(axis=0))
for i in range(4):
  print(meta[np.argmax(grads[:,i])])

print()
print(displacements.mean(axis=0))
print(displacements.min(axis=0))
print(displacements.max(axis=0))
print(displacements.std(axis=0))

for i in range(4):
  print(meta[np.argmax(displacements[:,i])])