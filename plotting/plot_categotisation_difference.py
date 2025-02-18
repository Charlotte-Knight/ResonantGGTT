import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (10,8)

import numpy as np
import scipy.stats as sps

import sys

def getBoundaries(line):
  # a = line.split("[")[1].split("]")[0]
  # a = a.split(" ")
  # while len(a) != 7:
  #   a.remove("")
  # print(a)
  # return [float(num) for num in a]
  return eval("["+"".join(line.split("[")[1]))

best_limit = []
cat_limit = []
boundaries = []
n_bkg = []

with open(sys.argv[1], "r") as f:
  lines = f.readlines()
for i in range(0, len(lines), 7):
  n_bkg_tmp = []
  for j in range(7):
    if j==2:
      print(lines[i+j].split(" ")[0])
      best_limit.append(float(lines[i+j].split(" ")[0]))
      boundaries.append(getBoundaries(lines[i+j]))
    elif j==3:
      print(lines[i+j].split(",")[0][1:])
      cat_limit.append(float(lines[i+j].split(",")[0][1:]))
    elif j>4:
      n_bkg_tmp.append(int(lines[i+j].split(" ")[2]))
  n_bkg.append(n_bkg_tmp)

masses = [260, 270, 280, 290, 300,320,350,400,450,500,550,600,650,700,750,800,900, 1000]
best_limit = best_limit[1:] + [best_limit[0]]
cat_limit = cat_limit[1:] + [cat_limit[0]]
boundaries = boundaries[1:] + [boundaries[0]]
n_bkg = n_bkg[1:] + [n_bkg[0]]

best_limit = np.array(best_limit) * 2
cat_limit = np.array(cat_limit) * 2

print(list(100*abs(best_limit-cat_limit)/best_limit))

plt.plot(masses, best_limit, label="Absolute optimal")
plt.plot(masses, cat_limit, label=r"Using $m_X=280$ cats")
plt.legend()
plt.xlabel(r"$m_X$")
plt.ylabel("Approximate 95% CL limit")
plt.savefig("cat_optimal.pdf")
plt.clf()

# boundaries = np.array(boundaries)
# n_bkg = np.array(n_bkg)

# for i in range(1, 3):
#   plt.plot(masses, boundaries[:, i], label=str(i))
#   plt.xlabel(r"$m_X$")
#   plt.ylabel("Boundary %d"%i)
# plt.legend()
# plt.savefig("boundary_%d.png"%i)
# plt.clf()

# for i in range(3):
#   plt.plot(masses, n_bkg[:, i])
#   plt.xlabel(r"$m_X$")
#   plt.ylabel("N bkg in sidebands %d"%i)
#   plt.savefig("n_bkg_%d.png"%i)
#   plt.clf()