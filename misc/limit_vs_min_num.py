import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
import numpy as np

def getLimits(results_path):
  with open(results_path, "r") as f:
    results = f.readlines()

  masses = []
  for line in results:
    if "combine" not in line: continue
    m = line.split(".")[0]
    mx = int(m.split("mx")[1].split("my")[0])
    my = int(m.split("my")[1].split("mh")[0])
    mh = 125
    if [mx, my, mh] not in masses:
      masses.append([mx, my, mh])

  limits = np.zeros((5, len(masses)))
  limits_toys = np.zeros((5, len(masses)))
  limits_toys_uncert = np.zeros((5, len(masses)))

  for line in results:
    if "combine" not in line: continue
    m = line.split(".")[0]
    mx = int(m.split("mx")[1].split("my")[0])
    my = int(m.split("my")[1].split("mh")[0])
    mh = 125
    idx1 = masses.index([mx, my, mh])
    if "2.5%" in line:
      idx2=0
    elif "16.0%" in line:
      idx2=1
    elif "50.0%" in line:
      idx2=2
    elif "84.0%" in line:
      idx2=3
    elif "97.5%" in line:
      idx2=4
    
    if "no" in line:
      pass
    elif "toys" in line:
      limit_uncert = line.split("r < ")[1].split("+/-")
      limit, uncert = float(limit_uncert[0]), float(limit_uncert[1].split("@")[0])
      limits_toys[2][idx1] = limit
      limits_toys_uncert[2][idx1] = uncert
    else:
      limit = float(line.split("r < ")[1])
      limits[idx2][idx1] = limit

  masses = np.array(masses)

  return masses, limits, limits_toys, limits_toys_uncert

all_limits = []
all_limits_toys = []
all_limits_toys_uncert = []
N = []
for i in range(2, 21, 2):
  results_path = "/home/hep/mdk16/PhD/ggtt/finalfits_try2/CMSSW_10_2_13/src/flashggFinalFit/Outputs/LimitVsMinNum/%d/Combine/Results/combine_results_summary.txt"%i
  masses, limits, limits_toys, limits_toys_uncert = getLimits(results_path)
  all_limits.append(limits[2])
  all_limits_toys.append(limits_toys[2])
  all_limits_toys_uncert.append(limits_toys_uncert[2])
  N.append(i)

print(all_limits)
print(all_limits_toys)

# for i in [25, 30]:
#   results_path = "/home/hep/mdk16/PhD/ggtt/finalfits_try2/CMSSW_10_2_13/src/flashggFinalFit/Outputs/LimitVsMinNum/%d/Combine/Results/combine_results_summary.txt"%i
#   masses, limits, limits_no_sys = getLimits(results_path)
#   all_limits.append(limits_no_sys[2])
#   N.append(i)

all_limits = np.array(all_limits)
all_limits_toys = np.array(all_limits_toys)
all_limits_toys_uncert = np.array(all_limits_toys_uncert)
N = np.array(N)

f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

axs[0].plot(N, all_limits[:,0], 'r--', label=r"$m_X=1000$ GeV")
axs[0].plot(N, all_limits[:,1], 'g--', label=r"$m_X=400$ GeV")
axs[0].plot(N, all_limits[:,2], 'b--', label=r"$m_X=600$ GeV")

axs[0].errorbar(N-0.05, all_limits_toys[:,0], all_limits_toys_uncert[:,0], fmt='ro', linestyle='')
axs[0].errorbar(N, all_limits_toys[:,1], all_limits_toys_uncert[:,1], fmt='go', linestyle='')
axs[0].errorbar(N+0.05, all_limits_toys[:,2], all_limits_toys_uncert[:,2], fmt='bo', linestyle='')
axs[0].legend(ncol=3)

axs[0].text(2, 0.6, r"$m_Y=90$ GeV", )

idx = np.where(N==10)[0][0]
axs[1].plot(N, all_limits[:,0]/all_limits[idx,0], label="1000", marker='o')
axs[1].plot(N, all_limits[:,1]/all_limits[idx,1], label="300", marker='o')
axs[1].plot(N, all_limits[:,2]/all_limits[idx,2], label="600", marker='o')
#axs[1].plot(N, all_limits[:,0]/min(all_limits[:,0]), label=r"$m_X=1000$")
#axs[1].plot(N, all_limits[:,1]/min(all_limits[:,1]), label=r"$m_X=300$")
#axs[1].plot(N, all_limits[:,2]/min(all_limits[:,2]), label=r"$m_X=600$")

axs[0].set_ylabel(r"Expected 95% CL Limits" +"\n" + r"on $\sigma*BR$ [fb]")
axs[1].set_xlabel(r"Minimum $N$ in sidebands")
axs[1].set_ylabel(r"Ratio to $N=10$")

plt.savefig("Outputs/NMSSM_Y_tautau/LimitVsMinNum/plot.png")
