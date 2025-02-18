import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
#plt.rcParams["figure.figsize"] = (12.5,10)
plt.rcParams["figure.figsize"] = (15,12.5)
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
  results_path = "/home/hep/mdk16/PhD/ggtt/finalfits_try2/CMSSW_10_2_13/src/flashggFinalFit/Outputs/LimitVsMinNum_BetterToys/%d/Combine/Results/combine_results_summary.txt"%i
  masses, limits, limits_toys, limits_toys_uncert = getLimits(results_path)
  all_limits.append(limits[2])
  all_limits_toys.append(limits_toys[2])
  all_limits_toys_uncert.append(limits_toys_uncert[2])
  N.append(i)

all_limits = np.array(all_limits)
all_limits_toys = np.array(all_limits_toys)
all_limits_toys_uncert = np.array(all_limits_toys_uncert)
N = np.array(N)

print(all_limits)
print(all_limits_toys[:,0])
print(all_limits_toys_uncert[:,0])
print(all_limits_toys_uncert[:,0]/all_limits_toys[:,0])

f, axs = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

axs[0].plot(N, all_limits[:,0], 'r--', label=r"$m_X=1000$ GeV")
axs[0].plot(N, all_limits[:,1], 'g--', label=r"$m_X=400$ GeV")
axs[0].plot(N, all_limits[:,2], 'b--', label=r"$m_X=600$ GeV")

axs[0].errorbar(N-0.05, all_limits_toys[:,0], all_limits_toys_uncert[:,0], fmt='rx', linestyle='')
axs[0].errorbar(N, all_limits_toys[:,1], all_limits_toys_uncert[:,1], fmt='gx', linestyle='')
axs[0].errorbar(N+0.05, all_limits_toys[:,2], all_limits_toys_uncert[:,2], fmt='bx', linestyle='')
axs[0].legend(ncol=2, loc="upper left")

axs[0].text(2, 0.6, r"$m_Y=90$ GeV", )
axs[0].set_ylabel(r"Expected 95% CL Limits" +"\n" + r"on $\sigma*BR$ [fb]")
axs[0].set_ylim(top=0.4)


idx = np.where(N==20)[0][0]
r = lambda i: all_limits_toys[:,i]/all_limits_toys[idx,i]
# uncertainty in ratio
u = lambda i: r(i) * np.sqrt( (all_limits_toys_uncert[:,i]/all_limits_toys[:,i])**2 + (all_limits_toys_uncert[idx,i]/all_limits_toys[idx,i])**2  )

axs[1].errorbar(N-0.05, r(0), u(0),  label="1000", fmt='rx-')
axs[1].errorbar(N, r(1), u(1), label="300", fmt='gx-')
axs[1].errorbar(N+0.05, r(2), u(2), label="600", fmt='bx-')
axs[1].plot(N, np.ones_like(N), 'k--')
axs[1].set_ylabel(r"Ratio to $N=20$")

print(u(0))

r = lambda i: all_limits[:,i]/all_limits_toys[:,i]
# uncertainty in ratio
u = lambda i: r(i) * np.sqrt( (all_limits_toys_uncert[:,i]/all_limits_toys[:,i])**2 )

axs[2].errorbar(N-0.05, r(0), u(0),  label="1000", fmt='rx-')
axs[2].errorbar(N, r(1), u(1), label="300", fmt='gx-')
axs[2].errorbar(N+0.05, r(2), u(2), label="600", fmt='bx-')
axs[2].plot(N, np.ones_like(N), 'k--')

axs[2].set_xlabel(r"Minimum $N$ in sidebands")
axs[2].set_ylabel(r"Asym. / Toys")


plt.savefig("Outputs/NMSSM_Y_tautau/LimitVsMinNum_BetterToys/plot.png")
