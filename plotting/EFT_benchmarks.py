import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
import uproot
import sys
import common

def getLimits(fname):
  f = uproot.open(fname)
  return f["limit/limit"].array()*1000

limit_files = {fname.split("bm")[1].split(".")[0]: fname for fname in sys.argv[1:]}

#bms = limit_files.keys()
bms = ["1", "2", "3", "4", "5", "6", "7", "8", "8a", "9", "10", "11", "12"]

limits = {bm: getLimits(fname) for bm,fname in limit_files.items()}

print(limits)

xticks = []
labels = []

for i, bm in enumerate(bms):
  xticks.append(i)
  labels.append(bm)

  print(bm, ",%d,"%limits[bm][5], "%d"%limits[bm][2])

  w = 0.35

  x = [i-w, i+w]

  if i == 0:
    plt.plot(x, [limits[bm][5], limits[bm][5]], 'k', label="Observed", zorder=3)
    plt.plot(x, [limits[bm][2], limits[bm][2]], 'k', linestyle="dashed", label="Median expected", zorder=3)
    plt.fill_between(x, limits[bm][1], limits[bm][3], color=(0, 0.8, 0), label="68% expected", zorder=2)
    plt.fill_between(x, limits[bm][0], limits[bm][4], color=(1, 0.8, 0), label="95% expected", zorder=1)

  else:
    plt.plot(x, [limits[bm][5], limits[bm][5]], 'k', zorder=3)
    plt.plot(x, [limits[bm][2], limits[bm][2]], 'k', linestyle="dashed", zorder=3)
    plt.fill_between(x, limits[bm][1], limits[bm][3], color=(0, 0.8, 0), zorder=2)
    plt.fill_between(x, limits[bm][0], limits[bm][4], color=(1, 0.8, 0), zorder=1)

plt.xticks(xticks, labels)
plt.yscale("log")

plt.xlabel("Benchmark hypothesis")
plt.ylabel(r"95% CL upper limit on $\sigma(pp \rightarrow HH)$ [fb]")
plt.legend(ncol=2)
plt.ylim(top=1.5e4)

#plt.gca().xaxis.grid(False, which='minor')
plt.gca().tick_params(axis='x', which='minor', bottom=False, top=False)

mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=2)
plt.savefig("eft_paper.pdf")
plt.savefig("eft_paper.png")

mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=2)
plt.savefig("eft_prelim.pdf")
plt.savefig("eft_prelim.png")