import matplotlib.patches as mpatches
import matplotlib.lines as lines
import matplotlib.collections as collections

import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
import uproot
import sys
import common
import numpy as np
import yaml

import scipy.interpolate as spi

class TheoryPredictionHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = plt.fill_between([x0, x0+width], [y0, y0], [y0+height, y0+height], 
                transform=handlebox.get_transform(), color="red", alpha=0.3)
        handlebox.add_artist(patch)
        line = lines.Line2D([x0+width*0.02, x0+0.98*width], [y0+height/2,y0+height/2], color="red",
                transform=handlebox.get_transform())
        handlebox.add_artist(line)
        return patch


# data = np.load(sys.argv[1])["data"]
# data = np.array([list(each) for each in data])
# kl = data[:,0]
# limits = data[:,1:].T
# print(limits)

with open(sys.argv[1], "r") as f:
  data = yaml.safe_load(f)

kl = np.array([each["value"] for each in data["independent_variables"][0]["values"]])

th = spi.CubicSpline(kl, np.array([each["value"] for each in data["dependent_variables"][0]["values"]]))
th_plus = spi.CubicSpline(kl, np.array([each["errors"][0]["asymerror"]["plus"] for each in data["dependent_variables"][0]["values"]]))
th_minus = spi.CubicSpline(kl, np.array([each["errors"][0]["asymerror"]["minus"] for each in data["dependent_variables"][0]["values"]]))

limit = spi.CubicSpline(kl, np.array([each["value"] for each in data["dependent_variables"][1]["values"]]))
limit_plus1 = spi.CubicSpline(kl, np.array([each["errors"][0]["asymerror"]["plus"] for each in data["dependent_variables"][1]["values"]]))
limit_minus1 = spi.CubicSpline(kl, np.array([each["errors"][0]["asymerror"]["minus"] for each in data["dependent_variables"][1]["values"]]))
limit_plus2 = spi.CubicSpline(kl, np.array([each["errors"][0]["asymerror"]["plus"] for each in data["dependent_variables"][2]["values"]]))
limit_minus2 = spi.CubicSpline(kl ,np.array([each["errors"][0]["asymerror"]["minus"] for each in data["dependent_variables"][2]["values"]]))

limit_obs = spi.CubicSpline(kl, np.array([each["value"] for each in data["dependent_variables"][3]["values"]]))

x = np.linspace(-15, 20, 1000)

obs, = plt.plot(x, limit_obs(x), 'k-',  lw=2, zorder=4, label="Observed")
exp, = plt.plot(x, limit(x), '--k', lw=2, zorder=3, label="Median expected")
pm1 = plt.fill_between(x, limit(x)+limit_minus1(x), limit(x)+limit_plus1(x), zorder=2, facecolor=(0, 0.8, 0), label="68% expected")
pm2 = plt.fill_between(x, limit(x)+limit_minus2(x), limit(x)+limit_plus2(x), zorder=1, facecolor=(1, 0.8, 0), label="95% expected")

legend1 = plt.gca().legend(handles=[obs, exp, pm1, pm2], loc="upper right", title="95% CL upper limits")
plt.gca().add_artist(legend1)

plt.plot(x, th(x), color="red")
theory = plt.fill_between(x, th(x)+th_minus(x), th(x)+th_plus(x), zorder=2, color="red", alpha=0.3, label="Theory prediction")
sm_prediction = plt.scatter([1], [th(1)], marker="*", s=200, color="red", label="SM prediction")
plt.gca().legend(handles=[theory, sm_prediction], loc=(0.26, 0.8), handler_map={collections.PolyCollection: TheoryPredictionHandler()})

plt.xlabel(r"$\kappa_\lambda$")
plt.ylabel(r"$\sigma(pp \rightarrow HH)$ [fb]")

plt.xlim(-15, 20)
plt.ylim(9, 2e5)
plt.yscale("log")

plt.savefig("test.png")

# mplhep.cms.label(llabel="", data=True, lumi=common.tot_lumi, loc=2)

# plt.savefig("kl_paper.pdf")
# plt.savefig("kl_paper.png")

# mplhep.cms.label(llabel="Preliminary", data=True, lumi=common.tot_lumi, loc=2)

# plt.savefig("kl_prelim.pdf")
# plt.savefig("kl_prelim.png")