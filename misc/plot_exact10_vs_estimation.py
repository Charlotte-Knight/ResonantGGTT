import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#limits_exact = pd.read_csv("Outputs/Graviton/LimitPlots/Limits_xs_br/limits.csv", index_col=0)
#limits_estimation = pd.read_csv("Outputs/Graviton/LimitPlots_MC_Fit_Final/Limits_xs_br/limits.csv", index_col=0)

limits_exact = pd.read_csv("Outputs_Feb/Graviton/LimitPlots/Limits_xs_br_no_res_bkg/param_test_results.csv", index_col=0)
#limits_estimation = pd.read_csv("Outputs/Graviton/LimitPlots_MC_Fit_Final_AdjError/Limits_xs_br_no_res_bkg/param_test_results.csv", index_col=0)
limits_estimation = pd.read_csv("Outputs/Graviton/LimitPlots/Limits_xs_br_no_res_bkg/param_test_results.csv", index_col=0)

#print(limits_exact)
#print(limits_estimation)

m1 = set(limits_exact.MX.unique())
m2 = set(limits_estimation.MX.unique())
m = np.array(list(m1.intersection(m2)), dtype="int64")
print(m)
print(limits_exact.MX.dtype)

limits_exact = limits_exact[np.isin(limits_exact.MX, m)]
limits_estimation = limits_estimation[np.isin(limits_estimation.MX, m)]

print(limits_exact)
print(limits_estimation)


limits_exact.reset_index(inplace=True)
limits_estimation.reset_index(inplace=True)

assert (limits_exact.MX == limits_estimation.MX).all()

MX = limits_exact.MX
ratio = limits_estimation["Expected 95% CL Limit [fb]"]/limits_exact["Expected 95% CL Limit [fb]"]

plt.scatter(MX, ratio)
plt.xlabel(r"$m_X$")
plt.ylabel("Limit Ratio (Estimation / require exact in data)")
plt.savefig("exact_vs_estimation_limit.pdf")