import json
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import common
import numpy as np
from optimisation.limit import calculateExpectedLimit

def getContributions(sig_proc, no_veto_n, no_veto_sumw, veto_n, veto_sumw):
  contributions = {}

  #procs = ["DY", "WW", "WZ", "ZZ"]
  procs = ["DY", "WZ", "ZZ"]

  for proc in procs:
    conversion = veto_sumw[sig_proc].loc[proc].to_numpy().sum() / no_veto_sumw[sig_proc].loc[proc+"invEveto"].to_numpy().sum()
    conversion_frac_uncert = np.sqrt( (1/veto_n[sig_proc].loc[proc].to_numpy().sum()) + (1/no_veto_n[sig_proc].loc[proc+"invEveto"].to_numpy().sum()) )
    print(proc, conversion, conversion_frac_uncert*conversion)

    #these numbers are for 85 < mgg < 95 but s and b from optim_results are for +-1 sigma so we need to multiply by 0.68
    contributions[proc] = no_veto_sumw[sig_proc].loc[proc+"invEveto"].to_numpy() * conversion * 0.68
    #if proc == "DY": contributions[proc] *= (1-0.967)
    #if proc == "ZZ": contributions[proc] *= (1-0.43)
    #if proc == "WZ": contributions[proc] *= (1-0.93)
    #if proc == "WW": contributions[proc] *= 0
    contributions[proc+"_uncert"] = np.nan_to_num(contributions[proc] * np.sqrt(conversion_frac_uncert**2 + (1/(no_veto_n[sig_proc].loc[proc+"invEveto"].to_numpy()+1e-8))))

  contributions["Total"] = np.sum([contributions[proc] for proc in procs], axis=0)
  contributions["Total_uncert"] = np.sqrt(np.sum([contributions[proc+"_uncert"]**2 for proc in procs], axis=0))

  return pd.DataFrame(contributions)  

with open("Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson/CatOptim/optim_results.json") as f:
  optim_results = json.load(f)
# with open("Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_no_1tau/CatOptim/optim_results.json") as f:
#   optim_results = json.load(f)

xs = pd.read_csv("Outputs_Sep/NMSSM_Y_gg_Low_Mass/Limits_xs_br/param_test_results.csv", index_col=0)

files = list(filter(lambda x: "csv" in x, os.listdir("Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson/yieldTables_n")))

veto_n = {f.split(".")[0]: pd.read_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson/yieldTables_n/{f}", index_col=0) for f in files}
veto_sumw = {f.split(".")[0]: pd.read_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson/yieldTables_sumw/{f}", index_col=0) for f in files}
no_veto_n = {f.split(".")[0]: pd.read_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_InvVeto/yieldTables_n/{f}", index_col=0) for f in files}
no_veto_sumw = {f.split(".")[0]: pd.read_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_InvVeto/yieldTables_sumw/{f}", index_col=0) for f in files}

# veto_n = {f.split(".")[0]: pd.read_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_no_1tau/yieldTables_n/{f}", index_col=0) for f in files}
# veto_sumw = {f.split(".")[0]: pd.read_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_no_1tau/yieldTables_sumw/{f}", index_col=0) for f in files}
# no_veto_n = {f.split(".")[0]: pd.read_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_InvVeto_no_1tau/yieldTables_n/{f}", index_col=0) for f in files}
# no_veto_sumw = {f.split(".")[0]: pd.read_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_InvVeto_no_1tau/yieldTables_sumw/{f}", index_col=0) for f in files}

for f in files:
  sig_proc = f.split(".")[0]
  MX, MY = common.get_MX_MY(sig_proc)
  
  exp_limit = xs[(xs.MX==MX)&(xs.MY==MY)].iloc[0]["Expected 95% CL Limit [fb]"]
  for each in optim_results:
    if each["sig_proc"] == sig_proc:
      #s = np.array(each["nsigs"][::-1]) * (1-0.46)
      s = np.array(each["nsigs"][::-1])
      b = np.array(each["nbkgs"][::-1])
      continue

  contributions = getContributions(sig_proc, no_veto_n, no_veto_sumw, veto_n, veto_sumw)
  contributions["Continuum"] = b

  nominal_limit = calculateExpectedLimit(s, b, np.zeros_like(b))
  #assert nominal_limit == exp_limit, print(nominal_limit, exp_limit)
  worsened_limit = calculateExpectedLimit(s, b+contributions["Total"], np.zeros_like(b))

  s_b = s/b
  frac_of_signal = contributions["Total"] / (s*exp_limit)

  print(f"\n{sig_proc}")
  print("-"*30)
  print(contributions)
  print()
  print(f"Nominal Limit: {nominal_limit:.3f}")
  print(f"Limit (b += DY and VV): {worsened_limit:.3f}")
  print(f"Fractional change: {(worsened_limit-nominal_limit)/nominal_limit:.3f}")
  print()
  #print(pd.DataFrame({"S/B (normalised to max)":(s_b)/max(s_b), "DY and VV as fraction of signal at exp. limit":frac_of_signal}))
  print(pd.DataFrame({"S/B (normalised to max as %)":100*(s_b)/max(s_b), "DY and VV as fraction of signal at exp. limit":frac_of_signal}))
  
  pretty_contributions = {}
  for col in contributions.columns:
    if "uncert" not in col:
      if col != "Continuum":
        new_col = []
        for i, row in contributions.iterrows():
          new_col.append(f'{row[col]:.2f} +- {row[col+"_uncert"]:.2f}')
        pretty_contributions[col] = new_col
      else:
        new_col = []
        for i, row in contributions.iterrows():
          new_col.append(f'{row[col]:.2f}')
        pretty_contributions[col] = new_col
  pretty_contributions = pd.DataFrame(pretty_contributions)
  print(pretty_contributions)
  #pretty_contributions.to_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_InvVeto/{f}_contributions.csv")
  #pretty_contributions.to_csv(f"Outputs_Sep/NMSSM_Y_gg_low_mass_DY_Diboson_InvVeto_no_1tau/{f}_contributions.csv")