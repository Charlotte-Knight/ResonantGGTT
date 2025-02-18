import pandas as pd
import sys
import json
import common
import os
import numpy as np

def getDYMC(proc_dict_loc, dataframe_loc):
  with open(proc_dict_loc, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]
  
  print(dataframe_loc)
  print(os.path.exists(dataframe_loc))
  possible_columns = common.getColumns(dataframe_loc)
  wanted_columns = ["process_id", "event", "LeadPhoton_pixelSeed", "LeadPhoton_electronVeto", "SubleadPhoton_pixelSeed", "SubleadPhoton_electronVeto", "year",
                    "LeadPhoton_pixelSeed", "SubleadPhoton_pixelSeed", "Diphoton_max_mvaID", "Diphoton_min_mvaID"]
  columns_to_load = set(wanted_columns).intersection(possible_columns)
  #df = pd.read_parquet(dataframe_loc, columns=list(columns_to_load))
  df = pd.read_parquet(dataframe_loc)
  
  # if "DY" in proc_dict.keys():
  #   return df[df.process_id==proc_dict["DY"]]
  # else:
  #   return df[df.process_id==proc_dict["DYinvEveto"]]
  return df[df.process_id==0]

def intersectionStats(df1, df2):
  df1 = set(df1.event)
  df2 = set(df2.event)
  print("df1 total N: %d"%len(df1))
  print("df2 total N: %d"%len(df2))

  print("df1 union df2: %d"%len(df1.union(df2)))

old_nominal_all = getDYMC(sys.argv[1], sys.argv[2])
old_inverted_all = getDYMC(sys.argv[3], sys.argv[4])
current_all = getDYMC(sys.argv[5], sys.argv[6])

for year in current_all.year.unique():
  print()
  print(year)
  old_nominal = old_nominal_all[old_nominal_all.year==year]
  old_inverted = old_inverted_all[old_inverted_all.year==year]
  current = current_all[current_all.year==year]

  print("Old Nominal N events: %d"%len(old_nominal))
  print("Old Inverted N events: %d"%len(old_inverted))
  print("Old total N events: %d"%(len(old_inverted)+len(old_nominal)))
  print("Current N events: %d"%len(current))

  print("veto = (df.LeadPhoton_pixelSeed==0)&(df.SubleadPhoton_pixelSeed==0)&(df.LeadPhoton_electronVeto==1)&(df.SubleadPhoton_electronVeto==1)")
  veto = (current.LeadPhoton_pixelSeed==0)&(current.SubleadPhoton_pixelSeed==0)&(current.LeadPhoton_electronVeto==1)&(current.SubleadPhoton_electronVeto==1)
  print("Current Nominal N events: %d"%sum(veto))
  print("Current Inverted N events: %d"%sum(~veto))

  intersectionStats(pd.concat([old_nominal, old_inverted]), current)

old_nominal = old_nominal_all
old_inverted = old_inverted_all
current = current_all

old_nominal = old_nominal[(old_nominal.LeadPhoton_pixelSeed==0)&(old_nominal.SubleadPhoton_pixelSeed==0)]

#old_nominal = old_nominal[(old_nominal.Diphoton_max_mvaID>-0.0)&(old_nominal.Diphoton_min_mvaID>-0.0)]
#old_inverted = old_inverted[(old_inverted.Diphoton_max_mvaID>-0.0)&(old_inverted.Diphoton_min_mvaID>-0.0)]
#current = current[(current.Diphoton_max_mvaID>-0.0)&(current.Diphoton_min_mvaID>-0.0)]


print()
print("all years")
print("Old Nominal N events: %d"%len(old_nominal))
print("Old Inverted N events: %d"%len(old_inverted))
print("Old total N events: %d"%(len(old_inverted)+len(old_nominal)))
print("Current N events: %d"%len(current))

print("veto = (df.LeadPhoton_pixelSeed==0)&(df.SubleadPhoton_pixelSeed==0)&(df.LeadPhoton_electronVeto==1)&(df.SubleadPhoton_electronVeto==1)")
veto = (current.LeadPhoton_pixelSeed==0)&(current.SubleadPhoton_pixelSeed==0)&(current.LeadPhoton_electronVeto==1)&(current.SubleadPhoton_electronVeto==1)
#veto = (current.LeadPhoton_electronVeto==1)&(current.SubleadPhoton_electronVeto==1)
#veto = (current.LeadPhoton_pixelSeed==0)&(current.SubleadPhoton_pixelSeed==0)
print("Current Nominal N events: %d"%sum(veto))
print("Current Inverted N events: %d"%sum(~veto))
intersectionStats(pd.concat([old_nominal, old_inverted]), current)
intersectionStats(old_nominal, current[veto])

old_nominal["isin_current"] = np.isin(old_nominal.event, current.event)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.hist(old_nominal.Diphoton_max_mvaID, bins=100, range=(-1, 1), label="Old Nominal", histtype="step", density=True)
plt.hist(current[veto].Diphoton_max_mvaID, bins=100, range=(-1, 1), label="Current", histtype="step", density=True)
plt.title("Max mvaID")
plt.legend()
plt.savefig("max_mvaid.png")
plt.clf()

plt.hist(old_nominal.Diphoton_min_mvaID, bins=100, range=(-1, 1), label="Old Nominal", histtype="step", density=True)
plt.hist(current[veto].Diphoton_min_mvaID, bins=100, range=(-1, 1), label="Current", histtype="step", density=True)
plt.title("Min mvaID")
plt.legend()
plt.savefig("min_mvaid.png")
plt.clf()

plt.hist(old_nominal.LeadPhoton_mvaID, bins=100, range=(-1, 1), label="Old Nominal", histtype="step", density=True)
plt.hist(current[veto].LeadPhoton_mvaID, bins=100, range=(-1, 1), label="Current", histtype="step", density=True)
plt.title("LeadPhoton mvaID")
plt.legend()
plt.savefig("lead_mvaid.png")
plt.clf()

plt.hist(old_nominal.SubleadPhoton_mvaID, bins=100, range=(-1, 1), label="Old Nominal", histtype="step", density=True)
plt.hist(current[veto].SubleadPhoton_mvaID, bins=100, range=(-1, 1), label="Current", histtype="step", density=True)
plt.title("SubleadPhoton mvaID")
plt.legend()
plt.savefig("sublead_mvaid.png")
plt.clf()