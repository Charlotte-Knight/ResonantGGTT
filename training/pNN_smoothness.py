import pandas as pd
import pickle
import sys

from pyparsing import makeXMLTags
import common
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (8, 6)
import numpy as np
plt.rcParams['figure.constrained_layout.use'] = True

def getScores(model, df, features, m=300):
  df.loc[:, "MX"] = m
  df.loc[:, "MY"] = 125.0
  return model.predict_proba(df[features])[:,1]

def turningpoints(lst):
  dx = np.diff(lst)
  return np.sum(dx[1:] * dx[:-1] < 0)

def loadDataFrame(train_features):
  columns_to_load = ["Diphoton_mass", "weight_central", "process_id", "category", "event", "year"] + train_features
  columns_to_load = set(columns_to_load)

  print(">> Loading dataframe")
  df = pd.read_parquet(sys.argv[1], columns=columns_to_load)
  df.rename({"weight_central": "weight"}, axis=1, inplace=True)
  with open(sys.argv[2]) as f:
    proc_dict = json.load(f)['sample_id_map']

  sig_procs_to_keep = ["XToHHggTauTau_M300"]

  sig_ids = [proc_dict[proc] for proc in sig_procs_to_keep]
  bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"] if proc in proc_dict.keys()]
  data_ids = [proc_dict["Data"]]
  needed_ids = sig_ids+bkg_ids+data_ids
  
  reversed_proc_dict = {proc_dict[key]:key for key in proc_dict.keys()}
  for i in df.process_id.unique():
    if i in needed_ids: print("> %s"%(reversed_proc_dict[i]).ljust(30), "kept")
    else: print("> %s"%(reversed_proc_dict[i]).ljust(30), "removed")
  df = df[df.process_id.isin(needed_ids)] #drop uneeded processes

  df["y"] = 0
  df.loc[df.process_id.isin(sig_ids), "y"] = 1

  return df, proc_dict

with open(sys.argv[3], "rb") as f:
  model = pickle.load(f)

features = common.train_features["important_17_corr"] + ["MX", "MY"]
df, proc_dict = loadDataFrame(features)

#select data
df = df[df.process_id==0]

mx = np.arange(260, 1000)

pd.options.mode.chained_assignment = None 

score_260 = getScores(model, df, features, 260)
df = df[score_260>0.9]

scores = np.array([getScores(model, df, features, m)for m in mx]).T

dx = np.diff(scores, axis=1)
turning_points = np.sum(dx[:, 1:] * dx[:, :-1] < 0, axis=1)

df["turning_points"] = turning_points


plt.hist2d(np.mean(scores, axis=1), turning_points)
plt.colorbar()
plt.savefig("smoothness/score_hist2d.png")
plt.clf()

corr = df.corr(method="spearman")["turning_points"].sort_values()
print(corr)

for point in np.unique(df.turning_points):
  plt.hist(df[df.turning_points==point].reco_MggtauMET_mgg, bins=10, range=(0, 4))
  plt.savefig("smoothness/corr%d.png"%point)
  plt.clf()

reco_mx = df["reco_MggtauMET_mgg"].to_numpy()[turning_points >= 2]
scores = scores[turning_points >= 2]
turning_points = turning_points[turning_points >= 2]
print(len(df))
print(len(scores))

plt.hist(scores.flatten(), bins=100, range=(0, 1))
plt.savefig("smoothness/score_hist.png")
plt.clf()

for i, score in enumerate(scores):
  if i > 100: break
  #print(i)
  plt.plot(mx, scores[i], label="%d %.1f"%(turning_points[i], reco_mx[i]))
  plt.xlabel(r"$m_X$")
  plt.ylabel(r"$f(\vec{x}; m_X)$")
  plt.legend()
  plt.savefig("smoothness/%d.png"%i)
  plt.clf()