import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import pandas as pd

import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import common
import json

def getDefColumns(df):
  def_columns = []
  for col in df.columns:
    if (df[col]==-9.0).sum() > 0:
      def_columns.append(col)
  return def_columns

# def makeClusters(d, threshold, features):
#   clusters = []
#   covered_features = []
#   for i, feature in enumerate(features):
#     if feature not in covered_features:
#       idx = np.where(d[i] < threshold)[0]
#       new_cluster = list(np.array(features)[idx])
#       clusters.append(new_cluster)
#       covered_features += new_cluster

#   return clusters

def unravel(l):
  new_list = []
  for each in l:
    if type(each) is list:
      new_list += unravel(each)
    else:
      new_list.append(each)
  return new_list

def makeClusters(d, features):
  clusters_set = [[[i] for i in range(len(features))]]

  for i in range(len(d)):
    new_clusters = clusters_set[-1].copy()
    idx1, idx2 = int(d[i][0]), int(d[i][1])
    new_clusters.append([new_clusters[idx1], new_clusters[idx2]])
    clusters_set.append(new_clusters)

  cleaned_clusters_set = []
  for clusters in clusters_set:
    new_clusters = []
    found_features = set()
    for i in range(len(clusters)-1, -1, -1):
      before_len = len(found_features)
      found_features.update(unravel(clusters[i]))

      if len(found_features) != before_len:
        new_clusters.append(unravel(clusters[i]))

    cleaned_clusters_set.append(new_clusters)

  for i, each1 in enumerate(cleaned_clusters_set):
    for j, each2 in enumerate(each1):
      for k, each3 in enumerate(each2):
        each2[k] = features[each3]

  return cleaned_clusters_set

def getImportantFeatures(clusters, feature_importance):
  important_features = []
  for cluster in clusters:
    most_important = ["", -1]
    for feature in cluster:
      if feature_importance[feature] > most_important[1]:
        most_important = [feature, feature_importance[feature]]
    important_features.append(most_important[0])
  return important_features

#features = common.train_features["important_17_corr"]
features = common.train_features["important_17"] 
print(len(features))
#features = ["category", "Diphoton_sublead_lepton_deta", "Diphoton_dR", "Diphoton_sublead_lepton_dR", "LeadPhoton_sublead_lepton_dR", "ditau_pt", "jet_2_btagDeepFlavB", "lead_lepton_mass", "LeadPhoton_ditau_dR"]
load_features = features + ["process_id", "Diphoton_mass"]
df = pd.read_parquet(sys.argv[1], columns=load_features)

with open(sys.argv[2], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

sig_procs_to_keep = ["XToHHggTauTau_M300"]
sig_ids = [proc_dict[proc] for proc in sig_procs_to_keep]
bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"] if proc in proc_dict.keys()]
data_ids = [proc_dict["Data"]]

# df = df[df.process_id.isin(sig_ids+bkg_ids)]

df_sig = df[df.process_id.isin(sig_ids)]
df_sig = df_sig[df_sig.category!=8]

df_bkg = df[df.process_id.isin(bkg_ids)]
df_bkg = df_bkg[df_bkg.category!=8]
df_bkg = df_bkg.sample(n=len(df_sig), random_state=42)

df = pd.concat([df_sig, df_bkg])

#df = df[df.process_id.isin(sig_ids)]
df = df[df.category!=8]
print(np.unique(df.process_id, return_counts=True))
print(df.process_id.isin(sig_ids).sum())
print(df.process_id.isin(bkg_ids).sum())
#print(len(df))
#df = df[df.b_jet_1_btagDeepFlavB != -9]
#print(len(df))
df["reco_MX_MET_mgg"] = df["reco_MX_MET"] / df["Diphoton_mass"]

features.remove("reco_MX_MET")
features.remove("reco_MX")
features.append("reco_MX_MET_mgg")
print(len(features))
df = df[features]
df.rename(columns={"reco_MX_MET_mgg": "reco_MX_mgg"}, inplace=True)

def_columns = getDefColumns(df)
print(def_columns)
X = df.drop(def_columns, axis=1)
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plt.figure(figsize=(8, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

labels = [common.latex_dict[each] if each in common.latex_dict.keys() else each for each in X.columns]

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
# dendro = hierarchy.dendrogram(
#     dist_linkage, labels=list(X.columns), ax=ax1, leaf_rotation=90
# )
dendro = hierarchy.dendrogram(
    dist_linkage, labels=labels, ax=plt.gca(), leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))
plt.tight_layout()
#plt.xticks(rotation="45", ha="right")

xticks = plt.gca().get_xticklabels()
removed_features = [
  "$\\Delta R(\\gamma\\gamma)$",
  "$\\Delta\\phi(\\gamma\\gamma, \\tau_1)$",
  "$\\Delta\\phi(\\gamma\\gamma, \\tau_2)$"
]
for label in xticks:
  if label.get_text() in removed_features:
    label.set_color("red")

plt.savefig("dendrogram.pdf")
plt.clf()

ax2 = plt.gca()
pos = ax2.imshow(abs(corr[dendro["leaves"], :][:, dendro["leaves"]]))
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
plt.colorbar(pos, ax=ax2, shrink=0.75)
plt.tight_layout()

xticks = plt.gca().get_xticklabels()
yticks = plt.gca().get_yticklabels()
for label in xticks:
  if label.get_text() in removed_features:
    label.set_color("red")
for label in yticks:
  if label.get_text() in removed_features:
    label.set_color("red")

#ax1.set_xticks(labelrotation=45, ha='right')

plt.savefig("correlation.pdf")

clusters_set = makeClusters(dist_linkage, X.columns)

for i, clusters in enumerate(clusters_set):
  print("i: %d   No.clusters: %d"%(i, len(clusters)))
  #for each in clusters:
  #  print(each)

# with open(sys.argv[3], "r") as f:
#   feature_importance = json.load(f)[0]

# clusters = clusters_set[14]
# print(clusters)
# important_features = getImportantFeatures(clusters, feature_importance)
# print(important_features)
# print(len(important_features))