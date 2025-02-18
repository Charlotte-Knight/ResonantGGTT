from concurrent.futures import process
import pandas as pd
import sys
import numpy as np
import processInputs.mass_variables
import common

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def add_ditau_phi(df):
  tau1_px = df.lead_lepton_pt * np.cos(df.lead_lepton_phi)
  tau1_py = df.lead_lepton_pt * np.sin(df.lead_lepton_phi)
  tau2_px = df.sublead_lepton_pt * np.cos(df.sublead_lepton_phi)
  tau2_py = df.sublead_lepton_pt * np.sin(df.sublead_lepton_phi)

  ditau_px = tau1_px + tau2_px
  ditau_py = tau1_py + tau2_py
  df["ditau_phi"] = np.arctan2(ditau_py, ditau_px)
  df.loc[df.category==8, "ditau_phi"] = common.dummy_val

def add_ditau_mass(df):
  tau1_px = df.lead_lepton_pt * np.cos(df.lead_lepton_phi)
  tau1_py = df.lead_lepton_pt * np.sin(df.lead_lepton_phi)
  tau1_pz = df.lead_lepton_pt * np.sinh(df.lead_lepton_eta)
  tau1_P = df.lead_lepton_pt * np.cosh(df.lead_lepton_eta)
  tau1_E = np.sqrt(1.77**2 + tau1_P**2)

  tau2_px = df.sublead_lepton_pt * np.cos(df.sublead_lepton_phi)
  tau2_py = df.sublead_lepton_pt * np.sin(df.sublead_lepton_phi)
  tau2_pz = df.sublead_lepton_pt * np.sinh(df.sublead_lepton_eta)
  tau2_P = df.sublead_lepton_pt * np.cosh(df.sublead_lepton_eta)
  tau2_E = np.sqrt(1.77**2 + tau2_P**2)

  MET_px = df.MET_pt * np.cos(df.MET_phi)
  MET_py = df.MET_pt * np.sin(df.MET_phi)
  MET_eta = np.arcsinh( (tau1_pz+tau2_pz) / (df.lead_lepton_pt+df.sublead_lepton_pt) )
  MET_eta = df.lead_lepton_eta
  MET_pz = df.MET_pt * np.sinh(MET_eta)
  #MET_pz = 0
  #MET_pz = tau1_pz + tau1_pz
  MET_E = np.sqrt(MET_px**2 + MET_py**2 + MET_pz**2)

  ditau_px = tau1_px + tau2_px + MET_px
  ditau_py = tau1_py + tau2_py + MET_py
  ditau_pz = tau1_pz + tau2_pz + MET_pz
  ditau_E = tau1_E + tau2_E + MET_E

  # ditau_px = tau1_px + tau2_px
  # ditau_py = tau1_py + tau2_py
  # ditau_pz = tau1_pz + tau2_pz
  # ditau_E = tau1_E + tau2_E

  df["ditau_mass_2"] = np.sqrt(ditau_E**2 - ditau_px**2 - ditau_py**2 - ditau_pz**2)

def add_reco_MX_met4(df):
  H1_px = df.Diphoton_pt * np.cos(df.Diphoton_phi)
  H1_py = df.Diphoton_pt * np.sin(df.Diphoton_phi)
  H1_pz = df.Diphoton_pt * np.sinh(df.Diphoton_eta)
  H1_P = df.Diphoton_pt * np.cosh(df.Diphoton_eta)
  H1_E = np.sqrt(H1_P**2 + df.Diphoton_mass**2)

  H2_px = df.ditau_pt * np.cos(df.ditau_phi) + df.MET_pt * np.cos(df.MET_phi)
  H2_py = df.ditau_pt * np.sin(df.ditau_phi) + df.MET_pt * np.sin(df.MET_phi)
  H2_pz = df.ditau_pt * np.sinh(df.ditau_eta) * 2
  H2_P = np.sqrt(H2_px**2 + H2_py**2 + H2_pz**2)
  H2_E = np.sqrt(H2_P**2 + df.ditau_mass**2)

  HH_px = H1_px + H2_px
  HH_py = H1_py + H2_py
  HH_pz = H1_pz + H2_pz
  HH_E = H1_E + H2_E

  df["reco_MX_MET"] = np.sqrt(HH_E**2 - HH_px**2 - HH_py**2 - HH_pz**2)

def add_reco_MX_met4_2(df):
  H1_px = df.Diphoton_pt * np.cos(df.Diphoton_phi)
  H1_py = df.Diphoton_pt * np.sin(df.Diphoton_phi)
  H1_pz = df.Diphoton_pt * np.sinh(df.Diphoton_eta)
  H1_P = df.Diphoton_pt * np.cosh(df.Diphoton_eta)
  H1_E = np.sqrt(H1_P**2 + 125**2)

  H2_px = df.ditau_pt * np.cos(df.ditau_phi) + df.MET_pt * np.cos(df.MET_phi)
  H2_py = df.ditau_pt * np.sin(df.ditau_phi) + df.MET_pt * np.sin(df.MET_phi)
  H2_pz = df.ditau_pt * np.sinh(df.ditau_eta) * 2
  H2_P = np.sqrt(H2_px**2 + H2_py**2 + H2_pz**2)
  H2_E = np.sqrt(H2_P**2 + 125**2)

  HH_px = H1_px + H2_px
  HH_py = H1_py + H2_py
  HH_pz = H1_pz + H2_pz
  HH_E = H1_E + H2_E

  df["reco_MX_MET_2"] = np.sqrt(HH_E**2 - HH_px**2 - HH_py**2 - HH_pz**2)

def add_reco_Mgtt_met(df):
  H1_px = df.Diphoton_pt * np.cos(df.Diphoton_phi)
  H1_py = df.Diphoton_pt * np.sin(df.Diphoton_phi)
  H1_pz = df.Diphoton_pt * np.sinh(df.Diphoton_eta)
  H1_P = df.Diphoton_pt * np.cosh(df.Diphoton_eta)
  H1_E = np.sqrt(H1_P**2 + df.Diphoton_mass**2)

  H2_px = df.lead_lepton_pt * np.cos(df.lead_lepton_phi) + df.MET_pt * np.cos(df.MET_phi)
  H2_py = df.lead_lepton_pt * np.sin(df.lead_lepton_phi) + df.MET_pt * np.sin(df.MET_phi)
  H2_pz = df.lead_lepton_pt * np.sinh(df.lead_lepton_eta) * 2
  H2_P = np.sqrt(H2_px**2 + H2_py**2 + H2_pz**2)
  H2_E = np.sqrt(H2_P**2 + 0**2)

  HH_px = H1_px + H2_px
  HH_py = H1_py + H2_py
  HH_pz = H1_pz + H2_pz
  HH_E = H1_E + H2_E

  df["reco_MX_MET"] = np.sqrt(HH_E**2 - HH_px**2 - HH_py**2 - HH_pz**2)

def add_reco_Mgtt_met_2(df):
  H1_px = df.Diphoton_pt * np.cos(df.Diphoton_phi)
  H1_py = df.Diphoton_pt * np.sin(df.Diphoton_phi)
  H1_pz = df.Diphoton_pt * np.sinh(df.Diphoton_eta)
  H1_P = df.Diphoton_pt * np.cosh(df.Diphoton_eta)
  H1_E = np.sqrt(H1_P**2 + df.Diphoton_mass**2)

  H2_px = df.lead_lepton_pt * np.cos(df.lead_lepton_phi) + df.MET_pt * np.cos(df.MET_phi) + df.jet_1_pt * np.cos(df.lead_lepton_phi)
  H2_py = df.lead_lepton_pt * np.sin(df.lead_lepton_phi) + df.MET_pt * np.sin(df.MET_phi) + df.jet_1_pt * np.sin(df.lead_lepton_phi)
  H2_pz = df.lead_lepton_pt * np.sinh(df.lead_lepton_eta) * 2 + df.jet_1_pt * np.sinh(df.jet_1_eta)
  H2_P = np.sqrt(H2_px**2 + H2_py**2 + H2_pz**2)
  H2_E = np.sqrt(H2_P**2 + 0**2)

  HH_px = H1_px + H2_px
  HH_py = H1_py + H2_py
  HH_pz = H1_pz + H2_pz
  HH_E = H1_E + H2_E

  df["reco_MX_MET_2"] = np.sqrt(HH_E**2 - HH_px**2 - HH_py**2 - HH_pz**2)

# def add_Mggt_met1(df):
#   H1_px = df.Diphoton_pt * np.cos(df.Diphoton_phi)
#   H1_py = df.Diphoton_pt * np.sin(df.Diphoton_phi)
#   H1_pz = df.Diphoton_pt * np.sinh(df.Diphoton_eta)
#   H1_P = df.Diphoton_pt * np.cosh(df.Diphoton_eta)
#   H1_E = np.sqrt(H1_P**2 + df.Diphoton_mass**2)

#   H2_px = df.lead_lepton_pt * np.cos(df.lead_lepton_phi)
#   H2_py = df.lead_lepton_pt * np.sin(df.lead_lepton_phi)
#   H2_pz = df.lead_lepton_pt * np.sinh(df.lead_lepton_eta)
#   H2_P = df.lead_lepton_pt * np.cosh(df.lead_lepton_eta)
#   H2_E = np.sqrt(H2_P**2 + 0**2)

#   MET_px = df.MET_pt * np.cos(df.MET_phi)
#   MET_py = df.MET_pt * np.sin(df.MET_phi)
#   MET_pz = 0
#   MET_E = np.sqrt(MET_px**2 + MET_py**2)

#   HH_px = H1_px + H2_px + MET_px
#   HH_py = H1_py + H2_py + MET_py
#   HH_pz = H1_pz + H2_pz + MET_pz
#   HH_E = H1_E + H2_E + MET_E

#   df["reco_MX_MET_2"] = np.sqrt(HH_E**2 - HH_px**2 - HH_py**2 - HH_pz**2)

columns = ["process_id", "category", "weight_central"]
vars = ["pt", "phi", "eta"]
parts = ["Diphoton", "lead_lepton", "sublead_lepton"]
columns += ["%s_%s"%(part, var) for part in parts for var in vars]
columns += ["MET_pt", "MET_phi", "jet_1_pt", "jet_1_eta", "ditau_pt", "ditau_eta", "ditau_mass"]
columns += ["Diphoton_mass"]

df = pd.read_parquet(sys.argv[1], columns=columns)

mass_id = 38
mlow = 0
mtop = 500

df = df[df.process_id.isin([18,mass_id])]
df = df[df.category != 8]
print(df.loc[df.process_id==18, "weight_central"].sum())
print(df.loc[df.process_id==mass_id, "weight_central"].sum())
df.loc[df.process_id==mass_id, "weight_central"] *= df.loc[df.process_id==18, "weight_central"].sum() / df.loc[df.process_id==mass_id, "weight_central"].sum()
print(df.loc[df.process_id==mass_id, "weight_central"].sum())
add_ditau_phi(df)
add_reco_MX_met4(df)
add_reco_MX_met4_2(df)
#add_reco_Mgtt_met(df)
#add_reco_Mgtt_met_2(df)
#add_Mggt_met1(df)
#add_ditau_mass(df)

#df = df[df.process_id==mass_id]

# for cat in df.category.unique():
#   plt.hist(df[df.category==cat].ditau_mass, bins=50, range=(0, mtop), histtype='step', weights=df[df.category==cat].weight_central, label="visible")
#   plt.hist(df[df.category==cat].ditau_mass_2, bins=50, range=(0, mtop), histtype='step', weights=df[df.category==cat].weight_central, label="with MET")
#   #plt.hist(df[df.process_id==18].ditau_mass, bins=50, range=(0, mtop), histtype='step', weights=df[df.process_id==18].weight_central)
#   #plt.hist(df[df.process_id==18].ditau_mass_2, bins=50, range=(0, mtop), histtype='step', weights=df[df.process_id==18].weight_central)
#   plt.legend()
#   plt.savefig("ditau_cat%d.png"%cat)
#   plt.clf()

plt.hist(df.reco_MX_MET, bins=50, range=(mlow, mtop), histtype='step', weights=df.weight_central, label="nom")
plt.hist(df.reco_MX_MET_2, bins=50, range=(mlow, mtop), histtype='step', weights=df.weight_central, label="mh")
plt.hist(df[df.process_id==18].reco_MX_MET, bins=50, range=(mlow, mtop), histtype='step', weights=df[df.process_id==18].weight_central, label="nom")
plt.hist(df[df.process_id==18].reco_MX_MET_2, bins=50, range=(mlow, mtop), histtype='step', weights=df[df.process_id==18].weight_central, label="mh")
print(df.reco_MX_MET.std(), df.reco_MX_MET_2.std())
plt.legend()
plt.savefig("mx.png")

df_sig = df[df.process_id==mass_id]
window = (df_sig.reco_MX_MET.mean()-df_sig.reco_MX_MET.std(), df_sig.reco_MX_MET.mean()+df_sig.reco_MX_MET.std())
s = df.loc[(df.process_id==mass_id)&(df.reco_MX_MET>window[0])&(df.reco_MX_MET<window[1]), "weight_central"].sum()
b = df.loc[(df.process_id==18)&(df.reco_MX_MET>window[0])&(df.reco_MX_MET<window[1]), "weight_central"].sum()
print(window, s, b, s/b)

window = (df_sig.reco_MX_MET_2.mean()-df_sig.reco_MX_MET_2.std(), df_sig.reco_MX_MET_2.mean()+df_sig.reco_MX_MET_2.std())
s = df.loc[(df.process_id==mass_id)&(df.reco_MX_MET_2>window[0])&(df.reco_MX_MET_2<window[1]), "weight_central"].sum()
b = df.loc[(df.process_id==18)&(df.reco_MX_MET_2>window[0])&(df.reco_MX_MET_2<window[1]), "weight_central"].sum()
print(window, s, b, s/b)