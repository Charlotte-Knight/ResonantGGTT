import correctionlib
import numpy as np
import os
import pandas as pd

# def getSystNames(era):
#   up_names = []
#   up_names += [f"stat_ptbin{i}_up" for i in range(1, 10)]
#   up_names += [f"syst_{era}_up", "syst_alleras_up"]
#   up_names += [f"stat_highpT_bin{i}_up" for i in range(1, 3)]
#   up_names += ["syst_highpT_up", "syst_highpT_extrap_up"]

#   down_names = [s.replace("up", "down") for s in up_names]
#   return up_names + down_names

def getSystNames():
  up_names = []
  up_names += [f"stat_ptbin{i}_up" for i in range(1, 10)]
  for era in ["2016_preVFP", "2016_postVFP", "2017", "2018"]:
    up_names += [f"syst_{era}_up", "syst_alleras_up"]
  up_names += [f"stat_highpT_bin{i}_up" for i in range(1, 3)]
  up_names += ["syst_highpT_up", "syst_highpT_extrap_up"]

  down_names = [s.replace("up", "down") for s in up_names]
  return up_names + down_names

def checkSystNames(corr, names):
  for name in names:
    found_changed = False
    for pt in np.arange(20, 202, 1.):
      central = corr.evaluate(pt, 0, 5, "Loose", "VVLoose", "nom", "pt")
      syst = corr.evaluate(pt, 0, 5, "Loose", "VVLoose", name, "pt")
      if central != syst:
        found_changed = True
        break
    assert found_changed

def undoTauSF(df):
  print("\n".join(df.columns))
  tau_central_columns = list(filter(lambda x: ("idDeepTauVSjet" in x) and ("central" in x), df.columns))
  for column in tau_central_columns:
    df.loc[:, "weight_central"] /= df.loc[:,column]

  # add column to keep track whether an event had a tau sf
  df["tauIDsf_applied"] = (df[tau_central_columns[0]] != 1.0)

  tau_columns = list(filter(lambda x: "idDeepTauVSjet" in x, df.columns))
  df.drop(columns=tau_columns, inplace=True)

def applyTauCorrection(df, corr, name="nom"):
  pt = np.array([df.lead_lepton_pt, df.sublead_lepton_pt])
  id = np.array([df.lead_lepton_id, df.sublead_lepton_id])
  genmatch = 5 * (abs(id) == 15) # set genmatch to 5 if lepton is hadronic tau (at reco)

  """
  Lepton_id != genmatched ID
  For the 1tau_h categories, we can find out whether the tau was genmatched by checking if a tauID sf was applied previously.
  For 2tau_h, we can find out which events had a double mismatching.
  But we cannot find out if one if the taus are mismatched.

  Make assumption - no occasions of a single tau_h mismatch in 2tau category (happens ~2% of time)
  """
  genmatch *= df.tauIDsf_applied 

  nom_corrections = corr.evaluate(pt, 0, genmatch, "Loose", "VVLoose", "nom", "pt")
  nom_event_corrections = nom_corrections[0,:] * nom_corrections[1,:]
  
  if name == "nom":
    df.loc[:, "weight_central"] *= nom_event_corrections
    df["weight_idDeepTauVSjet_central"] = nom_event_corrections
  else:
    corrections = corr.evaluate(pt, 0, genmatch, "Loose", "VVLoose", name, "pt")
    event_corrections = corrections[0,:] * corrections[1,:]
    df[f"weight_idDeepTauVSjet_{name}"] = event_corrections / nom_event_corrections

def fixTauSF(df, do_syst):
  era_dict = {
    b"2016UL_post": "2016postVFP",
    b"2016UL_pre": "2016preVFP",
    b"2017": "2017",
    b"2018": "2018"
  }
  
  skimmed_df = df.loc[:, ["weight_central", "lead_lepton_pt", "lead_lepton_id", "sublead_lepton_pt", "sublead_lepton_id", "tauIDsf_applied"]]

  era_dfs = []

  for era in df.year.unique():
    era_df = skimmed_df.loc[df.year==era, :]
    
    cset = correctionlib.CorrectionSet.from_file(f"processInputs/jsonpog-integration/POG/TAU/{era_dict[era]}_UL/tau.json")
    corr = cset["DeepTau2017v2p1VSjet"]
    
    applyTauCorrection(era_df, corr, "nom")
    
    if do_syst:
      syst_names = getSystNames()
      for name in syst_names:
        applyTauCorrection(era_df, corr, name)

    era_dfs.append(era_df)

  df_tauID = pd.concat(era_dfs)
  return df_tauID


