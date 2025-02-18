import correctionlib
import numpy as np
import os
import pandas as pd
import sys
import common

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
  tau_central_columns = list(filter(lambda x: ("idDeepTauVSjet" in x) and ("central" in x), df.columns))
  for column in tau_central_columns:
    df.loc[:, "weight_central"] /= df[column]

  # add column to keep track whether an event had a tau sf
  df["tauIDsf_applied"] = (df[tau_central_columns[0]] != 1.0)

  tau_columns = list(filter(lambda x: "idDeepTauVSjet" in x, df.columns))
  df.drop(columns=tau_columns, inplace=True)

def applyTauCorrection(df, corr, name, year):
  pt = np.array([df.loc[df.year==year, "lead_lepton_pt"], df.loc[df.year==year, "sublead_lepton_pt"]])
  id = np.array([df.loc[df.year==year, "lead_lepton_id"], df.loc[df.year==year, "sublead_lepton_id"]])
  genmatch = 5 * (abs(id) == 15) # set genmatch to 5 if lepton is hadronic tau (at reco)

  """
  Lepton_id != genmatched ID
  For the 1tau_h categories, we can find out whether the tau was genmatched by checking if a tauID sf was applied previously.
  For 2tau_h, we can find out which events had a double mismatching.
  But we cannot find out if one if the taus are mismatched.

  Make assumption - no occasions of a single tau_h mismatch in 2tau category (happens ~2% of time)
  """
  genmatch *= df.loc[df.year==year, "tauIDsf_applied"]

  nom_corrections = corr.evaluate(pt, 0, genmatch, "Loose", "VVLoose", "nom", "pt")
  nom_event_corrections = nom_corrections[0,:] * nom_corrections[1,:]
  
  if name == "nom":
    df.loc[df.year==year, "weight_central"] *= nom_event_corrections
    df.loc[df.year==year, "weight_idDeepTauVSjet_central"] = nom_event_corrections
  else:
    corrections = corr.evaluate(pt, 0, genmatch, "Loose", "VVLoose", name, "pt")
    event_corrections = corrections[0,:] * corrections[1,:]
    #df.loc[df.year==year, f"weight_idDeepTauVSjet_{name}"] = event_corrections / nom_event_corrections
    df.loc[df.year==year, f"weight_idDeepTauVSjet_{name}"] = event_corrections

def initTauColumns(df):
  df["weight_idDeepTauVSjet_central"] = -9999
  syst_names = getSystNames()
  for name in syst_names:
    df[f"weight_idDeepTauVSjet_{name}"] = -9999

def fixTauSF(df, do_syst):
  era_dict = {
    b"2016UL_pos": "2016postVFP",
    b"2016UL_pre": "2016preVFP",
    b"2017": "2017",
    b"2018": "2018"
  }
  
  initTauColumns(df)

  for era in df.year.unique():
    print(era)
    pog_era = era_dict[era]

    cset = correctionlib.CorrectionSet.from_file(f"processInputs/jsonpog-integration/POG/TAU/{pog_era}_UL/tau.json")
    corr = cset["DeepTau2017v2p1VSjet"]
    
    applyTauCorrection(df, corr, "nom", era)
    print(df)

    if do_syst:
      syst_names = getSystNames()
      for name in syst_names:
        print(name)
        applyTauCorrection(df, corr, name, era)

if __name__ == "__main__":
  inf = sys.argv[1]
  outf = sys.argv[2]

  all_columns = common.getColumns(inf)

  columns = ["weight_central", "lead_lepton_pt", "lead_lepton_id", "sublead_lepton_pt", "sublead_lepton_id", "year"] + list(filter(lambda x: ("idDeepTauVSjet" in x) and ("central" in x), all_columns))
  if "year" not in all_columns:
    columns.remove("year")
    must_add_year = True
  else:
    must_add_year = False

  df = pd.read_parquet(inf, columns=columns)
  if must_add_year and "2018" in inf:
    df.loc[:, "year"] = b"2018"
  elif must_add_year:
    raise Exception()
    
  print(df)
  undoTauSF(df)
  print(df)

  do_syst = "nominal" in inf
  fixTauSF(df, do_syst)

  # now read in all columns except weight_central and tauIDvsJet columns
  tau_columns = list(filter(lambda x: "idDeepTauVSjet" in x, all_columns))
  columns = list(set(all_columns).difference(tau_columns+list(df.columns)))
  
  df_rest = pd.read_parquet(inf, columns=columns)
  df = pd.concat([df, df_rest], axis=1)
  df.to_parquet(outf)