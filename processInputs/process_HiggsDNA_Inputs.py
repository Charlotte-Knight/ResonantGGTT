import pandas as pd
import argparse
import numpy as np
import json
import common
import mass_variables
import sys
import tauID

import tracemalloc
get_memory = lambda: np.array(tracemalloc.get_traced_memory())/1024**3

dphi = lambda x, y: abs(x-y) - 2*(abs(x-y) - np.pi) * (abs(x-y) // np.pi)

def add_Deltas(df):
  print(">> Adding delta variables")
  df["Diphoton_deta"] = df.LeadPhoton_eta-df.SubleadPhoton_eta
  df["Diphoton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi, df.SubleadPhoton_phi)**2 + (df.LeadPhoton_eta-df.SubleadPhoton_eta)**2 )

  df["ditau_deta"] = df.lead_lepton_eta - df.sublead_lepton_eta
  df.loc[df.category==8, "ditau_deta"] = common.dummy_val
  df.loc[df.category==8, "ditau_dphi"] = common.dummy_val

  df["Diphoton_lead_lepton_deta"] = df.Diphoton_eta-df.lead_lepton_eta
  df["Diphoton_lead_lepton_dphi"] = dphi(df.Diphoton_phi,df.lead_lepton_phi)
  df["Diphoton_lead_lepton_dR"] = np.sqrt( dphi(df.Diphoton_phi,df.lead_lepton_phi)**2 + (df.Diphoton_eta-df.lead_lepton_eta)**2 )

  df["Diphoton_sublead_lepton_deta"] = df.Diphoton_eta-df.sublead_lepton_eta
  df["Diphoton_sublead_lepton_dphi"] = dphi(df.Diphoton_phi,df.sublead_lepton_phi)
  df["Diphoton_sublead_lepton_dR"] = np.sqrt( dphi(df.Diphoton_phi,df.sublead_lepton_phi)**2 + (df.Diphoton_eta-df.sublead_lepton_eta)**2 )
  df.loc[df.category==8, "Diphoton_sublead_lepton_deta"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_sublead_lepton_dphi"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_sublead_lepton_dR"] = common.dummy_val

  df["Diphoton_ditau_deta"] = df.Diphoton_eta-df.ditau_eta
  df["Diphoton_ditau_dphi"] = dphi(df.Diphoton_phi,df.ditau_phi)
  df["Diphoton_ditau_dR"] = np.sqrt( dphi(df.Diphoton_phi,df.ditau_phi)**2 + (df.Diphoton_eta-df.ditau_eta)**2 )
  df.loc[df.category==8, "Diphoton_ditau_deta"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_ditau_dphi"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_ditau_dR"] = common.dummy_val

  #zgamma variables
  df["LeadPhoton_ditau_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.ditau_phi)**2 + (df.LeadPhoton_eta-df.ditau_eta)**2 )
  df["SubleadPhoton_ditau_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.ditau_phi)**2 + (df.SubleadPhoton_eta-df.ditau_eta)**2 )
  df["LeadPhoton_lead_lepton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.lead_lepton_phi)**2 + (df.LeadPhoton_eta-df.lead_lepton_eta)**2 )
  df["SubleadPhoton_lead_lepton_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.lead_lepton_phi)**2 + (df.SubleadPhoton_eta-df.lead_lepton_eta)**2 )
  df["LeadPhoton_sublead_lepton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.sublead_lepton_phi)**2 + (df.LeadPhoton_eta-df.sublead_lepton_eta)**2 )
  df["SubleadPhoton_sublead_lepton_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.sublead_lepton_phi)**2 + (df.SubleadPhoton_eta-df.sublead_lepton_eta)**2 )
  df.loc[df.category==8, "LeadPhoton_ditau_dR"] = common.dummy_val
  df.loc[df.category==8, "SubleadPhoton_ditau_dR"] = common.dummy_val
  df.loc[df.category==8, "LeadPhoton_sublead_lepton_dR"] = common.dummy_val
  df.loc[df.category==8, "SubleadPhoton_sublead_lepton_dR"] = common.dummy_val

def add_Deltas_less(df):
  print(">> Adding delta variables") 

  df["Diphoton_ditau_deta"] = df.Diphoton_eta-df.ditau_eta
  df.loc[df.category==8, "Diphoton_ditau_deta"] = common.dummy_val
  df.loc[df.ditau_mass==common.dummy_val, "Diphoton_ditau_deta"] = common.dummy_val

  df["Diphoton_lead_lepton_deta"] = df.Diphoton_eta-df.lead_lepton_eta
  df["Diphoton_lead_lepton_dphi"] = dphi(df.Diphoton_phi,df.lead_lepton_phi)
  df["Diphoton_lead_lepton_dR"] = np.sqrt( dphi(df.Diphoton_phi,df.lead_lepton_phi)**2 + (df.Diphoton_eta-df.lead_lepton_eta)**2 )

  df["Diphoton_sublead_lepton_deta"] = df.Diphoton_eta-df.sublead_lepton_eta
  df["Diphoton_sublead_lepton_dphi"] = dphi(df.Diphoton_phi,df.sublead_lepton_phi)
  df["Diphoton_sublead_lepton_dR"] = np.sqrt( dphi(df.Diphoton_phi,df.sublead_lepton_phi)**2 + (df.Diphoton_eta-df.sublead_lepton_eta)**2 )
  df.loc[df.category==8, "Diphoton_sublead_lepton_deta"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_sublead_lepton_dphi"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_sublead_lepton_dR"] = common.dummy_val
  df.loc[df.ditau_mass==common.dummy_val, "Diphoton_sublead_lepton_deta"] = common.dummy_val
  df.loc[df.ditau_mass==common.dummy_val, "Diphoton_sublead_lepton_dphi"] = common.dummy_val
  df.loc[df.ditau_mass==common.dummy_val, "Diphoton_sublead_lepton_dR"] = common.dummy_val

  #zgamma variables
  df["LeadPhoton_ditau_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.ditau_phi)**2 + (df.LeadPhoton_eta-df.ditau_eta)**2 )
  df["SubleadPhoton_ditau_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.ditau_phi)**2 + (df.SubleadPhoton_eta-df.ditau_eta)**2 )
  df["LeadPhoton_lead_lepton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.lead_lepton_phi)**2 + (df.LeadPhoton_eta-df.lead_lepton_eta)**2 )
  df["SubleadPhoton_lead_lepton_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.lead_lepton_phi)**2 + (df.SubleadPhoton_eta-df.lead_lepton_eta)**2 )
  df["LeadPhoton_sublead_lepton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.sublead_lepton_phi)**2 + (df.LeadPhoton_eta-df.sublead_lepton_eta)**2 )
  df["SubleadPhoton_sublead_lepton_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.sublead_lepton_phi)**2 + (df.SubleadPhoton_eta-df.sublead_lepton_eta)**2 )
  df.loc[df.category==8, "LeadPhoton_ditau_dR"] = common.dummy_val
  df.loc[df.category==8, "SubleadPhoton_ditau_dR"] = common.dummy_val
  df.loc[df.category==8, "LeadPhoton_sublead_lepton_dR"] = common.dummy_val
  df.loc[df.category==8, "SubleadPhoton_sublead_lepton_dR"] = common.dummy_val
  df.loc[df.ditau_mass==common.dummy_val, "LeadPhoton_ditau_dR"] = common.dummy_val
  df.loc[df.ditau_mass==common.dummy_val, "SubleadPhoton_ditau_dR"] = common.dummy_val
  df.loc[df.ditau_mass==common.dummy_val, "LeadPhoton_sublead_lepton_dR"] = common.dummy_val
  df.loc[df.ditau_mass==common.dummy_val, "SubleadPhoton_sublead_lepton_dR"] = common.dummy_val

def add_MET_variables(df):
  print(">> Adding MET variables")
  # met_dphi variables already exist for diphoton and lead_lepton
  df["ditau_met_dPhi"] = dphi(df.MET_phi, df.ditau_phi)
  df.loc[df.category==8, "ditau_met_dPhi"] = common.dummy_val

  df["sublead_lepton_met_dPhi"] = dphi(df.MET_phi, df.sublead_lepton_phi)
  df.loc[df.category==8, "sublead_lepton_met_dPhi"] = common.dummy_val

def add_ditau_phi(df):
  print(">> Adding ditau phi")
  tau1_px = df.lead_lepton_pt * np.cos(df.lead_lepton_phi)
  tau1_py = df.lead_lepton_pt * np.sin(df.lead_lepton_phi)
  tau2_px = df.sublead_lepton_pt * np.cos(df.sublead_lepton_phi)
  tau2_py = df.sublead_lepton_pt * np.sin(df.sublead_lepton_phi)

  ditau_px = tau1_px + tau2_px
  ditau_py = tau1_py + tau2_py
  df["ditau_phi"] = np.arctan2(ditau_py, ditau_px)
  df.loc[df.category==8, "ditau_phi"] = common.dummy_val

def applyPixelVeto(df):
  print(">> Applying Pixel Veto")
  pixel_veto = (df.LeadPhoton_pixelSeed==0) & (df.SubleadPhoton_pixelSeed==0)
  return df[pixel_veto]

def checkNanAndInf(df):
  print(">> Checking for any NaN or inf")
  pd.options.mode.use_inf_as_na = True
  
  row_containing_null_selection = df.isnull().any(axis=1)
  if row_containing_null_selection.sum() > 100:
    print(df[row_containing_null_selection])
    raise Exception("More than 100 rows with NaNs in")
  else:
    df.drop(df.index[row_containing_null_selection], axis=0, inplace=True)

  #assert not df.isnull().any().any(), print("> Offending columns:\n"+str(df.isnull().sum()[df.isnull().sum()>0])) 

def selectSigProcs(df, proc_dict, sig_procs):
  print(">> Filtering signal processes")
  data_bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"]+["Data"]]
  sig_proc_ids = [proc_dict[proc] for proc in sig_procs]
  wanted_ids = data_bkg_ids + sig_proc_ids

  absent_proc_ids = set(wanted_ids).difference(df.process_id.unique())
  if 0 in absent_proc_ids:
    print("> Warning: data is not in this parquet file")
    absent_proc_ids.remove(0)
  assert len(absent_proc_ids) == 0, print("Requesting processes (process_id in %s) that are not in the dataframe"%(str(absent_proc_ids)))
  unwanted_ids = set(df.process_id.unique()).difference(wanted_ids)
  reverse_proc_dict = {proc_dict[name]:name for name in proc_dict.keys()}
  print("> Dropping processes:\n "+"\n ".join([reverse_proc_dict[i] for i in unwanted_ids]))

  return df[df.process_id.isin(wanted_ids)]

def memEfficientRead(parquet_input, columns=None):
  """
  Reads parquet one columns at a time and converts any float64 to float32
  """
  print(">> Reading parquet file", flush=True)
  dfs = []
  if columns == None:
    columns = common.getColumns(parquet_input)
  
  for col in (columns):
    print(col, flush=True)
    df = pd.read_parquet(parquet_input, columns=[col])
    if df[col].dtype == "float64":
      df = df.astype("float32")

    if col == "year":
      df.loc[df.year==b"2016UL_pre", "year"] = "2016"
      df.loc[df.year==b"2016UL_pos", "year"] = "2016"
      df = df.astype("uint16")
    if col == "process_id":
      df = df.astype("int16")
    elif col == "category":
      df = df.astype("uint8")
    elif (col.split("_")[-1] == "id") or (col.split("_")[-1] == "charge"):
      df = df.astype("int8")
      
    dfs.append(df)

  df = pd.concat(dfs, axis=1)
  df.reset_index(inplace=True)

  return df


# def memEfficientRead(parquet_input, columns=None):
#   """
#   Reads parquet one columns at a time and converts any float64 to float32
#   """
#   print(">> Reading parquet file", flush=True)
#   if columns == None:
#     columns = common.getColumns(parquet_input)

#   print("Reading parquet...", flush=True)
#   df = pd.read_parquet(parquet_input, columns=columns)
#   df.info()
#   print("Read", flush=True)
  
#   for i, col in enumerate((df.columns)):
#     print(i, col, flush=True)
#     print(df[col].dtype, flush=True)
#     assert not df[col].isna().any()
#     print("finish check")

#     if df[col].dtype == "float64":
#       print("Converting to float32", flush=True)
#       df[col] = df[col].astype("float32")

#     if col == "year":
#       print(1)
#       df.loc[df.year==b"2016UL_pre", "year"] = "2016"
#       df.loc[df.year==b"2016UL_pos", "year"] = "2016"
#       df[col] = df[col].astype("uint16")
#     if col == "process_id":
#       print(2)
#       df[col] = df[col].astype("int16")
#     elif col == "category":
#       print(3)
#       df[col] = df[col].astype("uint8")
#     elif (col.split("_")[-1] == "id") or (col.split("_")[-1] == "charge"):
#       print(4)
#       df[col] = df[col].astype("int8")

#     print("here")

#   return df

def dropUnwantedFeatures(df, keep_features):
  if keep_features != None:
    print(">> Dropping unwanted features")
    keep_features += ["Diphoton_mass", "MX", "MY", "event", "year", "category", "process_id"] #add other neccessary columns
    keep_features = list(set(keep_features)) #remove overlap in columns
    not_to_keep = [col for col in df.columns if col not in keep_features]
    df.drop(not_to_keep, axis=1, inplace=True)

def renameSVFitFeatures(df):
  print(">> Renaming SVFit features")
  # Adjust naming to line up with naming scheme we had before
  df.loc[:, "eta_tautau_SVFit_bdt"] = df.eta_tautau_SVFit_bdt * np.sign(df.Diphoton_eta)
  df.loc[df.category==8, "eta_tautau_SVFit_bdt"] = common.dummy_val
  df.loc[df.m_tautau_SVFit==common.dummy_val, "eta_tautau_SVFit_bdt"] = common.dummy_val
  df.rename({"eta_tautau_SVFit_bdt":"eta_tautau_SVFit"}, axis=1, inplace=True)

  df.loc[:, "ditau_phi"] = df.Diphoton_phi - df.dPhi_ggtautau_SVFit
  df.loc[df.category==8, "ditau_phi"] = common.dummy_val
  df.loc[df.m_tautau_SVFit==common.dummy_val, "ditau_phi"] = common.dummy_val

  mapper = {
    "pt_tautau_SVFit":"ditau_pt",
    "eta_tautau_SVFit":"ditau_eta",
    "m_tautau_SVFit":"ditau_mass",
    "dR_tautau_SVFit":"ditau_dR",
    "dR_ggtautau_SVFit":"Diphoton_ditau_dR",
    "dPhi_tautau_SVFit":"ditau_dphi",
    "dPhi_ggtautau_SVFit":"Diphoton_ditau_dphi",
    "MET_ll_dPhi":"ditau_met_dPhi"
  }
  to_drop = []
  for col, new_name in mapper.items():
    if new_name in df.columns:
      to_drop.append(new_name)
  df.drop(to_drop, axis=1, inplace=True)

  df.rename(mapper, axis=1, inplace=True)

def main(parquet_input, parquet_output, summary_input, do_test, keep_features, sig_procs=None, drop_pixel_veto=False, undo_LHE_weights=False, remove_out_of_sync=False):
  with open(summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]
  
  if keep_features != None:
    keep_features = common.train_features[keep_features]

  # everything needed to do the calculations + desired features for later
  # wanted_columns = list(set(common.train_features[keep_features]).intersection(common.getColumns(parquet_input)))
  # wanted_columns += ["m_tautau_SVFit", "dR_tautau_SVFit", "dR_ggtautau_SVFit", "dPhi_tautau_SVFit", "dPhi_ggtautau_SVFit"]
  # wanted_columns += ["Diphoton_mass", "event", "year", "category", "process_id"]
  # wanted_columns += ["LeadPhoton_pixelSeed", "SubleadPhoton_pixelSeed", "lead_lepton_id", "sublead_lepton_id", "lead_lepton_charge", "sublead_lepton_charge"]
  # wanted_columns += ["lead_lepton_phi", "lead_lepton_pt", "lead_lepton_eta", "sublead_lepton_phi", "sublead_lepton_pt", "sublead_lepton_eta", "MET_phi"]
  # wanted_columns += ["LeadPhoton_eta", "SubleadPhoton_eta", "LeadPhoton_phi", "SubleadPhoton_phi"]
  # wanted_columns += ["Diphoton_eta", "Diphoton_phi", "Diphoton_pt"]
  # wanted_columns += ["ditau_eta", "pt_tautau_SVFit", "eta_tautau_SVFit_bdt"]
  # wanted_columns = list(set(wanted_columns))
  
  # wanted_columns = None

  # load in everything that isn't a weight (can be a smaller list if needed for memory issues)
  wanted_columns = list(filter(lambda x: "weight" not in x.lower(), common.getColumns(parquet_input)))

  if not do_test:
    df = memEfficientRead(parquet_input, columns=wanted_columns)
  else:
    df = common.getTestSample(parquet_input, columns=wanted_columns)

  renameSVFitFeatures(df)

  print(">> All columns:\n "+"\n ".join(sorted(df.columns)))

  # make selections
  if sig_procs is not None:
    df = selectSigProcs(df, proc_dict, sig_procs)
  if not drop_pixel_veto:
    df = applyPixelVeto(df)
  if remove_out_of_sync:
    out_of_sync_s = (df.category != 8) & (df.ditau_mass == common.dummy_val)
    df = df[~out_of_sync_s]

  print(">> Adding MX and MY values")
  common.add_MX_MY(df, proc_dict)

  keep_features_is_subset = lambda: len(set(keep_features).difference(df.columns)) == 0
  if (keep_features is None) or not keep_features_is_subset():
    #add_ditau_phi(df)
    #add_MET_variables(df)
    #add_Deltas(df)
    #mass_variables.add_reco_MX_MET(df)
    #mass_variables.add_reco_MX_SVFit(df)
    #df.rename({"reco_MX_SVFit": "reco_MX_MET", "reco_MX_SVFit_mgg": "reco_MX_MET_mgg"}, inplace=True)
    
    mass_variables.add_reco_MX(df)
    add_Deltas_less(df)
    if keep_features != None:
      assert keep_features_is_subset(), print("Missing features:"+str(set(keep_features).difference(df.columns)))

  dropUnwantedFeatures(df, keep_features)

  print(">> Columns to output w/o weights:\n "+"\n ".join(sorted(df.columns)))

  # load in all the weights
  wcols = list(filter(lambda x: "weight" in x, common.getColumns(parquet_input)))
  if not do_test:
    dfw = memEfficientRead(parquet_input, columns=wcols)
  else:
    dfw = common.getTestSample(parquet_input, columns=wcols)
  
  print("Finded dropped", flush=True)
  print(df.index.is_unique)
  print(dfw.index.is_unique)
  dropped_indices = set(dfw.index).difference(df.index)
  print("Dropping indices")
  dfw.drop(index = dropped_indices, inplace=True)

  print("Reindexing", flush=True)
  #dfw = dfw.loc[df.index]
  print(df)
  print(dfw)
  assert (df.index == dfw.index).all()
  print("Finished reindexing", flush=True)

  checkNanAndInf(df)

  if undo_LHE_weights:
    lhe_cols = list(filter(lambda x: "lhe" in x, wcols))
    centrals = list(filter(lambda x: "central" in x, lhe_cols))
    for col in centrals:
      dfw.loc[:, "weight_central"] /= dfw[col]
      dfw.loc[:, "weight_central_no_lumi"] /= dfw[col]
    
    dfw.loc[:, lhe_cols] = 1
    dfw.drop(lhe_cols, axis=1, inplace=True)

  df = pd.concat([df, dfw], axis=1)

  df.info(max_cols=200)

  print(">> Final columns:\n "+"\n ".join(sorted(df.columns)))
  print(">> Outputing dataframe with %d events"%len(df))

  df.to_parquet(parquet_output)
  return df

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--parquet-output', '-o', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--test', action="store_true", default=False)
  parser.add_argument('--keep-features', '-f', type=str, default=None)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)
  parser.add_argument('--sig-procs', '-p', type=str, nargs="+", default=None)
  parser.add_argument('--dropPixelVeto', action="store_true", default=False)
  parser.add_argument('--undo-LHE-weights', action="store_true")
  parser.add_argument('--remove-out-of-sync', action="store_true", help="Remove events where HiggsDNA and c++ looper is out of sync (where SVFit variables have not been added")

  args = parser.parse_args()

  if args.sig_procs is not None:
    args.sig_procs = common.expandSigProcs(args.sig_procs)

  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
  else:
    main(args.parquet_input, args.parquet_output, args.summary_input, args.test, args.keep_features, args.sig_procs, args.dropPixelVeto, args.undo_LHE_weights, args.remove_out_of_sync)