from multiprocessing import process
from unicodedata import numeric
import pandas as pd
import numpy as np
from xgboost import plot_importance, train
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
import torch

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
from plotting.training_plots import plotOutputScore
from plotting.training_plots import plotROC
from plotting.training_plots import plotLoss

import common
import models
import preprocessing

import itertools

import copy
import pickle

from scipy.interpolate import interp1d

import tracemalloc
tracemalloc.start()
TRACK_MEMORY = True

def log_memory():
  mem_use = np.array(tracemalloc.get_traced_memory())/1024**3
  print("Current memory usage (GB): %.2f,  Peak memory usage (GB): %.2f"%tuple(mem_use)) 

def getProcId(proc_dict, proc):
  if proc in proc_dict.keys():
    return proc_dict[proc]
  else:
    print(f'Warning: the process "{proc}" is not in the proc_dict')
    return None

def getGJetIds(proc_dict):
  gjet_ids = [getProcId(proc_dict, proc) for proc in common.bkg_procs["GJets"]] + [getProcId(proc_dict, "TTJets")] + [getProcId(proc_dict, "DiPhoton_Low")]
  gjet_ids = [each for each in gjet_ids if each != None]
  return gjet_ids

def loadDataFrame(args, train_features):
  print(">> Loading dataframe")
  if TRACK_MEMORY:  log_memory()
  columns_to_load = ["Diphoton_mass", "weight_central", "process_id", "category", "event", "year"] + train_features
  #columns_to_load += ["LeadPhoton_pixelSeed", "LeadPhoton_electronVeto", "SubleadPhoton_pixelSeed", "SubleadPhoton_electronVeto"] # for DY studies
  columns_to_load = set(columns_to_load)

  
  df = pd.read_parquet(args.parquet_input, columns=columns_to_load)
  if args.dataset_fraction != 1.0:
    df = df.sample(frac=args.dataset_fraction)
  
  df.rename({"weight_central": "weight"}, axis=1, inplace=True)
  
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)['sample_id_map']

  sig_procs_to_keep = set(args.train_sig_procs + args.eval_sig_procs)
  #sig_procs_to_keep.add("NMSSM_XYH_Y_gg_H_WW_MX_300_MY_100")

  sig_ids = [getProcId(proc_dict, proc) for proc in sig_procs_to_keep]
  bkg_ids = [getProcId(proc_dict, proc) for proc in common.bkg_procs["all"]]
  data_ids = [getProcId(proc_dict, "Data")]
  needed_ids = sig_ids+bkg_ids+data_ids
  needed_ids = [each for each in needed_ids if each != None] 

  reversed_proc_dict = {proc_dict[key]:key for key in proc_dict.keys()}
  for i in df.process_id.unique():
    if i in needed_ids: 
      print("> %s"%(reversed_proc_dict[i]).ljust(40), "kept")
    else: 
      print("> %s"%(reversed_proc_dict[i]).ljust(40), "removed")
  df = df[df.process_id.isin(needed_ids)] #drop uneeded processes

  df["y"] = 0
  df.loc[df.process_id.isin(sig_ids), "y"] = 1

  df.drop_duplicates(subset=["process_id", "year", "event", "Diphoton_mass"], inplace=True)

  if TRACK_MEMORY:  log_memory()
  return df, proc_dict

def train_test_split_consistent(MC, test_size, random_state):
  """
  Must ensure that for a given seed, the same splitting occurs for each process.
  If you do not consider this, then the splitting for mx=300 will be different if mx=500
  is left out of training for example.
  """
  print(">> Splitting dataset with %d%% as test"%int(test_size*100))
  if TRACK_MEMORY:  log_memory()

  train_dfs = []
  test_dfs = []

  idx = np.arange(len(MC))
  for proc in MC.process_id.unique():
    train_df, test_df = train_test_split(idx[MC.process_id==proc], test_size=test_size, random_state=random_state)
    train_dfs.append(train_df)
    test_dfs.append(test_df)

  if TRACK_MEMORY:  log_memory()
  return MC.iloc[np.concatenate(train_dfs)], MC.iloc[np.concatenate(test_dfs)]

def cv_fold_consistent(args, train_df, test_df):
  train_dfs = []
  test_dfs = []

  for proc in train_df.process_id.unique():
    proc_df = train_df[train_df.process_id==proc]

    fold_i, n_folds = args.cv_fold.split("/")
    fold_i, n_folds = int(fold_i), int(n_folds)
    kf = KFold(n_splits=n_folds)
    train_idx, test_idx = [each for each in kf.split(np.arange(len(proc_df)))][fold_i-1]
    
    train_proc_df, test_proc_df = proc_df.iloc[train_idx], proc_df.iloc[test_idx]

    train_dfs.append(train_proc_df)
    test_dfs.append(test_proc_df)

  return pd.concat(train_dfs), pd.concat(test_dfs)

#should work with BDT or NN
# def addScores(args, model, train_features, train_df, test_df, data, MX_MY_to_eval=None):
#   pd.options.mode.chained_assignment = None

#   if not args.parquetSystematic: dfs = [train_df, test_df, data]
#   else:                          dfs = [train_df, test_df]

#   #evaluate at nominal mass points
#   if MX_MY_to_eval is None:
#     MX_MY_to_eval = []
#     for sig_proc in args.eval_sig_procs:
#       MX, MY = common.get_MX_MY(sig_proc)
#       MX_MY_to_eval.append([MX, MY])

#   for MX, MY in MX_MY_to_eval:
#     #sig_proc = "XToHHggTauTau_M%d"%MX
#     sig_proc = "NMSSM_XYH_Y_gg_H_tautau_MX_%d_MY_%d"%(MX, MY)
#     print(sig_proc, MX, MY)
#     for df in dfs:
#       df.loc[:, "MX"] = MX
#       df.loc[:, "MY"] = MY
#       df["score_%s"%sig_proc] = model.predict_proba(df[train_features])[:,1]
      
#       #df.loc[:, "score_%s"%sig_proc] = (df["score_%s"%sig_proc] - df["score_%s"%sig_proc].min()) #rescale so everything within 0 and 1
#       #df.loc[:, "score_%s"%sig_proc] = (df["score_%s"%sig_proc] / df["score_%s"%sig_proc].max())
#       assert (df["score_%s"%sig_proc] < 0).sum() == 0
#       assert (df["score_%s"%sig_proc] > 1).sum() == 0

#   pd.options.mode.chained_assignment = "warn"

#specific to the paramNN, will not work with BDT
def addScores(args, model, train_features, train_df, test_df, data, MX_MY_to_eval=None):
  print(">> Adding scores (evaluating)")
  if TRACK_MEMORY:  log_memory()
  pd.options.mode.chained_assignment = None

  if len(data) > 0:
    dfs = [train_df, test_df, data]
  else:
    dfs = [train_df, test_df]
  
  #evaluate at nominal mass points
  if MX_MY_to_eval is None:
    MX_MY_to_eval = []
    for sig_proc in args.eval_sig_procs:
      MX, MY = common.get_MX_MY(sig_proc)
      if (MX, MY) not in MX_MY_to_eval:
        MX_MY_to_eval.append((MX, MY))

  for df in dfs:
    df.loc[:, "MX"] = MX_MY_to_eval[0][0]
    df.loc[:, "MY"] = MX_MY_to_eval[0][1]
  for i, (MX, MY) in enumerate(MX_MY_to_eval):
    for df in dfs:
      df.iloc[i, df.columns.get_loc("MX")] = MX
      df.iloc[i, df.columns.get_loc("MY")] = MY

  for df in dfs:
    all_predictions = model.predict_proba(df[train_features])
    for i, (MX, MY) in enumerate(MX_MY_to_eval):
      sig_proc = common.get_sig_proc(args.train_sig_procs[0], MX, MY)

      print(">", sig_proc, MX, MY)
      df["score_%s"%sig_proc] = all_predictions[i][:,1]
      assert (df["score_%s"%sig_proc] < 0).sum() == 0
      assert (df["score_%s"%sig_proc] > 1).sum() == 0
      assert (df["score_%s"%sig_proc].isna()).sum() == 0

  pd.options.mode.chained_assignment = "warn"
  if TRACK_MEMORY:  log_memory()

def addTransformedScores(args, df, bkg, drop_scores):
  """
  Transforms scores such that bkg is flat.
  """
  if TRACK_MEMORY:  log_memory()

  if args.outputTransformCDFs:
    os.makedirs(args.outputTransformCDFs, exist_ok=True)

  cdfs = {}
  if args.loadTransformCDFs is not None:
    print(">> Loading transform cdfs")
    for score_name in filter(lambda x: x.split("_")[0]=="score", df.columns):
      print(">", score_name)

      with open(os.path.join(args.loadTransformCDFs, score_name+".npy"), "rb") as f:
        bkg_score, bkg_cdf = np.load(f)
      bkg_cdf_spline = interp1d(bkg_score, bkg_cdf, kind='linear')
      cdfs[score_name] = bkg_cdf_spline

  else:
    print(">> Creating transform pdfs")
    for score_name in filter(lambda x: x.split("_")[0]=="score", df.columns):
      print(">", score_name)

      bkg_score = bkg[score_name].to_numpy()
      bkg_weight = bkg["weight"].to_numpy()
      s = np.argsort(bkg_score)
      bkg_weight = bkg_weight[s]
      bkg_score = bkg_score[s]

      # only use positive weighted events to create cdf
      bkg_score = bkg_score[bkg_weight>0]
      bkg_weight = bkg_weight[bkg_weight>0]

      bkg_cdf = np.cumsum(bkg_weight)
      bkg_cdf = bkg_cdf / bkg_cdf[-1]
      assert bkg_cdf[-1] == 1.0

      # interpolation expectes unique values of bkg_score -> remove repeat instances of 0 and 1
      s = ~((bkg_score == 0) | (bkg_score == 1))
      bkg_cdf = bkg_cdf[s]
      bkg_score = bkg_score[s]
      f32 = lambda x: np.array([x], dtype="float32")
      bkg_score = np.concatenate((f32(0.0), bkg_score, f32(1.0)))
      bkg_cdf = np.concatenate((f32(0.0), bkg_cdf, f32(1.0)))

      bkg_cdf_spline = interp1d(bkg_score, bkg_cdf, kind='linear')
      cdfs[score_name] = bkg_cdf_spline
    
      if args.outputTransformCDFs is not None:
        with open(os.path.join(args.outputTransformCDFs, score_name+".npy"), "wb") as f:
          np.save(f, np.array([bkg_score, bkg_cdf]))

  print(">> Transforming scores")
  for score_name in filter(lambda x: x.split("_")[0]=="score", df.columns):
    print(">", score_name)
    sig_proc = "_".join(score_name.split("_")[1:])
    intermediate_name = "intermediate_transformed_score_%s"%sig_proc

    bkg_cdf_spline = cdfs[score_name]
    
    # plot the cdfs (used for debugging)
    # x = np.linspace(0, 1, 100)
    # plt.clf()
    # plt.plot(x, bkg_cdf_spline(x))
    # plt.savefig("cdf/cdf_%s.png"%intermediate_name)
    # plt.clf()
    
    df[intermediate_name] = bkg_cdf_spline(df[score_name])
    
    assert df[intermediate_name].isnull().sum() == 0 # check for NaNs in transformed score
    assert (df[intermediate_name] < 0).sum() == 0
    assert (df[intermediate_name] > 1).sum() == 0

    if drop_scores:
      df.drop(score_name, axis=1, inplace=True)

  if TRACK_MEMORY:  log_memory()

def doROC(args, train_df, test_df, sig_proc, proc_dict):
  #select just bkg and sig_proc
  train_df = train_df[(train_df.y==0)|(train_df.process_id==proc_dict[sig_proc])]
  test_df = test_df[(test_df.y==0)|(test_df.process_id==proc_dict[sig_proc])]

  if args.remove_gjets:
    gjet_ids = getGJetIds(proc_dict)
    train_df = train_df[~train_df.process_id.isin(gjet_ids)]
    test_df = test_df[~test_df.process_id.isin(gjet_ids)]

  train_fpr, train_tpr, t = roc_curve(train_df.y, train_df["score_%s"%sig_proc], sample_weight=train_df.weight)
  test_fpr, test_tpr, t = roc_curve(test_df.y, test_df["score_%s"%sig_proc], sample_weight=test_df.weight)
  train_auc, test_auc = plotROC(train_fpr, train_tpr, test_fpr, test_tpr, os.path.join(args.outdir, sig_proc))
  return train_auc, test_auc

def importance_getter(model, X=None, y=None, w=None):
  model.importance_type = "gain"
  return model.feature_importances_

  # from sklearn.inspection import permutation_importance
  # print(sorted(list(X.columns)))
  # print(X.columns)
  # r = permutation_importance(model, X, y, sample_weight=w, n_repeats=5, random_state=0, n_jobs=-1)
  # print(r)
  # return r["importances_mean"]
  
def featureImportance(args, model, train_features, X=None, y=None, w=None):
  f, ax = plt.subplots(constrained_layout=True)
  f.set_size_inches(10, 20)

  plot_importance(model, ax)
  plt.savefig(os.path.join(args.outdir, args.train_sig_procs[0], "feature_importance.png"))
  plt.close()

  feature_importances = pd.Series(importance_getter(model, X, y, w), index=train_features)
  feature_importances.sort_values(ascending=False, inplace=True)
  print(feature_importances)
  with open(os.path.join(args.outdir, args.train_sig_procs[0], "feature_importances.json"), "w") as f:
    json.dump([feature_importances.to_dict(), feature_importances.index.to_list()], f, indent=4)

def findMassOrdering(args, model, train_df):
  """Find out order of sig procs in the train and test loss arrays from NN training"""
  sig_proc_ordering = ["" for proc in args.train_sig_procs]
  for proc in args.train_sig_procs:
    MX, MY = common.get_MX_MY(proc)
    dummy_X = train_df.iloc[0:1].copy()
    dummy_X.loc[:, "MX"] = MX
    dummy_X.loc[:, "MY"] = MY
    
    trans_dummy_X = model["transformer"].transform(dummy_X)
    for i, mass in enumerate(model["classifier"].mass_key):
      if abs(trans_dummy_X[0,-2:] - mass).sum() < 1e-4: #if found mass match
        sig_proc_ordering[i] = proc
        break
  return sig_proc_ordering

def evaluatePlotAndSave(args, proc_dict, model, train_features, train_df, test_df, data):
  models.setSeed(args.seed)
  addScores(args, model, train_features, train_df, test_df, data)

  if not args.skipPlots:
    print(">> Plotting ROC curves")
    metrics = {}
    for sig_proc in args.eval_sig_procs:
      print(">", sig_proc)
      train_auc, test_auc = doROC(args, train_df, test_df, sig_proc, proc_dict)
      metrics["AUC/%d_%d_train_auc"%common.get_MX_MY(sig_proc)] = train_auc
      metrics["AUC/%d_%d_test_auc"%common.get_MX_MY(sig_proc)] = test_auc
    print(1)
    if hasattr(model["classifier"], "addHyperparamMetrics"):
     model["classifier"].addHyperparamMetrics(metrics)
    print(2)
    if hasattr(model["classifier"], "train_loss"):
      print(3)
      print(">> Plotting loss curves")
      train_loss = model["classifier"].train_loss
      validation_loss = model["classifier"].validation_loss
      plotLoss(train_loss.sum(axis=1), validation_loss.sum(axis=1), args.outdir)
      for i, proc in enumerate(findMassOrdering(args, model, train_df)):
        plotLoss(train_loss[:,i], validation_loss[:,i], os.path.join(args.outdir, proc))
  
  #make sure the writer is closed at this point
  if hasattr(model["classifier"], "writer") and model["classifier"] is not None:
    model["classifier"].writer.close()

  if args.only_ROC: 
    return None
  
  if args.extra_masses is not None:
    with open(args.extra_masses, "r") as f:
      MX_MY_to_eval = json.load(f)
    addScores(args, model, train_features, train_df, test_df, data, MX_MY_to_eval)

  print(">> Creating output dataframe")
  if args.outputOnlyTest:
    output_df = pd.concat([test_df, data])
    output_df.loc[output_df.process_id!=proc_dict["Data"], "weight"] /= args.test_size #scale signal by amount thrown away
  else:
    output_df = pd.concat([test_df, train_df, data])
  print("5 len(df)=%d"%len(output_df))

  del train_df, test_df, data
  output_bkg_MC = output_df[(output_df.y==0) & (output_df.process_id != proc_dict["Data"])]

  transform_bkg = output_bkg_MC

  if args.remove_gjets:
    transform_proc_dict = proc_dict
    gjet_ids = getGJetIds(transform_proc_dict)
    transform_bkg = transform_bkg[~transform_bkg.process_id.isin(gjet_ids)]

  print(">> Transforming scores")
  addTransformedScores(args, output_df, transform_bkg, drop_scores=args.skipPlots)
  output_data = output_df[output_df.process_id == proc_dict["Data"]]
  output_bkg_MC = output_df[(output_df.y==0) & (output_df.process_id != proc_dict["Data"])]
  
  if not args.skipPlots:
    print(">> Plotting output scores")
    for sig_proc in args.eval_sig_procs:
      print(sig_proc)
      output_sig = output_df[output_df.process_id == proc_dict[sig_proc]]
      with np.errstate(divide='ignore', invalid='ignore'): plotOutputScore(output_data, output_sig, output_bkg_MC, proc_dict, sig_proc, os.path.join(args.outdir, sig_proc))

  columns_to_keep = ["Diphoton_mass", "weight", "process_id", "category", "event", "year", "y"]
  #columns_to_keep += ["LeadPhoton_pixelSeed", "LeadPhoton_electronVeto", "SubleadPhoton_pixelSeed", "SubleadPhoton_electronVeto"] # for DY studies
  for column in output_df:
    if "intermediate" in column: columns_to_keep.append(column)
  columns_to_keep = set(columns_to_keep)
  output_df = output_df[columns_to_keep]

  if not args.dropSystematicWeights:
    print(">> Loading all systematic weights")
    weight_columns = sorted(list(filter(lambda x: "weight" in x, common.getColumns(args.parquet_input))))
    dfw = pd.read_parquet(args.parquet_input, columns=weight_columns)
    dfw = dfw.loc[output_df.index]
    output_df = pd.concat([output_df, dfw], axis=1)
  
  print(">> Outputting parquet file")
  output_df.to_parquet(os.path.join(args.outdir, args.outputName))

def main(args):
  os.makedirs(args.outdir, exist_ok=True)
  for sig_proc in args.eval_sig_procs:
    os.makedirs(os.path.join(args.outdir, sig_proc), exist_ok=True)

  models.setSeed(args.seed)

  train_features = common.train_features[args.train_features].copy()
  if "Param" in args.model: 
    train_features += ["MX", "MY"]
  print(">> Training features:\n "+"\n ".join(train_features))

  df, proc_dict = loadDataFrame(args, train_features)

  if args.feature_importance:
    train_features += ["random"]
    df["random"] = np.random.random(size=len(df))

  #shuffle bkg masses if 
  if ("Param" in args.model) and (not args.skipPlots):
    s = (df.y==0)&(df.process_id!=proc_dict["Data"]) # select bkg MC
    df.loc[s, "MX"] = np.random.choice(np.unique(df.loc[df.y==1, "MX"]), size=sum(s))
    df.loc[s, "MY"] = np.random.choice(np.unique(df.loc[df.y==1, "MY"]), size=sum(s))

  MC = df.loc[~(df.process_id==proc_dict["Data"])]
  data = df.loc[df.process_id==proc_dict["Data"]]
  del df

  train_df, test_df = train_test_split_consistent(MC, test_size=args.test_size, random_state=1)
  
  if args.data_as_bkg:
    train_df = pd.concat([train_df[train_df.y==1], data])
  if args.cv_fold is not None:
    train_df, test_df = cv_fold_consistent(args, train_df, test_df)
  
  # remove negative weights from training
  s = train_df.weight>0
  if not args.outputOnlyTest:
    # keep discarded training events in data as a trick to retain all events when outputting
    data = pd.concat([data, train_df[~s]]) 
  train_df = train_df[s] 

  if args.remove_gjets:
    gjet_ids = getGJetIds(proc_dict)
    s = ~train_df.process_id.isin(gjet_ids)
    if not args.outputOnlyTest:
      # keep discarded training events in data as a trick to retain all events when outputting
      data = pd.concat([data, train_df[~s]]) 
    train_df = train_df[s] 

  if not args.loadModel:
    if "Param" in args.model: classifier = getattr(models, args.model)(n_params=2, n_sig_procs=len(args.train_sig_procs), n_features=preprocessing.getNTransformedFeatures(train_df, train_features), hyperparams=args.hyperparams)
    else:                     classifier = getattr(models, args.model)(args.hyperparams)

    if args.drop_preprocessing:
      to_numpy = preprocessing.FunctionTransformer(lambda X, y=None: X.to_numpy())
      model = Pipeline([('to_numpy', to_numpy), ('classifier', classifier)])
      #model = Pipeline([('classifier', classifier)])
    else:
      numeric_features, categorical_features = preprocessing.autoDetermineFeatureTypes(train_df, train_features)
      print("> Numeric features:\n" +"\n ".join(numeric_features))
      print("> Categorical features:\n "+"\n ".join(categorical_features))
      model = Pipeline([('transformer', preprocessing.Transformer(numeric_features, categorical_features)), ('classifier', classifier)])

    sumw_before = train_df.weight.sum() # keep track for sanity check

    # reweight bkg mc per category to data
    # must be careful that these altered weights are not outputeed
    train_df_weight_copy = train_df.weight.copy()
    # for cat in train_df.category.unique():
    #   bkg_mc_s = (train_df.category==cat)&(train_df.y==0)
    #   data_s = data.category==cat
    #   train_df.loc[bkg_mc_s, "weight"] *= ((1-args.test_size)*data.loc[data_s, "weight"].sum()) / train_df.loc[bkg_mc_s, "weight"].sum()

    # # up weight single higgs bkg
    # for proc, proc_id in proc_dict.items():
    #   if "M125" in proc:
    #     train_df.loc[train_df.process_id==proc_id, "weight"] *= 80.0/3.0

    train_sig_ids = [proc_dict[sig_proc] for sig_proc in args.train_sig_procs]
    s = train_df.y==0 | train_df.process_id.isin(train_sig_ids)
    fit_params = {"classifier__w": train_df[s]["weight"]}
    if not args.drop_preprocessing: 
      fit_params["transformer__w"] = train_df[s]["weight"]
    if hasattr(model["classifier"], "setOutdir"): 
      model["classifier"].setOutdir(args.outdir)
    
    print(">> Training")
    if TRACK_MEMORY:  log_memory()
    model.fit(train_df[s][train_features], train_df[s]["y"], **fit_params)
    print(">> Training complete")
    if TRACK_MEMORY:  log_memory()

    train_df.loc[:, "weight"] = train_df_weight_copy
    print(train_df[["weight", "y"]])

    assert sumw_before == train_df.weight.sum() # sanity check

  else:
    with open(args.loadModel, "rb") as f:
      model = pickle.load(f)

  if args.feature_importance:
    featureImportance(args, classifier.model, train_features, train_df[s][train_features], train_df[s]["y"], train_df[s]["weight"])

  evaluatePlotAndSave(args, proc_dict, model, train_features, train_df, test_df, data)

  if args.outputModel is not None:
      with open(args.outputModel, "wb") as f:
        pickle.dump(model, f)

def doParamTests(parser, args):
  args.do_param_tests = False
  
  #training on all
  args_copy = copy.deepcopy(args)
  args_copy.outdir = os.path.join(args.outdir, "all")
  start(parser, common.parserToList(args_copy))

  #training on individual
  if not args.skip_only_test:
    for sig_proc in args.train_sig_procs:
      args_copy = copy.deepcopy(args)
      args_copy.outdir = os.path.join(args.outdir, "only")
      args_copy.train_sig_procs = [sig_proc]
      args_copy.eval_sig_procs = [sig_proc]
      start(parser, common.parserToList(args_copy))

  #skip one
  for sig_proc in args.train_sig_procs:
    args_copy = copy.deepcopy(args)
    args_copy.outdir = os.path.join(args.outdir, "skip")
    args_copy.train_sig_procs.remove(sig_proc)
    args_copy.eval_sig_procs = [sig_proc]
    start(parser, common.parserToList(args_copy))


def doHyperParamSearch(parser, args):
  with open(args.hyperparams_grid, "r") as f:
    grid = json.load(f)
  args.hyperparams_grid = None

  original_outdir = args.outdir

  keys, values = zip(*grid.items())
  experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
  for i, experiment in enumerate(experiments):
    args_copy = copy.deepcopy(args)

    args_copy.outdir = os.path.join(original_outdir, "experiment_%d"%i)
    os.makedirs(args_copy.outdir, exist_ok=True)

    hyperparams_path = os.path.join(args_copy.outdir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
      json.dump(experiment, f, indent=4)
    args_copy.hyperparams = hyperparams_path

    start(parser, common.parserToList(args_copy))

def doCV(parser, args):
  original_outdir = args.outdir
  n_folds = args.do_cv
  
  args.do_cv = 0

  for i in range(1, n_folds+1):
    args_copy = copy.deepcopy(args)
    args_copy.outdir = os.path.join(original_outdir, "cv_fold_%d"%i)
    args_copy.cv_fold = "%d/%d"%(i, n_folds)

    start(parser, common.parserToList(args_copy))

def mergeBatchSplit(outdir, outputName):
  parquet_dir = os.path.join(outdir, outputName.split(".parquet")[0])
  parquet_files = os.listdir(parquet_dir)
  parquet_files = sorted(list(filter(lambda x: ".parquet" in x, parquet_files)))

  dfs = []  

  for i, f in enumerate(parquet_files):
    print(f, flush=True)
    path = os.path.join(parquet_dir, f)
    if i == 0:
      dfs.append(pd.read_parquet(path))
      rnd_idx_drop = np.random.choice(dfs[-1].index, size=int(len(dfs[-1])/2), replace=False)
      dfs[-1].drop(index=rnd_idx_drop, inplace=True)
      dfs[-1].loc[:, "weight"] *= 2
    else:
      score_columns = list(filter(lambda x:"score" in x, common.getColumns(path)))
      dfs.append(pd.read_parquet(path, columns=score_columns))
      dfs[-1].drop(index=rnd_idx_drop, inplace=True)
      #assert ((dfs[-1].index == dfs[0].index).all())
      

  print("Concating", flush=True)
  merged_df = pd.concat(dfs, axis=1)
  print("Outputting", flush=True)
  merged_df.to_parquet(os.path.join(outdir, outputName))

def start(parser, args=None):
  args = parser.parse_args(args)

  if args.batch_split or args.merge_batch_split:
    assert args.loadModel is not None
    assert args.hyperparams_grid is None
    assert not args.do_param_tests
    assert args.do_cv == 0

  if args.eval_sig_procs == None:
    args.eval_sig_procs = args.train_sig_procs
  args.train_sig_procs = common.expandSigProcs(args.train_sig_procs)
  args.train_sig_procs_exclude = common.expandSigProcs(args.train_sig_procs_exclude)
  args.eval_sig_procs = common.expandSigProcs(args.eval_sig_procs)
  
  args.train_sig_procs = list(filter(lambda x: x not in args.train_sig_procs_exclude, args.train_sig_procs))

  """
  if args.feature_importance:
    assert args.model == "BDT"
    assert len(args.train_sig_procs) == len(args.eval_sig_procs) == 1
    assert args.drop_preprocessing
  """

  if args.hyperparams_grid != None:
    assert args.hyperparams == None
    doHyperParamSearch(parser, args)
    return True

  if args.do_param_tests:
    #assert args.batch
    doParamTests(parser, args)
    return True

  if args.do_cv > 0:
    doCV(parser, args)
    return True
  
  if args.batch:
    if (not args.batch_split) or (args.merge_batch_split):
      common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
      return True
    else:
      if args.extra_masses is not None:
        with open(args.extra_masses, "r") as f:
          extra_masses = json.load(f)
        
        example_sig_proc = args.eval_sig_procs[0]
        for MX, MY in extra_masses:
          args.eval_sig_procs.append(common.get_sig_proc(example_sig_proc, MX, MY))
        args.extra_masses = None

      args.outdir = os.path.join(args.outdir, args.outputName.split(".parquet")[0])
      os.makedirs(args.outdir, exist_ok=True)
      eval_sig_procs_copy = args.eval_sig_procs.copy()
      outputName_copy = args.outputName
      for i, proc in enumerate(sorted(eval_sig_procs_copy)):
        if i != 0:
          args.dropSystematicWeights = True

        args.eval_sig_procs = [proc]
        args.outputName = outputName_copy.split(".parquet")[0] + f"_{proc}.parquet"
        common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
      return True

  if args.merge_batch_split:
    mergeBatchSplit(args.outdir, args.outputName)
    return True

  if args.hyperparams != None:
    with open(args.hyperparams, "r") as f:
      args.hyperparams = json.load(f)

  print(">> Will train on:\n "+"\n ".join(args.train_sig_procs))
  print(">> Will evaluate on:\n "+"\n ".join(args.eval_sig_procs))

  tracemalloc.start()

  df = main(args)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--train-sig-procs', '-p', type=str, nargs="+", required=True)
  parser.add_argument('--train-sig-procs-exclude', type=str, nargs="+", default=[])
  parser.add_argument('--eval-sig-procs', type=str, nargs="+")
  parser.add_argument('--train-features', type=str, default="all")
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--model', type=str, default="BDT")
  parser.add_argument('--outputOnlyTest', action="store_true", default=False)
  parser.add_argument('--test-size', type=float, default=0.5)
  parser.add_argument('--drop-preprocessing', action="store_true")
  parser.add_argument('--data-as-bkg', action="store_true", help="Use data as training background")

  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)

  parser.add_argument('--feature-importance', action="store_true")
  parser.add_argument('--do-param-tests', action="store_true")
  parser.add_argument('--skip-only-test', action="store_true")
  parser.add_argument('--only-ROC', action="store_true")
  parser.add_argument('--remove-gjets', action="store_true")
  parser.add_argument('--dataset-fraction', type=float, default=1.0, help="Only use a fraction of the whole dataset.")

  parser.add_argument('--hyperparams',type=str, default=None)
  parser.add_argument('--hyperparams-grid', type=str, default=None)

  parser.add_argument('--do-cv', type=int, default=0, help="Give a non-zero number which specifies the number of folds to do for cv. Will then run script over all folds.")
  parser.add_argument('--cv-fold', type=str, default=None, help="If doing cross-validation, specify the number of folds and which to run on. Example: '--cv-fold 2/5' means the second out of five folds.")

  parser.add_argument('--outputModel', type=str, default=None)
  parser.add_argument('--loadModel', type=str, default=None)
  parser.add_argument('--outputName', type=str, default="output.parquet")
  parser.add_argument('--skipPlots', action="store_true")

  parser.add_argument('--extra-masses', type=str, default=None)

  parser.add_argument('--dropSystematicWeights', action="store_true")

  parser.add_argument('--loadTransformCDFs', type=str, default=None)
  parser.add_argument('--outputTransformCDFs', type=str, default=None)

  parser.add_argument('--batch-split', action="store_true")
  parser.add_argument('--merge-batch-split', action="store_true")

  #import cProfile
  #cProfile.run('start(parser)', 'restats')
  start(parser)
