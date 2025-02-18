import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import pandas as pd
import numpy as np

import os

import json

from plotting.plot_input_features import plot_feature
from training.auc import getAUC

def plotOutputScore(data, sig, bkg, proc_dict, sig_proc, savein, blinded=True, skip_zoom=False):
  bkg_rw = bkg.copy()
  bkg_rw.loc[:, "weight"] *= data.loc[:, "weight"].sum() / bkg_rw.loc[:, "weight"].sum()

  for column in data.columns:
    if ("score" in column) & (sig_proc in column):
      if blinded:
        data_v = data[data[column]<0.88]
      else:
        data_v = data

      plot_feature(data_v, bkg_rw, sig, proc_dict, sig_proc, column, 25, (0,1), os.path.join(savein, column), delete_zeros=True)
      if not skip_zoom:
        plot_feature(data_v, bkg_rw, sig, proc_dict, sig_proc, column, 25, (0.99,1), os.path.join(savein, column+"_zoom"), delete_zeros=True)

def plotROC(train_fpr, train_tpr, test_fpr, test_tpr, savein):
  #train_auc = np.trapz(train_tpr, train_fpr)
  #test_auc = np.trapz(test_tpr, test_fpr)
  train_auc = getAUC(train_fpr, train_tpr)
  test_auc = getAUC(test_fpr, test_tpr)

  # wanted_fpr = np.linspace(0, 1, 1001) #save slices in roc curve
  # save_package = {
  #   "train_auc": train_auc,
  #   "test_auc": test_auc,
  #   "train_fpr": list(train_fpr[np.searchsorted(train_fpr, wanted_fpr)]),
  #   "train_tpr": list(train_tpr[np.searchsorted(train_fpr, wanted_fpr)]),
  #   "test_fpr": list(test_fpr[np.searchsorted(test_fpr, wanted_fpr)]),
  #   "test_tpr": list(test_tpr[np.searchsorted(test_fpr, wanted_fpr)])
  # }

  save_package = {
    "train_auc": train_auc,
    "test_auc": test_auc,
    "train_fpr": list(train_fpr),
    "train_tpr": list(train_tpr),
    "test_fpr": list(test_fpr),
    "test_tpr": list(test_tpr)
  }
  with open(os.path.join(savein, "ROC.json"), "w") as f:
    json.dump(save_package, f, indent=4)

  save_package = {
    "train_auc": train_auc,
    "test_auc": test_auc,
  }
  with open(os.path.join(savein, "ROC_skimmed.json"), "w") as f:
    json.dump(save_package, f, indent=4)

  plt.plot(train_fpr, train_tpr, label="Train AUC = %.4f"%train_auc)
  plt.plot(test_fpr, test_tpr, label="Test AUC = %.4f"%test_auc)
  plt.xlabel("False positive rate")
  plt.ylabel("True positive rate")
  plt.legend()
  plt.savefig(os.path.join(savein, "ROC.png"))

  #plt.xlim(left=0.1)
  #plt.ylim(bottom=min(test_tpr[test_fpr>0.1]), top=1+(1-min(test_tpr[test_fpr>0.1]))*0.1)
  #plt.savefig(os.path.join(savein, "ROC_zoom.png"))
  #plt.xscale("log")
  #plt.savefig(os.path.join(savein, "ROC_log.png"))
  plt.clf()

  plt.plot(train_tpr, train_fpr, label="Train AUC = %.4f"%train_auc)
  plt.plot(test_tpr, test_fpr, label="Test AUC = %.4f"%test_auc)
  plt.ylabel("Background efficiency")
  plt.xlabel("Signal efficiency")
  plt.legend()
  plt.yscale("log")
  plt.ylim(bottom=1e-3, top=2)
  left_sig_eff = train_tpr[np.argmin(abs(train_fpr-1e-3))]
  plt.xlim(left=left_sig_eff, right=1+0.08*(1-left_sig_eff))
  plt.savefig(os.path.join(savein, "ROC_new.png"))
  plt.clf()

  return train_auc, test_auc

def plotLoss(train_loss, validation_loss, savein):
  n_epochs = len(train_loss)

  plt.plot(np.arange(n_epochs), train_loss, label="Train")
  plt.plot(np.arange(n_epochs), validation_loss, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(os.path.join(savein, "loss.png"))
  plt.clf()

if __name__=="__main__":
  import common
  import sys
  import os

  os.makedirs(sys.argv[3], exist_ok=True)

  columns = common.getColumns(sys.argv[1])

  with open(sys.argv[2], "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  sig_procs = [col.split("score_")[1] for col in columns if "score" in col]
  print(len(sig_procs))
  for i, proc in enumerate(sorted(sig_procs)):
    if len(sys.argv) > 4:
      task_id = int(sys.argv[4]) - 1
      if i != task_id:
        continue

    print(proc)
    columns = ["weight", "process_id", "intermediate_transformed_score_%s"%proc]
    df = pd.read_parquet(sys.argv[1], columns=columns)

    data = df[df.process_id==0]
    bkg = df[df.process_id > 0]
    sig = df[df.process_id==proc_dict[proc]]

    plotOutputScore(data, sig, bkg, proc_dict, proc, sys.argv[3], blinded=False, skip_zoom=False)