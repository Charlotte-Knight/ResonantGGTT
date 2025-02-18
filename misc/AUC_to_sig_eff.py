import json
import numpy as np

import sys

def getSigEff(roc, bkg_eff):
  fpr = np.array(roc["test_fpr"])
  tpr = np.array(roc["test_tpr"])

  idx = np.argmin(abs(fpr-bkg_eff))
  return tpr[idx]

if __name__=="__main__":
  bkg_eff = 0.01

  with open(sys.argv[1], "r") as f:
    roc1 = json.load(f)
  with open(sys.argv[2], "r") as f:
    roc2 = json.load(f)

  print("ROC1 Test AUC: %.4f"%roc1["test_auc"])
  print("ROC2 Test AUC: %.4f"%roc2["test_auc"])
  print("AUC difference: %.4f"%(roc2["test_auc"]-roc1["test_auc"]))
  #for bkg_eff in [0.5, 0.2, 0.1, 0.05, 0.01, 0.001]:
  for bkg_eff in [0.1, 0.01, 0.001]:
    print("At %.4f bkg efficiency"%bkg_eff)
    sig_eff_1 = getSigEff(roc1, bkg_eff)
    sig_eff_2 = getSigEff(roc2, bkg_eff)
    print("ROC1 Sig Eff: %.4f"%sig_eff_1)
    print("ROC2 Sig Eff: %.4f"%sig_eff_2)
    print("Sig eff diff: %.4f"%(sig_eff_2-sig_eff_1))