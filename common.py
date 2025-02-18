import os
import htcondor

# get LOW_MASS_MODE from environment variable if exists, otherwise default to hard-coded value
LOW_MASS_MODE = os.getenv("LOW_MASS_MODE")
if LOW_MASS_MODE is None:
  LOW_MASS_MODE = False
else:
  LOW_MASS_MODE = (LOW_MASS_MODE == "True")
print('\033[93m')
if LOW_MASS_MODE: print("Running in LOW MASS MODE")
else:             print("Running in HIGH MASS MODE")
print('\033[0m')

dummy_val = -9.0 #value used for missing variables, e.g. sublead_lepton_pt when there is only one lepton

if not LOW_MASS_MODE:  
  lumi_table = {
    2016: 36.31,
    2017: 41.48,
    2018: 59.83
  }
else:
  lumi_table = {
    2016: 36.31,
    2017: 41.48,
    2018: 54.67
  }
tot_lumi = "%.3g"%(lumi_table[2016]+lumi_table[2017]+lumi_table[2018])

category_map = {
  1: "tau/mu",
  2: "tau/ele",
  3: "tau/tau",
  4: "mu/mu",
  5: "ele/ele",
  6: "mu/ele",
  7: "tau/iso_track",
  8: "tau"
}

sig_procs = {
  #"Graviton": [f'{label}{MX}' for label in ["GluGluToBulkGravitonToHHTo2G2Tau_M-", "GluGluToBulkGravitonToHHTo2G2WTo2G2L2Nu_M-", "GluGluToBulkGravitonToHHTo2G2WTo2G2Q1L1Nu_M-"] for MX in [260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]],
  #"Radion": [f'{label}{MX}' for label in ["GluGluToBulkRadionToHHTo2G2Tau_M-", "GluGluToBulkRadionToHHTo2G2WTo2G2L2Nu_M-", "GluGluToBulkRadionToHHTo2G2WTo2G2Q1L1Nu_M-"] for MX in [260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]],
  "Graviton": [f'{label}{MX}' for label in ["GluGluToBulkGravitonToHHTo2G2Tau_M-"] for MX in [260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]],
  "Radion": [f'{label}{MX}' for label in ["GluGluToRadionToHHTo2G2Tau_M-"] for MX in [260, 270, 280, 290, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]],
  "NMSSM_Y_tautau": ['NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_500', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_600', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_700', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_800', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_500', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_500', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_600', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_500', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_600', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_700', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_90']
}
#sig_procs["all"] = sig_procs["X_HH"] + sig_procs["X_YH_Y_gg"] + sig_procs["X_YH_Y_tautau"]
if LOW_MASS_MODE:
  sig_procs["NMSSM_Y_gg"] = ['NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_90']
else:
  #sig_procs["NMSSM_Y_gg"] = ['NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_700', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_800', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_700']
  sig_procs["NMSSM_Y_gg"] = ['NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_700', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_800', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_700']

sig_procs["all"] = sig_procs["Graviton"] + sig_procs["Radion"] + sig_procs["NMSSM_Y_tautau"] + sig_procs["NMSSM_Y_gg"]

# if not LOW_MASS_MODE:
#   bkg_procs = {
#     'Diphoton': ['DiPhoton'],
#     'GJets': ['GJets_HT-100To200', 'GJets_HT-200To400', 'GJets_HT-400To600', 'GJets_HT-40To100', 'GJets_HT-600ToInf'],
#     'TT': ['TTGG', 'TTGamma', 'TTJets'],
#     'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
#     'VGamma': ['WGamma', 'ZGamma'],
#   }
# else:
#   bkg_procs = {
#     #'Diphoton': ['DiPhoton40to80','DiPhoton'],
#     'Diphoton': ['DiPhoton'],
#     'GJets': ['GJets_HT-100To200', 'GJets_HT-200To400', 'GJets_HT-400To600', 'GJets_HT-40To100', 'GJets_HT-600ToInf'],
#     'TT': ['TTGG', 'TTGamma', 'TTJets'],
#     'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
#     'VGamma': ['WGamma', 'ZGamma'],
#     'DY': ['DY'],
#     'VV': ['WW', 'ZZ', 'WZ']
#     #'DY': ['DYinvEveto'],
#     #'VV': ['WWinvEveto', 'ZZinvEveto', 'WZinvEveto']
#   }

if not LOW_MASS_MODE:
  bkg_procs = {
    'Diphoton': ['DiPhoton'],
    'GJets': ['GJets'],
    #'TT': ['TTGG', 'TTGamma', 'TTJets'],
    'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
    'VGamma': ['WGamma', 'ZGamma', 'TTGG', 'TTGamma', 'TTJets']
    #'VGamma': ['WGamma', 'ZGamma', 'TTGG']
  }
else:
  bkg_procs = {
    #'Diphoton': ['DiPhoton40to80','DiPhoton'],
    'Diphoton': ['DiPhoton', "DiPhoton_Low"],
    'GJets': ['GJets'],
    #'TT': ['TTGG', 'TTGamma', 'TTJets'],
    'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
    'VGamma': ['WGamma', 'ZGamma', 'TTGG', 'TTGamma', 'TTJets'],
    'DY': ['DY'],
    'VV': ['WW', 'ZZ', 'WZ']
  }


bkg_procs["all"] = [proc for key in bkg_procs.keys() for proc in bkg_procs[key]]

bkg_procs_latex = {
  "Diphoton": r"$\gamma\gamma$ + jets",
  "GJets": r"$\gamma$ + jets",
  "TT": r"$t\bar{t}$ + jets",
  #"VGamma": r"$V + \gamma$",
  "VGamma": r"$V + \gamma$" + " & " + r"$t\bar{t} + jets$",
  #"SM Higgs": "Single Higgs",
  "SM Higgs": "SM H",
  "DY": "DY",
  "VV": r"$VV$"
}

weights_systematics = ['weight_tau_idDeepTauVSe_sf_AnalysisTau_central', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_down', 'weight_muon_id_sfSTAT_SelectedMuon_down', 'weight_central_initial', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_up', 'weight_muon_iso_sfSTAT_SelectedMuon_down', 'weight_muon_id_sfSYS_SelectedMuon_central', 'weight_muon_iso_sfSYS_SelectedMuon_up', 'weight_electron_id_sf_SelectedElectron_up', 'weight_puWeight_up', 'weight_electron_id_sf_SelectedElectron_central', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_up', 'weight_muon_iso_sfSYS_SelectedMuon_down', 'weight_puWeight_down', 'weight_electron_id_sf_SelectedElectron_down', 'weight_central_no_lumi', 'weight_muon_id_sfSYS_SelectedMuon_up', 'weight_muon_id_sfSTAT_SelectedMuon_central', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_central', 'weight_trigger_sf_central', 'weight_electron_veto_sf_Diphoton_Photon_central', 'weight_muon_iso_sfSTAT_SelectedMuon_central', 'weight_L1_prefiring_sf_up', 'weight_electron_veto_sf_Diphoton_Photon_down', 'weight_muon_id_sfSTAT_SelectedMuon_up', 'weight_electron_veto_sf_Diphoton_Photon_up', 'weight_muon_iso_sfSYS_SelectedMuon_central', 'weight_puWeight_central', 'weight_L1_prefiring_sf_down', 'weight_muon_id_sfSYS_SelectedMuon_down', 'weight_muon_iso_sfSTAT_SelectedMuon_up', 'weight_trigger_sf_down', 'weight_trigger_sf_up', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_down', 'weight_L1_prefiring_sf_central', 'weight_tau_idDeepTauVSe_sf_AnalysisTau_down', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_central', 'weight_tau_idDeepTauVSe_sf_AnalysisTau_up']

train_features = {
  'base': ['Diphoton_eta', 'Diphoton_phi', 'Diphoton_helicity', 'Diphoton_pt_mgg', 'Diphoton_max_mvaID', 'Diphoton_min_mvaID', 'Diphoton_dPhi', 'LeadPhoton_pt_mgg', 'LeadPhoton_eta', 'LeadPhoton_phi', 'LeadPhoton_mvaID', 'SubleadPhoton_pt_mgg', 'SubleadPhoton_eta', 'SubleadPhoton_phi', 'SubleadPhoton_mvaID', 'n_electrons', 'n_muons', 'n_taus', 'n_iso_tracks', 'n_jets', 'n_bjets', 'MET_pt', 'MET_phi', 'diphoton_met_dPhi', 'lead_lepton_met_dphi', 'ditau_dphi', 'ditau_deta', 'ditau_dR', 'ditau_mass', 'ditau_pt', 'ditau_eta', 'lead_lepton_pt', 'lead_lepton_eta', 'lead_lepton_phi', 'lead_lepton_mass', 'lead_lepton_charge', 'lead_lepton_id', 'sublead_lepton_pt', 'sublead_lepton_eta', 'sublead_lepton_phi', 'sublead_lepton_mass', 'sublead_lepton_charge', 'sublead_lepton_id', 'category', 'jet_1_pt', 'jet_1_eta', 'jet_1_btagDeepFlavB', 'jet_2_pt', 'jet_2_eta', 'jet_2_btagDeepFlavB', 'b_jet_1_btagDeepFlavB', 'dilep_leadpho_mass', 'dilep_subleadpho_mass', 'year'],
  'additional': ['reco_MggtauMET_mgg', 'reco_MX', 'reco_MX_MET', 'Diphoton_deta', 'Diphoton_ditau_deta', 'LeadPhoton_lead_lepton_dR', 'Diphoton_lead_lepton_dR', 'LeadPhoton_sublead_lepton_dR', 'Diphoton_sublead_lepton_deta', 'Diphoton_sublead_lepton_dphi', 'Diphoton_lead_lepton_deta', 'SubleadPhoton_lead_lepton_dR', 'Diphoton_lead_lepton_dphi', 'LeadPhoton_ditau_dR', 'SubleadPhoton_ditau_dR', 'SubleadPhoton_sublead_lepton_dR', 'sublead_lepton_met_dPhi', 'ditau_met_dPhi', 'Diphoton_dR', 'Diphoton_sublead_lepton_dR', 'ditau_phi', 'Diphoton_ditau_dphi', 'Diphoton_ditau_dR']
}
train_features['all'] = train_features['base'] + train_features['additional']
train_features['important_corr'] = ["Diphoton_dR", "LeadPhoton_sublead_lepton_dR", "ditau_pt", "jet_2_btagDeepFlavB", "lead_lepton_mass", "LeadPhoton_ditau_dR", "lead_lepton_pt", "dilep_subleadpho_mass", "ditau_mass", "lead_lepton_charge", "dilep_leadpho_mass", "Diphoton_helicity", "LeadPhoton_pt_mgg", "MET_pt", "reco_MX_MET", "Diphoton_lead_lepton_dR", "ditau_met_dPhi", "b_jet_1_btagDeepFlavB", "sublead_lepton_id", "Diphoton_ditau_dR", "n_bjets", "SubleadPhoton_sublead_lepton_dR", "category", "ditau_dR", "jet_1_btagDeepFlavB", "jet_1_pt", "diphoton_met_dPhi", "Diphoton_pt_mgg", "reco_MggtauMET_mgg", "lead_lepton_met_dphi", "SubleadPhoton_pt_mgg", "sublead_lepton_pt", "SubleadPhoton_lead_lepton_dR", "LeadPhoton_lead_lepton_dR", "Diphoton_sublead_lepton_dR"]
train_features['important_30'] = ["Diphoton_sublead_lepton_deta", "Diphoton_dR", "Diphoton_sublead_lepton_dR", "LeadPhoton_sublead_lepton_dR", "ditau_pt", "jet_2_btagDeepFlavB", "lead_lepton_mass", "LeadPhoton_ditau_dR", "lead_lepton_pt", "n_electrons", "Diphoton_ditau_dphi", "dilep_subleadpho_mass", "ditau_mass", "Diphoton_sublead_lepton_dphi", "lead_lepton_charge", "dilep_leadpho_mass", "Diphoton_helicity", "ditau_deta", "n_taus", "Diphoton_lead_lepton_dphi", "LeadPhoton_pt_mgg", "MET_pt", "reco_MX_MET", "Diphoton_lead_lepton_dR", "ditau_met_dPhi", "b_jet_1_btagDeepFlavB", "sublead_lepton_id", "Diphoton_ditau_dR", "n_bjets", "Diphoton_lead_lepton_deta", "SubleadPhoton_sublead_lepton_dR", "category", "ditau_dR", "jet_1_btagDeepFlavB", "reco_MX", "jet_1_pt", "sublead_lepton_eta", "ditau_dphi", "diphoton_met_dPhi", "Diphoton_pt_mgg", "lead_lepton_eta", "reco_MggtauMET_mgg", "lead_lepton_met_dphi", "Diphoton_ditau_deta", "SubleadPhoton_pt_mgg", "sublead_lepton_pt", "SubleadPhoton_lead_lepton_dR", "n_iso_tracks", "Diphoton_dPhi", "LeadPhoton_lead_lepton_dR"]
train_features['important_20'] = ["Diphoton_helicity", "ditau_deta", "Diphoton_lead_lepton_deta", "category", "SubleadPhoton_sublead_lepton_dR", "SubleadPhoton_lead_lepton_dR", "LeadPhoton_lead_lepton_dR", "lead_lepton_mass", "SubleadPhoton_pt_mgg", "lead_lepton_pt", "ditau_mass", "LeadPhoton_pt_mgg", "MET_pt", "Diphoton_lead_lepton_dR", "ditau_dR", "Diphoton_pt_mgg", "reco_MX_MET", "ditau_pt", "ditau_met_dPhi", "Diphoton_ditau_dphi", "Diphoton_dPhi", "Diphoton_lead_lepton_dphi", "Diphoton_dR", "Diphoton_sublead_lepton_dR", "reco_MX", "ditau_dphi", "jet_1_pt", "LeadPhoton_ditau_dR", "reco_MggtauMET_mgg", "Diphoton_sublead_lepton_deta", "diphoton_met_dPhi", "dilep_leadpho_mass", "Diphoton_sublead_lepton_dphi", "Diphoton_ditau_deta"]
train_features['important_15'] = ["Diphoton_dR", "Diphoton_lead_lepton_dphi", "diphoton_met_dPhi", "SubleadPhoton_pt_mgg", "reco_MggtauMET_mgg", "ditau_dphi", "ditau_mass", "Diphoton_lead_lepton_dR", "lead_lepton_pt", "Diphoton_lead_lepton_deta", "LeadPhoton_pt_mgg", "SubleadPhoton_lead_lepton_dR", "Diphoton_ditau_deta", "reco_MX", "ditau_pt", "ditau_dR", "Diphoton_sublead_lepton_dphi", "dilep_leadpho_mass", "MET_pt", "Diphoton_pt_mgg", "LeadPhoton_lead_lepton_dR", "ditau_met_dPhi", "LeadPhoton_ditau_dR", "jet_1_pt", "category", "Diphoton_dPhi", "reco_MX_MET", "Diphoton_sublead_lepton_dR", "Diphoton_sublead_lepton_deta", "ditau_deta"]
train_features['important_16'] = ["LeadPhoton_ditau_dR", "dilep_leadpho_mass", "ditau_mass", "Diphoton_lead_lepton_deta", "ditau_dphi", "Diphoton_sublead_lepton_dR", "ditau_dR", "Diphoton_ditau_deta", "SubleadPhoton_lead_lepton_dR", "reco_MggtauMET_mgg", "reco_MX_MET", "Diphoton_dR", "category", "LeadPhoton_lead_lepton_dR", "lead_lepton_pt", "MET_pt", "Diphoton_pt_mgg", "reco_MX", "ditau_pt", "ditau_deta", "SubleadPhoton_pt_mgg", "Diphoton_dPhi", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "Diphoton_lead_lepton_dphi", "LeadPhoton_pt_mgg", "diphoton_met_dPhi", "Diphoton_sublead_lepton_deta", "Diphoton_sublead_lepton_dphi", "jet_1_pt"]
train_features['important_17'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_MET", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "Diphoton_lead_lepton_dphi", "ditau_pt", "category", "reco_MX", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "Diphoton_dR", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "reco_MggtauMET_mgg", "Diphoton_sublead_lepton_dphi", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['important_17_corr'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_MET", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "reco_MggtauMET_mgg", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['important_17_corr_no_mggtau'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_MET_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]

# same as important_17_corr_no_mggtau except change reco_MX_MET_mgg -> reco_MX_mgg (svfit has MET incorporated)
train_features['preapproval'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]

train_features['test_no_lead_lepton_mass'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_MET_mgg", "Diphoton_ditau_deta", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['test_no_the_rest'] = ["LeadPhoton_pt_mgg", "ditau_mass", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "Diphoton_pt_mgg", "reco_MX_MET_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]

train_features['svfit_investigation_all'] = ["LeadPhoton_pt_mgg", "ditau_mass", "ditau_mass_SVFit", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "ditau_dR_SVFit", "Diphoton_dPhi", "ditau_deta", "ditau_deta_SVFit", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_MET_mgg", "reco_MX_SVFit_mgg", "Diphoton_ditau_deta", "Diphoton_ditau_dR_SVFit", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "ditau_dphi_SVFit", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi", "Diphoton_ditau_dphi_SVFit"]
train_features['svfit_investigation_visible'] = train_features['important_17_corr_no_mggtau']
train_features['svfit_investigation_svfit'] = ["LeadPhoton_pt_mgg", "ditau_mass_SVFit", "SubleadPhoton_lead_lepton_dR", "ditau_dR_SVFit", "Diphoton_dPhi", "ditau_deta_SVFit", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_SVFit_mgg", "Diphoton_ditau_dR_SVFit", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi_SVFit", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi_SVFit"]
train_features['svfit_investigation_hybrid'] = ["LeadPhoton_pt_mgg", "ditau_mass_SVFit", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "ditau_dR_SVFit", "Diphoton_dPhi", "ditau_deta", "ditau_deta_SVFit", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_SVFit_mgg", "Diphoton_ditau_deta", "Diphoton_ditau_dR_SVFit", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "ditau_dphi_SVFit", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi", "Diphoton_ditau_dphi_SVFit"]

train_features['preapproval_half'] = train_features['preapproval'][:int(len(train_features['preapproval'])/2)]

# testing the mgg sculpting
# each row has one more feature removed
train_features['preapproval_no_MX'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['preapproval_no_dilep_leadpho_mass'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['preapproval_no_Diphoton_dPhi'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['preapproval_no_Diphoton_pt_mgg'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['preapproval_no_dilep_leadpho_mass_w_MX'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]

train_features['preapproval_dilep_leadpho_mass_mgg'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass_mgg", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['preapproval_dilep_leadpho_mass_rotate'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass_rotate", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]


latex_dict = {'LeadPhoton_pt_mgg': r'$p_T^{\gamma_1}/m_{\gamma\gamma}$', 'ditau_mass': r'$m_{\tau\tau}$', 'SubleadPhoton_lead_lepton_dR': r'$\Delta R(\gamma_2, \tau_1)$', 'ditau_dR': r'$\Delta R(\tau\tau)$', 'Diphoton_dPhi': r'$\Delta\phi(\gamma\gamma)$', 'ditau_deta': r'$\Delta \eta(\tau\tau)$', 'LeadPhoton_lead_lepton_dR': r'$\Delta R(\gamma_1, \tau_1)$', 'Diphoton_pt_mgg': r'$p_T^{\gamma\gamma}/m_{\gamma\gamma}$', 'reco_MX_MET_mgg': r'$m(\gamma\gamma\tau\tau \mathrm{MET})/m_{\gamma\gamma}$', 'Diphoton_ditau_deta': r'$\Delta \eta(\gamma\gamma, \tau\tau)$', 'lead_lepton_mass': r'$m_{\tau_1}$', 'diphoton_met_dPhi': r'$\Delta \phi(\gamma\gamma, \mathrm{MET})$', 'ditau_pt': r'$p_T^{\tau\tau}$', 'category': r'$\mathrm{category}$', 'Diphoton_sublead_lepton_dR': r'$\Delta R(\gamma\gamma, \tau_2)$', 'MET_pt': r'$p_T^\mathrm{MET}$', 'jet_1_pt': r'$p_T^{j_1}$', 'ditau_dphi': r'$\Delta \phi(\tau\tau)$', 'dilep_leadpho_mass': r'$m(\tau\tau\gamma_1)$', 'lead_lepton_pt': r'$p_T^{\tau_1}$', 'ditau_met_dPhi': r'$\Delta \phi(\tau\tau, \mathrm{MET})$', 'Diphoton_lead_lepton_dR': r'$\Delta R(\gamma\gamma, \tau_1)$', 'LeadPhoton_ditau_dR': r'$\Delta R(\gamma_1, \tau\tau)$', 'Diphoton_lead_lepton_deta': r'$\Delta \eta(\gamma\gamma, \tau_1)$', 'Diphoton_sublead_lepton_deta': r'$\Delta \eta(\gamma\gamma, \tau_2)$', 'SubleadPhoton_pt_mgg': r'$p_T^{\gamma_2}/m_{\gamma\gamma}$', 'Diphoton_ditau_dphi': r'$\Delta \phi(\gamma\gamma, \tau\tau)$', 'Diphoton_mass': r'$m_{\gamma\gamma}$',
              'b_jet_1_btagDeepFlavB':'Lead b-jet b-tag score', 'bdt_score':'BDT Score', 'dilep_subleadpho_mass': r'$m(\tau\tau\gamma_2)$', 'Diphoton_dR': r'$\Delta R(\gamma\gamma)$', 'Diphoton_eta': r'$\eta^{\gamma\gamma}$', 'Diphoton_helicity': r'$|\cos($helicity angle$(\gamma\gamma))|$', 'ditau_eta': r'$\eta^{\tau\tau}$', 'dR_ggtautau_SVFit': r'$\Delta R(\gamma\gamma, \tau\tau^\mathregular{SV Fit})$', 'dR_tautau_SVFit': r'$\Delta R(\tau\tau^{SV Fit})$', 'eta_tautau_SVFit_bdt': r'$\eta^{\tau\tau} \mathrm{SV Fit}$', 'gg_tt_CS': r'$\cos \theta^{CS} (\tau\tau^{SV Fit}, \gamma\gamma)$', 'gg_tt_hel': r'$|\cos($helicity angle$(\gamma\gamma, \tau\tau))|$', 'jet_1_btagDeepFlavB': 'Lead jet b-tag score', 'jet_1_eta': r'$\eta^{j_1}$', 'jet_2_btagDeepFlavB': 'Sublead jet b-tag score', 'jet_2_eta': r'$\eta^{j_2}$', 'jet_2_pt': r'$p_T^{j_2}$', 'lead_lepton_eta': r'$\eta^{\tau_1}$', 'lead_lepton_met_dphi': r'$\Delta\phi (\tau_1, \mathrm{MET})$', 'LeadPhoton_eta': r'$\eta^{\gamma_1}$', 'LeadPhoton_pixelSeed': r'$\gamma_1 \mathrm{Pixel Seed}$', 'm_tautau_SVFit': r'$m_{\tau\tau}^{SV Fit}$', 'MET_ll_dPhi': r'$\Delta\phi (\tau\tau, \mathrm{MET})$', 'mX': r'$m_X$', 'n_bjets': r'$N_{\mathrm{b-jets}}$', 'n_electrons': r'$N_e$', 'n_iso_tracks': r'$N_\mathrm{iso tracks}$', 'n_jets': r'$N_j$', 'n_muons': r'$N_\mu$', 'n_taus': r'$N_\tau$', 'pt_tautau_SVFit': r'$p_T^{\tau\tau} SV Fit}$', 'sublead_lepton_eta': r'$\eta^{\tau_2}$', 'sublead_lepton_pt': r'$p_T^{\tau_2}$', 'SubleadPhoton_eta':r'$\eta^{\gamma_2}$', 'SubleadPhoton_pixelSeed': r'$\gamma_2 \mathrm{pixel seed}$',
              'reco_MX_mgg': r'$m(\gamma\gamma\tau\tau) / m_{\gamma\gamma}$',
              'reco_MX': r'$m(\gamma\gamma\tau\tau)$',
              'reco_MggtauMET_mgg': r'$m(\gamma\gamma\tau\mathrm{MET}) / m_{\gamma\gamma}$',
              'Diphoton_lead_lepton_dphi': r'$\Delta\phi(\gamma\gamma, \tau_1)$',
              'Diphoton_sublead_lepton_dphi': r'$\Delta\phi(\gamma\gamma, \tau_2)$',
              'reco_MX_mgg': r'$m(\gamma\gamma\tau\tau) / m_{\gamma\gamma}$',}

import fnmatch
def expandSigProcs(sig_procs_to_expand):
  expanded_sig_procs = []
  for sig_proc in sig_procs_to_expand:
    expanded_to_add = list(filter(lambda string: fnmatch.fnmatch(string, sig_proc), sig_procs["all"]))
    if len(expanded_to_add) == 0:
      expanded_sig_procs.append(sig_proc)
    else:
      expanded_sig_procs.extend(expanded_to_add)
    
  return expanded_sig_procs

def expandColumns(columns):
  expanded_columns = []
  for col in columns:
    if col in train_features:
      expanded_columns.extend(train_features[col])
    else:
      expanded_columns.append(col)
  return expanded_columns

def get_MX_MY(sig_proc):
  if "NMSSM" in sig_proc:
    split_name = sig_proc.split("_")
    MX = float(split_name[7])
    MY = float(split_name[9])
  elif "XToHHggTauTau" in sig_proc:
    MX = float(sig_proc.split("_")[1][1:])
    MY = 125.0
  elif "HHTo2G" in sig_proc:
    MX = float(sig_proc.split("-")[1])
    MY = 125.0
  else:
    raise Exception("Unexpected signal process: %s"%sig_proc)
  return MX, MY

def get_sig_proc(example_sig_proc, mx, my):
  if "XToHHggTauTau" in example_sig_proc:
    return "XToHHggTauTau_M%d"%mx
  elif "NMSSM_XYH_Y_gg_H_tautau" in example_sig_proc:
    return "NMSSM_XYH_Y_gg_H_tautau_MX_%d_MY_%d"%(mx, my)
  elif "NMSSM_XYH_Y_tautau_H_gg" in example_sig_proc:
    return "NMSSM_XYH_Y_tautau_H_gg_MX_%d_MY_%d"%(mx, my)
  elif "GluGluToBulkGravitonToHHTo2G2Tau_M" in example_sig_proc:
    return "GluGluToBulkGravitonToHHTo2G2Tau_M-%d"%mx
  elif "GluGluToRadionToHHTo2G2Tau_M" in example_sig_proc:
    return "GluGluToRadionToHHTo2G2Tau_M-%d"%mx

def add_MX_MY(df, proc_dict):
  #get sig_procs in parquet file
  sig_procs_in = [sig_proc for sig_proc in sig_procs["all"] if sig_proc in proc_dict.keys()]

  df["MX"] = np.single(dummy_val)
  df["MY"] = np.single(dummy_val)
  
  for sig_proc in sig_procs_in:
    MX, MY = get_MX_MY(sig_proc)
    
    df.loc[df.process_id==proc_dict[sig_proc], "MX"] = MX
    df.loc[df.process_id==proc_dict[sig_proc], "MY"] = MY

def parserToList(args):
  names = list(filter(lambda x: x[0] != "_", dir(args)))
  #print(names)
  l = []
  for name in names:
    value = getattr(args, name)
    if type(value) in [list, tuple]:
      if len(value) > 0: l.append("--%s %s"%(name.replace("_", "-"), " ".join([str(v) for v in value])))
    elif value is True:
      l.append("--%s"%name.replace("_", "-"))
    elif type(value) in [str, float, int]:
      l.append("--%s %s"%(name.replace("_", "-"), value))
    elif value is False:
      pass
    elif value is None:
      pass
    else:
      print("Dont know what to do with %s of type %s"%(name, type(value)))

  l_split = []
  for each in l:
    l_split.extend(each.split(" "))
  return l_split

def submitToBatch(argv, extra_memory=False, quick=False):
  COMMAND = argv
  COMMAND.remove("--batch")
  COMMAND = " ".join(COMMAND)
  
  print('\033[93m')
  if LOW_MASS_MODE and ("low_mass" not in COMMAND.lower()):
    print("WARNING: running in LOW_MASS_MODE but no low mass directories found in commnad")
  elif ("low_mass" in COMMAND.lower()) and not LOW_MASS_MODE:
    print("WARNING: low mass directories referred to but LOW_MASS_MODE is not activated")
  print('\033[0m')

  PWD = os.getcwd()

  with open("batch_template.sh", "r") as f:
    template = f.read()
  filled_template = template % {
    "PWD": PWD,
    "COMMAND": COMMAND,
    "LOW_MASS_MODE": LOW_MASS_MODE
  }

  os.makedirs("batch", exist_ok=True)
  
  script_name = argv[0].replace("/", "_").replace(".", "_")
  existing_files = list(filter(lambda x: (script_name in x) and (x[-3:] == ".sh"), os.listdir("batch")))

  script_path = os.path.join(os.getcwd(), "batch", "%s_%d.sh"%(script_name, len(existing_files)+1))
  with open(script_path, "w") as f:
    f.write(filled_template)

  if type(extra_memory) == bool:
    if extra_memory: slots = 2
    else:            slots = 1
  else:
    slots = extra_memory

  #if not quick:  submit_command = f"qsub -q hep.q -l h_vmem=24G -l h_rt=3:0:0 -pe hep.pe {slots} -o {script_path}.out -e {script_path}.err {script_path}"
  #else:          submit_command = f"qsub -q hep.q -l h_vmem=24G -l h_rt=600 -o {script_path}.out -e {script_path}.err {script_path}"
  #if not quick:  submit_command = f"qsub -q gpu.q -l h_vmem=24G -l h_rt=3:0:0 -pe hep.pe {slots} -o {script_path}.out -e {script_path}.err {script_path}"
  #else:          submit_command = f"qsub -q gpu.q -l h_vmem=24G -l h_rt=600 -o {script_path}.out -e {script_path}.err {script_path}"
  #print(submit_command)
  #os.system(submit_command)

  sub = {
          "executable": f"{script_path}",
          "max_retries": "1",
          "log": f"{script_path}.log",
          "output": f"{script_path}.out",
          "error": f"{script_path}.err",
          "+MaxRuntime": 3600,
          "+OS": '"EL7"',
          "request_memory": "8G",
          "request_cpus" : slots,
  }
  with open(f"{script_path}.sub", "w") as f:
    lines = [f"{key} = {value}" for key, value in sub.items()]
    lines.append("queue")
    f.write("\n".join(lines))
  
  #schedd = htcondor.Schedd()
  #schedd.submit(sub)

from pyarrow.parquet import ParquetFile
def getColumns(parquet_file):
  pf = ParquetFile(parquet_file) 
  columns = [each.name for each in pf.schema]
  if "__index_level_0__" in columns:
    columns.remove("__index_level_0__")
  return columns

import pyarrow
def getTestSample(parquet_input, size=100, columns=None):
  pf = pyarrow.parquet.ParquetFile(parquet_input) 
  iter = pf.iter_batches(batch_size = size, columns=columns)
  first_rows = next(iter) 
  return pyarrow.Table.from_batches([first_rows]).to_pandas()

import json
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def getAllMasses(optim_results):
  return [get_MX_MY(optim_result["sig_proc"]) for optim_result in optim_results]

def getMassesToRun(mass_points, optim_results):
  if mass_points is not None:
    masses = []
    new_optim_results = []

    for mass_point in mass_points:
      MX, MY = mass_point.split(",")
      MX = float(MX)
      MY = float(MY)
      masses.append((MX, MY))

      for optim_result in optim_results:
        if get_MX_MY(optim_result["sig_proc"]) == (MX, MY):
          new_optim_results.append(optim_result)

    optim_results = new_optim_results

  else:
    masses = getAllMasses(optim_results)

  return masses, optim_results

def filterOptimResults(optim_results, mass_points):
  new_optim_results = []
  for mass_point in mass_points:
    MX, MY = mass_point.split(",")
    MX = float(MX)
    MY = float(MY)

    for optim_result in optim_results:
      if get_MX_MY(optim_result["sig_proc"]) == (MX, MY):
        new_optim_results.append(optim_result)
  optim_results = new_optim_results
  return optim_results