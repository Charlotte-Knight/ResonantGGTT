import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import common

df = pd.read_parquet(sys.argv[1], columns=["LeadPhoton_pt_mgg", "SubleadPhoton_pt_mgg", "weight_central", "process_id", "Diphoton_mass"])
with open(sys.argv[2], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

#check mx=300 signals
plt.hist(df[df.process_id==proc_dict["Data"]].LeadPhoton_pt_mgg, range=(0,2), bins=50, label="data", histtype="step")
plt.hist(df[df.process_id==proc_dict["NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70"]].LeadPhoton_pt_mgg, range=(0,2), bins=50, label="mx_300_my_70", histtype="step")
plt.hist(df[df.process_id==proc_dict["NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_150"]].LeadPhoton_pt_mgg, range=(0,2), bins=50, label="mx_300_my_150", histtype="step")
plt.hist(df[df.process_id==proc_dict["NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_600"]].LeadPhoton_pt_mgg, range=(0,2), bins=50, label="mx_1000_my_600", histtype="step")
plt.plot([0.33, 0.33],plt.ylim())
plt.legend()
plt.savefig("pt_mgg_lead.png")
plt.clf()
plt.hist(df[df.process_id==proc_dict["Data"]].SubleadPhoton_pt_mgg, range=(0,2), bins=50, label="data", histtype="step")
plt.hist(df[df.process_id==proc_dict["NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70"]].SubleadPhoton_pt_mgg, range=(0,2), bins=50, label="mx_300_my_70", histtype="step")
plt.hist(df[df.process_id==proc_dict["NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_150"]].SubleadPhoton_pt_mgg, range=(0,2), bins=50, label="mx_300_my_150", histtype="step")
plt.hist(df[df.process_id==proc_dict["NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_600"]].SubleadPhoton_pt_mgg, range=(0,2), bins=50, label="mx_1000_my_600", histtype="step")
plt.plot([0.25, 0.25],plt.ylim())
plt.legend()
plt.savefig("pt_mgg_sublead.png")
plt.clf()

s = (df.Diphoton_mass < 80)
plt.hist(df[s & df.process_id==proc_dict["Data"]].LeadPhoton_pt_mgg, range=(0,2), bins=50, label="lead", histtype="step")
plt.hist(df[s & df.process_id==proc_dict["Data"]].SubleadPhoton_pt_mgg, range=(0,2), bins=50, label="sublead", histtype="step")
plt.legend()
plt.savefig("pt_mgg_in_mgg.png")
plt.clf()


plt.hist(df[df.process_id==proc_dict["Data"]].Diphoton_mass, range=(55, 155), bins=100, histtype="step", label="unselected")
s = (df.LeadPhoton_pt_mgg > 0.2) & (df.SubleadPhoton_pt_mgg > 0.2)
plt.hist(df[s & (df.process_id==proc_dict["Data"])].Diphoton_mass, range=(55, 155), bins=100, histtype="step", label="selected")
plt.savefig("pt_mgg_mgg.png")