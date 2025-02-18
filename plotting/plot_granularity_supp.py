import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
import json
import numpy as np
import os
import pandas as pd

def loadMasses(indir):
  with open(os.path.join(indir, "all_masses_0p1.json"), "r") as f:
    all_masses = np.array(json.load(f))
  
  with open(os.path.join(indir, "extra_masses_0p1.json"), "r") as f:
    extra_masses = np.array(json.load(f))
  
  return all_masses, extra_masses

def plotXHH(indir):
  all_masses, extra_masses = loadMasses(indir)
  plt.figure(figsize=(10,4))
  plt.subplots_adjust(left=0.05, right=0.95, bottom=0.3, top=0.8)
  mplhep.cms.label(llabel="Supplementary", loc=0)
  plt.scatter(all_masses[:,0], all_masses[:,1], marker=".", label="Nominal Mass Points")
  plt.scatter(extra_masses[:,0], extra_masses[:,1], marker=".", label="Intermediate Mass Points")
  plt.legend(ncol=2)
  plt.xlabel(r"$m_X$ [GeV]")
  plt.tick_params(left=False, right=False, labelleft=False)
  plt.tick_params(which="minor", left=False, right=False, labelleft=False)
  #plt.tight_layout()
  plt.savefig(os.path.join(indir, "supplementary.png"))
  plt.savefig(os.path.join(indir, "supplementary.pdf"))

def plotXYH(indir):
  all_masses, extra_masses = loadMasses(indir)
  plt.figure(figsize=(10,9))
  mplhep.cms.label(llabel="Supplementary", loc=0)
  plt.scatter(all_masses[:,0], all_masses[:,1], marker=".", label="Nominal Mass Points")
  plt.scatter(extra_masses[:,0], extra_masses[:,1], marker=".", label="Intermediate Mass Points")
  plt.xlabel(r"$m_X$ [GeV]")
  plt.ylabel(r"$m_Y$ [GeV]")
  
  if "Low_Mass" in indir:
    plt.ylim(top=145)
  plt.legend(loc="upper left")
  
  plt.tight_layout()
  plt.savefig(os.path.join(indir, "supplementary.png"))
  plt.savefig(os.path.join(indir, "supplementary.pdf")) 

def getNumResults(csv_file):
  df = pd.read_csv(csv_file)
  return len(df)

plotXHH("Outputs/Radion/LimitGranularity")
plotXHH("Outputs/Graviton/LimitGranularity")
plotXYH("Outputs/Y_tautau/LimitGranularity")
plotXYH("Outputs/Y_gg_Low_Mass/LimitGranularity")
plotXYH("Outputs/Y_gg_High_Mass/LimitGranularity")

print(f"Radion: {getNumResults('Outputs/Radion_Feb24/LimitPlots_Toys_May30_filtered/Limits_xs/limits.csv')}")
print(f"Graviton: {getNumResults('Outputs/Graviton_Feb24/LimitPlots_Toys_May30_filtered/Limits_xs/limits.csv')}")
print(f"Y_tautau: {getNumResults('Outputs/Y_tautau_Feb24/LimitPlots_Toys_May30_filtered/Limits_xs/limits.csv')}")
print(f"Y_gg_Low_Mass: {getNumResults('Outputs/Y_gg_Low_Mass_Feb24/LimitPlots_Toys_May30_filtered/Limits_xs/limits.csv')}")
print(f"Y_gg_High_Mass: {getNumResults('Outputs/Y_gg_High_Mass_Feb24/LimitPlots_Toys_May30_filtered/Limits_xs/limits.csv')}")