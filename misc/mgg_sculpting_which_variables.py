import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import common
import argparse
import scipy.stats as sps

def main(args):
  columns_to_read = ["Diphoton_mass", "weight_central", "process_id"] + common.train_features[args.features]
  df = pd.read_parquet(args.parquet_input, columns=columns_to_read)
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  #bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["Diphoton"]]
  #bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"]]
  # df = df[df.process_id.isin(bkg_ids)]

  #all background and sampled according to exp n
  dfs = []
  for bkg_proc in common.bkg_procs["all"]:
    bkg_df = df[df.process_id == proc_dict[bkg_proc]]
    bkg_df = bkg_df.sample(int(bkg_df.weight_central.sum()), replace=True)
    print(len(bkg_df))
    dfs.append(bkg_df)
  df = pd.concat(dfs)

  #df = df[df.process_id == proc_dict["NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_300"]]

  # for col in common.train_features[args.features]:
  #   col_quant = df[col].quantile((0.01, 0.99))
  #   plt.hist2d(df.Diphoton_mass, df[col], weights=df.weight_central, range=((100, 500), col_quant))
  #   plt.ylabel(col)
  #   plt.xlabel("Diphoton mass")
  #   plt.savefig(f"corr_{col}.png")
  #   plt.clf()

  for col in common.train_features[args.features]:
    print(col)
    col_quant = list(df.loc[df[col] != -9, col].quantile(np.linspace(0.01, 0.99, 6)))
    mass_edges = np.linspace(100, 500, 40)
    # plt.hist2d(df.Diphoton_mass, df[col], weights=df.weight_central, bins=(mass_edges, col_quant))
    # plt.ylabel(col)
    
    # print(col_quant)
    # for i in range(len(col_quant)-1):
    #   df_quant = df[(df[col]>col_quant[i]) & (df[col]<col_quant[i+1])]
    #   plt.hist(df_quant.Diphoton_mass, range=(100, 400), bins=50, histtype="step", label=f"Quant {i}")

    # plt.xlabel("Diphoton mass")
    # plt.legend()
    # plt.title(col)

    df_less = df[df[col] != -9].iloc[:]
    # df_less = df_less[(df_less[col]>col_quant[0])&(df_less[col]<col_quant[-1])]
    # if len(df_less) == 0:
    #   df_less = df[df[col] != -9].iloc[:1000]

    # if col == "dilep_leadpho_mass":
    #   #df_less.loc[:, col] /= df_less.Diphoton_mass
    #   df_less.loc[:, col] -= 0.638 * df_less.Diphoton_mass

    plt.scatter(df_less.Diphoton_mass, df_less[col])
    plt.xlabel("Diphoton mass")
    plt.ylabel(col)
    p = np.polyfit(df_less.Diphoton_mass, df_less[col], 1)
    print(p)
    plt.plot(mass_edges, np.polyval(p, mass_edges), label=f"r= {sps.pearsonr(df_less.Diphoton_mass, df_less[col])}")
    #plt.xlim(100, 200)
    plt.legend()
    plt.savefig(f"corr_{col}.png")
    plt.clf()

  plt.scatter(df[(df.reco_MX_mgg!=-9)].reco_MX_mgg, df[(df.reco_MX_mgg!=-9)].Diphoton_mass)
  plt.savefig(f"corr_MX_mgg.png")

  


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--features', '-f', type=str, required=True)
  
  args = parser.parse_args()
  
  df = main(args)