import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (10,8)

import argparse
import pandas as pd
import json 
import numpy as np

def loadDataFrame(args):
  df = pd.read_parquet(args.parquet_input)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  data = df[df.process_id == proc_dict["Data"]]
  sig = df[df.process_id == proc_dict[args.sig_proc]]
  bkg = df[(df.y==0) & (df.process_id != proc_dict["Data"])]

  return data, sig, bkg, proc_dict

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--sig-proc', '-p', type=str, required=True)
  parser.add_argument('--pres', type=float, default=(100,180), nargs=2)
  parser.add_argument('--sr', type=float, default=(115,135), nargs=2)
  parser.add_argument('--score', type=str, default="score")
  args = parser.parse_args()

  args.score = "%s_%s"%(args.score, args.sig_proc)

  data, sig, bkg, proc_dict = loadDataFrame(args)
  data = data[(data.Diphoton_mass>args.pres[0])&(data.Diphoton_mass<args.pres[1])]
  data = data[(data.Diphoton_mass<args.sr[0])|(data.Diphoton_mass>args.sr[1])]

  data.sort_values(args.score, ascending=False, inplace=True)
  N = 200
  data = data.iloc[:N]

  nbkg = np.arange(0, N, 1)
  nsig = [sig[sig[args.score]>data.iloc[i][args.score]].weight.sum() for i in range(N)]

  #print(nbkg, nsig)

  plt.plot(nbkg, nsig)
  plt.savefig("sig_eff_vs_nbkg.png")