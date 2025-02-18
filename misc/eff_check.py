import pandas as pd
import sys
import json
import numpy as np

df = pd.read_parquet(sys.argv[1])
with open(sys.argv[2]) as f:
  proc_dict = json.load(f)['sample_id_map']

for proc in proc_dict.keys():
  print(proc, df[df.process_id==proc_dict[proc]].weight_central.sum())

