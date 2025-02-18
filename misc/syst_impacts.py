import sys
import pandas as pd
import numpy as np
import common

columns = common.getColumns(sys.argv[1])
columns = list(filter(lambda x: "weight" in x, columns)) + ["process_id"]

df = pd.read_parquet(sys.argv[1], columns=columns)
print(df)
#df = df[df.process_id < 0]
df = df[(df.process_id >= 5) & (df.process_id <= 44)]

syst_columns = np.array(list(filter(lambda x: "_up" in x, df.columns)))

impacts = []
for col in syst_columns:
  mean_variations = []
  for proc in df.process_id.unique():
    up = abs(1 - df.loc[df.process_id==proc, col].mean())
    #print(proc, up)
    down = abs(1 - df.loc[df.process_id==proc, col.replace("_up", "_down")].mean())
    mean_variations.append((up+down)/2)

  impacts.append([min(mean_variations), max(mean_variations)])

impacts = np.array(impacts)
s = np.argsort(impacts[:,1])[::-1]

for i, col in enumerate(syst_columns[s]):
  syst_name = col.replace("weight_", "").replace("_up", "")
  print("\n" + syst_name)
  print(f"Min/max variation over processes: {impacts[s][i][0]:.3f}/{impacts[s][i][1]:.3f}")



