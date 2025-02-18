import pandas as pd
import sys
import numpy as np

columns = ["process_id", "event", "Diphoton_mass"]

select = lambda df, i: df[df.process_id==i]

df1 = select(pd.read_parquet(sys.argv[1], columns=columns), int(sys.argv[3]))
df2 = select(pd.read_parquet(sys.argv[2], columns=columns), int(sys.argv[3]))

# print("len(df1)=%d"%len(df1))
# print("len(df2)=%d"%len(df2))
# n_overlap = sum(np.isin(df1.event, df2.event))
# percentage = (n_overlap / len(df1))*100
# print("Overlapping events: %d, %.2f%%"%(n_overlap, percentage))

print(df1.Diphoton_mass.mean(), df2.Diphoton_mass.mean(), df2.Diphoton_mass.mean()-df1.Diphoton_mass.mean())
print(df1.Diphoton_mass.std(), df2.Diphoton_mass.std(), df2.Diphoton_mass.std()-df1.Diphoton_mass.std())
